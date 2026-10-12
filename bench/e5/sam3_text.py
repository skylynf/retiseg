"""SAM 3 text prompts: the third model of the E5 zero-shot tier a (design/15 section 5.4; e5_frozen.yaml zero_shot.sam3).

Three phrases per class are registered. One run is one phrase index (0, 1, 2)
over the four classes, one set, split and canvas variant: class c gets the map
of phrases[c][index]. Per image: canvas -> G1 windows (1008, overlap 0.5) ->
processor.set_image(PIL window) once per window -> set_text_prompt once per
class -> pixel score = max over the kept queries of query score x mask
probability (the query score is already probability x presence,
sam3_image_processor.py) -> canvas by max -> paste -> write_prob. The
confidence threshold is 0 so every query stays; the default 0.5 would cut the
PR curve. Sources a_sam3_ph<index>; the main result a_sam3_selected is
assembled from the per-class phrases chosen on the validation pool
(zero_shot.sam3.selected), copying maps, no new inference.

Phrase selection (D2): select_phrases pools the pixel histograms of the pool
runs of every phrase index and takes, per class, the phrase with the highest
AUPR; it prints the yaml to paste into e5_frozen.yaml.

Environment: retiseg-sam3 (Python 3.12, torch >= 2.7, CUDA >= 12.6;
bench/scripts/cluster_env.sh install_sam3). SAM 3 is under the SAM License.

    python -m bench.e5.sam3_text --phrase 0 --set DDR --split val
    python -m bench.e5.sam3_text --select                      # after the pool runs of the three indices
    python -m bench.e5.sam3_text --assemble --set IDRiD --split test
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES, load_split
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.e5 import outputs as out
from bench.e5.amg import write_maps
from bench.models import foundation

TIER = "a"
MODEL = "SAM 3"
PHRASE_INDICES = (0, 1, 2)
KIND_TEXT, KIND_SELECTED = "text", "text_selected"
SELECTED = "selected"


def spec(frozen):
    return frozen["zero_shot"]["sam3"]


def phrases_for(frozen, index):
    index = int(index)
    if index not in PHRASE_INDICES:
        raise ValueError(f"phrase index must be one of {PHRASE_INDICES}, got {index}")
    return {cls: spec(frozen)["phrases"][cls][index] for cls in LESION_CLASSES}


def source_name(index):
    return out.source_name(TIER, MODEL, f"ph{int(index)}")


def selected_source():
    return out.source_name(TIER, MODEL, SELECTED)


class Sam3Scorer:
    """Sam3Processor behind a two-call interface: set_image(PIL) once per window, score(phrase) per class."""

    def __init__(self, processor):
        self.processor = processor
        self.state = None

    def set_image(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("SAM 3 takes a PIL image (set_image misreads the size of a numpy array, design/15 section 5.4)")
        self.state = self.processor.set_image(image)

    def score(self, phrase):
        state = self.processor.set_text_prompt(prompt=phrase, state=self.state)
        masks, scores = state["masks_logits"], state["scores"]  # (N, 1, H, W) probabilities, (N,)
        if scores.numel() == 0:
            return np.zeros((state["original_height"], state["original_width"]), dtype=np.float32)
        prob = (masks[:, 0] * scores.view(-1, 1, 1)).amax(dim=0)
        return np.clip(prob.float().cpu().numpy(), 0.0, 1.0)


def build_scorer(frozen, device=None, window=None):
    """The official model and processor at the registered confidence threshold; returns (scorer, record)."""
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sam3 = spec(frozen)
    checkpoint = foundation.weights(sam3["checkpoint"])
    resolution = int(window if window is not None else geo.window_size(frozen, MODEL))
    with foundation.official("sam3", purge=("sam3",)):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, device=device)
        processor = Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=float(sam3["confidence_threshold"]))
    record = {
        "checkpoint": out.checkpoint_record(checkpoint),
        "device": str(device),
        "resolution": resolution,
        "confidence_threshold": float(sam3["confidence_threshold"]),
        "pixel_score": sam3["pixel_score"],
        "license": "SAM License (official_code/sam3/README.md)",
    }
    return Sam3Scorer(processor), record


def window_scores(scorer, cv, window, boxes, phrases):
    """{cls: canvas map} of one image: one set_image per window, one text prompt per class; returns (maps, seconds per window)."""
    maps = {cls: np.zeros(cv.hw, dtype=np.float32) for cls in LESION_CLASSES}
    seconds = []
    for top, left in boxes:
        if window is None:
            piece, valid = cv.image, cv.hw
        else:
            piece, valid = geo.crop_window(cv.image, top, left, window)
        started = time.perf_counter()
        scorer.set_image(Image.fromarray(np.ascontiguousarray(piece)))
        rows, cols = slice(top, top + valid[0]), slice(left, left + valid[1])
        for cls in LESION_CLASSES:
            prob = np.asarray(scorer.score(phrases[cls]), dtype=np.float32)[: valid[0], : valid[1]]
            np.maximum(maps[cls][rows, cols], np.clip(prob, 0.0, 1.0), out=maps[cls][rows, cols])
        seconds.append(time.perf_counter() - started)
    return maps, seconds


def _ids(data_dir, split, limit):
    all_ids = load_split(data_dir, split)
    ids = all_ids if limit is None else all_ids[: int(limit)]
    return ids, all_ids[len(ids):]


def planned(frozen, index, set_name, split, variant=geo.MAIN):
    """(source, directory) of one phrase run."""
    source = source_name(index)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if fz.is_pool_split(split) else set_name
    return source, out.out_dir(TIER, source, out.set_dirname(base, canvas_variants))


def planned_selected(set_name, split, variant=geo.MAIN):
    """The assembled a_sam3_selected directory of a set and split."""
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if fz.is_pool_split(split) else set_name
    return out.out_dir(TIER, selected_source(), out.set_dirname(base, canvas_variants))


def run(index, set_name, split, variant=geo.MAIN, geometry=geo.G1, device=None, scorer=None, frozen=None, data_dir=None,
        long_side=None, window=None, evaluate=True, drop_maps=(), limit=None, pool=None, assemble=True):
    """Predict (and evaluate) one phrase index; assembles the selected source when the selection is filled and the three indices exist."""
    frozen = frozen or fz.load_frozen()
    if geometry not in geo.GEOMETRIES:
        raise ValueError(f"geometry must be one of {geo.GEOMETRIES}, got {geometry!r}")
    fz.assert_ready(frozen, TIER, split, model=MODEL)
    phrases = phrases_for(frozen, index)
    source, directory = planned(frozen, index, set_name, split, variant)
    pool_run = fz.is_pool_split(split)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if pool_run else set_name
    data_dir = Path(data_dir) if data_dir is not None else geo.data_dir(set_name)
    if long_side is None:
        long_side = geo.NATIVE if geometry == geo.G2 else geo.long_side_for(set_name, variant, frozen)
    else:
        geo.long_side_for(set_name, variant, frozen)
    window_size = None if geometry == geo.G2 else int(window if window is not None else geo.window_size(frozen, MODEL))
    ids, excluded = _ids(data_dir, split, limit)
    if out.needs_prediction(directory, limit):
        if scorer is None:
            scorer, record = build_scorer(frozen, device, window_size)
        else:
            record = {"checkpoint": None, "device": str(device), "scorer": type(scorer).__name__,
                      "confidence_threshold": float(spec(frozen)["confidence_threshold"]), "pixel_score": spec(frozen)["pixel_score"]}
        n_windows, seconds = 0, []
        started = time.perf_counter()
        for position, image_id in enumerate(ids):
            cv = geo.canvas(data_dir, image_id, long_side, masks=False)
            boxes = [(0, 0)] if window_size is None else geo.windows(cv.hw, window_size, geo.overlap(frozen))
            maps, window_seconds = window_scores(scorer, cv, window_size, boxes, phrases)
            n_windows += len(boxes)
            seconds.extend(window_seconds)
            write_maps(out.pred_dir(directory, split), image_id, cv, maps)
            if position % 20 == 0:
                print(f"{source} {base} {position + 1}/{len(ids)} windows {len(boxes)}", flush=True)
        out.write_predicted(
            directory,
            {
                "tier": TIER, "model": MODEL, "kind": KIND_TEXT, "source": source, "set": set_name, "split": split, "variant": variant,
                "variants": canvas_variants, "geometry": geometry, "phrase_index": int(index), "phrases": phrases,
                "canvas_long_side": "native" if long_side == geo.NATIVE else int(long_side),
                "window": window_size, "overlap": None if window_size is None else geo.overlap(frozen),
                "n_images": len(ids), "limit": limit, "n_windows": n_windows,
                "predict_seconds": round(time.perf_counter() - started, 1),
                "seconds_per_window_mean": round(float(np.mean(seconds)), 3) if seconds else None,
                **record, **out.environment(),
            },
        )
    assembled = None
    if assemble and _selection_filled(frozen):
        assembled = assemble_selected(set_name, split, variant, frozen, data_dir, evaluate=False, limit=limit)
    if evaluate:
        from bench.eval.evaluate import load_protocol

        proto = load_protocol(fz.REPO / "bench" / "eval" / "protocol.yaml")
        for path, src in [(directory, source)] + ([(assembled, selected_source())] if assembled is not None else []):
            if out.is_done(path, limit):
                continue
            predicted_limit, excluded_eval = out.evaluation_scope(path, ids + excluded)
            thresholds = None if pool_run else out.thresholds_for_test(TIER, src, frozen, proto, pool=pool, limit=predicted_limit)
            out.evaluate_dir(path, set_name, split, data_dir=data_dir, proto=proto, val_thresholds=thresholds, drop_maps=drop_maps, exclude=excluded_eval)
    return {"source": source, "directory": str(directory), "assembled": None if assembled is None else str(assembled), "n_images": len(ids)}


def _selection_filled(frozen):
    try:
        fz.selected_phrases(frozen)
    except ValueError:
        return False
    return True


def selected_indices(frozen):
    """{cls: phrase index} of the phrases chosen on the pool."""
    chosen = fz.selected_phrases(frozen)
    return {cls: spec(frozen)["phrases"][cls].index(chosen[cls]) for cls in LESION_CLASSES}


def assemble_selected(set_name, split, variant=geo.MAIN, frozen=None, data_dir=None, evaluate=True, drop_maps=(), pool=None, limit=None):
    """Copy, per class, the maps of the selected phrase's run into a_sam3_selected; None while an index is missing."""
    frozen = frozen or fz.load_frozen()
    indices = selected_indices(frozen)
    pool_run = fz.is_pool_split(split)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    set_dir = out.set_dirname(out.pool_set_dirname(set_name) if pool_run else set_name, canvas_variants)
    sources = {cls: out.out_dir(TIER, source_name(indices[cls]), set_dir) for cls in LESION_CLASSES}
    for path in sources.values():
        if not (path / "predicted.json").is_file() or not out.pred_dir(path, split).is_dir():
            return None  # that phrase index has not run (or its maps are gone) for this set
    records = {cls: out.read_predicted(path) for cls, path in sources.items()}
    limits = {records[cls].get("limit") for cls in LESION_CLASSES}
    if len(limits) != 1:
        raise ValueError(f"the phrase runs of {set_dir} were made with different --limit values {limits}; redo them alike")
    run_limit = limits.pop()
    target = planned_selected(set_name, split, variant)
    if not out.needs_prediction(target, run_limit):
        return target
    maps_dir = out.pred_dir(target, split)
    for cls in LESION_CLASSES:
        source_maps = out.pred_dir(sources[cls], split) / cls
        shutil.copytree(source_maps, maps_dir / cls, dirs_exist_ok=True)
    first = records["MA"]
    out.write_predicted(
        target,
        {
            "tier": TIER, "model": MODEL, "kind": KIND_SELECTED, "source": selected_source(), "set": set_name, "split": split,
            "variant": variant, "variants": canvas_variants, "geometry": first.get("geometry"),
            "phrases": fz.selected_phrases(frozen), "phrase_index": indices,
            "assembled_from": {cls: str(sources[cls]) for cls in LESION_CLASSES},
            "canvas_long_side": first.get("canvas_long_side"), "window": first.get("window"), "overlap": first.get("overlap"),
            "n_images": first.get("n_images"), "limit": run_limit, "checkpoint": first.get("checkpoint"),
            "confidence_threshold": first.get("confidence_threshold"), "pixel_score": first.get("pixel_score"), **out.environment(),
        },
    )
    if evaluate:
        from bench.eval.evaluate import load_protocol

        proto = load_protocol(fz.REPO / "bench" / "eval" / "protocol.yaml")
        data_dir = Path(data_dir) if data_dir is not None else geo.data_dir(set_name)
        _ids_done, excluded = _ids(data_dir, split, run_limit)
        thresholds = None if pool_run else out.thresholds_for_test(TIER, selected_source(), frozen, proto, pool=pool, limit=run_limit)
        out.evaluate_dir(target, set_name, split, data_dir=data_dir, proto=proto, val_thresholds=thresholds, drop_maps=drop_maps, exclude=excluded)
    return target


def select_phrases(frozen=None, proto=None, pool=None):
    """Per class, the phrase index with the highest pooled AUPR on the pool runs; returns (chosen, table).

    chosen: {cls: {"index", "phrase", "aupr"}}; table: {cls: [aupr of index 0, 1, 2]} (None where a run is missing).
    """
    from bench.eval.evaluate import load_protocol

    frozen = frozen or fz.load_frozen()
    proto = proto if proto is not None else load_protocol(fz.REPO / "bench" / "eval" / "protocol.yaml")
    table = {cls: [] for cls in LESION_CLASSES}
    for index in PHRASE_INDICES:
        auprs = out.pool_class_aupr(TIER, source_name(index), frozen, proto, pool=pool)
        for cls in LESION_CLASSES:
            table[cls].append(None if auprs is None else auprs[cls])
    chosen = {}
    for cls in LESION_CLASSES:
        values = table[cls]
        if any(v is None for v in values):
            raise FileNotFoundError(f"the pool runs of every phrase index are needed before selecting; {cls} has {values}")
        best = int(np.argmax(values))  # ties to the lowest index
        chosen[cls] = {"index": best, "phrase": spec(frozen)["phrases"][cls][best], "aupr": float(values[best])}
    return chosen, table


def format_selected_yaml(chosen, table):
    """The lines to paste under zero_shot.sam3 in e5_frozen.yaml (selected: {...}) with the pool AUPRs as comments."""
    lines = ["    selected:  # chosen on the validation pool by per-class AUPR (bench/e5/sam3_text.py --select)"]
    for cls in LESION_CLASSES:
        auprs = ", ".join(f"{v:.4f}" for v in table[cls])
        lines.append(f"      {cls}: {chosen[cls]['phrase']}   # index {chosen[cls]['index']}; pool AUPR per index {auprs}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench.e5.sam3_text", description="SAM 3 text prompts on one set and split.")
    parser.add_argument("--phrase", type=int, choices=PHRASE_INDICES, help="phrase index of every class")
    parser.add_argument("--select", action="store_true", help="choose the phrases on the validation pool and print the yaml")
    parser.add_argument("--assemble", action="store_true", help="assemble a_sam3_selected for --set / --split from the phrase runs")
    parser.add_argument("--set")
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default=geo.MAIN)
    parser.add_argument("--geometry", default=geo.G1, choices=geo.GEOMETRIES)
    parser.add_argument("--device")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--drop-maps")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    drop = [item.strip() for item in (args.drop_maps or "").split(",") if item.strip()]
    if args.select:
        chosen, table = select_phrases()
        print(format_selected_yaml(chosen, table))
        return
    if args.set is None:
        parser.error("--set is required unless --select")
    if args.assemble:
        print(assemble_selected(args.set, args.split, args.variant, evaluate=not args.predict_only, drop_maps=drop, limit=args.limit))
        return
    if args.phrase is None:
        parser.error("--phrase is required for a run")
    print(run(args.phrase, args.set, args.split, args.variant, args.geometry, args.device, evaluate=not args.predict_only, drop_maps=drop, limit=args.limit))


if __name__ == "__main__":
    main()
