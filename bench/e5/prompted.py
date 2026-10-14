"""Point and box prompts from the labels: the E5 oracle tier b (design/15 section 6; e5_frozen.yaml oracle).

One run is one model and prompt kind (SAM ViT-B boxes; SAM 2.1 points or
boxes; MedSAM boxes), one set and split, one canvas variant. The prompts are
read from the file bench/e5/prompts.py wrote for that set, split and variant;
every model reads the same file. This tier reads the test labels: it is an
upper bound and is never ranked with trained models.

Decoding, as registered in e5_frozen.yaml oracle.decode: each prompt is
decoded once, in the G1 window whose edges are farthest from the box centre,
with one set_image per window per image; a point keeps the mask with the
highest predicted IoU of the multimask output, a box asks for one mask; SAM
and SAM 2 logits go through a sigmoid; MedSAM follows MedSAM_Inference.py
(input 1024 with per-image min-max to [0, 1], box only, sigmoid of the 256
output then bilinear). The map of a class at prompt count k is the pixelwise
maximum over the prompts of rank < k ("all": every prompt), so the k levels
are prefix maxima filled in one pass; a pixel no prompt covers scores 0. A box
reaches the decoder with its right and bottom as inclusive pixel indices
(decoder_box): the prompt file stores them exclusive, the SAM family and MedSAM
read inclusive ones (review fix 6, B5; e5_frozen.yaml oracle.box).

Directories (bench/e5/outputs.py): <base>[__<canvas variant>]__k<k> for every
k of the grid on a pool run and on the curve sets (e5_frozen.yaml oracle.curve_sets:
IDRiD, DDR), for k = all alone on the other five sets (D6; review fix 9 / B3);
<base> = <set> or <set>-val for a pool run; sources b_sam_vitb_box,
b_sam2_hieral21_point, b_medsam_vitb_box.

    python -m bench.e5.prompted --model "SAM 2" --kind boxes --set IDRiD --split test
        [--variant fov] [--device cuda:0] [--predict-only] [--drop-maps ...] [--limit 2]
"""

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from bench.common.io import LESION_CLASSES, load_split
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.e5 import outputs as out
from bench.e5 import prompts as pr
from bench.e5.amg import BACKBONE_TOKEN, write_maps
from bench.models import foundation

TIER = "b"
BOXES, POINTS = "boxes", "points"
KIND_TOKEN = {BOXES: "box", POINTS: "point"}
SAM, SAM2, MEDSAM = "SAM", "SAM 2", "MedSAM"
BOX_CONVENTION = "file: [x0, y0, x1, y1] with x1, y1 exclusive; decoder: [x0, y0, x1-1, y1-1] inclusive pixel indices (e5_frozen.yaml oracle.box)"
WINDOW_CHOICE = "a window holding the whole box, largest box-to-edge margin, ties to the first; else the largest overlap (e5_frozen.yaml oracle.window_choice)"
SAM2_CONFIG = {"sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml", "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml"}
SAM_MODEL_TYPE = {"sam_vit_h_4b8939.pth": "vit_h", "sam_vit_b_01ec64.pth": "vit_b"}
MEDSAM_INPUT = 1024


def kinds_of(frozen, model):
    models = frozen["oracle"]["models"]
    if model not in models:
        raise ValueError(f"the oracle tier runs {list(models)}, got {model!r}")
    return list(models[model])


def check_kind(frozen, model, kind):
    kinds = kinds_of(frozen, model)
    if kind not in kinds:
        raise ValueError(f"{model} runs {kinds} in the oracle tier, not {kind!r}")


def checkpoint_name(frozen, model):
    return frozen["oracle"]["checkpoints"][model]


def source_name(frozen, model, kind):
    name = checkpoint_name(frozen, model)
    return out.source_name(TIER, model, BACKBONE_TOKEN.get(name, Path(name).stem.lower()), KIND_TOKEN[kind])


def k_token(k):
    return f"k{k}"


def box_center(box):
    x0, y0, x1, y1 = box
    return (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0


def decoder_box(box, top, left):
    """The box as the decoders take it: window coordinates, right and bottom as inclusive pixel indices.

    The prompt file stores [x0, y0, x1, y1] with x1 and y1 exclusive (prompts.py,
    the numpy slice convention same_info.py relies on). SAM's prompt encoder reads
    box corners as pixel indices (+0.5 to the pixel centre), its own
    batched_mask_to_box returns the inclusive maximum index, and MedSAM was trained
    on np.max of the component's indices, so the decoders get x1 - 1 and y1 - 1: a
    one-pixel component becomes a box of zero width at that pixel, not one spanning
    two pixels. Registered in e5_frozen.yaml oracle.box (2026-10-12, review fix 6).
    """
    x0, y0, x1, y1 = box
    return [x0 - left, y0 - top, x1 - 1 - left, y1 - 1 - top]


def choose_window(boxes, window, center):
    """Index of the window containing the centre with the largest margin to its edges; ties to the first.

    Used only as the fall-back of ``window_for_box`` for a box no window holds whole.
    """
    cx, cy = center
    best, best_margin = None, None
    for index, (top, left) in enumerate(boxes):
        if top <= cy < top + window and left <= cx < left + window:
            margin = min(cy - top, top + window - 1 - cy, cx - left, left + window - 1 - cx)
            if best_margin is None or margin > best_margin:
                best, best_margin = index, margin
    if best is None:  # a centre outside every window: the nearest window centre
        best = int(np.argmin([(top + window / 2 - cy) ** 2 + (left + window / 2 - cx) ** 2 for top, left in boxes]))
    return best


def window_for_box(boxes, window, box):
    """(index, contained) of the window a box prompt is decoded in.

    A window that holds the whole box is preferred (exclusive right and bottom,
    as the prompt file stores them); among those, the one whose edges are
    farthest from the box (the smallest of the four box-to-edge distances is
    largest); ties to the first. Only when no window holds the box (a component
    wider or taller than the window allows) the window that keeps most of the
    box is taken, ties broken by the centre margin of ``choose_window``, and
    ``contained`` is False so the run can count it. Registered 2026-10-14 before
    any oracle run (e5_frozen.yaml oracle.window_choice): the earlier rule chose
    by the centre's smallest margin alone, which near a canvas edge ties every
    window of a row and sent part of the large boxes to a window that cut them.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    win = int(window)
    best, best_margin = None, None
    for index, (top, left) in enumerate(boxes):
        if left <= x0 and x1 <= left + win and top <= y0 and y1 <= top + win:
            margin = min(x0 - left, left + win - x1, y0 - top, top + win - y1)
            if best_margin is None or margin > best_margin:
                best, best_margin = index, margin
    if best is not None:
        return best, True
    overlaps = [
        max(0, min(x1, left + win) - max(x0, left)) * max(0, min(y1, top + win) - max(y0, top))
        for top, left in boxes
    ]
    most = max(overlaps)
    candidates = [index for index, area in enumerate(overlaps) if area == most]
    if len(candidates) == 1:
        return candidates[0], False
    centre_pick = choose_window([boxes[i] for i in candidates], win, box_center(box))
    return candidates[centre_pick], False


# ---- decoders: one window at a time, prompts in window coordinates, probabilities in [0, 1] ----


def _sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))


class SamDecoder:
    """segment_anything / sam2 image predictors share this interface."""

    def __init__(self, predictor):
        self.predictor = predictor

    def set_image(self, image):
        self.predictor.set_image(image)

    def box(self, box):
        masks, _ious, _low = self.predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False, return_logits=True)
        return _sigmoid(masks[0])

    def point(self, xy):
        masks, ious, _low = self.predictor.predict(
            point_coords=np.asarray([xy], dtype=np.float32), point_labels=np.ones(1, dtype=np.int64), multimask_output=True, return_logits=True
        )
        return _sigmoid(masks[int(np.argmax(ious))])


class MedSamDecoder:
    """MedSAM_Inference.py: 1024 input with per-image min-max to [0, 1]; box only; sigmoid of the 256 logits, bilinear to the window."""

    def __init__(self, model, device):
        self.model, self.device, self.embedding, self.hw = model, device, None, None

    def set_image(self, image):
        import torch
        import torch.nn.functional as F

        self.hw = tuple(image.shape[:2])
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()[None]
        if self.hw != (MEDSAM_INPUT, MEDSAM_INPUT):  # G2 only; a G1 window is already 1024
            tensor = F.interpolate(tensor, (MEDSAM_INPUT, MEDSAM_INPUT), mode="bicubic", align_corners=False, antialias=True).clamp(0, 255)
        tensor = (tensor - tensor.min()) / torch.clamp(tensor.max() - tensor.min(), min=1e-8)  # MedSAM_Inference.py:124-125
        with torch.no_grad():
            self.embedding = self.model.image_encoder(tensor.to(self.device))

    def box(self, box):
        import torch
        import torch.nn.functional as F

        height, width = self.hw
        scale = np.array([MEDSAM_INPUT / width, MEDSAM_INPUT / height] * 2, dtype=np.float32)
        box_1024 = torch.as_tensor(np.asarray(box, dtype=np.float32) * scale, device=self.device)[None, None, :]  # (1, 1, 4)
        with torch.no_grad():
            sparse, dense = self.model.prompt_encoder(points=None, boxes=box_1024, masks=None)
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=self.embedding, image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False,
            )
            prob = F.interpolate(torch.sigmoid(low_res_logits), size=(height, width), mode="bilinear", align_corners=False)
        return prob[0, 0].cpu().numpy()

    def point(self, xy):
        raise ValueError("MedSAM takes box prompts only (design/15 section 6.1)")


def build_decoder(model, frozen, device=None):
    """The official predictor of the model behind the decoder interface; returns (decoder, record)."""
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = checkpoint_name(frozen, model)
    checkpoint = foundation.weights(name)
    record = {"checkpoint": out.checkpoint_record(checkpoint), "device": str(device)}
    if model == SAM:
        with foundation.official("segment-anything", purge=("segment_anything",)):
            from segment_anything import SamPredictor, sam_model_registry
        network = sam_model_registry[SAM_MODEL_TYPE[name]](checkpoint=str(checkpoint)).to(device).eval()
        return SamDecoder(SamPredictor(network)), record
    if model == SAM2:
        with foundation.official("sam2", purge=("sam2",)):
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            network = build_sam2(SAM2_CONFIG[name], str(checkpoint), device=device)
        return SamDecoder(SAM2ImagePredictor(network)), record
    if model == MEDSAM:
        with foundation.official("MedSAM", purge=("segment_anything",)):
            from segment_anything import sam_model_registry  # MedSAM's fork, as MedSAM_Inference.py:112
        network = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device).eval()
        record["medsam_resize"] = "torch bicubic with antialias when the input is not 1024 (G2); G1 windows are 1024"
        return MedSamDecoder(network, device), record
    raise ValueError(f"no decoder for {model!r}")


# ---- one image --------------------------------------------------------------------------------


def decode_image(decoder, cv, window, boxes, prompts_by_class, kind, k_levels):
    """{cls: {k: canvas map}} of one image plus counts and seconds; one set_image per window that has prompts."""
    maps = {cls: {k: np.zeros(cv.hw, dtype=np.float32) for k in k_levels} for cls in LESION_CLASSES}
    per_window = defaultdict(list)
    uncontained = 0
    for cls in LESION_CLASSES:
        for prompt in prompts_by_class.get(cls, []):
            if window is None:
                index = 0
            else:
                index, contained = window_for_box(boxes, window, prompt["box"])
                uncontained += 0 if contained else 1
            per_window[index].append((cls, prompt))
    stats = {"n_prompts": 0, "n_windows_encoded": 0, "encode_seconds": 0.0, "decode_seconds": 0.0, "n_prompts_box_uncontained": uncontained}
    for index in sorted(per_window):
        top, left = boxes[index]
        if window is None:
            piece, valid = cv.image, cv.hw
        else:
            piece, valid = geo.crop_window(cv.image, top, left, window)
        started = time.perf_counter()
        decoder.set_image(np.ascontiguousarray(piece))
        stats["encode_seconds"] += time.perf_counter() - started
        stats["n_windows_encoded"] += 1
        rows, cols = slice(top, top + valid[0]), slice(left, left + valid[1])
        for cls, prompt in per_window[index]:
            started = time.perf_counter()
            if kind == BOXES:
                prob = decoder.box(decoder_box(prompt["box"], top, left))
            else:
                px, py = prompt["point"]
                prob = decoder.point([px - left, py - top])
            stats["decode_seconds"] += time.perf_counter() - started
            stats["n_prompts"] += 1
            prob = np.clip(np.asarray(prob, dtype=np.float32)[: valid[0], : valid[1]], 0.0, 1.0)
            for k in k_levels:
                if k == "all" or int(prompt["rank"]) < int(k):
                    np.maximum(maps[cls][k][rows, cols], prob, out=maps[cls][k][rows, cols])
    return maps, stats


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planned(frozen, model, kind, set_name, split, variant=geo.MAIN):
    """(source, {k: path}) of one run."""
    source = source_name(frozen, model, kind)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if fz.is_pool_split(split) else set_name
    return source, {k: out.out_dir(TIER, source, out.set_dirname(base, canvas_variants + [k_token(k)])) for k in fz.k_levels_for(frozen, set_name, split)}


def run(model, kind, set_name, split, variant=geo.MAIN, geometry=geo.G1, device=None, decoder=None, frozen=None,
        data_dir=None, long_side=None, window=None, prompts_dir=None, evaluate=True, drop_maps=(), limit=None, pool=None, resume=True):
    """Predict (and evaluate) one run; directories that already have predicted.json are left alone.

    Every image's mask files are checked against the sha256 the prompt file recorded
    (prompts.verify_masks). ``resume`` skips the images whose maps every pending k
    directory already holds (review fix 12).
    """
    frozen = frozen or fz.load_frozen()
    check_kind(frozen, model, kind)
    if geometry not in geo.GEOMETRIES:
        raise ValueError(f"geometry must be one of {geo.GEOMETRIES}, got {geometry!r}")
    fz.assert_ready(frozen, TIER, split, set_name=set_name, prompts_dir=prompts_dir, variant=variant)
    prompt_path = fz.prompt_file(set_name, split, prompts_dir, variant)
    doc = pr.load(prompt_path)
    if doc["set"] != set_name or doc["split"] != split or doc["variant"] != variant:
        raise ValueError(f"{prompt_path} is for {doc['set']}/{doc['split']}/{doc['variant']}, not {set_name}/{split}/{variant}")
    if long_side is None:
        long_side = geo.NATIVE if geometry == geo.G2 else geo.long_side_for(set_name, variant, frozen)
    if int(doc["long_side"]) != int(long_side):
        raise ValueError(f"{prompt_path} was generated on a {doc['long_side']} canvas, this run uses {long_side}; regenerate it")
    pool_run = fz.is_pool_split(split)
    k_levels = fz.k_levels_for(frozen, set_name, split)
    source, dirs = planned(frozen, model, kind, set_name, split, variant)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if pool_run else set_name
    pending = [k for k, path in dirs.items() if out.needs_prediction(path, limit)]
    data_dir = Path(data_dir) if data_dir is not None else geo.data_dir(set_name)
    window_size = None if geometry == geo.G2 else int(window if window is not None else geo.window_size(frozen, model))
    all_ids = load_split(data_dir, split)
    ids = all_ids if limit is None else all_ids[: int(limit)]
    excluded = all_ids[len(ids):]
    if pending:
        if decoder is None:
            decoder, record = build_decoder(model, frozen, device)
        else:
            record = {"checkpoint": None, "device": str(device), "decoder": type(decoder).__name__}
        totals = {"n_prompts": 0, "n_windows_encoded": 0, "encode_seconds": 0.0, "decode_seconds": 0.0, "n_windows": 0, "n_prompts_box_uncontained": 0}
        resumed = 0
        started = time.perf_counter()
        for index, image_id in enumerate(ids):
            entry = doc["images"].get(image_id)
            if entry is None:
                raise ValueError(f"{prompt_path} has no entry for {image_id}; regenerate it")
            pr.verify_masks(doc, data_dir, image_id)
            if resume and all(out.maps_written(out.pred_dir(dirs[k], split), image_id) for k in pending):
                resumed += 1
                continue
            cv = geo.canvas(data_dir, image_id, long_side, masks=False)
            if tuple(entry["canvas_hw"]) != cv.hw:
                raise ValueError(f"{prompt_path} has no entry for {image_id} on a {cv.hw} canvas; regenerate it")
            boxes = [(0, 0)] if window_size is None else geo.windows(cv.hw, window_size, geo.overlap(frozen))
            by_class = {cls: entry["classes"][cls]["prompts"] for cls in LESION_CLASSES}
            maps, stats = decode_image(decoder, cv, window_size, boxes, by_class, kind, k_levels)
            for key, value in stats.items():
                totals[key] += value
            totals["n_windows"] += len(boxes)
            for k in pending:
                write_maps(out.pred_dir(dirs[k], split), image_id, cv, {cls: maps[cls][k] for cls in LESION_CLASSES})
            if index % 20 == 0:
                print(f"{source} {base} {index + 1}/{len(ids)} prompts {stats['n_prompts']} windows {stats['n_windows_encoded']}/{len(boxes)}", flush=True)
        elapsed = time.perf_counter() - started
        environment = out.environment()
        for k in pending:
            out.write_predicted(
                dirs[k],
                {
                    "tier": TIER, "model": model, "kind": kind, "source": source, "set": set_name, "split": split, "variant": variant,
                    "variants": canvas_variants + [k_token(k)], "k": k, "geometry": geometry,
                    "canvas_long_side": "native" if long_side == geo.NATIVE else int(long_side),
                    "window": window_size, "overlap": None if window_size is None else geo.overlap(frozen),
                    "prompt_file": str(prompt_path), "prompt_file_sha256": _file_sha256(prompt_path), "prompt_rule": doc["rule"],
                    "box_convention": BOX_CONVENTION,
                    "n_images": len(ids), "limit": limit, "resumed_images": resumed, "n_windows": totals["n_windows"],
                    "n_windows_encoded": totals["n_windows_encoded"], "n_prompts": totals["n_prompts"],
                    "window_choice": WINDOW_CHOICE, "n_prompts_box_uncontained": totals["n_prompts_box_uncontained"],
                    "predict_seconds": round(elapsed, 1), "encode_seconds_total": round(totals["encode_seconds"], 1),
                    "decode_seconds_total": round(totals["decode_seconds"], 1),
                    "seconds_per_prompt_mean": round(totals["decode_seconds"] / totals["n_prompts"], 4) if totals["n_prompts"] else None,
                    **record, **environment,
                },
            )
    if evaluate:
        from bench.eval.evaluate import load_protocol

        proto = load_protocol(fz.REPO / "bench" / "eval" / "protocol.yaml")
        for k, path in dirs.items():
            if out.is_done(path, limit):
                continue
            predicted_limit, excluded = out.evaluation_scope(path, all_ids)
            thresholds = None if pool_run else out.thresholds_for_test(TIER, source, frozen, proto, pool=pool, variants=[k_token(k)], limit=predicted_limit)
            out.evaluate_dir(path, set_name, split, data_dir=data_dir, proto=proto, val_thresholds=thresholds, drop_maps=drop_maps, exclude=excluded)
    return {"source": source, "directories": [str(path) for path in dirs.values()], "n_images": len(ids)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench.e5.prompted", description="Oracle point / box prompts on one set and split.")
    parser.add_argument("--model", required=True, choices=(SAM, SAM2, MEDSAM))
    parser.add_argument("--kind", required=True, choices=(BOXES, POINTS))
    parser.add_argument("--set", required=True)
    parser.add_argument("--split", default="test", help="test, or val for a pool run")
    parser.add_argument("--variant", default=geo.MAIN)
    parser.add_argument("--device")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--drop-maps")
    parser.add_argument("--limit", type=int, help="first N images only (smoke); redone in full by a later run without --limit")
    args = parser.parse_args(argv)
    drop = [item.strip() for item in (args.drop_maps or "").split(",") if item.strip()]
    print(run(args.model, args.kind, args.set, args.split, args.variant, geo.G1, args.device,
              evaluate=not args.predict_only, drop_maps=drop, limit=args.limit))


if __name__ == "__main__":
    main()
