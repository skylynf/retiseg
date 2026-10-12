"""SAM and SAM 2.1 automatic masks: the E5 zero-shot tier a (design/15 section 5; e5_frozen.yaml zero_shot).

One run is one model, one point density of the registered grid, one set and
split, one geometry (G1 windows, G2 whole crop) and one canvas variant. Per
image: canvas (geometry.canvas) -> windows (geometry.windows, padded crops) ->
generator.generate on each window -> Mask records at the window offsets
(score_rules) -> maps -> paste onto the original image -> write_prob.

Directories of one run (bench/e5/outputs.py; <base> is <set>, or <set>-val for
a pool run, plus the canvas variant):

    pool run (val split of IDRiD or DDR, decision D2): for every cap of the grid,
        <base>[__cap<f>]                 the any-lesion map of the masks after the cap (no token for cap 1.0)
        <base>[__cap<f>]__oracle_class   the same masks credited by overlap with the labels (upper bound)
        <base>[__cap<f>]__stability      the same masks scored by stability_score
    test run (the selected pair only, decision D1): <base>, <base>__oracle_class, <base>__stability
        with the selected cap applied; the cap is recorded in predicted.json, not in the name, so the
        main result of a set is always the plain <base> directory.

The any-lesion map is stored under all four class names (D8). The registered
budget rule (zero_shot.grid.budget_rule) skips the per-degree main variant of
TJDR_uwf when the selected density is the largest of the grid. Seconds per
window are recorded in predicted.json so the server smoke yields the real cost.

    python -m bench.e5.amg --model SAM --density 64 --set DDR --split val
    python -m bench.e5.amg --model "SAM 2" --density 64 --set DiaRetDB1 --split test --geometry G1 [--variant fov]
        [--device cuda:0] [--predict-only] [--drop-maps FGADR,Retinal-Lesions] [--limit 2]
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from bench import e3a
from bench.common.io import LESION_CLASSES, load_split, write_prob
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.e5 import outputs as out
from bench.e5 import score_rules as sr
from bench.models import foundation

TIER = "a"
SAM, SAM2 = "SAM", "SAM 2"
MODELS = (SAM, SAM2)
SPEC_KEY = {SAM: "sam", SAM2: "sam2"}
BACKBONE_TOKEN = {
    "sam_vit_h_4b8939.pth": "vith",
    "sam_vit_b_01ec64.pth": "vitb",
    "sam2.1_hiera_large.pt": "hieral21",
    "sam2_hiera_large.pt": "hieral20",
    "medsam_vit_b.pth": "vitb",
}
SAM_MODEL_TYPE = {"sam_vit_h_4b8939.pth": "vit_h", "sam_vit_b_01ec64.pth": "vit_b"}
SAM2_CONFIG = {"sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml", "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml"}
ANY, ORACLE, STABILITY = "any_lesion", "oracle_class", "stability"
KINDS = (ANY, ORACLE, STABILITY)


def spec(frozen, model):
    if model not in MODELS:
        raise ValueError(f"automatic masks run for {MODELS}, got {model!r}")
    return frozen["zero_shot"][SPEC_KEY[model]]


def generator_params(frozen, model, density):
    """points_per_side plus the library defaults registered for the model; nothing else is set."""
    density = int(density)
    densities = [int(d) for d in frozen["zero_shot"]["grid"]["points_per_side"]]
    if density not in densities:
        raise ValueError(f"density {density} is not in the registered grid {densities}")
    return {"points_per_side": density, **dict(spec(frozen, model)["generator_defaults"])}


def cap_token(fraction):
    """"cap0.02" for a cap below 1.0; "" for no cap."""
    fraction = float(fraction)
    return "" if fraction >= 1.0 else f"cap{fraction:g}"


def variant_tokens(canvas_variants, cap, kind):
    tokens = list(canvas_variants)
    if cap_token(cap):
        tokens.append(cap_token(cap))
    if kind != ANY:
        tokens.append(kind)
    return tokens


def source_name(frozen, model, density):
    checkpoint = spec(frozen, model)["checkpoint"]
    return out.source_name(TIER, model, BACKBONE_TOKEN.get(checkpoint, Path(checkpoint).stem.lower()), f"p{int(density)}")


def budget_rule_skips(frozen, split, density, set_name, variant):
    """The registered rule: the largest density does not run the per-degree main variant of a per-degree set."""
    grid = frozen["zero_shot"]["grid"]
    if fz.is_pool_split(split) or variant != geo.MAIN or not grid.get("budget_rule"):
        return False
    largest = max(int(d) for d in grid["points_per_side"])
    return int(density) == largest and set_name in tuple(e3a.load_frozen().get("per_degree_sets") or ())


def _sam2_postprocessing_record(official):
    """Whether sam2's CUDA connected-components op is importable (hole filling and small-region removal rely on it)."""
    try:
        with official("sam2", purge=("sam2",)):
            from sam2 import _C  # noqa: F401
        available = True
    except Exception:  # noqa: BLE001  (the op is optional in the official code too)
        available = False
    return {"connected_components_op": available}


def build_generator(model, density, frozen, device=None):
    """The official generator with the registered parameters; returns (generator, record for predicted.json)."""
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    params = generator_params(frozen, model, density)
    checkpoint_name = spec(frozen, model)["checkpoint"]
    checkpoint = foundation.weights(checkpoint_name)
    record = {"checkpoint": out.checkpoint_record(checkpoint), "params": params, "device": str(device), "sam2_postprocessing": None}
    if model == SAM:
        with foundation.official("segment-anything", purge=("segment_anything",)):
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        network = sam_model_registry[SAM_MODEL_TYPE[checkpoint_name]](checkpoint=str(checkpoint)).to(device).eval()
        return SamAutomaticMaskGenerator(network, **params), record
    with foundation.official("sam2", purge=("sam2",)):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        network = build_sam2(SAM2_CONFIG[checkpoint_name], str(checkpoint), device=device)
    record["sam2_postprocessing"] = {"apply_postprocessing": True, "min_mask_region_area": params.get("min_mask_region_area", 0),
                                     **_sam2_postprocessing_record(foundation.official)}
    return SAM2AutomaticMaskGenerator(network, **params), record


def window_masks(generator, cv, window, boxes):
    """Masks of every window of a canvas, at their offsets; returns (masks, seconds per window)."""
    masks, seconds = [], []
    for top, left in boxes:
        piece = cv.image if window is None else geo.crop_window(cv.image, top, left, window)[0]
        started = time.perf_counter()
        records = generator.generate(np.ascontiguousarray(piece))
        seconds.append(time.perf_counter() - started)
        masks.extend(sr.Mask.from_record(record, top, left) for record in records)
    return masks, seconds


def maps_of(kind, masks, cv):
    """{cls: canvas map} of one output kind."""
    if kind == ANY:
        return sr.as_four(sr.masks_to_score(masks, cv.hw))
    if kind == STABILITY:
        return sr.as_four(sr.masks_to_score(masks, cv.hw, key=sr.STABILITY))
    if kind == ORACLE:
        return sr.oracle_class(masks, cv.masks, cv.hw)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def write_maps(maps_dir, image_id, cv, class_maps):
    for cls in LESION_CLASSES:
        write_prob(maps_dir, cls, image_id, geo.paste(class_maps[cls], cv))


def targets_for(frozen, split, canvas_variants, selected_cap=None):
    """[(variant tokens, cap, kind)]: every cap of the grid on the pool (cap in the name), the selected cap on a test split (not in the name)."""
    pool_run = fz.is_pool_split(split)
    caps = [float(c) for c in frozen["zero_shot"]["grid"]["area_cap_fraction"]] if pool_run else [float(selected_cap)]
    return [(variant_tokens(canvas_variants, cap if pool_run else 1.0, kind), cap, kind) for cap in caps for kind in KINDS]


def planned(frozen, model, density, set_name, split, variant=geo.MAIN):
    """(source, {dirname: (tokens, cap, kind)}, {dirname: path}) of one run; a test split needs grid.selected."""
    pool_run = fz.is_pool_split(split)
    selected_cap = None if pool_run else fz.selected_pair(frozen)[1]
    source = source_name(frozen, model, density)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if pool_run else set_name
    directories = {out.set_dirname(base, tokens): (tokens, cap, kind) for tokens, cap, kind in targets_for(frozen, split, canvas_variants, selected_cap)}
    return source, directories, {name: out.out_dir(TIER, source, name) for name in directories}


def run(model, density, set_name, split, geometry=geo.G1, variant=geo.MAIN, device=None, generator=None, frozen=None,
        data_dir=None, long_side=None, window=None, evaluate=True, drop_maps=(), limit=None, pool=None):
    """Predict (and evaluate) one run; directories that already have predicted.json are left alone.

    ``generator``, ``data_dir``, ``long_side``, ``window`` and ``pool`` override
    the registered model, data and geometry (tests and smoke).
    """
    frozen = frozen or fz.load_frozen()
    if geometry not in geo.GEOMETRIES:
        raise ValueError(f"geometry must be one of {geo.GEOMETRIES}, got {geometry!r}")
    fz.assert_ready(frozen, TIER, split, model=model)
    density = int(density)
    params = generator_params(frozen, model, density)
    pool_run = fz.is_pool_split(split)
    selected_cap = None
    if not pool_run:
        chosen_density, selected_cap = fz.selected_pair(frozen)
        if density != chosen_density:
            raise ValueError(f"a test split runs the selected density {chosen_density}, not {density}")
    source = source_name(frozen, model, density)
    if budget_rule_skips(frozen, split, density, set_name, variant):
        print(f"{source} {set_name}: skipped by the registered budget rule (largest density, per-degree main variant)", flush=True)
        return {"source": source, "skipped": "budget_rule", "directories": []}
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if pool_run else set_name
    _source, directories, dirs = planned(frozen, model, density, set_name, split, variant)
    pending = [name for name, path in dirs.items() if out.needs_prediction(path, limit)]
    data_dir = Path(data_dir) if data_dir is not None else geo.data_dir(set_name)
    if long_side is None:
        long_side = geo.NATIVE if geometry == geo.G2 else geo.long_side_for(set_name, variant, frozen)
    else:
        geo.long_side_for(set_name, variant, frozen)  # the variant must be a registered one
    window_size = None if geometry == geo.G2 else int(window if window is not None else geo.window_size(frozen, model))
    all_ids = load_split(data_dir, split)
    ids = all_ids if limit is None else all_ids[: int(limit)]
    excluded = all_ids[len(ids):]  # a smoke run evaluates the predicted images only
    if pending:
        if generator is None:
            generator, record = build_generator(model, density, frozen, device)
        else:
            record = {"checkpoint": None, "params": params, "device": str(device), "sam2_postprocessing": None, "generator": type(generator).__name__}
        n_windows, n_masks, seconds = 0, 0, []
        started = time.perf_counter()
        for index, image_id in enumerate(ids):
            cv = geo.canvas(data_dir, image_id, long_side, masks=True)
            boxes = [(0, 0)] if window_size is None else geo.windows(cv.hw, window_size, geo.overlap(frozen))
            masks, window_seconds = window_masks(generator, cv, window_size, boxes)
            n_windows += len(boxes)
            n_masks += len(masks)
            seconds.extend(window_seconds)
            fov_pixels = int(np.count_nonzero(cv.fov))
            kept = {}
            for name in pending:
                _tokens, cap, kind = directories[name]
                if cap not in kept:
                    kept[cap] = sr.area_cap(masks, cap, fov_pixels, cv.hw)
                write_maps(out.pred_dir(dirs[name], split), image_id, cv, maps_of(kind, kept[cap], cv))
            if index % 20 == 0:
                print(f"{source} {base} {index + 1}/{len(ids)} windows {len(boxes)} masks {len(masks)}", flush=True)
        elapsed = time.perf_counter() - started
        environment = out.environment()
        for name in pending:
            tokens, cap, kind = directories[name]
            out.write_predicted(
                dirs[name],
                {
                    "tier": TIER, "model": model, "source": source, "set": set_name, "split": split, "variant": variant,
                    "variants": tokens, "geometry": geometry,
                    "canvas_long_side": "native" if long_side == geo.NATIVE else int(long_side),
                    "window": window_size, "overlap": None if window_size is None else geo.overlap(frozen),
                    "density": density, "area_cap_fraction": cap, "kind": kind,
                    "pixel_score": sr.STABILITY if kind == STABILITY else frozen["zero_shot"]["pixel_score"],
                    "n_images": len(ids), "limit": limit, "n_windows": n_windows, "n_masks": n_masks,
                    "predict_seconds": round(elapsed, 1),
                    "generate_seconds_total": round(float(sum(seconds)), 1),
                    "seconds_per_window_mean": round(float(np.mean(seconds)), 3) if seconds else None,
                    **record, **environment,
                },
            )
    if evaluate:
        from bench.eval.evaluate import load_protocol

        proto = load_protocol(fz.REPO / "bench" / "eval" / "protocol.yaml")
        for name, path in dirs.items():
            if out.is_done(path, limit):
                continue
            thresholds = None
            if not pool_run:
                tokens, cap, kind = directories[name]
                thresholds = out.pool_thresholds(TIER, source, frozen, proto, pool=pool, variants=variant_tokens([], cap, kind))
            out.evaluate_dir(path, set_name, split, data_dir=data_dir, proto=proto, val_thresholds=thresholds, drop_maps=drop_maps, exclude=excluded)
    return {"source": source, "skipped": None, "directories": [str(path) for path in dirs.values()], "n_images": len(ids)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench.e5.amg", description="SAM / SAM 2.1 automatic masks on one set and split.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--density", required=True, type=int, help="points_per_side from the registered grid")
    parser.add_argument("--set", required=True, help="a test set of bench/configs/e3a_frozen.yaml")
    parser.add_argument("--split", default="test", help="test, or val for a pool run")
    parser.add_argument("--geometry", default=geo.G1, choices=geo.GEOMETRIES)
    parser.add_argument("--variant", default=geo.MAIN, help="main, or a comparison variant such as fov for TJDR_uwf")
    parser.add_argument("--device")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--drop-maps", help="comma-separated test sets whose maps are removed after evaluation")
    parser.add_argument("--limit", type=int, help="first N images only (smoke); the directory is predicted again by a later run without --limit")
    args = parser.parse_args(argv)
    drop = [item.strip() for item in (args.drop_maps or "").split(",") if item.strip()]
    summary = run(args.model, args.density, args.set, args.split, args.geometry, args.variant, args.device,
                  evaluate=not args.predict_only, drop_maps=drop, limit=args.limit)
    print(summary)


if __name__ == "__main__":
    main()
