"""The same-information reference of docs/paper/16 (row E5b): a supervised model's own test maps
restricted to the union of the same k oracle boxes, then evaluated.

For each image, class and k, the model's original-resolution probability map
(a finished prob_test directory, e.g. runs/BS_m2mrf_ddr_seed0/prob_test or an
E3a directory) is multiplied by the union of the first k boxes of the prompt
file, mapped from the canvas to the original image; everything outside the
boxes scores 0. This is CPU post-processing of maps that live on the server;
no model runs. The result is the reference line of figure 5b, evaluated with
the fixed threshold only.

Directories (bench/e5/outputs.py): b_<model>_ref_<tokens>/<set>[__<variant>]__k<k>.

    python -m bench.e5.same_info --model M2MRF --tokens ddr seed0 --pred-dir runs/BS_m2mrf_ddr_seed0/prob_test --set DDR --split test
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from bench.common.io import LESION_CLASSES, PROB_LEVELS, load_split, read_fov, read_prob_quantized, write_prob
from bench.data.fov import crop_box
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.e5 import outputs as out
from bench.e5 import prompts as pr
from bench.e5.prompted import k_token

TIER = "b"


def canvas_box_to_original(box, canvas_hw, crop):
    """A canvas box [x0, y0, x1, y1] on the original image: scaled into the crop box, outer-rounded, clipped."""
    y0, y1, x0, x1 = crop
    scale_y = (y1 - y0) / float(canvas_hw[0])
    scale_x = (x1 - x0) / float(canvas_hw[1])
    bx0, by0, bx1, by1 = box
    return [
        int(np.clip(x0 + np.floor(bx0 * scale_x), x0, x1)),
        int(np.clip(y0 + np.floor(by0 * scale_y), y0, y1)),
        int(np.clip(x0 + np.ceil(bx1 * scale_x), x0, x1)),
        int(np.clip(y0 + np.ceil(by1 * scale_y), y0, y1)),
    ]


def box_union_mask(prompts, k, canvas_hw, crop, original_hw):
    """Union of the first k prompts' boxes (rank order; "all": every one) on the original image."""
    mask = np.zeros(original_hw, dtype=bool)
    for prompt in pr.take(prompts, k):
        ox0, oy0, ox1, oy1 = canvas_box_to_original(prompt["box"], canvas_hw, crop)
        mask[oy0:oy1, ox0:ox1] = True
    return mask


def planned(frozen, model, tokens, set_name, split, variant=geo.MAIN):
    """(source, {k: path}) of one reference run."""
    source = out.source_name(TIER, model, "ref", *tokens)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if fz.is_pool_split(split) else set_name
    return source, {k: out.out_dir(TIER, source, out.set_dirname(base, canvas_variants + [k_token(k)])) for k in fz.k_levels(frozen)}


def run(model, tokens, pred_dir, set_name, split, variant=geo.MAIN, frozen=None, data_dir=None, prompts_dir=None,
        evaluate=True, drop_maps=(), limit=None):
    """Restrict the reference maps to the k-box unions and evaluate; directories with predicted.json are left alone."""
    frozen = frozen or fz.load_frozen()
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"{pred_dir} is not a prediction directory")
    fz.assert_ready(frozen, TIER, split, set_name=set_name, prompts_dir=prompts_dir, variant=variant)
    prompt_path = fz.prompt_file(set_name, split, prompts_dir, variant)
    doc = pr.load(prompt_path)
    if doc["set"] != set_name or doc["split"] != split or doc["variant"] != variant:
        raise ValueError(f"{prompt_path} is for {doc['set']}/{doc['split']}/{doc['variant']}, not {set_name}/{split}/{variant}")
    k_levels = fz.k_levels(frozen)
    source, dirs = planned(frozen, model, tokens, set_name, split, variant)
    canvas_variants = [] if variant == geo.MAIN else [variant]
    base = out.pool_set_dirname(set_name) if fz.is_pool_split(split) else set_name
    pending = [k for k, path in dirs.items() if out.needs_prediction(path, limit)]
    data_dir = Path(data_dir) if data_dir is not None else geo.data_dir(set_name)
    all_ids = load_split(data_dir, split)
    ids = all_ids if limit is None else all_ids[: int(limit)]
    excluded = all_ids[len(ids):]
    if pending:
        started = time.perf_counter()
        n_boxes = 0
        for image_id in ids:
            entry = doc["images"][image_id]
            fov = read_fov(data_dir, image_id)
            crop = crop_box(fov)
            canvas_hw = tuple(entry["canvas_hw"])
            for cls in LESION_CLASSES:
                reference = read_prob_quantized(pred_dir, cls, image_id).astype(np.float32) / PROB_LEVELS
                prompts = entry["classes"][cls]["prompts"]
                n_boxes += len(prompts)
                for k in pending:
                    kept = reference * box_union_mask(prompts, k, canvas_hw, crop, fov.shape)
                    write_prob(out.pred_dir(dirs[k], split), cls, image_id, kept)
        environment = out.environment()
        for k in pending:
            out.write_predicted(
                dirs[k],
                {
                    "tier": TIER, "model": model, "kind": "same_information_reference", "source": source, "set": set_name,
                    "split": split, "variant": variant, "variants": canvas_variants + [k_token(k)], "k": k,
                    "reference_pred_dir": str(pred_dir), "prompt_file": str(prompt_path),
                    "prompt_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                    "n_images": len(ids), "limit": limit, "n_boxes_total": n_boxes,
                    "predict_seconds": round(time.perf_counter() - started, 1), **environment,
                },
            )
    if evaluate:
        from bench.eval.evaluate import load_protocol

        proto = load_protocol(fz.REPO / "bench" / "eval" / "protocol.yaml")
        for k, path in dirs.items():
            if not out.is_done(path, limit):
                _predicted_limit, excluded = out.evaluation_scope(path, all_ids)
                out.evaluate_dir(path, set_name, split, data_dir=data_dir, proto=proto, drop_maps=drop_maps, exclude=excluded,
                                 record={"thresholds_note": "reference line: fixed threshold only"})
    return {"source": source, "directories": [str(path) for path in dirs.values()], "n_images": len(ids)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench.e5.same_info", description="Supervised maps restricted to the same k oracle boxes.")
    parser.add_argument("--model", required=True, choices=("M2MRF", "nnU-Net"))
    parser.add_argument("--tokens", nargs="+", required=True, help="source tokens, e.g. ddr seed0 or default ddr")
    parser.add_argument("--pred-dir", required=True, help="the model's finished original-resolution maps for this set and split")
    parser.add_argument("--set", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default=geo.MAIN)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--drop-maps")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    drop = [item.strip() for item in (args.drop_maps or "").split(",") if item.strip()]
    print(run(args.model, args.tokens, args.pred_dir, args.set, args.split, args.variant,
              evaluate=not args.predict_only, drop_maps=drop, limit=args.limit))


if __name__ == "__main__":
    main()
