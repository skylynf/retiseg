"""Tier b, oracle prompts from the test labels (design/15 section 6; e5_frozen.yaml ``oracle``).

    python -m bench.e5 prompted --model sam|sam2|medsam --kind box|point --set IDRiD [--split test] [--k 1,5,10,20,all]
                                [--geometry G1|G2] [--variant vit_h]

For every image and class the frozen prompt list (bench.e5.prompts) is decoded
one prompt at a time, in the frozen order, in the window whose edges are
farthest from the box centre (G1; the encoder runs once per window). The
class map is the pixelwise maximum of the prompts' probability maps. Because
smaller k is a prefix of larger k, one pass writes every k: the running
maximum is snapshotted after the k-th prompt. An upper bound: it reads the
test labels and is never ranked with the trained rows.

Per prompt, the Dice and IoU of the thresholded (0.5) mask against the ground
truth of that class inside the box are kept (inbox.json), to separate "the model
cannot segment a lesion it is pointed at" from "there are more lesions than
prompts" (figure 5b text).

Output: runs/E5/b/<model>[__G2]/<kind>/k<k>/<set>_<split>/{<CLS>/<id>.png, paper.json, done.json}; inbox.json
in the k=all directory. The results.csv variant is oracle:<kind>:<k>.
"""

import time

import numpy as np

from bench.common.io import LESION_CLASSES
from bench.e5 import common, prompts

K_DEFAULT = (1, 5, 10, 20, "all")


def out_dir(model_name, kind, k, test_set, split, geometry="G1"):
    name = model_name if geometry == "G1" else f"{model_name}__{geometry}"
    return common.OUT_ROOT / "b" / name / kind / f"k{k}" / common.set_dirname(test_set, split)


def parse_k(text):
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append("all" if item == "all" else int(item))
    return tuple(out)


def group_by_window(items, wins):
    """{window index: [(rank, item), ...]} keeping the frozen order inside each window."""
    groups = {}
    for rank, item in enumerate(items):
        groups.setdefault(common.window_of(item["box"], wins), []).append((rank, item))
    return groups


def in_box_scores(prob, gt_crop):
    pred = prob >= 0.5
    inter = int(np.count_nonzero(pred & gt_crop))
    p, g = int(pred.sum()), int(gt_crop.sum())
    dice = 2.0 * inter / (p + g) if p + g else float("nan")
    iou = inter / (p + g - inter) if p + g - inter else float("nan")
    return dice, iou


def decode_class(adapter, image, wins, items, kind, geometry, ks, gt=None):
    """Running-maximum maps of one class for every k in ``ks`` plus the per-prompt in-box scores.

    Prompts are decoded window by window (one encoder pass per window). A prompt's contribution is
    order independent (pixelwise maximum), so the map for a bound b is the maximum over the prompts of
    frozen rank < b whatever the decoding order: one running map per bound, updated by every prompt
    below it."""
    canvas_h, canvas_w = image.shape[:2]
    n = len(items)
    bounds = sorted({n if k == "all" else min(int(k), n) for k in ks})
    running = {b: np.zeros((canvas_h, canvas_w), dtype=np.float32) for b in bounds}
    scores = []
    for window_index, group in sorted(group_by_window(items, wins).items()):
        top, left, h, w = wins[window_index]
        crop = common.window_crop(image, top, left, h, w) if geometry == "G1" else image
        adapter.set_image(crop)
        for rank, item in group:
            if kind == "box":
                x0, y0, x1, y1 = item["box"]
                prob, score = adapter.predict_box([x0 - left, y0 - top, x1 - left, y1 - top])
            else:
                x, y = item["point"]
                prob, score = adapter.predict_point([x - left, y - top])
            prob = np.asarray(prob, dtype=np.float32)
            for b in bounds:
                if rank < b:
                    common.paste_max(running[b], prob, top, left)
            if gt is not None:
                # the box clipped to the window (a component larger than the window is scored on its visible part)
                x0, y0, x1, y1 = item["box"]
                cy0, cy1 = max(y0, top), min(y1, top + prob.shape[0], canvas_h)
                cx0, cx1 = max(x0, left), min(x1, left + prob.shape[1], canvas_w)
                dice, iou = in_box_scores(prob[cy0 - top : cy1 - top, cx0 - left : cx1 - left], gt[cy0:cy1, cx0:cx1])
                scores.append({"rank": rank, "area": item["area"], "score": score, "dice": dice, "iou": iou})
    return {k: running[n if k == "all" else min(int(k), n)] for k in ks}, scores


def run(model_name, kind, test_set, split="test", ks=K_DEFAULT, geometry="G1", variant="vit_h", limit=None, adapter=None):
    frozen = common.load_frozen()
    allowed = {"SAM": "sam", "SAM 2": "sam2", "MedSAM": "medsam"}
    kinds = {allowed[m]: v for m, v in frozen["oracle"]["models"].items()}
    if f"{kind}s" not in kinds.get(model_name, ()):
        raise ValueError(f"{model_name} with {kind} prompts is not in e5_frozen.yaml oracle.models: {kinds}")
    ks = tuple(ks)
    dirs = {k: out_dir(model_name, kind, k, test_set, split, geometry) for k in ks}
    pending_ks = [k for k in ks if not common.is_done(dirs[k])]
    if not pending_ks:
        print(f"{dirs[ks[0]].parent.parent}: every k done", flush=True)
        return dirs
    table = prompts.load(test_set, split)
    ids = common.split_ids(test_set, split)
    if limit:
        ids = ids[: int(limit)]
    if adapter is None:
        from bench.e5 import models

        adapter = models.build(model_name, variant=variant)
    inbox = {}
    started = time.perf_counter()
    for image_id in ids:
        if all((dirs[k] / "SE" / f"{image_id}.png").is_file() for k in pending_ks):
            continue
        canvas = common.load_canvas(test_set, image_id, masks=True)
        image = canvas["image"]
        canvas_hw = tuple(canvas["canvas_hw"])
        wins = common.windows(canvas_hw[0], canvas_hw[1], adapter.window) if geometry == "G1" else [(0, 0, *canvas_hw)]
        maps = {k: {} for k in pending_ks}
        inbox[image_id] = {}
        for cls, gt in zip(LESION_CLASSES, canvas["masks"]):
            items = table["images"][image_id][cls]
            per_k, scores = decode_class(adapter, image, wins, items, kind, geometry, pending_ks, gt=np.asarray(gt, dtype=bool))
            for k in pending_ks:
                maps[k][cls] = per_k[k]
            inbox[image_id][cls] = {"n_components": len(items), "prompts": scores}
        for k in pending_ks:
            common.write_maps(dirs[k], image_id, canvas, maps[k])
        print(f"{model_name}/{kind} {test_set} {image_id}: {len(wins)} windows, "
              f"{ {cls: len(table['images'][image_id][cls]) for cls in LESION_CLASSES} } prompts", flush=True)
    if limit:
        return dirs
    summary = _inbox_summary(inbox)
    for k in pending_ks:
        common.write_json(dirs[k] / "paper.json", common.paper_record(model_name, f"oracle:{kind}:{k}", test_set))
        common.write_json(common.done_marker(dirs[k]), {
            "experiment": "E5", "tier": "b", "reads_labels": True, "model": model_name, "variant": variant, "kind": kind, "k": k,
            "geometry": geometry, "window": adapter.window if geometry == "G1" else None, "test_set": test_set,
            "split": split, "n_images": len(ids), "prompts_sha256": table.get("sha256"), "in_box": summary,
            "seconds": round(time.perf_counter() - started, 1), "environment": common.environment(),
        })
    if "all" in pending_ks:
        common.write_json(dirs["all"] / "inbox.json", {"summary": summary, "images": inbox})
    print(f"{model_name}/{kind} {test_set}_{split}: {len(ids)} images, in-box Dice {summary}", flush=True)
    return dirs


def _inbox_summary(inbox):
    out = {}
    for cls in LESION_CLASSES:
        dice = [p["dice"] for img in inbox.values() for p in img[cls]["prompts"] if p["dice"] == p["dice"]]
        comps = [img[cls]["n_components"] for img in inbox.values()]
        out[cls] = {"mean_dice": float(np.mean(dice)) if dice else None, "n_prompts": len(dice),
                    "mean_components_per_image": float(np.mean(comps)) if comps else 0.0}
    return out
