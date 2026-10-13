"""Tier a, from stored automatic masks to probability maps (design/15 section 5.3; e5_frozen.yaml ``zero_shot``).

    python -m bench.e5 score --model sam --generator default --set DDR [--split test] [--geometry G1]
                             [--output any|oracle_class] [--score predicted_iou|stability_score] [--cap train|0.005|0.02]
    python -m bench.e5 areacap            # the training-set area cap for e5_frozen.yaml (IDRiD + DDR train canvases)

Rules (all offline, same masks):
    any           a pixel's score is the largest score of the masks covering it, 0 where no mask does; the same
                  map is written to the four classes and scored with the protocol's any-lesion AUPR (main) and
                  per class. No label is read.
    oracle_class  every mask goes to the ground-truth class it overlaps most; a mask overlapping none goes to all
                  four (a false positive everywhere). Reads the test labels: an upper bound for the class
                  assignment, the localisation is still the model's.
    score         predicted_iou (main) or stability_score.
    cap           masks larger than the cap are dropped before scoring: "train" is the largest lesion component
                  of the IDRiD and DDR training canvases (e5_frozen.yaml area_cap, filled by `areacap`), a
                  fraction is of the canvas field-of-view area (docs/paper/16: 0.5 %, 2 %, 100 % = no cap).

Output: runs/E5/a/<model>/<generator>[__G2]/<set>_<split>/<rule>/{<CLS>/<id>.png, paper.json, done.json};
``rule`` is any|oracle_class[+stability][+cap_train|+cap_0.5pct|...]. bench.e5.score evaluates the directory.
"""

import time

import numpy as np

from bench.common.io import LESION_CLASSES
from bench.e5 import amg, common, masks


def rule_name(output="any", score="predicted_iou", cap=None):
    name = output
    if score != "predicted_iou":
        name += "+stability"
    if cap is not None:
        name += "+cap_" + ("train" if cap == "train" else f"{float(cap) * 100:g}pct")
    return name


def variant_name(output, generator, geometry, score, cap):
    """The results.csv variant: zero_shot:auto_any_lesion | zero_shot:auto_oracle_class, with modifiers."""
    base = "zero_shot:auto_any_lesion" if output == "any" else "zero_shot:auto_oracle_class"
    mods = []
    if generator != "default":
        mods.append(generator)
    if geometry != "G1":
        mods.append(geometry)
    if score != "predicted_iou":
        mods.append("stability")
    if cap is not None:
        mods.append("cap_" + ("train" if cap == "train" else f"{float(cap) * 100:g}pct"))
    return ":".join([base] + mods)


def cap_pixels(cap, canvas, frozen):
    if cap is None:
        return None
    if cap == "train":
        value = frozen["zero_shot"].get("area_cap")
        if value is None:
            raise ValueError("e5_frozen.yaml zero_shot.area_cap is not filled; run python -m bench.e5 areacap first")
        return int(value)
    return int(round(float(cap) * int(np.asarray(canvas["canvas_fov"], dtype=bool).sum())))


def keep_indices(store, cap_px):
    if cap_px is None:
        return np.arange(masks.count(store))
    return np.nonzero(store["area"] <= cap_px)[0]


def any_map(store, score="predicted_iou", cap_px=None):
    out = np.zeros(tuple(int(v) for v in store["canvas_hw"]), dtype=np.float32)
    scores = store[score]
    for index in keep_indices(store, cap_px):
        crop, y0, y1, x0, x1 = masks.crop_of(store, int(index))
        region = out[y0:y1, x0:x1]
        np.maximum(region, np.where(crop, np.float32(scores[index]), np.float32(0)), out=region)
    return out


def assign_classes(store, gt_masks, indices):
    """Index of the ground-truth class each mask overlaps most, or -1 when it overlaps none."""
    assigned = np.full(masks.count(store), -1, dtype=np.int64)
    for index in indices:
        crop, y0, y1, x0, x1 = masks.crop_of(store, int(index))
        overlap = [int(np.count_nonzero(crop & gt[y0:y1, x0:x1])) for gt in gt_masks]
        if max(overlap) > 0:
            assigned[index] = int(np.argmax(overlap))
    return assigned


def oracle_maps(store, gt_masks, score="predicted_iou", cap_px=None):
    hw = tuple(int(v) for v in store["canvas_hw"])
    out = {cls: np.zeros(hw, dtype=np.float32) for cls in LESION_CLASSES}
    indices = keep_indices(store, cap_px)
    assigned = assign_classes(store, gt_masks, indices)
    scores = store[score]
    for index in indices:
        crop, y0, y1, x0, x1 = masks.crop_of(store, int(index))
        targets = LESION_CLASSES if assigned[index] < 0 else (LESION_CLASSES[assigned[index]],)
        patch = np.where(crop, np.float32(scores[index]), np.float32(0))
        for cls in targets:
            region = out[cls][y0:y1, x0:x1]
            np.maximum(region, patch, out=region)
    return out


def run(model_name, generator, test_set, split="test", geometry="G1", output="any", score="predicted_iou", cap=None):
    frozen = common.load_frozen()
    source = amg.out_dir(model_name, generator, test_set, split, geometry)
    if not common.is_done(source):
        raise FileNotFoundError(f"{source} has no done.json; run python -m bench.e5 amg first")
    directory = source / rule_name(output, score, cap)
    if common.is_done(directory):
        print(f"{directory}: done", flush=True)
        return directory
    ids = common.split_ids(test_set, split)
    started = time.perf_counter()
    n_masks, n_dropped = 0, 0
    for image_id in ids:
        store = masks.load(source / "masks" / f"{image_id}.npz")
        canvas = common.load_canvas(test_set, image_id, masks=output == "oracle_class")
        cap_px = cap_pixels(cap, canvas, frozen)
        kept = keep_indices(store, cap_px)
        n_masks += masks.count(store)
        n_dropped += masks.count(store) - len(kept)
        if output == "any":
            maps = {"any": any_map(store, score, cap_px)}
        else:
            gt = [np.asarray(m, dtype=bool) for m in canvas["masks"]]
            maps = oracle_maps(store, gt, score, cap_px)
        common.write_maps(directory, image_id, canvas, maps)
    variant = variant_name(output, generator, geometry, score, cap)
    common.write_json(directory / "paper.json", common.paper_record(model_name, variant, test_set))
    common.write_json(common.done_marker(directory), {
        "experiment": "E5", "tier": "a", "rule": rule_name(output, score, cap), "output": output, "score": score,
        "cap": cap, "reads_labels": output == "oracle_class", "model": model_name, "generator": generator,
        "geometry": geometry, "test_set": test_set, "split": split, "n_images": len(ids), "n_masks": n_masks,
        "n_masks_dropped_by_cap": n_dropped, "seconds": round(time.perf_counter() - started, 1),
        "variant": variant, "environment": common.environment(),
    })
    print(f"{directory}: {len(ids)} images, {n_masks} masks, {n_dropped} dropped by the cap", flush=True)
    return directory


def training_area_cap(sources=(("IDRiD", "train"), ("DDR", "train"))):
    """Largest 8-connected lesion component (canvas pixels) over the training canvases of both sources."""
    best = {"area": 0}
    per_class = {cls: 0 for cls in LESION_CLASSES}
    for test_set, split in sources:
        for image_id in common.split_ids(test_set, split):
            canvas = common.load_canvas(test_set, image_id, masks=True)
            for cls, mask in zip(LESION_CLASSES, canvas["masks"]):
                labels, n = common.components(mask, 8)
                if n == 0:
                    continue
                areas = np.bincount(labels.ravel())[1:]
                largest = int(areas.max())
                per_class[cls] = max(per_class[cls], largest)
                if largest > best["area"]:
                    best = {"area": largest, "source": test_set, "image": image_id, "cls": cls}
    print(f"largest training lesion component on the {common.CANVAS} canvas: {best}; per class {per_class}")
    print(f"append to bench/configs/e5_frozen.yaml under zero_shot:  area_cap: {best['area']}   # {best.get('source')} "
          f"{best.get('image')} {best.get('cls')}, python -m bench.e5 areacap")
    return best, per_class
