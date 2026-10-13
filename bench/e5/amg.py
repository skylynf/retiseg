"""Tier a, automatic masks (design/15 section 5.1-5.3; e5_frozen.yaml ``zero_shot``).

    python -m bench.e5 amg --model sam|sam2 --generator default|more_masks --set DDR [--split test] [--geometry G1|G2]

For every image the canvas is cut into windows (G1: native window, overlap 0.5;
G2: the whole field-of-view crop as one window, the library's own resize), the
automatic mask generator runs per window with the frozen parameters, and the
masks are stored on the canvas with predicted_iou and stability_score
(bench.e5.masks). No label is read. The scoring rules that turn the masks into
probability maps are offline (bench.e5.rules), so a new rule never reruns the
model.

Output: runs/E5/a/<model>/<generator>[__G2]/<set>_<split>/masks/<id>.npz, done.json (parameters,
environment, per-image mask counts and seconds).
"""

import time

import numpy as np

from bench.e5 import common, masks


def generator_params(frozen, model, generator):
    params = dict(frozen["zero_shot"][model]["generator"][generator])
    build_kwargs = {k: params.pop(k) for k in ("apply_postprocessing",) if k in params}
    return params, build_kwargs


def out_dir(model, generator, test_set, split, geometry="G1"):
    name = generator if geometry == "G1" else f"{generator}__{geometry}"
    return common.OUT_ROOT / "a" / model / name / common.set_dirname(test_set, split)


def masks_from_generator(records, top, left, canvas_hw, window_index):
    """Generator output of one window -> rows for bench.e5.masks.pack, clipped at the canvas edge."""
    canvas_h, canvas_w = canvas_hw
    out = []
    for record in records:
        seg = np.asarray(record["segmentation"], dtype=bool)
        h = min(seg.shape[0], canvas_h - top)
        w = min(seg.shape[1], canvas_w - left)
        if h <= 0 or w <= 0:
            continue
        out.append((seg[:h, :w], top, left, record.get("predicted_iou", 0.0), record.get("stability_score", 0.0), window_index))
    return out


def run(model_name, generator, test_set, split="test", geometry="G1", limit=None, adapter=None):
    frozen = common.load_frozen()
    params, build_kwargs = generator_params(frozen, model_name, generator)
    directory = out_dir(model_name, generator, test_set, split, geometry)
    if common.is_done(directory):
        print(f"{directory}: done", flush=True)
        return directory
    ids = common.split_ids(test_set, split)
    if limit:
        ids = ids[: int(limit)]
    mask_dir = directory / "masks"
    todo = [image_id for image_id in ids if not (mask_dir / f"{image_id}.npz").is_file()]
    if adapter is None and todo:
        from bench.e5 import models

        adapter = models.build(model_name, **build_kwargs)
    counts, seconds = {}, {}
    for image_id in ids:
        path = mask_dir / f"{image_id}.npz"
        if path.is_file():
            store = masks.load(path)
            counts[image_id] = masks.count(store)
            continue
        started = time.perf_counter()
        canvas = common.load_canvas(test_set, image_id, masks=False)
        image = canvas["image"]
        canvas_hw = tuple(canvas["canvas_hw"])
        if geometry == "G1":
            wins = common.windows(canvas_hw[0], canvas_hw[1], adapter.window)
        else:
            wins = [(0, 0, canvas_hw[0], canvas_hw[1])]
        rows = []
        for index, (top, left, h, w) in enumerate(wins):
            crop = common.window_crop(image, top, left, h, w) if geometry == "G1" else image
            rows += masks_from_generator(adapter.auto_masks(crop, params), top, left, canvas_hw, index)
        store = masks.pack(rows, canvas_hw)
        masks.save(path, store)
        counts[image_id] = masks.count(store)
        seconds[image_id] = round(time.perf_counter() - started, 2)
        print(f"{model_name}/{generator} {test_set} {image_id}: {len(wins)} windows, {counts[image_id]} masks, "
              f"{seconds[image_id]} s", flush=True)
    if limit:
        return directory   # a smoke run never writes the marker
    common.write_json(common.done_marker(directory), {
        "experiment": "E5", "tier": "a", "model": model_name, "generator": generator, "geometry": geometry,
        "params": params, "build": build_kwargs, "window": None if geometry != "G1" else adapter.window if adapter else common.WINDOW[model_name],
        "overlap": common.OVERLAP, "test_set": test_set, "split": split, "n_images": len(ids),
        "masks_per_image": counts, "seconds_per_image": seconds, "environment": common.environment(),
    })
    print(f"{directory}: {len(ids)} images, mean {np.mean(list(counts.values())):.1f} masks per image", flush=True)
    return directory
