"""Compact store of automatic masks on the canvas (tier a), one .npz per image.

A mask is kept as its bounding box plus the bit-packed crop, with the generator's
scores, so the scoring rules (bench.e5.rules) can be rerun offline without the
model: {"boxes": (N, 4) int32 [y0, y1, x0, x1), "offsets": (N + 1,) int64 into
"bits" (packed uint8), "predicted_iou", "stability_score", "area" (canvas pixels),
"window" (index of the window the mask came from), "canvas_hw"}.
"""

import numpy as np

FIELDS = ("predicted_iou", "stability_score", "area", "window")


def pack(masks, canvas_hw):
    """``masks``: iterable of (bool canvas crop, y0, x0, predicted_iou, stability_score, window_index)."""
    boxes, bits, offsets, scores = [], [], [0], {k: [] for k in FIELDS}
    for crop, y0, x0, iou, stab, window in masks:
        crop = np.asarray(crop, dtype=bool)
        if not crop.any():
            continue
        ys, xs = np.nonzero(crop)
        sub = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        boxes.append([y0 + ys.min(), y0 + ys.max() + 1, x0 + xs.min(), x0 + xs.max() + 1])
        packed = np.packbits(sub.ravel())
        bits.append(packed)
        offsets.append(offsets[-1] + packed.size)
        scores["predicted_iou"].append(float(iou))
        scores["stability_score"].append(float(stab))
        scores["area"].append(int(sub.sum()))
        scores["window"].append(int(window))
    return {
        "boxes": np.asarray(boxes, dtype=np.int32).reshape(-1, 4),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "bits": np.concatenate(bits) if bits else np.zeros(0, dtype=np.uint8),
        "predicted_iou": np.asarray(scores["predicted_iou"], dtype=np.float32),
        "stability_score": np.asarray(scores["stability_score"], dtype=np.float32),
        "area": np.asarray(scores["area"], dtype=np.int64),
        "window": np.asarray(scores["window"], dtype=np.int32),
        "canvas_hw": np.asarray(canvas_hw, dtype=np.int64),
    }


def save(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **store)


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def count(store):
    return int(store["boxes"].shape[0])


def crop_of(store, index):
    """(bool crop, y0, y1, x0, x1) of mask ``index``."""
    y0, y1, x0, x1 = (int(v) for v in store["boxes"][index])
    n = (y1 - y0) * (x1 - x0)
    packed = store["bits"][store["offsets"][index] : store["offsets"][index + 1]]
    crop = np.unpackbits(packed)[:n].reshape(y1 - y0, x1 - x0).astype(bool)
    return crop, y0, y1, x0, x1


def iter_masks(store):
    for index in range(count(store)):
        yield index, crop_of(store, index)


def full_mask(store, index):
    crop, y0, y1, x0, x1 = crop_of(store, index)
    out = np.zeros(tuple(int(v) for v in store["canvas_hw"]), dtype=bool)
    out[y0:y1, x0:x1] = crop
    return out
