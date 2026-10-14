"""Canvases and windows of the E5 inference tiers (design/15 section 4; e5_frozen.yaml geometry).

A test image is shown on the same canvas as every B-std model: the field-of-view
crop resized so its longer side is 1440 (bench/configs/e3a_frozen.yaml
canvas_reference). TJDR_uwf is the exception fixed in e5_frozen.yaml
geometry.G1 (D7, 2026-10-12): 4256 for the main variant, 1440 for the "fov"
variant. The canvas stays uint8 RGB here; SAM, SAM 2, SAM 3 and MedSAM apply
their own normalization.

G1 cuts the canvas into the model's native windows (1024, SAM 3 1008) with
overlap 0.5 on the grid of bench.infer: stride window x (1 - overlap), the last
window flush with the far edge, a canvas smaller than the window padded with
zeros at the bottom and right so every window has the same size and the same
point density. G2 hands the whole native-resolution crop to the model's own
resize, as a user of the release would.

``paste`` brings a canvas-resolution score map back onto the original image
the way bench.predict does: bilinear to the crop box, 0 outside the field of
view. Files are written elsewhere (bench/e5/outputs.py) through
bench.common.io.write_prob.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from bench import e3a
from bench.common.io import LESION_CLASSES, load_split, read_fov, read_mask
from bench.data.b1_input import crop_resize, read_image
from bench.infer import window_starts

G1 = "G1"
G2 = "G2"
GEOMETRIES = (G1, G2)
MAIN = e3a.MAIN
NATIVE = 0  # long side 0: no resize (bench.data.b1_input._scaled_hw)


def _e3a_frozen():
    return e3a.load_frozen()


def test_sets():
    """The seven test-set names of bench/configs/e3a_frozen.yaml, in its order."""
    return list(_e3a_frozen()["test_sets"])


def data_dir(set_name):
    frozen = _e3a_frozen()
    if set_name not in frozen["test_sets"]:
        raise ValueError(f"unknown test set {set_name!r}; known {list(frozen['test_sets'])}")
    return e3a.test_data_dir(set_name, frozen)


def image_ids(set_name, split):
    return load_split(data_dir(set_name), split)


def variants_for(set_name, frozen):
    """["main"] plus the comparison variants e5_frozen.yaml registers for this set (TJDR_uwf: "fov")."""
    special = frozen["geometry"][G1].get(set_name)
    if not isinstance(special, dict):
        return [MAIN]
    return [MAIN] + list(special.get("comparison_variant") or {})


def long_side_for(set_name, variant, frozen):
    """Long side of the G1 canvas: e3a canvas_reference, or the e5_frozen.yaml entry of the set."""
    e3a_frozen = _e3a_frozen()
    if set_name not in e3a_frozen["test_sets"]:
        raise ValueError(f"unknown test set {set_name!r}; known {list(e3a_frozen['test_sets'])}")
    special = frozen["geometry"][G1].get(set_name)
    if not isinstance(special, dict):
        if variant != MAIN:
            raise ValueError(f"{set_name} has no variant {variant!r}")
        return int(e3a_frozen["canvas_reference"])
    if variant == MAIN:
        return int(special["long_side"])
    variants = special.get("comparison_variant") or {}
    if variant not in variants:
        raise ValueError(f"{set_name} has no variant {variant!r}; registered {list(variants)}")
    return int(variants[variant])


def window_size(frozen, model):
    table = frozen["geometry"][G1]["window"]
    if model not in table:
        raise ValueError(f"no G1 window for {model!r}; registered {list(table)}")
    return int(table[model])


def overlap(frozen):
    return float(frozen["geometry"][G1]["overlap"])


@dataclass
class Canvas:
    image: np.ndarray        # uint8 (H, W, 3), the field-of-view crop at the canvas size
    fov: np.ndarray          # bool (H, W), the field of view on the canvas
    masks: list | None       # four bool (H, W) arrays in LESION_CLASSES order, when requested
    crop: tuple              # (y0, y1, x0, x1) of the field-of-view box on the original image
    original_hw: tuple
    original_fov: np.ndarray  # bool, original size
    long_side: int

    @property
    def hw(self):
        return tuple(self.image.shape[:2])


def canvas(dataset_dir, image_id, long_side, masks=False):
    """The canvas of one prepared image; ``long_side`` 0 keeps the native crop size (G2)."""
    image = read_image(dataset_dir, image_id)
    fov = read_fov(dataset_dir, image_id)
    label = [read_mask(dataset_dir, cls, image_id, shape=image.shape[:2]) for cls in LESION_CLASSES] if masks else None
    out = crop_resize(image, fov, int(long_side), None, label)
    return Canvas(
        image=np.ascontiguousarray(out["image"], dtype=np.uint8),
        fov=np.asarray(out["canvas_fov"], dtype=bool),
        masks=None if out["masks"] is None else [np.asarray(m, dtype=bool) for m in out["masks"]],
        crop=tuple(int(v) for v in out["crop"]),
        original_hw=tuple(int(v) for v in out["original_hw"]),
        original_fov=np.asarray(fov, dtype=bool),
        long_side=int(long_side),
    )


def windows(canvas_hw, window, overlap_fraction):
    """(top, left) of every G1 window, on the grid of bench.infer.canvas_logits.

    Stride is window x (1 - overlap) rounded; the last window is flush with the
    far edge; a side shorter than the window gets the single start 0 (the crop
    is padded by ``crop_window``).
    """
    height, width = int(canvas_hw[0]), int(canvas_hw[1])
    win = int(window)
    if not 0.0 <= float(overlap_fraction) < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap_fraction}")
    stride = max(1, int(round(win * (1.0 - float(overlap_fraction)))))
    return [
        (top, left)
        for top in window_starts(max(height, win), win, stride)
        for left in window_starts(max(width, win), win, stride)
    ]


def crop_window(array, top, left, window):
    """A window x window crop of an (H, W[, C]) array, zero-padded at the bottom and right.

    Returns (crop, (valid_height, valid_width)): the part of the crop that is
    canvas, not padding.
    """
    win = int(window)
    height, width = array.shape[:2]
    top, left = int(top), int(left)
    if top < 0 or left < 0 or top >= max(height, 1) or left >= max(width, 1):
        raise ValueError(f"window start ({top}, {left}) outside a {height}x{width} canvas")
    piece = array[top : top + win, left : left + win]
    valid = (int(piece.shape[0]), int(piece.shape[1]))
    if valid == (win, win):
        return piece, valid
    out = np.zeros((win, win) + array.shape[2:], dtype=array.dtype)
    out[: valid[0], : valid[1]] = piece
    return out, valid


def plan(frozen, geometry, model, canvas_hw):
    """(window, [(top, left)]) for a canvas: G1 is the model's windows, G2 one pass over the whole crop."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    if geometry == G2:
        return None, [(0, 0)]
    win = window_size(frozen, model)
    return win, windows(canvas_hw, win, overlap(frozen))


def _resize_plane(plane, height, width):
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    if plane.shape == (height, width):
        return np.clip(plane, 0.0, 1.0)
    resized = Image.fromarray(plane).resize((width, height), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def paste(canvas_map, cv):
    """A canvas-resolution map in [0, 1] placed on the original image: bilinear to the crop box, 0 outside the FOV."""
    canvas_map = np.asarray(canvas_map, dtype=np.float32)
    if canvas_map.shape != cv.hw:
        raise ValueError(f"map {canvas_map.shape} is not the canvas size {cv.hw}")
    y0, y1, x0, x1 = cv.crop
    placed = np.zeros(cv.original_hw, dtype=np.float32)
    placed[y0:y1, x0:x1] = _resize_plane(canvas_map, y1 - y0, x1 - x0)
    placed[~cv.original_fov] = 0.0
    return placed
