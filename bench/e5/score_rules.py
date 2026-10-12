"""Masks to pixel scores for the E5 zero-shot tier (design/15 section 5.3; e5_frozen.yaml zero_shot).

An automatic mask generator returns, per window, masks with a predicted IoU and a
stability score. The rules registered for turning them into probability maps:

    pixel score   the maximum, over every mask covering the pixel across all windows,
                  of the mask's predicted_iou (``key``; stability_score is the reported
                  alternative); a pixel no mask covers scores 0; values are clipped to [0, 1]
    any lesion    that single map is the main zero-shot result and is stored in all four
                  class directories (D8), so bench.eval's AUPR_any_lesion is its pooled AUPR
    oracle class  each mask goes to the ground-truth class it overlaps most (ties to the
                  first of MA, HE, EX, SE); a mask overlapping no class goes to all four as a
                  false positive; reads the test labels, so it is an upper bound only
    area cap      masks larger than fraction x the canvas field-of-view pixels are dropped
                  before scoring (fractions of e5_frozen.yaml zero_shot.grid.area_cap_fraction;
                  1.0 is no cap); mask area is counted on the canvas, padding excluded

``combine_max`` is the shared combiner: pieces placed at window offsets, the canvas
map is their pixelwise maximum. The prompted and SAM 3 tiers reuse it with
probabilities instead of IoU-valued masks.
"""

from dataclasses import dataclass

import numpy as np

from bench.common.io import LESION_CLASSES

PREDICTED_IOU = "predicted_iou"
STABILITY = "stability_score"
SCORE_KEYS = (PREDICTED_IOU, STABILITY)


@dataclass(frozen=True)
class Mask:
    """One generated mask in window coordinates, placed at (top, left) on the canvas."""

    segmentation: np.ndarray  # bool (h, w), window coordinates
    top: int
    left: int
    predicted_iou: float
    stability_score: float

    @classmethod
    def from_record(cls, record, top=0, left=0):
        """From a SamAutomaticMaskGenerator / SAM2AutomaticMaskGenerator record."""
        return cls(
            segmentation=np.asarray(record["segmentation"], dtype=bool),
            top=int(top),
            left=int(left),
            predicted_iou=float(record["predicted_iou"]),
            stability_score=float(record["stability_score"]),
        )

    def score(self, key=PREDICTED_IOU):
        if key not in SCORE_KEYS:
            raise ValueError(f"score key must be one of {SCORE_KEYS}, got {key!r}")
        return float(getattr(self, key))

    def footprint(self, canvas_hw):
        """(rows, cols, segmentation) clipped to the canvas; padding beyond the canvas is dropped."""
        return _clip(self.segmentation, self.top, self.left, canvas_hw)

    def canvas_area(self, canvas_hw):
        _rows, _cols, piece = self.footprint(canvas_hw)
        return int(np.count_nonzero(piece))


def _clip(array, top, left, canvas_hw):
    height, width = int(canvas_hw[0]), int(canvas_hw[1])
    top, left = int(top), int(left)
    if top < 0 or left < 0:
        raise ValueError(f"window offset ({top}, {left}) is negative")
    h = min(array.shape[0], height - top)
    w = min(array.shape[1], width - left)
    if h <= 0 or w <= 0:
        return slice(top, top), slice(left, left), array[:0, :0]
    return slice(top, top + h), slice(left, left + w), array[:h, :w]


def combine_max(canvas_hw, pieces):
    """A float32 canvas map: the pixelwise maximum of ``pieces`` = iterable of (top, left, array), 0 elsewhere."""
    out = np.zeros((int(canvas_hw[0]), int(canvas_hw[1])), dtype=np.float32)
    for top, left, array in pieces:
        rows, cols, piece = _clip(np.asarray(array, dtype=np.float32), top, left, canvas_hw)
        if piece.size:
            np.maximum(out[rows, cols], piece, out=out[rows, cols])
    return out


def masks_to_score(masks, canvas_hw, key=PREDICTED_IOU):
    """The class-agnostic map: max over covering masks of their ``key``, clipped to [0, 1]."""
    pieces = ((m.top, m.left, m.segmentation.astype(np.float32) * np.clip(m.score(key), 0.0, 1.0)) for m in masks)
    return combine_max(canvas_hw, pieces)


def as_four(score):
    """The same map under every class name (D8): AUPR_any_lesion is then this map's pooled AUPR."""
    return {cls: score for cls in LESION_CLASSES}


def assign_classes(mask, gt_masks, canvas_hw):
    """Indices into LESION_CLASSES the mask is credited to: the class it overlaps most, or all four when none."""
    rows, cols, piece = mask.footprint(canvas_hw)
    overlaps = [int(np.count_nonzero(piece & np.asarray(gt, dtype=bool)[rows, cols])) for gt in gt_masks]
    best = max(overlaps) if overlaps else 0
    if best == 0:
        return list(range(len(LESION_CLASSES)))
    return [overlaps.index(best)]  # ties go to the first class in LESION_CLASSES order


def oracle_class(masks, gt_masks, canvas_hw, key=PREDICTED_IOU):
    """{cls: map}: each mask scored into the class it overlaps most (reads labels: upper bound only)."""
    if len(gt_masks) != len(LESION_CLASSES):
        raise ValueError(f"expected {len(LESION_CLASSES)} ground-truth masks, got {len(gt_masks)}")
    per_class = {cls: [] for cls in LESION_CLASSES}
    for mask in masks:
        piece = mask.segmentation.astype(np.float32) * np.clip(mask.score(key), 0.0, 1.0)
        for index in assign_classes(mask, gt_masks, canvas_hw):
            per_class[LESION_CLASSES[index]].append((mask.top, mask.left, piece))
    return {cls: combine_max(canvas_hw, pieces) for cls, pieces in per_class.items()}


def cap_pixels(fraction, fov_pixels):
    """The area cap in canvas pixels, or None for fraction 1.0 (no cap)."""
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"area cap fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return None
    return int(np.floor(fraction * int(fov_pixels)))


def area_cap(masks, fraction, fov_pixels, canvas_hw):
    """The masks whose canvas area is at most fraction x fov_pixels (all of them for fraction 1.0)."""
    cap = cap_pixels(fraction, fov_pixels)
    if cap is None:
        return list(masks)
    return [mask for mask in masks if mask.canvas_area(canvas_hw) <= cap]
