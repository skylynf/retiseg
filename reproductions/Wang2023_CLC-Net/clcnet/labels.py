"""Exclusive segmentation labels and 4-d presence labels."""

from __future__ import annotations

import numpy as np

from clcnet.assumptions import CLASSIFICATION_CLASSES, OVERLAP_PRIORITY, class_index


def exclusive_label(planes: dict[str, np.ndarray]) -> np.ndarray:
    """One label per pixel. Higher priority overwrites lower priority (A9)."""
    shape = next(iter(planes.values())).shape
    label = np.zeros(shape, dtype=np.int64)
    for name in reversed(OVERLAP_PRIORITY):
        label[planes[name] > 0] = class_index(name)
    return label


def presence_vector(planes: dict[str, np.ndarray]) -> np.ndarray:
    """4-d binary vector in the section 3.2 order MA, EX, HE, SE (A8)."""
    return np.asarray(
        [1.0 if np.any(planes[name] > 0) else 0.0 for name in CLASSIFICATION_CLASSES],
        dtype=np.float32,
    )
