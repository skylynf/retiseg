"""Otsu field-of-view crop. Assumption A13."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def luminance(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def otsu_threshold(gray_u8: np.ndarray) -> int:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = gray_u8.size
    if total == 0:
        return 0
    index = np.arange(256, dtype=np.float64)
    sum_all = (index * hist).sum()
    weight_back = 0.0
    sum_back = 0.0
    best_between = -1.0
    threshold = 0
    for value in range(256):
        weight_back += hist[value]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += value * hist[value]
        mean_back = sum_back / weight_back
        mean_fore = (sum_all - sum_back) / weight_fore
        between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if between > best_between:
            best_between = between
            threshold = value
    return threshold


def field_of_view(image: np.ndarray) -> np.ndarray:
    gray = np.clip(luminance(image), 0, 255).astype(np.uint8)
    # Foreground is strictly above the Otsu bin. A threshold of 0 would
    # otherwise mark every pixel, including a black background.
    mask = gray > otsu_threshold(gray)
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.size and float(border.mean()) > 0.5:
        mask = ~mask
    labeled, count = ndimage.label(mask)
    if count == 0:
        return np.ones(mask.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    mask = labeled == int(np.argmax(sizes))
    return ndimage.binary_fill_holes(mask)


def crop_to_fov(image: np.ndarray, planes: dict[str, np.ndarray]):
    """Return cropped image, cropped planes, fov mask, and the origin in the original image."""
    fov = field_of_view(image)
    ys, xs = np.nonzero(fov)
    if ys.size == 0:
        origin = (0, 0)
        fov = np.ones(image.shape[:2], dtype=bool)
        cropped = image.copy()
        cropped_planes = {name: plane.copy() for name, plane in planes.items()}
        return cropped, cropped_planes, fov, origin, image.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = image[y0:y1, x0:x1].copy()
    fov_c = fov[y0:y1, x0:x1]
    cropped[~fov_c] = 0
    cropped_planes = {}
    for name, plane in planes.items():
        cropped_planes[name] = plane[y0:y1, x0:x1].copy()
    return cropped, cropped_planes, fov_c, (y0, x0), image.shape[:2]
