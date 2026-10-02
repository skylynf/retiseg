"""Geometric augmentation on the contextual patch. Assumption A12.

The same warp is applied to the image and the lesion planes. The local
patch is the center 256 of the warped 512 patch, so the two branches
stay centered on each other.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from clcnet.assumptions import FLIP_PROBABILITY, ROTATION_DEGREES, SCALE_RANGE, SHIFT_FRACTION


def augment_context(image: np.ndarray, planes: np.ndarray, rng: np.random.Generator):
    """image is H,W,3 float in [0, 1]. planes is 4,H,W float or uint8."""
    image = np.array(image, copy=True)
    planes = np.array(planes, dtype=np.float32, copy=True)
    if rng.random() < FLIP_PROBABILITY:
        image = np.flip(image, axis=1)
        planes = np.flip(planes, axis=2)
    if rng.random() < FLIP_PROBABILITY:
        image = np.flip(image, axis=0)
        planes = np.flip(planes, axis=1)
    image = np.ascontiguousarray(image)
    planes = np.ascontiguousarray(planes)

    angle = float(rng.uniform(ROTATION_DEGREES[0], ROTATION_DEGREES[1]))
    scale = float(rng.uniform(SCALE_RANGE[0], SCALE_RANGE[1]))
    height, width = image.shape[:2]
    max_shift_y = SHIFT_FRACTION * height
    max_shift_x = SHIFT_FRACTION * width
    shift_y = float(rng.uniform(-max_shift_y, max_shift_y))
    shift_x = float(rng.uniform(-max_shift_x, max_shift_x))
    matrix, offset = _sampling_matrix(height, width, angle, scale, shift_y, shift_x)
    image = _warp(image, matrix, offset, order=1, cval=0.0)
    warped_planes = np.stack(
        [_warp(planes[index], matrix, offset, order=0, cval=0.0) for index in range(planes.shape[0])],
        axis=0,
    )
    warped_planes = (warped_planes > 0.5).astype(np.uint8)
    return image.astype(np.float32), warped_planes


def _sampling_matrix(height, width, angle_deg, scale, shift_y, shift_x):
    theta = np.deg2rad(angle_deg)
    factor = 1.0 / scale
    cosine = np.cos(theta)
    sine = np.sin(theta)
    matrix = factor * np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    center = np.array([(height - 1) / 2.0, (width - 1) / 2.0], dtype=np.float64)
    offset = center - matrix @ center - np.array([shift_y, shift_x], dtype=np.float64)
    return matrix, offset


def _warp(array, matrix, offset, order, cval):
    if array.ndim == 2:
        return ndimage.affine_transform(
            array,
            matrix,
            offset=offset,
            order=order,
            cval=cval,
            output_shape=array.shape,
        )
    channels = [
        ndimage.affine_transform(
            array[:, :, index],
            matrix,
            offset=offset,
            order=order,
            cval=cval,
            output_shape=array.shape[:2],
        )
        for index in range(array.shape[2])
    ]
    return np.stack(channels, axis=-1)
