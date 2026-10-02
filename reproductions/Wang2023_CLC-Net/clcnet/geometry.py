"""Patch windows. Local 256 with stride 128, context 512 centered on it."""

from __future__ import annotations

import numpy as np

from clcnet.assumptions import CONTEXT_SIZE, LOCAL_SIZE, PATCH_STRIDE


def window_starts(length: int, window: int = LOCAL_SIZE, stride: int = PATCH_STRIDE) -> list[int]:
    """Starts of a sliding window. A final border-aligned window is added when needed (A13)."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def context_origin(local_y: int, local_x: int) -> tuple[int, int]:
    margin = (CONTEXT_SIZE - LOCAL_SIZE) // 2
    return local_y - margin, local_x - margin


def crop_with_pad(array: np.ndarray, y: int, x: int, size: int, fill: float = 0):
    """Crop a square that may extend outside the array. Outside pixels are `fill` (A13, A19)."""
    if array.ndim == 2:
        output = np.full((size, size), fill, dtype=array.dtype)
    else:
        output = np.full((size, size, array.shape[2]), fill, dtype=array.dtype)
    src_y0 = max(y, 0)
    src_x0 = max(x, 0)
    src_y1 = min(y + size, array.shape[0])
    src_x1 = min(x + size, array.shape[1])
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return output
    dst_y0 = src_y0 - y
    dst_x0 = src_x0 - x
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    output[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return output


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    """Crop the spatial center of an H,W or H,W,C array."""
    y = (array.shape[0] - size) // 2
    x = (array.shape[1] - size) // 2
    return array[y : y + size, x : x + size]


def center_crop_planes(planes: np.ndarray, size: int) -> np.ndarray:
    """Crop the spatial center of a C,H,W stack."""
    y = (planes.shape[1] - size) // 2
    x = (planes.shape[2] - size) // 2
    return planes[:, y : y + size, x : x + size]
