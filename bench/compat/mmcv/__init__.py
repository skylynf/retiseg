"""Inference-only stand-in for mmcv 1.2.0.

The cloned M2MRF repository imports mmcv. Its network definitions stay in
official_code/M2MRF and are not copied here. This package only supplies the
constructors and image helpers that import and the IDRiD test pipeline call.
SyncBN is built as BatchNorm2d: evaluation uses the checkpoint's running
statistics, and this process is not distributed.
"""

import cv2
import numpy as np

__version__ = "1.2.0"

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def is_str(x):
    return isinstance(x, str)


def is_list_of(seq, expected_type):
    return isinstance(seq, list) and all(isinstance(item, expected_type) for item in seq)


def imread(path, flag="color"):
    flags = cv2.IMREAD_COLOR if flag == "color" else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _scale_size(size, scale):
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    width, height = size
    return int(width * float(scale[0]) + 0.5), int(height * float(scale[1]) + 0.5)


def rescale_size(old_size, scale, return_scale=False):
    """Same size rule as mmcv.image.geometric.rescale_size. old_size is (w, h)."""
    width, height = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"invalid scale {scale}")
        scale_factor = scale
    elif isinstance(scale, tuple):
        scale_factor = min(max(scale) / max(height, width), min(scale) / min(height, width))
    else:
        raise TypeError(f"scale must be a number or a pair, got {scale!r}")
    new_size = _scale_size((width, height), scale_factor)
    if return_scale:
        return new_size, scale_factor
    return new_size


def imresize(img, size, return_scale=False, interpolation="bilinear"):
    height, width = img.shape[:2]
    resized = cv2.resize(img, size, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized
    return resized, size[0] / width, size[1] / height


def imrescale(img, scale, return_scale=False, interpolation="bilinear"):
    height, width = img.shape[:2]
    new_size, scale_factor = rescale_size((width, height), scale, return_scale=True)
    resized = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return resized, scale_factor
    return resized


def imnormalize_(img, mean, std, to_rgb=True):
    mean = np.float64(np.asarray(mean).reshape(1, -1))
    stdinv = 1 / np.float64(np.asarray(std).reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def imnormalize(img, mean, std, to_rgb=True):
    return imnormalize_(img.copy().astype(np.float32), mean, std, to_rgb)


def imflip(img, direction="horizontal"):
    if direction == "horizontal":
        return np.flip(img, axis=1)
    if direction == "vertical":
        return np.flip(img, axis=0)
    raise ValueError(f"unknown flip direction {direction}")


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def impad(img, shape=None, padding=None, pad_val=0, padding_mode="constant"):
    """Pad like mmcv 1.2. ``shape`` is (height, width) and pads the right and bottom."""
    if (shape is None) == (padding is None):
        raise ValueError("set exactly one of shape and padding")
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])
    if isinstance(padding, (int, float)):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    elif len(padding) != 4:
        raise ValueError(f"padding must have length 2 or 4, got {padding}")
    modes = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    return cv2.copyMakeBorder(
        img,
        int(padding[1]),
        int(padding[3]),
        int(padding[0]),
        int(padding[2]),
        modes[padding_mode],
        value=pad_val,
    )


def imshow(*args, **kwargs):
    raise RuntimeError("imshow is not used for M2MRF evaluation")


def imwrite(*args, **kwargs):
    raise RuntimeError("imwrite is not used for M2MRF evaluation")
