"""Binary lesion masks from the files shipped with each release.

Palette and grayscale files are positive wherever the value is above zero.
Color files use the color channels only. An RGBA file keeps a constant alpha
of 255 on the background, so the alpha channel is not a lesion.
"""

import numpy as np


def binary_mask(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3].any(axis=-1)
    elif arr.ndim == 3:
        arr = arr.any(axis=-1)
    return arr > 0
