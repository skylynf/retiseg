"""Field-of-view mask at the original image resolution.

IDRiD and DDR publish no field-of-view mask. Their black borders sit near
luminance 0. Otsu's threshold splits a bright half of the retina from a
darker half, so lesions in the dark half fall outside the mask. A fixed
luminance of 10 is above that border and below the dark retina on the
training images where this happened. The foreground is the largest connected
region above that value, with holes filled. Author-weight metrics do not use
this mask; they count every pixel, as the source code does.
"""

import numpy as np
from scipy import ndimage


def _luminance(image):
    image = np.asarray(image, dtype=np.float32)
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


# Above the black border, below the dark retina on the training images
# where Otsu cut the disk in half (IDRiD_04, IDRiD_06, DDR 007-5625-300,
# 007-3377-200, 007-6713-400). Checked on the training splits only.
BORDER_LUMINANCE = 10


def field_of_view(image, threshold=BORDER_LUMINANCE):
    gray = _luminance(image)
    mask = gray > threshold
    labeled, count = ndimage.label(mask)
    if count == 0:
        return np.ones(mask.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    mask = labeled == int(np.argmax(sizes))
    return ndimage.binary_fill_holes(mask)


def lesion_outside(mask, fov, margin=2):
    """Count lesion pixels outside the field of view.

    ``margin`` is an 8-connected radius. Pixels farther out than that are
    called deep. A one- or two-pixel rim is the uncertainty of the threshold,
    not a lesion cut off inside the retina.
    """
    mask = np.asarray(mask, dtype=bool)
    fov = np.asarray(fov, dtype=bool)
    if mask.shape != fov.shape:
        raise ValueError(f"mask {mask.shape} != fov {fov.shape}")
    outside = mask & ~fov
    if margin > 0:
        near = ndimage.binary_dilation(fov, structure=np.ones((3, 3), dtype=bool), iterations=margin)
        deep = outside & ~near
    else:
        deep = outside
    return int(mask.sum()), int(outside.sum()), int(deep.sum())


def crop_box(fov):
    ys, xs = np.nonzero(fov)
    if ys.size == 0:
        height, width = fov.shape
        return 0, height, 0, width
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
