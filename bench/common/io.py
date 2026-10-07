"""Standardized on-disk layout shared by data preparation, prediction and evaluation.

Dataset directory (written by bench/data/*):
    <dataset_dir>/meta.json            {"name": str, "classes": [..], "granularity": "fine"|"coarse"}
    <dataset_dir>/splits.json          {"train": [ids], "val": [ids], "test": [ids]}
    <dataset_dir>/images/<id>.<ext>    original-resolution RGB image
    <dataset_dir>/masks/<CLS>/<id>.png binary mask; a missing file means no lesion of that class
    <dataset_dir>/fov/<id>.png         binary field-of-view mask
    <dataset_dir>/ignore/<CLS>/<id>.png optional; pixels left out of that class in evaluation
                                       and in the loss (Retinal-Lesions gray 127). Missing = none.
    <dataset_dir>/masks_ext/<CLS>/<id>.png classes outside the four; never in the four-class mean

Prediction directory (written by bench/predict.py):
    <pred_dir>/<CLS>/<id>.png          16-bit grayscale, probability = value / PROB_LEVELS,
                                       same height and width as the original image
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

LESION_CLASSES = ("MA", "HE", "EX", "SE")
PROB_LEVELS = 65535

# Order used by M2MRF tools/prepare_labels.py when merging classes into one label map;
# a later class overwrites an earlier one where masks overlap.
M2MRF_OVERWRITE_ORDER = ("EX", "HE", "SE", "MA")


def load_meta(dataset_dir):
    return json.loads((Path(dataset_dir) / "meta.json").read_text())


def load_split(dataset_dir, split):
    splits = json.loads((Path(dataset_dir) / "splits.json").read_text())
    if split not in splits:
        raise KeyError(f"split '{split}' not in {sorted(splits)} for {dataset_dir}")
    return list(splits[split])


def _read_binary(path):
    """Positive wherever a color or gray value is above zero.

    An RGBA file keeps a constant alpha of 255 on the background, so the alpha
    channel is not a lesion. This is the same rule as bench.data.masks.binary_mask.
    """
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3:
        arr = arr.any(axis=-1)
    return arr > 0


def read_mask(dataset_dir, cls, image_id, shape=None):
    path = Path(dataset_dir) / "masks" / cls / f"{image_id}.png"
    if not path.exists():
        if shape is None:
            raise FileNotFoundError(f"{path} missing and no shape given to build an empty mask")
        return np.zeros(shape, dtype=bool)
    return _read_binary(path)


def read_fov(dataset_dir, image_id):
    return _read_binary(Path(dataset_dir) / "fov" / f"{image_id}.png")


def read_ignore(dataset_dir, cls, image_id, shape):
    path = Path(dataset_dir) / "ignore" / cls / f"{image_id}.png"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    return _read_binary(path)


def image_path(dataset_dir, image_id):
    """The one image file of an id; IDRiD and DDR are .jpg, most other sources .png."""
    found = sorted((Path(dataset_dir) / "images").glob(f"{image_id}.*"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one image for {image_id} in {dataset_dir}/images, found {len(found)}")
    return found[0]


def write_prob(pred_dir, cls, image_id, prob):
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2:
        raise ValueError(f"expected a 2-D probability map, got shape {prob.shape}")
    if np.isnan(prob).any() or prob.min() < 0 or prob.max() > 1:
        raise ValueError("probabilities must be finite and within [0, 1]")
    q = np.rint(prob * PROB_LEVELS).astype(np.uint16)
    out = Path(pred_dir) / cls
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(q).save(out / f"{image_id}.png")


def read_prob_quantized(pred_dir, cls, image_id):
    """Return the stored integer levels in [0, PROB_LEVELS] as int64."""
    path = Path(pred_dir) / cls / f"{image_id}.png"
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"{path} must be single-channel, got shape {arr.shape}")
    q = arr.astype(np.int64)
    if q.min() < 0 or q.max() > PROB_LEVELS:
        raise ValueError(f"{path} has values outside [0, {PROB_LEVELS}]")
    return q


def apply_m2mrf_overwrite(masks):
    """Make per-class masks mutually exclusive the way M2MRF's single label map does."""
    out = {c: m.copy() for c, m in masks.items()}
    order = [c for c in M2MRF_OVERWRITE_ORDER if c in out]
    for i, later in enumerate(order):
        for earlier in order[:i]:
            out[earlier] &= ~masks[later]
    return out
