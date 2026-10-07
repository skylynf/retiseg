"""Training patches for B-std, cut from the 1440 field-of-view canvas.

One sample: pick a scale in scale_range, cut a patch-sized window from the
canvas resized by that scale, pad it to the patch size, flip, rotate by a
multiple of 90 degrees, then photometric distortion. The resize and the cut
are one PIL ``resize(box=...)`` call, so the full scaled canvas is never
built. Masks use nearest-neighbour. Padding is 0 after normalization and
background in the label.

Each sample's randomness comes from (seed, sample id) only. EpochSampler gives
sample ids that never repeat across epochs, so a resumed run, a different
worker count and persistent workers all draw the same patches.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from bench.data.b1_input import normalize


class EpochSampler(Sampler):
    """A fresh permutation of ``n`` sample ids per epoch, offset by epoch * n."""

    def __init__(self, n, seed):
        self.n = int(n)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        order = torch.randperm(self.n, generator=generator).tolist()
        base = self.epoch * self.n
        self.epoch += 1
        return iter([base + index for index in order])


def scaled_window(image, masks, patch, scale, rng):
    """Cut a patch from the canvas resized by ``scale``. Returns image, masks and the filled height and width."""
    height, width = image.shape[:2]
    patch_h, patch_w = int(patch[0]), int(patch[1])
    scaled_h = max(1, int(round(height * scale)))
    scaled_w = max(1, int(round(width * scale)))
    top = int(rng.integers(0, max(scaled_h - patch_h, 0) + 1))
    left = int(rng.integers(0, max(scaled_w - patch_w, 0) + 1))
    out_h = min(patch_h, scaled_h - top)
    out_w = min(patch_w, scaled_w - left)
    fy = height / scaled_h
    fx = width / scaled_w
    box = (left * fx, top * fy, (left + out_w) * fx, (top + out_h) * fy)
    window = np.asarray(Image.fromarray(image).resize((out_w, out_h), Image.Resampling.BILINEAR, box=box))
    window_masks = []
    for mask in masks:
        plane = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
        window_masks.append(np.asarray(plane.resize((out_w, out_h), Image.Resampling.NEAREST, box=box)) > 0)
    canvas = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    canvas[:out_h, :out_w] = window
    planes = []
    for plane in window_masks:
        full = np.zeros((patch_h, patch_w), dtype=bool)
        full[:out_h, :out_w] = plane
        planes.append(full)
    filled = np.zeros((patch_h, patch_w), dtype=bool)
    filled[:out_h, :out_w] = True
    return canvas, planes, filled


def _clip(array):
    return np.clip(array, 0, 255)


def photometric(image, rng, params):
    """mmseg PhotoMetricDistortion order: brightness, contrast (first or last), saturation, hue.

    Each operation fires with probability ``p``. Saturation and hue use PIL
    HSV, where hue spans 0-255.
    """
    p = float(params["p"])
    img = image.astype(np.float32)
    if rng.random() < p:
        delta = float(params["brightness_delta"])
        img = _clip(img + rng.uniform(-delta, delta))
    contrast_first = rng.random() < 0.5
    low, high = params["contrast_range"]
    contrast = rng.uniform(float(low), float(high)) if rng.random() < p else None
    if contrast_first and contrast is not None:
        img = _clip(img * contrast)
    saturation = None
    if rng.random() < p:
        low, high = params["saturation_range"]
        saturation = rng.uniform(float(low), float(high))
    hue = None
    if rng.random() < p:
        degrees = float(params["hue_delta_degrees"])
        hue = rng.uniform(-degrees, degrees) * 256.0 / 360.0
    if saturation is not None or hue is not None:
        hsv = np.asarray(Image.fromarray(img.astype(np.uint8)).convert("HSV")).astype(np.float32)
        if saturation is not None:
            hsv[..., 1] = _clip(hsv[..., 1] * saturation)
        if hue is not None:
            hsv[..., 0] = np.mod(hsv[..., 0] + hue, 256.0)
        img = np.asarray(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")).astype(np.float32)
    if not contrast_first and contrast is not None:
        img = _clip(img * contrast)
    return img.astype(np.uint8)


def augment_patch(image, masks, patch, rng, params):
    """Scale and cut, flip, rot90, photometric. Returns uint8 image, bool masks and the filled region."""
    low, high = params["scale_range"]
    scale = rng.uniform(float(low), float(high))
    image, masks, filled = scaled_window(image, masks, patch, scale, rng)
    if rng.random() < float(params["hflip"]):
        image, filled = image[:, ::-1], filled[:, ::-1]
        masks = [mask[:, ::-1] for mask in masks]
    if params.get("rot90"):
        if int(patch[0]) != int(patch[1]):
            raise ValueError(f"rot90 needs a square patch, got {tuple(patch)}")
        turns = int(rng.integers(0, 4))
        if turns:
            image, filled = np.rot90(image, turns), np.rot90(filled, turns)
            masks = [np.rot90(mask, turns) for mask in masks]
    image = photometric(np.ascontiguousarray(image), rng, params["photometric"])
    return image, [np.ascontiguousarray(mask) for mask in masks], np.ascontiguousarray(filled)


class PatchTrain(Dataset):
    """``samples_per_epoch`` patches per epoch. Sample id s reads canvas s mod len(source)."""

    def __init__(self, source, patch, samples_per_epoch, params, seed):
        if source.forward_size is not None:
            raise ValueError("B-std patches are cut from the diameter canvas; the source must not resize")
        self.source = source
        self.patch = (int(patch[0]), int(patch[1]))
        self.samples = int(samples_per_epoch)
        self.params = params
        self.seed = int(seed)

    def __len__(self):
        return self.samples

    def sample(self, sample_id):
        rng = np.random.default_rng([self.seed, int(sample_id)])
        local = int(sample_id) % self.samples
        image, masks = self.source.canvas(local % len(self.source))
        return augment_patch(image, masks, self.patch, rng, self.params)

    def __getitem__(self, sample_id):
        image, masks, filled = self.sample(sample_id)
        tensor = normalize(image, self.source.mean, self.source.std)
        tensor[:, ~torch.from_numpy(filled)] = 0.0
        target = torch.from_numpy(np.stack(masks).astype(np.float32))
        return tensor, target
