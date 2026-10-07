"""Training and prediction tensors for the B1 runner.

Reads dataset/prepared/IDRiD and dataset/prepared/DDR only. Images are the
JPEG files in images/. The field-of-view mask is the prepared luminance>10
mask. Labels are four overlapping classes in the order MA, HE, EX, SE.
Horizontal flip, probability 0.5, is the only augmentation, and only when
``augment`` is true. Validation and prediction leave the image unflipped.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bench.common.io import LESION_CLASSES, load_split, read_fov, read_mask
from bench.data.fov import crop_box

DATASETS = ("IDRiD", "DDR")
EXPECTED_COUNTS = {
    "IDRiD": {"train": 44, "val": 10, "test": 27},
    "DDR": {"train": 383, "val": 149, "test": 225},
}


def dataset_name(dataset_dir):
    path = Path(dataset_dir)
    if path.name not in DATASETS or path.parent.name != "prepared":
        raise ValueError(f"B1 reads dataset/prepared/IDRiD or dataset/prepared/DDR, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path.name


def _scaled_hw(height, width, diameter):
    scale = float(diameter) / max(int(height), int(width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _resize_rgb(image, height, width):
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))


def _resize_mask(mask, height, width):
    scaled = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
    )
    return scaled > 0


def resize_nearest_ids(ids, height, width):
    """Nearest resize of integer component ids, same sampler as ``_resize_mask``."""
    height, width = int(height), int(width)
    if tuple(ids.shape) == (height, width):
        return np.asarray(ids)
    image = Image.fromarray(np.ascontiguousarray(ids, dtype=np.int32), mode="I")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST))


def map_cropped_ids(labeled, diameter, forward_size=None):
    """Resize a cropped component-id map along the B1 canvas path.

    Long side to ``diameter``, then ``forward_size`` when the card sets one.
    An id that is absent afterwards was not sampled by nearest-neighbour.
    """
    canvas_h, canvas_w = _scaled_hw(labeled.shape[0], labeled.shape[1], diameter)
    ids = resize_nearest_ids(labeled, canvas_h, canvas_w)
    if forward_size is not None:
        ids = resize_nearest_ids(ids, int(forward_size[0]), int(forward_size[1]))
    return ids


def normalize(image, mean, std):
    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(tensor.transpose(2, 0, 1)))


def apply_forward_size(image, masks, forward_size):
    """Resize a canvas to the card's fixed input. None returns it unchanged."""
    if forward_size is None:
        return image, masks
    height, width = int(forward_size[0]), int(forward_size[1])
    image = _resize_rgb(image, height, width)
    if masks is not None:
        masks = [_resize_mask(mask, height, width) for mask in masks]
    return image, masks


def crop_resize(image, fov, diameter, forward_size=None, masks=None):
    """FOV bounding box, longer side to ``diameter``, then optional ``forward_size``.

    ``forward_size`` is (height, width). None leaves the diameter canvas as the
    network input. Masks use nearest-neighbour so a lesion pixel stays binary.
    """
    fov = np.asarray(fov, dtype=bool)
    if image.shape[:2] != fov.shape:
        raise ValueError(f"image {image.shape[:2]} != fov {fov.shape}")
    y0, y1, x0, x1 = crop_box(fov)
    cropped = image[y0:y1, x0:x1]
    canvas_h, canvas_w = _scaled_hw(cropped.shape[0], cropped.shape[1], diameter)
    image_out = _resize_rgb(cropped, canvas_h, canvas_w)
    masks_out = None
    if masks is not None:
        masks_out = [_resize_mask(mask[y0:y1, x0:x1], canvas_h, canvas_w) for mask in masks]
    canvas_fov = _resize_mask(fov[y0:y1, x0:x1], canvas_h, canvas_w)
    image_out, masks_out = apply_forward_size(image_out, masks_out, forward_size)
    network_hw = tuple(image_out.shape[:2])
    return {
        "image": image_out,
        "masks": masks_out,
        "crop": (y0, y1, x0, x1),
        "canvas_hw": (canvas_h, canvas_w),
        "network_hw": network_hw,
        "original_hw": image.shape[:2],
        "fov": fov,
        "canvas_fov": canvas_fov,
    }


def canvas_hw(dataset_dir, image_id, diameter):
    """Canvas size of one prepared image, from its field-of-view mask only."""
    y0, y1, x0, x1 = crop_box(read_fov(dataset_dir, image_id))
    return _scaled_hw(y1 - y0, x1 - x0, diameter)


def read_jpg(dataset_dir, image_id):
    path = Path(dataset_dir) / "images" / f"{image_id}.jpg"
    if not path.is_file():
        raise FileNotFoundError(f"prepared image must be a JPEG at {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def load_example(dataset_dir, image_id, diameter, mean, std, forward_size=None, masks=True):
    image = read_jpg(dataset_dir, image_id)
    fov = read_fov(dataset_dir, image_id)
    label = None
    if masks:
        label = [read_mask(dataset_dir, cls, image_id, shape=image.shape[:2]) for cls in LESION_CLASSES]
    canvas = crop_resize(image, fov, diameter, forward_size, label)
    canvas["image_id"] = image_id
    canvas["tensor"] = normalize(canvas["image"], mean, std)
    return canvas


def pad_to_multiple(tensor, multiple):
    """Pad the bottom and right with 0. ``tensor`` is already normalized."""
    multiple = int(multiple)
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)


def collate(items, pad_multiple=16):
    images, targets = zip(*items)
    height = max(tensor.shape[-2] for tensor in images)
    width = max(tensor.shape[-1] for tensor in images)
    height += (pad_multiple - height % pad_multiple) % pad_multiple
    width += (pad_multiple - width % pad_multiple) % pad_multiple

    def _one(tensor):
        return torch.nn.functional.pad(
            tensor,
            (0, width - tensor.shape[-1], 0, height - tensor.shape[-2]),
            value=0.0,
        )

    return torch.stack([_one(tensor) for tensor in images]), torch.stack([_one(tensor) for tensor in targets])


class PreparedSplit(Dataset):
    def __init__(self, dataset_dir, split, diameter, augment, mean, std, forward_size=None):
        self.root = Path(dataset_dir)
        self.name = dataset_name(self.root)
        if split == "train_official":
            raise ValueError("B1 uses splits.json train (IDRiD has 44 images), not train_official")
        if split not in EXPECTED_COUNTS[self.name]:
            raise ValueError(f"split must be one of {tuple(EXPECTED_COUNTS[self.name])}, got {split!r}")
        self.ids = load_split(self.root, split)
        expected = EXPECTED_COUNTS[self.name][split]
        if len(self.ids) != expected:
            raise ValueError(f"{self.name} {split} has {len(self.ids)} images, expected {expected}")
        self.diameter = int(diameter)
        self.augment = bool(augment)
        self.mean = mean
        self.std = std
        self.forward_size = None if forward_size is None else (int(forward_size[0]), int(forward_size[1]))
        from bench.data.canvas_cache import open_cache

        self.cache = open_cache(self.root, self.diameter, self.ids)

    def __len__(self):
        return len(self.ids)

    def canvas(self, index):
        """RGB canvas and four masks at the network input, before augmentation."""
        image_id = self.ids[index]
        if self.cache is not None:
            image, masks = self.cache.load(image_id)
            return apply_forward_size(image, masks, self.forward_size)
        canvas = load_example(
            self.root,
            image_id,
            self.diameter,
            self.mean,
            self.std,
            self.forward_size,
            masks=True,
        )
        return canvas["image"], canvas["masks"]

    def canvas_with_fov(self, index):
        """Canvas, four masks and the canvas field of view. Needs forward_size None."""
        if self.forward_size is not None:
            raise ValueError("canvas_with_fov is the diameter canvas; this split resizes to forward_size")
        image_id = self.ids[index]
        if self.cache is not None:
            image, masks = self.cache.load(image_id)
            return image, masks, self.cache.load_fov(image_id)
        canvas = load_example(self.root, image_id, self.diameter, self.mean, self.std, None, masks=True)
        return canvas["image"], canvas["masks"], canvas["canvas_fov"]

    def __getitem__(self, index):
        image, masks = self.canvas(index)
        if self.augment and np.random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            masks = [np.ascontiguousarray(mask[:, ::-1]) for mask in masks]
        target = torch.from_numpy(np.ascontiguousarray(np.stack(masks).astype(np.float32)))
        return normalize(image, self.mean, self.std), target
