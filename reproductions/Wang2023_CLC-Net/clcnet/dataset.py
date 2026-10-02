"""Patch dataset. One sample is a 256 local window and its 512 context window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from clcnet.assumptions import CONTEXT_SIZE, LOCAL_SIZE, PREPROCESS_ID, SEED
from clcnet.augment import augment_context
from clcnet.discover import LESIONS, ImageRecord, load_dataset
from clcnet.geometry import center_crop, center_crop_planes, context_origin, crop_with_pad, window_starts
from clcnet.labels import exclusive_label, presence_vector
from clcnet.preprocess import crop_to_fov

PLANE_ORDER = ("MA", "HE", "EX", "SE")


class LesionPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        dataset: str,
        split: str,
        train: bool,
        cache_dir,
        split_file=None,
        seed: int = SEED,
    ):
        self.train = train
        self.seed = seed
        self.epoch = 1
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        records = [
            record
            for record in load_dataset(Path(data_root), dataset, split_file)
            if record.split == split
        ]
        if not records:
            raise FileNotFoundError(f"no {dataset} images in split {split} under {data_root}")
        self.records = {record.id: record for record in records}
        self.prepared = {}
        self.patches = []
        for record in records:
            prepared = self._prepare(record)
            self.prepared[record.id] = prepared
            self.patches.extend(self._index_patches(record.id, prepared))
        if not self.patches:
            raise RuntimeError(f"{dataset} split {split} produced no tissue patches")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int):
        image_id, local_y, local_x = self.patches[index]
        prepared = self.prepared[image_id]
        context_y, context_x = context_origin(local_y, local_x)
        image = crop_with_pad(prepared["image"], context_y, context_x, CONTEXT_SIZE, fill=0)
        planes = np.stack(
            [
                crop_with_pad(prepared["planes"][name], context_y, context_x, CONTEXT_SIZE, fill=0)
                for name in PLANE_ORDER
            ],
            axis=0,
        )
        image = image.astype(np.float32) / 255.0
        if self.train:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, self.epoch, index])))
            image, planes = augment_context(image, planes, rng)
        local_image = center_crop(image, LOCAL_SIZE)
        local_planes = center_crop_planes(planes, LOCAL_SIZE)
        context_planes = {name: planes[i] for i, name in enumerate(PLANE_ORDER)}
        local_plane_dict = {name: local_planes[i] for i, name in enumerate(PLANE_ORDER)}
        sample = {
            "local_image": _image_tensor(local_image),
            "context_image": _image_tensor(image),
            "local_mask": torch.from_numpy(exclusive_label(local_plane_dict)),
            "context_mask": torch.from_numpy(exclusive_label(context_planes)),
            "local_cls": torch.from_numpy(presence_vector(local_plane_dict)),
            "context_cls": torch.from_numpy(presence_vector(context_planes)),
            "image_id": image_id,
            "y": local_y,
            "x": local_x,
        }
        return sample

    def full_target(self, image_id: str):
        prepared = self.prepared[image_id]
        return {
            "height": prepared["image"].shape[0],
            "width": prepared["image"].shape[1],
            "origin": prepared["origin"],
            "full_shape": prepared["full_shape"],
            "planes": prepared["planes"],
            "fov": prepared["fov"],
        }

    def _prepare(self, record: ImageRecord):
        cache_path = self.cache_dir / f"{PREPROCESS_ID}_{_safe(record.id)}.npz"
        if cache_path.exists():
            loaded = np.load(cache_path, allow_pickle=False)
            planes = {name: loaded[f"plane_{name}"] for name in LESIONS}
            return {
                "image": loaded["image"],
                "planes": planes,
                "fov": loaded["fov"],
                "origin": (int(loaded["origin_y"]), int(loaded["origin_x"])),
                "full_shape": (int(loaded["full_h"]), int(loaded["full_w"])),
            }
        image = np.asarray(Image.open(record.image_path).convert("RGB"))
        raw_planes = {}
        for lesion in LESIONS:
            raw_planes[lesion] = _read_mask(record.mask_paths.get(lesion), image.shape[:2])
        cropped, planes, fov, origin, full_shape = crop_to_fov(image, raw_planes)
        np.savez_compressed(
            cache_path,
            image=cropped,
            fov=fov.astype(np.uint8),
            origin_y=origin[0],
            origin_x=origin[1],
            full_h=full_shape[0],
            full_w=full_shape[1],
            **{f"plane_{name}": planes[name] for name in LESIONS},
        )
        return {
            "image": cropped,
            "planes": planes,
            "fov": fov,
            "origin": origin,
            "full_shape": full_shape,
        }

    def _index_patches(self, image_id: str, prepared):
        fov = prepared["fov"]
        height, width = fov.shape
        patches = []
        for y in window_starts(height):
            for x in window_starts(width):
                window = fov[y : y + LOCAL_SIZE, x : x + LOCAL_SIZE]
                if window.size == 0 or not np.any(window):
                    continue
                patches.append((image_id, y, x))
        return patches


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.transpose(np.ascontiguousarray(image), (2, 0, 1))
    return torch.from_numpy(array.astype(np.float32))


def _read_mask(source, shape):
    if source is None:
        return np.zeros(shape, dtype=np.uint8)
    paths = source if isinstance(source, (list, tuple)) else [source]
    mask = np.zeros(shape, dtype=np.uint8)
    for path in paths:
        array = np.asarray(Image.open(path))
        if array.ndim == 3:
            array = array.max(axis=2)
        if array.shape[:2] != shape:
            raise ValueError(f"mask {path} has shape {array.shape[:2]}, image has {shape}")
        mask = np.maximum(mask, (array > 0).astype(np.uint8))
    return mask


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
