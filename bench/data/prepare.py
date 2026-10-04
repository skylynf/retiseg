"""Write the original-resolution layout consumed by bench.common.io.

Images stay at the release resolution. Masks are one binary PNG per class;
an all-negative class is omitted, which the reader treats as a negative mask.
The FOV mask is computed here and stored beside the image. Lesion-size
statistics use only the training ids passed in, never the test split.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.data.fov import field_of_view
from bench.data.lesion_size import relative_diameters

PREPARED = {
    "IDRiD": "dataset/prepared/IDRiD",
    "DDR": "dataset/prepared/DDR",
}


def _write_binary(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def write_case(dest, image_id, image_bytes, image_ext, masks, fov):
    dest = Path(dest)
    image_path = dest / "images" / f"{image_id}.{image_ext}"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_bytes, (bytes, bytearray)):
        image_path.write_bytes(image_bytes)
    else:
        shutil.copyfile(image_bytes, image_path)
    for cls, mask in masks.items():
        if np.any(mask):
            _write_binary(dest / "masks" / cls / f"{image_id}.png", mask)
    _write_binary(dest / "fov" / f"{image_id}.png", fov)


def write_index(dest, name, splits):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(
        json.dumps({"name": name, "classes": list(LESION_CLASSES), "granularity": "fine"}, indent=2)
        + "\n"
    )
    (dest / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")


def measure_exported(dest, train_ids):
    dest = Path(dest)
    diameters = {cls: [] for cls in LESION_CLASSES}
    for image_id in train_ids:
        fov = np.asarray(Image.open(dest / "fov" / f"{image_id}.png")) > 0
        for cls in LESION_CLASSES:
            path = dest / "masks" / cls / f"{image_id}.png"
            if path.is_file():
                mask = np.asarray(Image.open(path)) > 0
            else:
                mask = np.zeros(fov.shape, dtype=bool)
            diameters[cls].extend(relative_diameters(mask, fov).tolist())
    return diameters


def export_split(dest, image_ids, load_case, train_ids, diameters):
    dest = Path(dest)
    train_ids = set(train_ids)
    for index, image_id in enumerate(image_ids, start=1):
        fov_path = dest / "fov" / f"{image_id}.png"
        if not fov_path.is_file():
            image_bytes, image_ext, image, masks = load_case(image_id)
            fov = field_of_view(image)
            write_case(dest, image_id, image_bytes, image_ext, masks, fov)
        else:
            fov = np.asarray(Image.open(fov_path)) > 0
            masks = {}
            for cls in LESION_CLASSES:
                path = dest / "masks" / cls / f"{image_id}.png"
                if path.is_file():
                    masks[cls] = np.asarray(Image.open(path)) > 0
                else:
                    masks[cls] = np.zeros(fov.shape, dtype=bool)
        if image_id in train_ids:
            for cls in LESION_CLASSES:
                diameters[cls].extend(relative_diameters(masks[cls], fov).tolist())
        if index % 20 == 0 or index == len(image_ids):
            print(f"{dest.name} {index}/{len(image_ids)}", flush=True)


def refresh_fovs(repo_root):
    """Rewrite stored FOV masks from the prepared images. Masks and JPEGs stay."""
    repo_root = Path(repo_root)
    for name, relative in PREPARED.items():
        dest = repo_root / relative
        images = sorted((dest / "images").glob("*.jpg"))
        for index, path in enumerate(images, start=1):
            image = np.asarray(Image.open(path).convert("RGB"))
            _write_binary(dest / "fov" / f"{path.stem}.png", field_of_view(image))
            if index % 40 == 0 or index == len(images):
                print(f"{name} fov {index}/{len(images)}", flush=True)


def prepare_idrid(repo_root):
    from bench.data.idrid import open_idrid

    data = open_idrid(repo_root)
    splits = data.splits()
    dest = Path(repo_root) / PREPARED["IDRiD"]
    write_index(dest, "IDRiD", {k: list(splits[k]) for k in ("train", "val", "test")})

    def load_case(image_id):
        path = data.image_path(image_id)
        image = data.read_image(image_id)
        masks = {cls: data.read_mask(image_id, cls) for cls in LESION_CLASSES}
        return path, "jpg", image, masks

    diameters = {cls: [] for cls in LESION_CLASSES}
    ids = list(splits["train"]) + list(splits["val"]) + list(splits["test"])
    export_split(dest, ids, load_case, splits["train"], diameters)
    return dest, diameters


def prepare_ddr(repo_root):
    from bench.data.ddr import open_ddr

    data = open_ddr(repo_root)
    groups = {
        "train": data.segmentation_ids("train"),
        "val": data.segmentation_ids("valid"),
        "test": data.segmentation_ids("test"),
    }
    dest = Path(repo_root) / PREPARED["DDR"]
    write_index(dest, "DDR", {k: list(v) for k, v in groups.items()})
    source_split = {}
    for split, official in (("train", "train"), ("val", "valid"), ("test", "test")):
        for image_id in groups[split]:
            source_split[image_id] = official

    def load_case(image_id):
        official = source_split[image_id]
        image = data.read_image(official, image_id)
        masks = {cls: data.read_mask(official, image_id, cls) for cls in LESION_CLASSES}
        return data.read_image_bytes(official, image_id), "jpg", image, masks

    diameters = {cls: [] for cls in LESION_CLASSES}
    ids = list(groups["train"]) + list(groups["val"]) + list(groups["test"])
    export_split(dest, ids, load_case, groups["train"], diameters)
    return dest, diameters
