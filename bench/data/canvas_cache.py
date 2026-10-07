"""Precomputed field-of-view canvases for training.

Each entry is exactly what ``bench.data.b1_input.crop_resize`` returns for one
prepared image at one diameter: the RGB canvas, the four nearest-resized
lesion masks and the nearest-resized field of view. Training then reads one small file per step instead of decoding
the original JPEG and four masks and resizing them again. Prediction and
evaluation do not read this cache; they stay at original resolution.

Layout: dataset/cache/<dataset>/d<diameter>/<id>.npz and manifest.json.
The manifest records the size and mtime of every source file. An entry whose
sources changed is rejected, not silently reused.

    python -m bench.data.canvas_cache --data dataset/prepared/IDRiD --diameter 1440
    python -m bench.data.canvas_cache --data dataset/prepared/DDR --diameter 1440

RETISEG_CANVAS_CACHE=off disables the cache in the loaders.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from bench.common.io import LESION_CLASSES, load_split, read_fov, read_mask

REPO = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO / "dataset" / "cache"
SPLITS = ("train", "val", "test")
# 2 added the canvas field of view. A manifest of another format is rebuilt.
FORMAT = 2


def cache_dir(dataset_dir, diameter):
    return CACHE_ROOT / Path(dataset_dir).name / f"d{int(diameter)}"


def _sources(dataset_dir, image_id):
    root = Path(dataset_dir)
    paths = [root / "images" / f"{image_id}.jpg", root / "fov" / f"{image_id}.png"]
    paths += [root / "masks" / cls / f"{image_id}.png" for cls in LESION_CLASSES]
    out = {}
    for path in paths:
        key = str(path.relative_to(root))
        if path.is_file():
            stat = path.stat()
            out[key] = [int(stat.st_size), int(stat.st_mtime_ns)]
        else:
            out[key] = None
    return out


def _canvas(dataset_dir, image_id, diameter):
    from bench.data.b1_input import crop_resize, read_jpg

    image = read_jpg(dataset_dir, image_id)
    fov = read_fov(dataset_dir, image_id)
    masks = [read_mask(dataset_dir, cls, image_id, shape=image.shape[:2]) for cls in LESION_CLASSES]
    return crop_resize(image, fov, diameter, None, masks)


def build(dataset_dir, diameter, splits=SPLITS, only_ids=None):
    dataset_dir = Path(dataset_dir)
    out = cache_dir(dataset_dir, diameter)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest and int(manifest.get("diameter", -1)) != int(diameter):
        raise ValueError(f"{manifest_path} is for diameter {manifest['diameter']}")
    if manifest.get("format") != FORMAT:
        manifest = {}
    entries = manifest.get("entries", {})
    written = 0
    for split in splits:
        for image_id in load_split(dataset_dir, split):
            if only_ids is not None and image_id not in only_ids:
                continue
            sources = _sources(dataset_dir, image_id)
            target = out / f"{image_id}.npz"
            if entries.get(image_id) == sources and target.is_file():
                continue
            canvas = _canvas(dataset_dir, image_id, diameter)
            masks = np.stack(canvas["masks"]).astype(bool)
            tmp = out / f"{image_id}.tmp.npz"
            np.savez(
                tmp,
                image=canvas["image"],
                masks=np.packbits(masks, axis=-1),
                mask_width=np.int64(masks.shape[-1]),
                fov=np.packbits(np.asarray(canvas["canvas_fov"], dtype=bool), axis=-1),
            )
            os.replace(tmp, target)
            entries[image_id] = sources
            written += 1
    manifest = {"dataset": dataset_dir.name, "diameter": int(diameter), "format": FORMAT, "entries": entries}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return written


class CanvasCache:
    """Read access for one dataset and diameter. Construction checks every requested id."""

    def __init__(self, dataset_dir, diameter, ids):
        self.dataset_dir = Path(dataset_dir)
        self.dir = cache_dir(dataset_dir, diameter)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        if int(manifest["diameter"]) != int(diameter):
            raise ValueError(f"cache {self.dir} has diameter {manifest['diameter']}, need {diameter}")
        if manifest.get("format") != FORMAT:
            raise ValueError(
                f"cache {self.dir} is format {manifest.get('format')}, need {FORMAT}; "
                "rebuild with python -m bench.data.canvas_cache"
            )
        for image_id in ids:
            if manifest["entries"].get(image_id) != _sources(self.dataset_dir, image_id):
                raise ValueError(
                    f"cache entry {image_id} in {self.dir} is missing or its sources changed; "
                    "rebuild with python -m bench.data.canvas_cache"
                )

    def load(self, image_id):
        with np.load(self.dir / f"{image_id}.npz") as data:
            image = data["image"]
            width = int(data["mask_width"])
            masks = np.unpackbits(data["masks"], axis=-1, count=width).astype(bool)
        return image, [masks[index] for index in range(masks.shape[0])]

    def load_fov(self, image_id):
        with np.load(self.dir / f"{image_id}.npz") as data:
            width = int(data["mask_width"])
            return np.unpackbits(data["fov"], axis=-1, count=width).astype(bool)


def open_cache(dataset_dir, diameter, ids):
    """A CanvasCache when one is built for every id, else None."""
    if os.environ.get("RETISEG_CANVAS_CACHE", "").lower() in ("off", "0", "false"):
        return None
    if not (cache_dir(dataset_dir, diameter) / "manifest.json").is_file():
        return None
    return CanvasCache(dataset_dir, diameter, ids)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--diameter", type=int, default=1440)
    args = parser.parse_args(argv)
    written = build(args.data, args.diameter)
    print(f"wrote {written} canvases to {cache_dir(args.data, args.diameter)}", flush=True)


if __name__ == "__main__":
    main()
