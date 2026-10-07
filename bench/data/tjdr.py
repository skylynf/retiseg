"""TJDR pixel labels, kept as three independent Google Drive zips.

The three files are separate archives, not one split zip. Official split is
train 448 and test 113. There is no validation set. Mask values are
EX=1, HE=2, MA=3, SE=4, background=0. Images are 2048 (about 35–50 degrees)
or 3912 (133 degrees). Those two fields are not one average.
"""

import io
import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.data import splits as split_lists

CLASS_VALUE = {"EX": 1, "HE": 2, "MA": 3, "SE": 4}
EXPECTED = {"train": 448, "test": 113}
# The two fields are exported as two datasets. Official test stays whole;
# 15% of each field's official training images become validation.
FIELDS = {"TJDR_std": "standard", "TJDR_uwf": "ultrawide"}
EXPECTED_FIELD = {
    "TJDR_std": {"train": 172, "val": 30, "test": 55},
    "TJDR_uwf": {"train": 209, "val": 37, "test": 58},
}
VAL_FRACTION = 0.15
FIELD_DEGREES = {"standard": "35-50", "ultrawide": "133"}


def _png_size(header):
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a png")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


class Tjdr:
    name = "TJDR"
    classes = LESION_CLASSES
    granularity = "fine"

    def __init__(self, root):
        self.root = Path(root)
        archives = sorted(self.root.glob("*.zip"))
        if len(archives) != 3:
            raise FileNotFoundError(f"TJDR expects three zip files in {self.root}")
        self._archives = [zipfile.ZipFile(path) for path in archives]
        self._index = {}
        for archive in self._archives:
            for name in archive.namelist():
                if name.endswith("/") or not name.endswith(".png"):
                    continue
                split, kind, filename = name.split("/")
                image_id = Path(filename).stem
                slot = self._index.setdefault(image_id, {})
                if kind in slot:
                    raise RuntimeError(f"duplicate {kind} for {image_id}")
                slot[kind] = (archive, name)
                slot["split"] = split
        for image_id, slot in self._index.items():
            if "image" not in slot or "annotation" not in slot:
                raise RuntimeError(f"{image_id} is missing an image or an annotation")

    def ids(self, split):
        return tuple(sorted(i for i, slot in self._index.items() if slot["split"] == split))

    def splits(self):
        return {"train": self.ids("train"), "test": self.ids("test")}

    def field(self, image_id):
        archive, name = self._index[image_id]["image"]
        with archive.open(name) as handle:
            width, height = _png_size(handle.read(24))
        if (width, height) == (2048, 2048):
            return "standard"
        if (width, height) == (3912, 3912):
            return "ultrawide"
        raise ValueError(f"{image_id} has unexpected size {width}x{height}")

    def read_image(self, image_id):
        archive, name = self._index[image_id]["image"]
        with Image.open(io.BytesIO(archive.read(name))) as image:
            return np.asarray(image.convert("RGB"))

    def read_mask(self, image_id, lesion):
        if lesion not in CLASS_VALUE:
            raise KeyError(lesion)
        archive, name = self._index[image_id]["annotation"]
        with Image.open(io.BytesIO(archive.read(name))) as image:
            labels = np.asarray(image)
        return labels == CLASS_VALUE[lesion]

    def read_masks(self, image_id):
        archive, name = self._index[image_id]["annotation"]
        with Image.open(io.BytesIO(archive.read(name))) as image:
            labels = np.asarray(image)
        if labels.ndim != 2:
            raise ValueError(f"{image_id} annotation is not a single-channel label map")
        unknown = sorted(set(np.unique(labels).tolist()) - {0, *CLASS_VALUE.values()})
        if unknown:
            raise ValueError(f"{image_id} annotation has values {unknown} outside 0-4")
        return {cls: labels == CLASS_VALUE[cls] for cls in LESION_CLASSES}

    def read_image_bytes(self, image_id):
        archive, name = self._index[image_id]["image"]
        return archive.read(name)

    def make_splits(self):
        """Generate both field lists in one draw: standard first, then ultrawide."""
        rng = split_lists.generator()
        official = self.splits()
        out = {}
        for name, field in FIELDS.items():
            train = [i for i in official["train"] if self.field(i) == field]
            kept, val = split_lists.holdout({field: train}, VAL_FRACTION, rng)
            test = sorted(i for i in official["test"] if self.field(i) == field)
            out[name] = {"train": kept, "val": val, "test": test}
        return out


SPLIT_RULE = (
    "Official TJDR test kept whole. Validation is 15% of the official training images of the same field, "
    "drawn with Generator(PCG64(20261003)): standard field first, then ultrawide."
)


def frozen_splits(name):
    if name not in FIELDS:
        raise KeyError(name)
    return split_lists.read_frozen(name)


def open_tjdr(repo_root):
    return Tjdr(Path(repo_root) / "dataset" / "TJDR")
