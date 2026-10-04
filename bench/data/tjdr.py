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

CLASS_VALUE = {"EX": 1, "HE": 2, "MA": 3, "SE": 4}
EXPECTED = {"train": 448, "test": 113}


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
        width, height = _png_size(archive.read(name)[:24])
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


def open_tjdr(repo_root):
    return Tjdr(Path(repo_root) / "dataset" / "TJDR")
