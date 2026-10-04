"""Official OIA-DDR release, stored as ten concatenated zip parts.

The parts are one zip, not ten independent archives. dataset/DDR/archive.zip
is a different, cropped grading dump and is not this release.

Validation masks live in 'segmentation label', not 'label'. Train and test
use 'label'. Every segmentation image has four mask files; an all-zero mask
means that class is absent.
"""

import io
import struct
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from bench.data.masks import binary_mask

SPLITS = ("train", "valid", "test")
EXPECTED_SEGMENTATION = {"train": 383, "valid": 149, "test": 225}
EXPECTED_GRADING = {"train": 6835, "valid": 2733, "test": 4105}
LABEL_DIR = {"train": "label", "test": "label", "valid": "segmentation label"}


class _Parts:
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.offsets = []
        acc = 0
        for size in self.sizes:
            self.offsets.append(acc)
            acc += size
        self.total = acc
        self.pos = 0
        self._fh = None
        self._idx = None

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.pos
        elif whence == 2:
            offset = self.total + offset
        self.pos = offset
        return offset

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        out = bytearray()
        while n and self.pos < self.total:
            index, local = self._locate(self.pos)
            if self._idx != index:
                if self._fh:
                    self._fh.close()
                self._fh = open(self.paths[index], "rb")
                self._idx = index
            self._fh.seek(local)
            chunk = self._fh.read(min(n, self.sizes[index] - local))
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def _locate(self, pos):
        for index, offset in enumerate(self.offsets):
            if pos < offset + self.sizes[index]:
                return index, pos - offset
        raise OSError(f"position {pos} is outside the {self.total}-byte archive")


def _names(parts):
    stream = _Parts(parts)
    stream.seek(-42, 2)
    locator = stream.read(20)
    if locator[:4] != b"PK\x06\x07":
        raise ValueError("OIA-DDR parts are not one zip64 archive")
    central_offset = struct.unpack("<Q", locator[8:16])[0]
    stream.seek(central_offset)
    header = stream.read(56)
    if header[:4] != b"PK\x06\x06":
        raise ValueError("missing zip64 central-directory header")
    _entries, size, start = struct.unpack("<QQQ", header[32:56])
    stream.seek(start)
    blob = stream.read(size)
    names = []
    pos = 0
    while pos + 46 <= len(blob) and blob[pos : pos + 4] == b"PK\x01\x02":
        name_len, extra_len, comment_len = struct.unpack("<HHH", blob[pos + 28 : pos + 34])
        names.append(blob[pos + 46 : pos + 46 + name_len].decode())
        pos += 46 + name_len + extra_len + comment_len
    if len(names) != 18249:
        raise ValueError(f"expected 18249 archive members, found {len(names)}")
    return names


class DdrRelease:
    name = "DDR"
    classes = ("MA", "HE", "EX", "SE")
    granularity = "fine"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.parts = tuple(self.directory / f"DDR-dataset.zip.{i:03d}" for i in range(1, 11))
        missing = [p.name for p in self.parts if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing OIA-DDR parts: {missing}")
        if self.parts[0].read_bytes()[:2] != b"PK":
            raise ValueError("DDR-dataset.zip.001 is not the start of a zip")
        self._names = None
        self._zip = None

    def names(self):
        if self._names is None:
            self._names = _names(self.parts)
        return self._names

    def segmentation_ids(self, split):
        if split not in SPLITS:
            raise KeyError(split)
        prefix = f"DDR-dataset/lesion_segmentation/{split}/image/"
        ids = [PurePosixPath(n).stem for n in self.names() if n.startswith(prefix) and not n.endswith("/")]
        return tuple(sorted(ids))

    def label_dir(self, split):
        return LABEL_DIR[split]

    def grading_count(self, split):
        prefix = f"DDR-dataset/DR_grading/{split}/"
        return sum(n.startswith(prefix) and not n.endswith("/") for n in self.names())

    def _archive(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(_Parts(self.parts))
        return self._zip

    def _image_member(self, split, image_id):
        return f"DDR-dataset/lesion_segmentation/{split}/image/{image_id}.jpg"

    def _mask_member(self, split, image_id, cls):
        return f"DDR-dataset/lesion_segmentation/{split}/{self.label_dir(split)}/{cls}/{image_id}.tif"

    def read_image_bytes(self, split, image_id):
        return self._archive().read(self._image_member(split, image_id))

    def read_image(self, split, image_id):
        with Image.open(io.BytesIO(self.read_image_bytes(split, image_id))) as image:
            return np.asarray(image.convert("RGB"))

    def read_mask(self, split, image_id, cls):
        if cls not in self.classes:
            raise KeyError(cls)
        raw = self._archive().read(self._mask_member(split, image_id, cls))
        with Image.open(io.BytesIO(raw)) as image:
            return binary_mask(np.asarray(image))


def open_ddr(repo_root):
    return DdrRelease(Path(repo_root) / "dataset" / "DDR" / "OIA-DDR")
