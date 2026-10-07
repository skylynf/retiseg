"""FGADR Seg-set, read from the release zip without unpacking.

1842 images, 1280x1280, inscribed circular field of view. Every image has
EX, SE, HE (folder Hemohedge_Masks) and MA masks; IRMA has 159 and NV 49.
Masks are L or RGB with equal channels, and every class has anti-aliased
edges with values 1-254, so a pixel is positive at >= 128, the midpoint.
The export report counts the pixels between 1 and 127 that this drops.
Filename suffix _1 (1000 images, grades 0-4), _2 (254) and _3 (588, grades
3-4 only) is treated as a source batch. One image per patient (Zhou 2021).
License: non-commercial research; no redistribution of the data or of data
derived from it. Probability maps on FGADR are not published.
"""

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.data import splits as split_lists

ARCHIVE = "FGADR-Seg-set_Release.zip"
FOLDERS = {
    "MA": "Microaneurysms_Masks",
    "HE": "Hemohedge_Masks",
    "EX": "HardExudate_Masks",
    "SE": "SoftExudate_Masks",
}
EXTRA_FOLDERS = {"IRMA": "IRMA_Masks", "NV": "Neovascularization_Masks"}
POSITIVE_AT = 128
EXPECTED_IMAGES = 1842
EXPECTED_GRADES = {0: 101, 1: 212, 2: 595, 3: 647, 4: 287}
TEST_FRACTION = 0.30
VAL_FRACTION = 0.15
SPLIT_RULE = (
    "Strata are filename suffix x DR grade. 30% of each stratum is test, then 15% of the rest is "
    "validation, both largest-remainder shares drawn from one Generator(PCG64(20261003)). "
    "One image per patient, so no grouping."
)


def binarize(arr):
    """Return (positive, dropped_pixel_count) for one FGADR mask array."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=-1)
    positive = arr >= POSITIVE_AT
    dropped = int(np.count_nonzero((arr > 0) & ~positive))
    return positive, dropped


class Fgadr:
    name = "FGADR"
    classes = LESION_CLASSES
    extra_classes = tuple(EXTRA_FOLDERS)
    granularity = "mixed"

    def __init__(self, root):
        self.root = Path(root)
        path = self.root / ARCHIVE
        if not path.is_file():
            raise FileNotFoundError(path)
        self._zip = zipfile.ZipFile(path)
        members = {}
        for name in self._zip.namelist():
            parts = name.split("/")
            if len(parts) == 3 and parts[0] == "Seg-set" and parts[2].endswith(".png"):
                members.setdefault(parts[1], set()).add(Path(parts[2]).stem)
        self._members = members
        self._ids = tuple(sorted(members.get("Original_Images", ())))
        if len(self._ids) != EXPECTED_IMAGES:
            raise RuntimeError(f"FGADR has {len(self._ids)} images, expected {EXPECTED_IMAGES}")
        for cls, folder in FOLDERS.items():
            if members.get(folder, set()) != set(self._ids):
                raise RuntimeError(f"FGADR {folder} does not cover every image")
        self._grades = self._read_grades()

    def _read_grades(self):
        text = self._zip.read("Seg-set/DR_Seg_Grading_Label.csv").decode("utf-8-sig")
        grades = {}
        for row in csv.reader(text.splitlines()):
            if len(row) < 2 or not row[0].strip():
                continue
            grades[Path(row[0].strip()).stem] = int(row[1])
        if set(grades) != set(self._ids):
            raise RuntimeError("FGADR grading table does not match the image list")
        counts = {g: sum(v == g for v in grades.values()) for g in EXPECTED_GRADES}
        if counts != EXPECTED_GRADES:
            raise RuntimeError(f"FGADR grade counts {counts} differ from the paper {EXPECTED_GRADES}")
        return grades

    def ids(self):
        return self._ids

    def grade(self, image_id):
        return self._grades[image_id]

    @staticmethod
    def suffix(image_id):
        return image_id.rsplit("_", 1)[1]

    def read_image_bytes(self, image_id):
        return self._zip.read(f"Seg-set/Original_Images/{image_id}.png")

    def read_image(self, image_id):
        with Image.open(io.BytesIO(self.read_image_bytes(image_id))) as image:
            return np.asarray(image.convert("RGB"))

    def _read(self, folder, image_id):
        if image_id not in self._members.get(folder, ()):
            return None, 0
        with Image.open(io.BytesIO(self._zip.read(f"Seg-set/{folder}/{image_id}.png"))) as image:
            return binarize(np.asarray(image))

    def read_masks(self, image_id):
        """Return (four-class masks, extra masks, dropped pixels per class)."""
        masks, extra, dropped = {}, {}, {}
        for cls, folder in FOLDERS.items():
            masks[cls], dropped[cls] = self._read(folder, image_id)
        for cls, folder in EXTRA_FOLDERS.items():
            mask, count = self._read(folder, image_id)
            if mask is not None:
                extra[cls] = mask
                dropped[cls] = count
        return masks, extra, dropped

    def make_splits(self):
        rng = split_lists.generator()
        strata = split_lists.by_stratum(self._ids, lambda i: f"{self.suffix(i)}_g{self.grade(i)}")
        rest, test = split_lists.holdout(strata, TEST_FRACTION, rng)
        strata = split_lists.by_stratum(rest, lambda i: f"{self.suffix(i)}_g{self.grade(i)}")
        train, val = split_lists.holdout(strata, VAL_FRACTION, rng)
        return {"train": train, "val": val, "test": test}


def open_fgadr(repo_root):
    return Fgadr(Path(repo_root) / "dataset" / "FGADR")
