"""Retinal-Lesions v20191227 (Wei 2020), 1593 EyePACS images at 896x896.

Every member of the release zip is ZipCrypto-encrypted. The reader takes the
unpacked directory dataset/retinal-lesions/retinal-lesions-v20191227 when it
exists, otherwise the zip with the password from RETISEG_RETLES_PASSWORD or
dataset/retinal-lesions/password.txt. dataset/ is not in git.

Masks are lesion_segs_896x896/<id>/<lesion>.png; a missing file is a negative
class, and 17 images have no mask at all. Value 255 is a lesion; 127 marks an
annotation that does not follow the AAO guideline, which the authors excluded,
so it becomes an ignore region. Any other value stops the export. Checked
2026-10-07: all 4143 masks are L 896x896 with only 0, 127 and 255; 127 occurs
in MA, HE, NV, vHE and pHE, never in EX, SE or FiP. 1505 patients, 88 with
both eyes.
vHE and pHE are not merged into HE (Playout 2024); they, NV and FiP are
extra classes. Ids keep the Kaggle "<patient>_<eye>" form, so both eyes of a
patient go to the same split.
"""

import csv
import io
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.data import splits as split_lists

TOP = "retinal-lesions-v20191227"
ARCHIVE = f"{TOP}.zip"
IMAGES = "images_896x896"
MASKS = "lesion_segs_896x896"
FILES = {
    "MA": "microaneurysm",
    "HE": "retinal_hemorrhage",
    "EX": "hard_exudate",
    "SE": "cotton_wool_spots",
}
EXTRA_FILES = {
    "vHE": "vitreous_hemorrhage",
    "pHE": "preretinal_hemorrhage",
    "NV": "neovascularization",
    "FiP": "fibrous_proliferation",
}
LESION_VALUE = 255
IGNORE_VALUE = 127
# dr_grades.csv: image id, kaggle label, our label. The README counts are "our label".
GRADE_COLUMN = "our label"
EXPECTED_IMAGES = 1593
EXPECTED_GRADES = {0: 166, 1: 337, 2: 929, 3: 99, 4: 62}
TEST_FRACTION = 0.30
VAL_FRACTION = 0.15
SPLIT_RULE = (
    "Patients (Kaggle id before _left/_right) are the unit. Strata are the patient's highest DR grade. "
    "30% of patients per stratum are test, then 15% of the remaining patients are validation, "
    "largest-remainder shares from one Generator(PCG64(20261003))."
)


def decode(arr, where=""):
    """Return (lesion, ignore) from one mask array."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        rgb = arr[..., :3]
        if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
            raise ValueError(f"{where} is a colour mask; expected gray levels 0, 127, 255")
        arr = rgb[..., 0]
    unknown = sorted(set(np.unique(arr).tolist()) - {0, IGNORE_VALUE, LESION_VALUE})
    if unknown:
        raise ValueError(f"{where} has values {unknown[:8]}; expected only 0, 127, 255")
    return arr == LESION_VALUE, arr == IGNORE_VALUE


def patient(image_id):
    return image_id.rsplit("_", 1)[0]


def _password(root):
    value = os.environ.get("RETISEG_RETLES_PASSWORD")
    if value:
        return value.encode()
    path = root / "password.txt"
    if path.is_file():
        return path.read_text().strip().encode()
    raise RuntimeError(
        f"{root / ARCHIVE} is encrypted. Unpack it to {root / TOP}, "
        "or set RETISEG_RETLES_PASSWORD, or write the password to password.txt beside the zip."
    )


class _Source:
    """Read members by path relative to the release top folder."""

    def __init__(self, root):
        root = Path(root)
        unpacked = root / TOP
        if (unpacked / IMAGES).is_dir():
            self._dir = unpacked
            self._zip = None
            self._names = [
                str(p.relative_to(unpacked)) for p in unpacked.rglob("*") if p.is_file()
            ]
        else:
            path = root / ARCHIVE
            if not path.is_file():
                raise FileNotFoundError(path)
            self._dir = None
            self._zip = zipfile.ZipFile(path)
            self._zip.setpassword(_password(root))
            prefix = f"{TOP}/"
            self._names = [n[len(prefix) :] for n in self._zip.namelist() if n.startswith(prefix) and not n.endswith("/")]

    def names(self):
        return self._names

    def read(self, relative):
        if self._dir is not None:
            return (self._dir / relative).read_bytes()
        return self._zip.read(f"{TOP}/{relative}")


class RetinalLesions:
    name = "Retinal-Lesions"
    classes = LESION_CLASSES
    extra_classes = tuple(EXTRA_FILES)
    granularity = "coarse"

    def __init__(self, root):
        self.root = Path(root)
        self._src = _Source(self.root)
        images, masks = set(), {}
        for name in self._src.names():
            parts = name.split("/")
            if len(parts) == 2 and parts[0] == IMAGES and parts[1].endswith(".jpg"):
                images.add(parts[1][: -len(".jpg")])
            elif len(parts) == 3 and parts[0] == MASKS and parts[2].endswith(".png"):
                masks.setdefault(parts[1], set()).add(parts[2][: -len(".png")])
        self._ids = tuple(sorted(images))
        if len(self._ids) != EXPECTED_IMAGES:
            raise RuntimeError(f"Retinal-Lesions has {len(self._ids)} images, expected {EXPECTED_IMAGES}")
        stray = sorted(set(masks) - images)
        if stray:
            raise RuntimeError(f"masks without an image, e.g. {stray[0]}")
        known = set(FILES.values()) | set(EXTRA_FILES.values())
        for image_id, names in masks.items():
            if names - known:
                raise RuntimeError(f"{image_id} has unknown mask files {sorted(names - known)}")
        self._masks = masks
        self._grades = None

    def ids(self):
        return self._ids

    def grades(self):
        if self._grades is None:
            text = self._src.read("dr_grades.csv").decode("utf-8-sig")
            rows = csv.DictReader(text.splitlines())
            if rows.fieldnames is None or GRADE_COLUMN not in rows.fieldnames:
                raise RuntimeError(f"dr_grades.csv has no '{GRADE_COLUMN}' column: {rows.fieldnames}")
            grades = {Path(row["image id"].strip()).stem: int(row[GRADE_COLUMN]) for row in rows}
            if set(grades) != set(self._ids):
                missing = sorted(set(self._ids) - set(grades))[:3]
                raise RuntimeError(f"dr_grades.csv does not match the image list, e.g. missing {missing}")
            counts = {g: sum(v == g for v in grades.values()) for g in EXPECTED_GRADES}
            if counts != EXPECTED_GRADES:
                raise RuntimeError(f"grade counts {counts} differ from the README {EXPECTED_GRADES}")
            self._grades = grades
        return self._grades

    def read_image_bytes(self, image_id):
        return self._src.read(f"{IMAGES}/{image_id}.jpg")

    def read_image(self, image_id):
        with Image.open(io.BytesIO(self.read_image_bytes(image_id))) as image:
            return np.asarray(image.convert("RGB"))

    def _read(self, image_id, stem, shape):
        if stem not in self._masks.get(image_id, ()):
            empty = np.zeros(shape, dtype=bool)
            return empty, empty
        with Image.open(io.BytesIO(self._src.read(f"{MASKS}/{image_id}/{stem}.png"))) as image:
            arr = np.asarray(image)
        if arr.shape[:2] != tuple(shape):
            raise ValueError(f"{image_id}/{stem} is {arr.shape[:2]}, image is {tuple(shape)}")
        return decode(arr, f"{image_id}/{stem}")

    def read_masks(self, image_id, shape):
        """Return (four-class masks, extra masks, ignore per class incl. extras)."""
        masks, extra, ignore = {}, {}, {}
        for cls, stem in FILES.items():
            masks[cls], ignore[cls] = self._read(image_id, stem, shape)
        for cls, stem in EXTRA_FILES.items():
            if stem in self._masks.get(image_id, ()):
                extra[cls], ignore[cls] = self._read(image_id, stem, shape)
        return masks, extra, ignore

    def make_splits(self):
        grades = self.grades()
        members = {}
        for image_id in self._ids:
            members.setdefault(patient(image_id), []).append(image_id)
        top = {p: max(grades[i] for i in ids) for p, ids in members.items()}
        rng = split_lists.generator()
        rest, test = split_lists.holdout(split_lists.by_stratum(members, lambda p: f"g{top[p]}"), TEST_FRACTION, rng)
        train, val = split_lists.holdout(split_lists.by_stratum(rest, lambda p: f"g{top[p]}"), VAL_FRACTION, rng)
        return {
            "train": split_lists.expand(train, members),
            "val": split_lists.expand(val, members),
            "test": split_lists.expand(test, members),
        }


def open_retlesion(repo_root):
    return RetinalLesions(Path(repo_root) / "dataset" / "retinal-lesions")
