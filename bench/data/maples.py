"""MAPLES-DR (Lepetit-Aimon et al., Scientific Data 2024): pixel labels for 198 MESSIDOR images.

The labels (figshare, doi 10.6084/m9.figshare.24328660, "AdditionalData")
were drawn at 1500 x 1500 on the field-of-view crop of each image; the
fundus images are MESSIDOR's (adcis.net licence, requested separately).
This module reads both directly, without the ``maples_dr`` package (which
needs pip, xlrd and rich; the server has no network), and reproduces the
package's ``resize=False`` geometry: every mask is resized (nearest) to the
MESSIDOR-ROIs.csv box of its image and placed there on a canvas of the
original MESSIDOR size (1440 x 960, 2240 x 1488 or 2304 x 1536).

Layout expected under the repository:
    dataset/maples-dr/      dataset_record.yaml, MESSIDOR-ROIs.csv, annotations/<Folder>/<name>.png
                            (the unpacked figshare archive; preannotations/ and the xls files are not read)
    dataset/messidor/       the MESSIDOR images, <name>.tif anywhere below: flat (the 200 of MAPLES-DR,
                            as on the server), in Base11/ ... Base34/ directories, or still inside
                            Base11.zip ... Base34.zip (as released); zip members are read in place.

Official split: 138 train, 60 test (dataset_record.yaml; the two duplicated
training images stay out). Classes: Microaneurysms -> MA, Hemorrhages -> HE,
Exudates -> EX, CottonWoolSpots -> SE. The dataset also marks red lesions
whose type was uncertain (RedUncertains: MA or HE) and bright ones
(BrightUncertains: EX, SE or drusen); those pixels go to the ignore layer of
the two classes they could belong to, so a prediction there counts neither
way. Drusens and Neovascularization go to masks_ext. MA, HE, EX and the
vessels were AI-pre-segmented and corrected by the retinologists; cotton
wool spots were drawn from scratch (paper, Methods); meta.json records it.
"""

import csv
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from bench.common.io import LESION_CLASSES

RAW = "dataset/maples-dr"
MESSIDOR = "dataset/messidor"
BIOMARKERS = {"MA": "Microaneurysms", "HE": "Hemorrhages", "EX": "Exudates", "SE": "CottonWoolSpots"}
UNCERTAIN = {"MA": "RedUncertains", "HE": "RedUncertains", "EX": "BrightUncertains", "SE": "BrightUncertains"}
EXTRA = {"DRUSEN": "Drusens", "NV": "Neovascularization"}
AI_ASSISTED = ("MA", "HE", "EX")   # pre-segmented by a network, then corrected; SE drawn from scratch
ANNOTATION_SIDE = 1500
EXPECTED = {"train": 138, "test": 60}
SPLIT_RULE = "official MAPLES-DR train (138) and test (60) lists of dataset_record.yaml; the two duplicated training images are left out"
GRADES = ("dr", "me")


def place_annotation(annotation, roi, shape):
    """A 1500 x 1500 label resized (nearest) into its MESSIDOR-ROIs box on an all-false canvas of ``shape``.

    ``roi`` is (x0, y0, x1, y1) with the right and bottom edges exclusive, as
    in MESSIDOR-ROIs.csv and the package's Rect.from_points; the same nearest
    resize the package applies with resize=False.
    """
    x0, y0, x1, y1 = (int(v) for v in roi)
    height, width = int(shape[0]), int(shape[1])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"ROI {roi} does not fit an image of {height}x{width}")
    annotation = np.asarray(annotation).astype(bool)
    if annotation.shape != (ANNOTATION_SIDE, ANNOTATION_SIDE):
        raise ValueError(f"annotation is {annotation.shape}, expected {ANNOTATION_SIDE}x{ANNOTATION_SIDE}")
    resized = Image.fromarray(annotation).resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    canvas = np.zeros((height, width), dtype=bool)
    canvas[y0:y1, x0:x1] = np.asarray(resized).astype(bool)
    return canvas


def case_from_sample(name, fundus, biomarker, grades=None):
    """One prepared case from a sample's masks.

    ``fundus`` is the RGB uint8 array at MESSIDOR resolution; ``biomarker(folder)``
    returns that annotation folder's boolean mask at the same resolution.
    Pure, so the mapping is testable without the data.
    """
    fundus = np.asarray(fundus)
    if fundus.ndim != 3 or fundus.shape[2] != 3 or fundus.dtype != np.uint8:
        raise ValueError(f"{name}: fundus must be RGB uint8, got {fundus.shape} {fundus.dtype}")
    shape = fundus.shape[:2]

    def read(folder):
        mask = np.asarray(biomarker(folder)).astype(bool)
        if mask.shape != shape:
            raise ValueError(f"{name} {folder}: mask {mask.shape} != image {shape}")
        return mask

    masks = {cls: read(folder) for cls, folder in BIOMARKERS.items()}
    uncertain = {folder: read(folder) for folder in sorted(set(UNCERTAIN.values()))}
    ignore = {cls: uncertain[UNCERTAIN[cls]] & ~masks[cls] for cls in LESION_CLASSES}
    extra = {cls: read(folder) for cls, folder in EXTRA.items()}
    buffer = BytesIO()
    Image.fromarray(fundus).save(buffer, format="PNG", optimize=False)
    stats = {}
    for key in GRADES:
        value = (grades or {}).get(key)
        if value:
            stats[key] = {str(value): 1}
    return {
        "bytes": buffer.getvalue(),
        "ext": "png",
        "image": fundus,
        "masks": masks,
        "binary": {"masks_ext": extra, "ignore": ignore},
        "stats": stats,
    }


def index_messidor(root, names):
    """{name: Path | (zip Path, member)} for every ``names`` entry found below ``root`` as <name>.tif."""
    root = Path(root)
    wanted = set(names)
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff") and path.stem in wanted:
            found.setdefault(path.stem, path)
    for archive in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                stem = Path(member).stem
                if not member.endswith("/") and Path(member).suffix.lower() in (".tif", ".tiff") and stem in wanted:
                    found.setdefault(stem, (archive, member))
    return found


def read_rois(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["name"]: {k: int(row[k]) for k in ("x0", "y0", "x1", "y1", "H", "W")} for row in rows}


def read_grades(raw):
    """DR and ME consensus grades from diagnosis_infos.xls when pandas can read it (xlrd); else nothing."""
    path = Path(raw) / "diagnosis_infos.xls"
    if not path.is_file():
        return {}
    try:
        import pandas as pd

        dr = pd.read_excel(path, sheet_name="DR", index_col=0)["Consensus"]
        me = pd.read_excel(path, sheet_name="ME", index_col=0)["Consensus"]
    except Exception:
        return {}
    out = {}
    for name in set(dr.index) | set(me.index):
        out[str(name)] = {"dr": str(dr.get(name, "")) or None, "me": str(me.get(name, "")) or None}
    return out


class MaplesDR:
    """The dataset from the unpacked figshare archive and the MESSIDOR images (directories or zips)."""

    extra_classes = tuple(EXTRA)
    granularity = "fine"

    def __init__(self, repo_root, raw=None, messidor=None):
        root = Path(repo_root)
        self.raw = Path(raw) if raw is not None else root / RAW
        self.messidor = Path(messidor) if messidor is not None else root / MESSIDOR
        record_path = self.raw / "dataset_record.yaml"
        if not record_path.is_file():
            raise FileNotFoundError(f"{record_path} is missing: unpack the MAPLES-DR figshare archive into {self.raw}")
        self.record = yaml.safe_load(record_path.read_text())
        self.rois = read_rois(self.raw / "MESSIDOR-ROIs.csv")
        self._grades = read_grades(self.raw)
        names = list(self.record["train"]) + list(self.record["test"])
        if not self.messidor.is_dir():
            raise FileNotFoundError(f"{self.messidor} is missing: the MESSIDOR images (Base11 ... Base34) come from adcis.net")
        self._images = index_messidor(self.messidor, names)
        missing = [name for name in names if name not in self._images]
        if missing:
            raise FileNotFoundError(f"{len(missing)} MESSIDOR images not found below {self.messidor}, e.g. {missing[:3]}")
        self._zips = {}

    def splits(self):
        train = list(self.record["train"])
        test = list(self.record["test"])
        if len(train) != EXPECTED["train"] or len(test) != EXPECTED["test"]:
            raise RuntimeError(f"MAPLES-DR official split moved: {len(train)} train, {len(test)} test (expected 138, 60)")
        if set(train) & set(test):
            raise RuntimeError("MAPLES-DR train and test lists overlap")
        # dataset_record.yaml: key is the image kept in the official 138, value is the extra copy already left out.
        copies = set((self.record.get("duplicates") or {}).values())
        if copies & (set(train) | set(test)):
            raise RuntimeError("a duplicated image is in the official lists")
        return {"train": train, "val": [], "test": test}

    def read_image_bytes(self, name):
        source = self._images[name]
        if isinstance(source, tuple):
            archive, member = source
            handle = self._zips.get(archive)
            if handle is None:
                handle = self._zips[archive] = zipfile.ZipFile(archive)
            return handle.read(member)
        return Path(source).read_bytes()

    def read_annotation(self, name, folder):
        """The 1500 x 1500 label of ``folder`` for ``name``; all-false when the file is absent (unsegmented)."""
        path = self.raw / "annotations" / folder / f"{name}.png"
        if not path.is_file():
            return np.zeros((ANNOTATION_SIDE, ANNOTATION_SIDE), dtype=bool)
        with Image.open(path) as handle:
            return np.asarray(handle.convert("1")).astype(bool)

    def read_case(self, name):
        roi = self.rois[name]
        with Image.open(BytesIO(self.read_image_bytes(name))) as handle:
            fundus = np.asarray(handle.convert("RGB"))
        if fundus.shape[:2] != (roi["H"], roi["W"]):
            raise ValueError(f"{name}: image is {fundus.shape[:2]}, MESSIDOR-ROIs.csv says {(roi['H'], roi['W'])}")
        box = (roi["x0"], roi["y0"], roi["x1"], roi["y1"])

        def biomarker(folder):
            return place_annotation(self.read_annotation(name, folder), box, fundus.shape[:2])

        return case_from_sample(name, fundus, biomarker, self._grades.get(name))


def open_maples(repo_root):
    return MaplesDR(repo_root)
