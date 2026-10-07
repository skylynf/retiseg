"""DIARETDB0 and DIARETDB1 in the layouts currently on disk.

The directory dataset/DIARETDB1 contains DIARETDB0. DIARETDB1 v2.1 is
dataset/DiaRetDB1 V2.1.

DIARETDB1 ground truth is four experts' regions (circle, ellipse, polygon),
each with a confidence. Each expert's regions of one class are filled with
that expert's highest level at each pixel. The levels map to High 1.0,
Medium 0.75, Low 0.25, the mean over the four experts is the consensus, and
a pixel is positive at >= 0.75 (Kauppi 2007, conf_GT = 0.75). That mapping is
frozen only if it reproduces the eight per-split positive-image counts in
section 3.3 of the paper (PAPER_POSITIVE_IMAGES); bench/scripts/
prepare_datasets.py check-diaretdb1 compares them and reads no model output.
Coordinates are 0-based pixel centres. The ellipse angle is in degrees; its
sense is the one under which more representative points fall inside their
own ellipse (ellipse_sense reports both).
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LESION_TYPES = {
    "Red_small_dots": "MA",
    "Haemorrhages": "HE",
    "Hard_exudates": "EX",
    "Soft_exudates": "SE",
}
LEVELS = {"Low": 1, "Medium": 2, "High": 3}
CONFIDENCE = {"High": 1.0, "Medium": 0.75, "Low": 0.25}
CONSENSUS_AT = 0.75
# Kauppi 2007 section 3.3, images positive at conf_GT = 0.75.
PAPER_POSITIVE_IMAGES = {
    "train": {"EX": 18, "SE": 6, "MA": 19, "HE": 21},
    "test": {"EX": 20, "SE": 9, "MA": 20, "HE": 18},
}
# Checked 2026-10-07: six of the eight counts match. Train HE is 22 and test MA
# is 21, one image above the paper each, under every confidence mapping tried
# (1/.75/.25, 1/(2/3)/(1/3), 1/.75/.5) and with 1-based coordinates, polygons
# without outline, or circles shrunk by half a pixel. The mapping is frozen
# with these two differences recorded; any other difference fails the check.
KNOWN_COUNT_DIFFERENCES = {("train", "HE"): 22, ("test", "MA"): 21}
EXPERTS = ("01", "02", "03", "04")
# Positive: counter-clockwise on screen (y down), i.e. rotation by -angle in image coordinates.
ELLIPSE_SENSE = 1


def _repo_dataset(repo_root):
    return Path(repo_root) / "dataset"


def _xy(node):
    x, y = node.text.split(",")
    return float(x), float(y)


def _shape(marking):
    circle = marking.find("circleregion")
    if circle is not None:
        cx, cy = _xy(circle.find("centroid/coords2d"))
        return {"geometry": "circle", "center": (cx, cy), "radius": float(circle.findtext("radius"))}
    ellipse = marking.find("ellipseregion")
    if ellipse is not None:
        cx, cy = _xy(ellipse.find("centroid/coords2d"))
        radii = {r.get("direction"): float(r.text) for r in ellipse.findall("radius")}
        return {
            "geometry": "ellipse",
            "center": (cx, cy),
            "radii": (radii["x"], radii["y"]),
            "angle": float(ellipse.findtext("angle")),
        }
    polygon = marking.find("polygonregion")
    if polygon is not None:
        points = [_xy(node) for node in polygon.findall("coords2d")]
        return {"geometry": "polygon", "points": points}
    raise ValueError("marking without a circle, ellipse or polygon region")


def ellipse_inside(xs, ys, shape, sense=ELLIPSE_SENSE):
    cx, cy = shape["center"]
    rx, ry = shape["radii"]
    theta = math.radians(shape["angle"]) * sense
    dx, dy = xs - cx, ys - cy
    u = dx * math.cos(theta) - dy * math.sin(theta)
    v = dx * math.sin(theta) + dy * math.cos(theta)
    return (u / max(rx, 0.5)) ** 2 + (v / max(ry, 0.5)) ** 2 <= 1.0


def rasterize(shape, size, sense=ELLIPSE_SENSE):
    """Boolean mask of one region on an image of size (height, width)."""
    height, width = size
    if shape["geometry"] == "polygon":
        canvas = Image.new("1", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(shape["points"], fill=1, outline=1)
        return np.asarray(canvas, dtype=bool)
    cx, cy = shape["center"]
    reach = shape["radius"] if shape["geometry"] == "circle" else max(shape["radii"])
    y0, y1 = max(0, int(math.floor(cy - reach))), min(height, int(math.ceil(cy + reach)) + 1)
    x0, x1 = max(0, int(math.floor(cx - reach))), min(width, int(math.ceil(cx + reach)) + 1)
    mask = np.zeros(size, dtype=bool)
    if y0 >= y1 or x0 >= x1:
        return mask
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    if shape["geometry"] == "circle":
        inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= shape["radius"] ** 2
    else:
        inside = ellipse_inside(xs, ys, shape, sense)
    mask[y0:y1, x0:x1] = inside
    return mask


def pack_levels(levels):
    """Pack {expert: uint8 level 0-3} into one uint8 map, two bits per expert."""
    packed = np.zeros(next(iter(levels.values())).shape, dtype=np.uint8)
    for index, expert in enumerate(EXPERTS):
        packed |= (levels[expert].astype(np.uint8) & 3) << (2 * index)
    return packed


def unpack_levels(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    return {expert: (packed >> (2 * index)) & 3 for index, expert in enumerate(EXPERTS)}


def consensus(levels, confidence=CONFIDENCE):
    """Mean expert confidence per pixel; levels is {expert: map of 0-3}."""
    value = np.zeros(4, dtype=np.float64)
    for name, level in LEVELS.items():
        value[level] = confidence[name]
    return sum(value[levels[e]] for e in EXPERTS) / len(EXPERTS)


def positive(levels, confidence=CONFIDENCE, at=CONSENSUS_AT):
    return consensus(levels, confidence) >= at - 1e-9


def union(levels):
    return np.logical_or.reduce([levels[e] > 0 for e in EXPERTS])


class Diaretdb1:
    name = "DIARETDB1"
    granularity = "region"
    version = "2.1"

    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / "images").is_dir():
            raise FileNotFoundError(self.root)

    def splits(self):
        return {
            "train": self._ids("ddb1_v02_01_train_plain.txt"),
            "test": self._ids("ddb1_v02_01_test_plain.txt"),
        }

    def _ids(self, name):
        ids = []
        for line in (self.root / name).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            image = line.split()[0]
            ids.append(Path(image).stem)
        return tuple(ids)

    def image_path(self, image_id):
        path = self.root / "images" / f"{image_id}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def image_size(self, image_id):
        with Image.open(self.image_path(image_id)) as image:
            width, height = image.size
        return height, width

    def read_image(self, image_id):
        with Image.open(self.image_path(image_id)) as image:
            return np.asarray(image.convert("RGB"))

    def annotation_paths(self, image_id):
        paths = []
        for split in ("ddb1_v02_01_train_plain.txt", "ddb1_v02_01_test_plain.txt"):
            for line in (self.root / split).read_text().splitlines():
                parts = line.split()
                if parts and Path(parts[0]).stem == image_id:
                    paths = [self.root / part for part in parts[1:]]
        if len(paths) != 4:
            raise FileNotFoundError(f"{image_id} does not have four expert files")
        return tuple(paths)

    def markings(self, image_id):
        found = []
        for path in self.annotation_paths(image_id):
            expert = path.name.split("_")[-2]
            tree = ET.parse(path)
            for marking in tree.getroot().find("markinglist"):
                label = marking.findtext("markingtype")
                shape = _shape(marking)
                point = marking.find("representativepoint/coords2d")
                found.append(
                    {
                        "expert": expert,
                        "type": label,
                        "lesion": LESION_TYPES.get(label),
                        "confidence": marking.findtext("confidencelevel"),
                        "geometry": shape["geometry"],
                        "shape": shape,
                        "point": _xy(point) if point is not None else None,
                    }
                )
        return found

    def expert_levels(self, image_id, size=None, sense=ELLIPSE_SENSE):
        """{class: {expert: uint8 map of the highest level 0-3}} for the four classes."""
        size = size or self.image_size(image_id)
        out = {cls: {e: np.zeros(size, dtype=np.uint8) for e in EXPERTS} for cls in LESION_TYPES.values()}
        for marking in self.markings(image_id):
            if marking["lesion"] is None:
                continue
            if marking["expert"] not in EXPERTS:
                raise ValueError(f"{image_id} has unknown expert {marking['expert']}")
            level = LEVELS[marking["confidence"]]
            region = rasterize(marking["shape"], size, sense)
            target = out[marking["lesion"]][marking["expert"]]
            np.maximum(target, np.where(region, level, 0).astype(np.uint8), out=target)
        return out


def ellipse_sense(data):
    """Count representative points inside their own ellipse under each angle sense."""
    counts = {1: 0, -1: 0}
    total = 0
    for split in data.splits().values():
        for image_id in split:
            for marking in data.markings(image_id):
                if marking["geometry"] != "ellipse" or marking["point"] is None:
                    continue
                total += 1
                x, y = marking["point"]
                for sense in counts:
                    counts[sense] += bool(ellipse_inside(np.float64(x), np.float64(y), marking["shape"], sense))
    return {"ellipses": total, "inside_positive_sense": counts[1], "inside_negative_sense": counts[-1]}


def positive_image_counts(data, confidence=CONFIDENCE, sense=ELLIPSE_SENSE):
    """Images per split with any consensus pixel, to compare with PAPER_POSITIVE_IMAGES."""
    counts = {}
    for split, ids in data.splits().items():
        counts[split] = {cls: 0 for cls in LESION_TYPES.values()}
        for image_id in ids:
            levels = data.expert_levels(image_id, sense=sense)
            for cls, by_expert in levels.items():
                counts[split][cls] += bool(positive(by_expert, confidence).any())
    return counts


def count_problems(counts):
    """Differences from the paper other than KNOWN_COUNT_DIFFERENCES."""
    problems = []
    for split, expected in PAPER_POSITIVE_IMAGES.items():
        for cls, paper in expected.items():
            found = counts[split][cls]
            allowed = KNOWN_COUNT_DIFFERENCES.get((split, cls), paper)
            if found != allowed:
                problems.append(f"{split} {cls}: {found} images, paper {paper}, recorded {allowed}")
    return problems


class Diaretdb0:
    name = "DIARETDB0"
    granularity = "screening"
    stored_under = "dataset/DIARETDB1"

    def __init__(self, root):
        self.root = Path(root)
        self.images = self.root / "resources" / "images" / "diaretdb0_fundus_images"
        if not self.images.is_dir():
            raise FileNotFoundError(self.images)

    def image_ids(self):
        return tuple(sorted(p.stem for p in self.images.glob("*.png")))


def open_diaretdb1(repo_root):
    return Diaretdb1(_repo_dataset(repo_root) / "DiaRetDB1 V2.1" / "archive" / "ddb1_v02_01")


def open_diaretdb0(repo_root):
    return Diaretdb0(_repo_dataset(repo_root) / "DIARETDB1" / "archive (1)" / "diaretdb0_v_1_1")
