"""DIARETDB0 and DIARETDB1 in the layouts currently on disk.

The directory dataset/DIARETDB1 contains DIARETDB0. DIARETDB1 v2.1 is
dataset/DiaRetDB1 V2.1. Pixel masks are not produced here: v2.1 stores
expert polygons and circles, and the consensus rule is not frozen.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

LESION_TYPES = {
    "Red_small_dots": "MA",
    "Haemorrhages": "HE",
    "Hard_exudates": "EX",
    "Soft_exudates": "SE",
}


def _repo_dataset(repo_root):
    return Path(repo_root) / "dataset"


class Diaretdb1:
    name = "DIARETDB1"
    granularity = "coarse"
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
                kind = "circle" if marking.find("circleregion") is not None else "polygon"
                label = marking.findtext("markingtype")
                found.append(
                    {
                        "expert": expert,
                        "type": label,
                        "lesion": LESION_TYPES.get(label),
                        "confidence": marking.findtext("confidencelevel"),
                        "geometry": kind,
                    }
                )
        return found


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
