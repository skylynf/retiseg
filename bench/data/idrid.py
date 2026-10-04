"""IDRiD segmentation release, read from the original directories.

Official split is train IDRiD_01–54 and test IDRiD_55–81.
There is no official validation set. Ten training images are held out with
Generator(PCG64(20261003)).choice on the sorted official training ids.
That list is fixed below so later NumPy changes cannot move it.
Optic-disc masks stay in the release and are not a lesion class.
A missing lesion file means that class is absent on that image.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.data.masks import binary_mask

VAL_SEED = 20261003
VAL_IDS = (
    "IDRiD_02",
    "IDRiD_13",
    "IDRiD_17",
    "IDRiD_27",
    "IDRiD_29",
    "IDRiD_34",
    "IDRiD_42",
    "IDRiD_47",
    "IDRiD_50",
    "IDRiD_54",
)

_FOLDERS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}
_SUFFIX = {"MA": "MA", "HE": "HE", "EX": "EX", "SE": "SE"}


def _official_train():
    return tuple(f"IDRiD_{i:02d}" for i in range(1, 55))


def _official_test():
    return tuple(f"IDRiD_{i:02d}" for i in range(55, 82))


class IdridSegmentation:
    name = "IDRiD"
    classes = LESION_CLASSES
    granularity = "fine"
    image_size = (4288, 2848)

    def __init__(self, root):
        self.root = Path(root)
        images = self.root / "1. Original Images"
        masks = self.root / "2. All Segmentation Groundtruths"
        if not images.is_dir() or not masks.is_dir():
            raise FileNotFoundError(f"not an IDRiD segmentation release: {self.root}")
        self._images = images
        self._masks = masks

    def splits(self):
        official = _official_train()
        val = VAL_IDS
        if set(val) - set(official) or len(val) != 10:
            raise RuntimeError("validation ids must be 10 images from the official training set")
        train = tuple(i for i in official if i not in set(val))
        return {
            "train": train,
            "val": val,
            "test": _official_test(),
            "train_official": official,
        }

    def image_path(self, image_id):
        split_dir = "a. Training Set" if image_id in set(_official_train()) else "b. Testing Set"
        path = self._images / split_dir / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def mask_path(self, image_id, cls):
        if cls not in _FOLDERS:
            raise KeyError(f"lesion class must be one of {LESION_CLASSES}")
        split_dir = "a. Training Set" if image_id in set(_official_train()) else "b. Testing Set"
        return self._masks / split_dir / _FOLDERS[cls] / f"{image_id}_{_SUFFIX[cls]}.tif"

    def read_image(self, image_id):
        return np.asarray(Image.open(self.image_path(image_id)))

    def read_mask(self, image_id, cls):
        path = self.mask_path(image_id, cls)
        if not path.is_file():
            with Image.open(self.image_path(image_id)) as image:
                width, height = image.size
            return np.zeros((height, width), dtype=bool)
        return binary_mask(np.asarray(Image.open(path)))


def open_idrid(repo_root):
    return IdridSegmentation(Path(repo_root) / "dataset" / "iDRID" / "A. Segmentation")
