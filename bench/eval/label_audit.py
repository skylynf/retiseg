"""Label geometry seen by B1 inputs. This does not score a model.

Lesion disappearance is the fraction of 8-connected components, inside the
field of view, that nearest-neighbour resize does not sample. The resize is
the training canvas: field-of-view crop, long side 1440, then the network's
forward size when it has one.

Softmax deletion is the fraction of positive pixels of each class removed
when a later class in EX, HE, SE, MA overwrites an earlier one. That map is
the training target of a softmax head. Evaluation keeps the original four
masks, including pixels this rule deletes.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from bench.common.io import LESION_CLASSES, load_split, read_fov, read_mask
from bench.common.labels import overwrite_removal_counts
from bench.data.b1_input import map_cropped_ids
from bench.data.fov import crop_box

_STRUCTURE = ndimage.generate_binary_structure(2, 2)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "input_label_audit.json"

# None keeps the long-side-1440 canvas. The squares are the cards that resize again.
FORWARD_SIZES = {
    "long_side_1440": None,
    "h2former_960": (960, 960),
    "fct_384": (384, 384),
    "swin_224": (224, 224),
}


def _empty_counts():
    return {cls: {"components": 0, "disappeared": 0} for cls in LESION_CLASSES}


def _empty_pixels():
    return {cls: {"positive": 0, "removed": 0} for cls in LESION_CLASSES}


def component_fate(mask, fov, diameter, forward_size):
    """Return (n_components, n_disappeared, resized_mask) for one class."""
    mask = np.asarray(mask, dtype=bool) & np.asarray(fov, dtype=bool)
    y0, y1, x0, x1 = crop_box(fov)
    cropped = mask[y0:y1, x0:x1]
    labeled, count = ndimage.label(cropped, structure=_STRUCTURE)
    mapped = map_cropped_ids(labeled, diameter, forward_size)
    if count == 0:
        return 0, 0, mapped > 0
    survived = set(np.unique(mapped).tolist()) - {0}
    disappeared = count - len(survived & set(range(1, count + 1)))
    return int(count), int(disappeared), mapped > 0


def _add_pixels(total, counts):
    for cls in LESION_CLASSES:
        total[cls]["positive"] += counts[cls]["positive"]
        total[cls]["removed"] += counts[cls]["removed"]


def _ratios(counts, numerator, denominator):
    out = {}
    for cls in LESION_CLASSES:
        denom = int(counts[cls][denominator])
        numer = int(counts[cls][numerator])
        out[cls] = {
            denominator: denom,
            numerator: numer,
            "ratio": None if denom == 0 else numer / denom,
        }
    return out


def audit_split(dataset_dir, split, diameter=1440):
    dataset_dir = Path(dataset_dir)
    disappearance = {name: _empty_counts() for name in FORWARD_SIZES}
    pixels = {name: _empty_pixels() for name in ("original_fov",) + tuple(FORWARD_SIZES)}
    n_images = 0
    for image_id in load_split(dataset_dir, split):
        n_images += 1
        fov = read_fov(dataset_dir, image_id)
        masks = [read_mask(dataset_dir, cls, image_id, shape=fov.shape) & fov for cls in LESION_CLASSES]
        original = torch.from_numpy(np.stack(masks)).unsqueeze(0)
        _add_pixels(pixels["original_fov"], overwrite_removal_counts(original))
        for name, forward_size in FORWARD_SIZES.items():
            resized = []
            for cls, mask in zip(LESION_CLASSES, masks):
                components, disappeared, mapped = component_fate(mask, fov, diameter, forward_size)
                disappearance[name][cls]["components"] += components
                disappearance[name][cls]["disappeared"] += disappeared
                resized.append(mapped)
            stacked = torch.from_numpy(np.stack(resized)).unsqueeze(0)
            _add_pixels(pixels[name], overwrite_removal_counts(stacked))
    return {
        "n_images": n_images,
        "lesion_disappearance": {
            name: _ratios(counts, "disappeared", "components") for name, counts in disappearance.items()
        },
        "softmax_overwrite_removed": {
            name: _ratios(counts, "removed", "positive") for name, counts in pixels.items()
        },
    }


def audit_datasets(repo=ROOT, diameter=1440):
    repo = Path(repo)
    payload = {
        "evaluation_labels": "original_four_class",
        "note": (
            "Disappearance is an 8-connected component inside the field of view "
            "that nearest-neighbour resize does not sample. "
            "softmax_overwrite_removed is the training target of an exclusive head. "
            "Evaluation does not use that target."
        ),
        "forward_sizes": {name: list(size) if size is not None else None for name, size in FORWARD_SIZES.items()},
        "datasets": {},
    }
    for name in ("IDRiD", "DDR"):
        root = repo / "dataset" / "prepared" / name
        payload["datasets"][name] = {
            split: audit_split(root, split, diameter) for split in ("train", "val", "test")
        }
    return payload


def write_audit(path=OUT, repo=ROOT):
    payload = audit_datasets(repo)
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main():
    path = write_audit()
    print(path)


if __name__ == "__main__":
    main()
