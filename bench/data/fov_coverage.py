"""Count lesion pixels that fall outside the stored field of view.

The acceptance rule uses the training split only. A class passes when fewer
than 0.1% of its training lesion pixels lie more than two pixels (8-connected)
outside the mask. Test counts are reported and do not enter the decision.
"""

import json
from pathlib import Path

import numpy as np

from bench.common.io import LESION_CLASSES, load_split, read_fov, read_mask
from bench.data.fov import lesion_outside

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("IDRiD", "DDR")
DEEP_MARGIN = 2
MAX_DEEP_FRACTION = 0.001
OUT = ROOT / "bench" / "eval" / "fov_coverage.json"


def _split_counts(dataset_dir, split):
    ids = load_split(dataset_dir, split)
    totals = {cls: np.zeros(3, dtype=np.int64) for cls in LESION_CLASSES}
    worst = {cls: {"image_id": None, "deep": 0, "lesion": 0} for cls in LESION_CLASSES}
    fov_pixels = 0
    image_pixels = 0
    for image_id in ids:
        fov = read_fov(dataset_dir, image_id)
        fov_pixels += int(fov.sum())
        image_pixels += int(fov.size)
        for cls in LESION_CLASSES:
            lesion, outside, deep = lesion_outside(read_mask(dataset_dir, cls, image_id, fov.shape), fov, DEEP_MARGIN)
            totals[cls] += (lesion, outside, deep)
            if deep > worst[cls]["deep"]:
                worst[cls] = {"image_id": image_id, "deep": int(deep), "lesion": int(lesion)}
    per_class = {}
    for cls in LESION_CLASSES:
        lesion, outside, deep = (int(v) for v in totals[cls])
        per_class[cls] = {
            "lesion_pixels": lesion,
            "outside_pixels": outside,
            "deep_pixels": deep,
            "outside_fraction": outside / lesion if lesion else None,
            "deep_fraction": deep / lesion if lesion else None,
            "worst_image": worst[cls],
        }
    return {
        "n_images": len(ids),
        "fov_fraction": fov_pixels / image_pixels if image_pixels else None,
        "per_class": per_class,
    }


def decide(train_report):
    failed = []
    for cls, row in train_report["per_class"].items():
        fraction = row["deep_fraction"]
        if fraction is not None and fraction >= MAX_DEEP_FRACTION:
            failed.append(cls)
    return len(failed) == 0, failed


def main():
    reports = {}
    accepted = True
    for name in DATASETS:
        dataset_dir = ROOT / "dataset" / "prepared" / name
        train = _split_counts(dataset_dir, "train")
        ok, failed = decide(train)
        accepted = accepted and ok
        reports[name] = {
            "train": train,
            "test_report_only": _split_counts(dataset_dir, "test"),
            "accepted_on_train": ok,
            "failed_classes": failed,
        }
    payload = {
        "rule": (
            f"On the training split only, every class must have deep_fraction < {MAX_DEEP_FRACTION}. "
            f"Deep means more than {DEEP_MARGIN} pixels outside the FOV in 8-connectivity. "
            "Test counts are not used for this decision."
        ),
        "accepted": accepted,
        "datasets": reports,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({"accepted": accepted, "out": str(OUT)}, indent=1))
    for name, report in reports.items():
        print(name, "train deep fractions:")
        for cls, row in report["train"]["per_class"].items():
            print(f"  {cls} {row['deep_fraction']} outside {row['outside_fraction']} worst {row['worst_image']}")


if __name__ == "__main__":
    main()
