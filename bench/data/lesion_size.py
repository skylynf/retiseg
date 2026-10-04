"""Freeze small/medium/large edges from training-set lesion diameters.

The rule is fixed before looking at a test image. Relative diameter is the
lesion's equivalent diameter divided by the FOV's equivalent diameter.
The small/medium edge is the median of training microaneurysms.
The medium/large edge is the median of training haemorrhages.
Each edge is rounded up to the next 0.001. If that does not leave the
haemorrhage edge strictly above the microaneurysm edge, the haemorrhage
75th percentile is used instead.
"""

import math

import numpy as np

from bench.eval.lesion import equivalent_diameter


def summarize(diameters):
    values = np.asarray(diameters, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    percentiles = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "n": int(values.size),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
    }


def _round_up_milli(value):
    return math.ceil(value * 1000.0 - 1e-12) / 1000.0


def freeze_edges(per_class):
    ma = per_class["MA"]
    he = per_class["HE"]
    if ma["n"] == 0 or he["n"] == 0:
        raise ValueError("training set has no microaneurysms or no haemorrhages")
    small = _round_up_milli(ma["p50"])
    large = _round_up_milli(he["p50"])
    source = "MA median and HE median, rounded up to 0.001"
    if large <= small:
        large = _round_up_milli(he["p75"])
        source = "MA median and HE 75th percentile, rounded up to 0.001"
    if large <= small:
        raise ValueError(f"size edges did not separate: small={small}, large={large}")
    return [0.0, small, large, math.inf], source


def component_areas(mask):
    from scipy import ndimage

    labeled, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(labeled.ravel())[1:]


def write_freeze(repo_root, out_path):
    """Refreeze size edges from prepared training masks. Test ids are not read."""
    import json
    from pathlib import Path

    from bench.common.io import LESION_CLASSES, load_split
    from bench.data.prepare import measure_exported

    repo_root = Path(repo_root)
    per_dataset = {}
    pooled_values = {cls: [] for cls in LESION_CLASSES}
    for name in ("IDRiD", "DDR"):
        dest = repo_root / "dataset" / "prepared" / name
        train_ids = load_split(dest, "train")
        values = measure_exported(dest, train_ids)
        per_dataset[name] = {cls: summarize(values[cls]) for cls in LESION_CLASSES}
        for cls in LESION_CLASSES:
            pooled_values[cls].extend(values[cls])
    pooled = {cls: summarize(pooled_values[cls]) for cls in LESION_CLASSES}
    edges, source = freeze_edges(pooled)
    payload = {
        "rule": source,
        "mask": "luminance > 10, largest component, holes filled. Training images only.",
        "splits": (
            "IDRiD train (44, official training set after the fixed validation holdout) "
            "and DDR official train (383). Validation and test images were not used."
        ),
        "edges": [edges[0], edges[1], edges[2], "inf"],
        "pooled": pooled,
        "per_dataset": per_dataset,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
    return edges, source


def relative_diameters(mask, fov):
    fov_area = int(np.count_nonzero(fov))
    if fov_area == 0:
        return np.zeros(0, dtype=np.float64)
    areas = component_areas(mask & fov)
    if areas.size == 0:
        return np.zeros(0, dtype=np.float64)
    return equivalent_diameter(areas) / equivalent_diameter(fov_area)
