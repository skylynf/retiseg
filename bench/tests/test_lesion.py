import numpy as np
import pytest

from bench.eval import lesion


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def test_match_counts_hits_misses_and_false_positives():
    shape = (200, 200)
    small = _disk(shape, 20, 20, 2)
    medium = _disk(shape, 100, 100, 8)
    large = _disk(shape, 160, 160, 25)
    gt = small | medium | large
    pred = medium | (large & _disk(shape, 160, 160, 5)) | _disk(shape, 20, 150, 4)
    m = lesion.match_lesions(pred, gt)
    assert m["gt_detected"].sum() == 2 and len(m["gt_detected"]) == 3
    assert m["n_pred"] == 3 and m["n_pred_tp"] == 2


def test_accumulator_strata_and_f1():
    shape = (200, 200)
    small = _disk(shape, 20, 20, 2)
    large = _disk(shape, 160, 160, 25)
    gt = small | large
    pred = large.copy()
    acc = lesion.LesionAccumulator([0.0, 0.05, np.inf], ["small", "large"])
    acc.add(lesion.match_lesions(pred, gt), fov_area=200 * 200)
    r = acc.result()
    assert r["n_gt_lesions"] == 2
    assert r["recall"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["by_size"]["small"] == {"n_lesions": 1, "recall": 0.0, "rel_diameter_range": [0.0, 0.05]}
    assert r["by_size"]["large"]["recall"] == 1.0


def test_no_predictions_gives_nan_precision():
    gt = _disk((50, 50), 25, 25, 5)
    acc = lesion.LesionAccumulator([0.0, np.inf], ["all"])
    acc.add(lesion.match_lesions(np.zeros_like(gt), gt), fov_area=2500)
    r = acc.result()
    assert r["recall"] == 0.0 and np.isnan(r["precision"]) and np.isnan(r["f1"])
