import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from bench.eval import imagelevel


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    labels = rng.random(60) < 0.4
    scores = np.round(rng.random(60) + 0.3 * labels, 1)
    fpr, tpr, thresholds = imagelevel.roc_points(scores, labels)
    assert imagelevel.auc(fpr, tpr) == pytest.approx(roc_auc_score(labels, scores))
    assert fpr[0] == 0 and tpr[0] == 0 and thresholds[0] == np.inf
    assert fpr[-1] == 1 and tpr[-1] == 1


def test_weighted_error_takes_the_best_true_point():
    # Scores 0.9, 0.8 positive; 0.7 negative; 0.6 positive; 0.1 negative.
    scores = [0.9, 0.8, 0.7, 0.6, 0.1]
    labels = [True, True, False, True, False]
    fpr, tpr, thresholds = imagelevel.roc_points(scores, labels)
    # Points: (0,0), (0,1/3), (0,2/3), (1/2,2/3), (1/2,1), (1,1).
    assert fpr.tolist() == [0, 0, 0, 0.5, 0.5, 1]
    w1 = imagelevel.weighted_error(fpr, tpr, thresholds, 1.0)
    assert w1["wer"] == pytest.approx(1 / 6)  # (0 + 1/3) / 2 at threshold 0.8
    assert w1["threshold"] == pytest.approx(0.8)
    w10 = imagelevel.weighted_error(fpr, tpr, thresholds, 10.0)
    assert w10["wer"] == pytest.approx(0.5 / 11)  # FNR 0 first reached at FPR 1/2
    w01 = imagelevel.weighted_error(fpr, tpr, thresholds, 0.1)
    assert w01["wer"] == pytest.approx(0.1 * (1 / 3) / 1.1)
    eer = imagelevel.equal_error(fpr, tpr, thresholds)
    # |FPR - FNR| is smallest at (1/2, 1/3), threshold 0.7.
    assert eer["eer"] == pytest.approx((0.5 + 1 / 3) / 2) and eer["threshold"] == pytest.approx(0.7)


def test_summarize_without_negative_images_is_undefined():
    out = imagelevel.summarize([0.2, 0.3], [True, True])
    assert out["n_negative"] == 0 and math.isnan(out["auc"]) and out["eer"] is None


def test_image_scores_max_and_top_k():
    q = np.array([[0, 10, 20], [30, 40, 50]])
    valid = np.array([[True, True, True], [True, True, False]])
    top_max, top_mean = imagelevel.image_scores(q, valid, 100, top_k=2)
    assert top_max == pytest.approx(0.4)
    assert top_mean == pytest.approx(0.35)
    assert imagelevel.image_scores(q, np.zeros_like(valid), 100) == (0.0, 0.0)
