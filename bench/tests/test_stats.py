import numpy as np
import pytest

from bench.eval import pixel, stats

L = 65535


def _per_image_hists(rng, n_images, shift):
    pos, neg = [], []
    for _ in range(n_images):
        gt = rng.random(4000) < 0.03
        q = np.rint(np.clip(rng.normal(0.3 + shift * gt, 0.2), 0, 1) * L).astype(np.int64)
        p, n = pixel.histograms(q, gt, np.ones_like(gt), L)
        pos.append(pixel.coarsen(p, 1024))
        neg.append(pixel.coarsen(n, 1024))
    return np.stack(pos), np.stack(neg)


def test_bootstrap_ci_brackets_point_estimate():
    pos, neg = _per_image_hists(np.random.default_rng(0), 30, 0.4)
    point = pixel.average_precision(pos.sum(0), neg.sum(0))
    lo, hi = stats.bootstrap_aupr_ci(pos, neg, n_boot=300)
    assert lo < point < hi


def test_paired_bootstrap_detects_better_model_and_not_identical_one():
    rng = np.random.default_rng(1)
    pos_a, neg_a = _per_image_hists(rng, 30, 0.6)
    pos_b, neg_b = _per_image_hists(rng, 30, 0.2)
    better = stats.paired_bootstrap_aupr(pos_a, neg_a, pos_b, neg_b, n_boot=300)
    assert better["delta"] > 0 and better["ci"][0] > 0
    same = stats.paired_bootstrap_aupr(pos_a, neg_a, pos_a, neg_a, n_boot=300)
    assert same["delta"] == 0 and same["ci"] == [0.0, 0.0]


def test_wilcoxon_drops_missing_pairs():
    x = np.array([0.5, 0.6, np.nan, 0.7, 0.8, 0.9, 0.65, 0.75])
    y = x - 0.1
    y[2] = 0.3
    r = stats.paired_wilcoxon(x, y)
    assert r["n"] == 7 and r["p"] < 0.05


def test_holm_matches_hand_computation():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_friedman_nemenyi_ranks_and_cd():
    scores = np.array([[0.9, 0.8, 0.85, 0.7], [0.5, 0.4, 0.45, 0.3], [0.7, 0.6, 0.65, 0.5]])
    r = stats.friedman_nemenyi(scores)
    assert r["avg_rank"] == [1.0, 3.0, 2.0]
    # Demsar 2006, Table 5: q_0.05 = 2.343 for k = 3.
    assert r["critical_difference"] == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 4)), abs=1e-3)
