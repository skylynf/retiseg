"""Uncertainty and comparison statistics over test images, seeds and datasets."""

import numpy as np
from scipy import stats as sps

from bench.eval.pixel import average_precision


def _bootstrap_weights(n_images, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_images, size=(n_boot, n_images))
    w = np.zeros((n_boot, n_images))
    np.add.at(w, (np.arange(n_boot)[:, None], idx), 1.0)
    return w


def _boot_ap(w, pos, neg):
    pooled_pos = w @ pos.astype(np.float64)
    pooled_neg = w @ neg.astype(np.float64)
    return np.array([average_precision(p, n) for p, n in zip(pooled_pos, pooled_neg)])


def bootstrap_aupr_ci(pos, neg, n_boot=2000, seed=0, ci=0.95):
    """Percentile CI of pooled AUPR, resampling test images. pos/neg: (n_images, n_bins)."""
    w = _bootstrap_weights(pos.shape[0], n_boot, seed)
    ap = _boot_ap(w, pos, neg)
    ap = ap[~np.isnan(ap)]
    a = (1 - ci) / 2
    return float(np.quantile(ap, a)), float(np.quantile(ap, 1 - a))


def paired_bootstrap_aupr(pos_a, neg_a, pos_b, neg_b, n_boot=2000, seed=0, ci=0.95):
    """AUPR(a) - AUPR(b) on the same test images, with the same resamples for both models."""
    if pos_a.shape != pos_b.shape:
        raise ValueError("models must be evaluated on the same images with the same binning")
    w = _bootstrap_weights(pos_a.shape[0], n_boot, seed)
    delta = _boot_ap(w, pos_a, neg_a) - _boot_ap(w, pos_b, neg_b)
    delta = delta[~np.isnan(delta)]
    point = average_precision(pos_a.sum(0), neg_a.sum(0)) - average_precision(pos_b.sum(0), neg_b.sum(0))
    a = (1 - ci) / 2
    p_two_sided = min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
    return {
        "delta": float(point),
        "ci": [float(np.quantile(delta, a)), float(np.quantile(delta, 1 - a))],
        "p_bootstrap": float(p_two_sided),
    }


def paired_wilcoxon(x, y):
    """Two-sided signed-rank test on per-image scores; pairs with a missing value are dropped."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 2 or np.allclose(x[ok], y[ok]):
        return {"n": int(ok.sum()), "statistic": float("nan"), "p": 1.0}
    res = sps.wilcoxon(x[ok], y[ok], alternative="two-sided")
    return {"n": int(ok.sum()), "statistic": float(res.statistic), "p": float(res.pvalue)}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def kendall_tau(a, b):
    res = sps.kendalltau(a, b)
    return {"tau": float(res.statistic), "p": float(res.pvalue)}


def friedman_nemenyi(scores, alpha=0.05):
    """scores: (n_models, n_datasets), higher is better. Rank 1 is best.

    Returns the Friedman test, average ranks, and the Nemenyi critical difference
    used for critical-difference diagrams.
    """
    scores = np.asarray(scores, dtype=np.float64)
    k, n = scores.shape
    ranks = np.apply_along_axis(lambda col: sps.rankdata(-col), 0, scores)
    avg_rank = ranks.mean(axis=1)
    if k >= 3 and n >= 2:
        fr = sps.friedmanchisquare(*scores)
        stat, p = float(fr.statistic), float(fr.pvalue)
    else:
        stat, p = float("nan"), float("nan")
    q_alpha = sps.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n)))
    return {"statistic": stat, "p": p, "avg_rank": avg_rank.tolist(), "critical_difference": cd}
