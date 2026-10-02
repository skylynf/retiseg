"""Pixel-level metrics computed from score histograms.

Scores are integer levels q in [0, L]; probability = q / L. A histogram pair
(pos, neg) of length L + 1 counts positive and negative pixels at each level.
Pooling over a test set is the sum of per-image histograms, so dataset-level
metrics never need all pixels in memory and are exact for the stored levels.
"""

import math

import numpy as np
from sklearn.metrics import auc


def histograms(q, gt, valid, levels):
    q = q[valid]
    g = gt[valid]
    pos = np.bincount(q[g], minlength=levels + 1)
    neg = np.bincount(q[~g], minlength=levels + 1)
    return pos.astype(np.int64), neg.astype(np.int64)


def coarsen(hist, n_bins):
    """Sum a length-(L+1) histogram into n_bins equal-width bins; exact regrouping of levels."""
    n_levels = hist.shape[-1]
    idx = (np.arange(n_levels) * n_bins) // n_levels
    out = np.zeros(hist.shape[:-1] + (n_bins,), dtype=np.int64)
    np.add.at(out, (..., idx), hist)
    return out


def average_precision(pos, neg):
    """Step-wise AP over all distinct score levels; equals sklearn.average_precision_score."""
    n_pos = pos.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    keep = (pos[::-1] + neg[::-1]) > 0
    tp, fp = tp[keep], fp[keep]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    d_recall = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(d_recall * precision))


def pr_curve(pos, neg, max_points=512):
    """Precision-recall points in order of decreasing threshold, thinned for plotting."""
    n_pos = pos.sum()
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    keep = (pos[::-1] + neg[::-1]) > 0
    tp, fp = tp[keep], fp[keep]
    precision = tp / (tp + fp)
    recall = tp / n_pos if n_pos > 0 else np.zeros_like(tp, dtype=float)
    if len(recall) > max_points:
        sel = np.unique(np.linspace(0, len(recall) - 1, max_points).round().astype(int))
        precision, recall = precision[sel], recall[sel]
    return recall.tolist(), precision.tolist()


def m2mrf_aupr(pos, neg, levels, n_thresholds=11):
    """Replica of M2MRF mmseg/core/evaluation/my_metrics.py::sigmoid_metrics AUPR.

    Positive iff prob > t for t in linspace(0, 1, n_thresholds); precision with no
    positive predictions is set to 1, recall with no ground truth to 0; the curve is
    integrated with the trapezoidal rule.
    """
    threshs = np.linspace(0, 1, n_thresholds)
    n_gt = pos.sum()
    tp_above = np.concatenate((np.cumsum(pos[::-1])[::-1], [0]))
    fp_above = np.concatenate((np.cumsum(neg[::-1])[::-1], [0]))
    ppv, sens = [], []
    for t in threshs:
        first = min(math.floor(t * levels) + 1, levels + 1)
        tp = tp_above[first]
        p = tp + fp_above[first]
        ppv.append(tp / p if p > 0 else 1.0)
        sens.append(tp / n_gt if n_gt > 0 else 0.0)
    return float(auc(np.array(sens), np.array(ppv)))


def level_for_threshold(t, levels):
    """Smallest level q with q / levels >= t."""
    return int(math.ceil(t * levels - 1e-9))


def confusion_at(pos, neg, q_min):
    tp = int(pos[q_min:].sum())
    fp = int(neg[q_min:].sum())
    fn = int(pos.sum()) - tp
    return tp, fp, fn


def overlap_scores(tp, fp, fn):
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    return {
        "dice": 2 * tp / denom_dice if denom_dice else float("nan"),
        "iou": tp / denom_iou if denom_iou else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def best_dice_threshold(pos, neg, levels):
    """Threshold maximizing pooled Dice; ties resolve to the highest threshold."""
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    n_pos = pos.sum()
    denom = tp + fp + n_pos
    dice = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    q = int(len(dice) - 1 - np.argmax(dice[::-1]))
    return q / levels
