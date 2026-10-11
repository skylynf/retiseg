"""Image-level detection: one score per image and class, against "the image has this lesion".

This is the DiaRetDB1 protocol (Kauppi 2007, section 4.1): each method gives
every test image a score per finding, the ROC is built from those scores,
and the weighted error rate

    WER(R) = (FPR + R * FNR) / (1 + R)

is reported for R = 0.1, 1 and 10, "computed from the nearest true points on
the ROC without interpolation". The paper does not say which point; here it
is the true operating point with the lowest WER(R) (the end points, where
every image is called negative or positive, are true points too), and that
point's FPR, FNR and threshold are kept with it. EER is the true point where
|FPR - FNR| is smallest, reported as (FPR + FNR) / 2.

Scores come from the probability maps (bench/eval/evaluate.py): the maximum
inside the valid region, and the mean of its top ``top_k`` values as a check.
An image is positive for a class when its mask has a positive pixel inside
the valid region.
"""

import math

import numpy as np


def image_scores(q, valid, levels, top_k=100):
    """(max, mean of the top_k) of the integer levels inside ``valid``, as probabilities."""
    values = q[valid]
    if values.size == 0:
        return 0.0, 0.0
    k = min(int(top_k), values.size)
    top = np.partition(values, values.size - k)[values.size - k :]
    return float(values.max()) / levels, float(top.mean()) / levels


def roc_points(scores, labels):
    """True ROC points, from "all negative" to "all positive".

    Returns (fpr, tpr, thresholds); an image is called positive when its
    score is >= threshold. The first threshold is +inf.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both positive and negative images")
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    distinct = np.flatnonzero(np.diff(s)) if s.size > 1 else np.zeros(0, dtype=int)
    ends = np.concatenate((distinct, [s.size - 1]))
    tp = np.cumsum(y)[ends]
    fp = np.cumsum(~y)[ends]
    tpr = np.concatenate(([0.0], tp / n_pos))
    fpr = np.concatenate(([0.0], fp / n_neg))
    thresholds = np.concatenate(([np.inf], s[ends]))
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Trapezoid area under the true ROC points (ties count one half, as in Mann-Whitney)."""
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def weighted_error(fpr, tpr, thresholds, ratio):
    """Lowest WER(ratio) over the true points, with that point."""
    fnr = 1.0 - np.asarray(tpr)
    wer = (np.asarray(fpr) + float(ratio) * fnr) / (1.0 + float(ratio))
    best = int(np.argmin(wer))
    return {
        "wer": float(wer[best]),
        "fpr": float(fpr[best]),
        "fnr": float(fnr[best]),
        "threshold": float(thresholds[best]),
    }


def equal_error(fpr, tpr, thresholds):
    fnr = 1.0 - np.asarray(tpr)
    gap = np.abs(np.asarray(fpr) - fnr)
    best = int(np.argmin(gap))
    return {
        "eer": float((fpr[best] + fnr[best]) / 2.0),
        "fpr": float(fpr[best]),
        "fnr": float(fnr[best]),
        "threshold": float(thresholds[best]),
    }


def summarize(scores, labels, ratios=(0.1, 1.0, 10.0)):
    """AUC, EER and WER(R) of one class, or NaNs when one side has no image."""
    labels = np.asarray(labels, dtype=bool)
    out = {"n_images": int(labels.size), "n_positive": int(labels.sum()), "n_negative": int((~labels).sum())}
    if out["n_positive"] == 0 or out["n_negative"] == 0:
        out.update({"auc": math.nan, "eer": None, "wer": {str(r): None for r in ratios}})
        return out
    fpr, tpr, thresholds = roc_points(scores, labels)
    out["auc"] = auc(fpr, tpr)
    out["eer"] = equal_error(fpr, tpr, thresholds)
    out["wer"] = {f"{float(r):g}": weighted_error(fpr, tpr, thresholds, r) for r in ratios}
    out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return out
