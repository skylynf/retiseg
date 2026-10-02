"""Lesion-wise AUPR, plus auxiliary Dice and IoU. Assumption A16."""

from __future__ import annotations

import numpy as np

from clcnet.assumptions import SEGMENTATION_CLASSES, class_index


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve, summed at each positive rank.

    Tied scores keep their incoming order (mergesort). Pixels with no
    positive label in the whole split return NaN and stay out of the mean.
    """
    truth = np.asarray(y_true).astype(np.uint8).ravel()
    score = np.asarray(y_score, dtype=np.float64).ravel()
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    truth = truth[order]
    tp = np.cumsum(truth)
    fp = np.cumsum(1 - truth)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    delta = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * delta))


def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred_bin).astype(bool)
    gt = np.asarray(gt_bin).astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    if tp + fp + fn == 0:
        return 1.0, 1.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    return float(dice), float(iou)


def score_split(probability_maps, binary_maps, lesions) -> dict:
    """probability_maps and binary_maps are lists aligned by image.

    Each probability map is C,H,W softmax. Each binary map is a dict of
    lesion name to a boolean H,W mask at the same resolution.
    """
    aupr = {}
    dice = {}
    iou = {}
    for lesion in lesions:
        index = class_index(lesion)
        scores = []
        truths = []
        dice_values = []
        iou_values = []
        for probs, masks in zip(probability_maps, binary_maps):
            scores.append(probs[index])
            truths.append(masks[lesion] > 0)
            prediction = probs.argmax(axis=0) == index
            one_dice, one_iou = dice_iou(prediction, masks[lesion] > 0)
            dice_values.append(one_dice)
            iou_values.append(one_iou)
        aupr[lesion] = average_precision(np.concatenate([t.ravel() for t in truths]), np.concatenate([s.ravel() for s in scores]))
        dice[lesion] = float(np.mean(dice_values)) if dice_values else float("nan")
        iou[lesion] = float(np.mean(iou_values)) if iou_values else float("nan")
    finite = [aupr[name] for name in lesions if np.isfinite(aupr[name])]
    return {
        "lesions": list(lesions),
        "AUPR": aupr,
        "AUPR_mean": float(np.mean(finite)) if finite else float("nan"),
        "Dice": dice,
        "IoU": iou,
        "AUPR_definition": "pooled pixels, stable sort on tied scores",
        "Dice_IoU_definition": "argmax class versus the original binary mask, then mean over images",
        "segmentation_classes": list(SEGMENTATION_CLASSES),
    }
