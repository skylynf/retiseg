"""Dice + weighted multi-class cross-entropy, and binary cross-entropy.

Equations (1) and (2) in section 3.1. The reductions and ? are assumption A10.
Contextual labels are downsampled with assumption A6.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clcnet.assumptions import DICE_EPS, OVERLAP_PRIORITY, SEGMENTATION_WEIGHTS, class_index


def downsample_label(target: torch.Tensor) -> torch.Tensor:
    """Max-pool each class over non-overlapping 2×2 blocks, then resolve ties.

    target: int64 tensor (N, H, W) with values in 0..4. H and W are even.
    """
    if target.ndim != 3:
        raise ValueError("target must be N,H,W")
    if target.shape[-1] % 2 or target.shape[-2] % 2:
        raise ValueError(f"spatial size must be even, got {tuple(target.shape[-2:])}")
    one_hot = F.one_hot(target.long(), num_classes=len(SEGMENTATION_WEIGHTS))
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    pooled = F.max_pool2d(one_hot, kernel_size=2, stride=2)
    out = torch.zeros(target.shape[0], target.shape[-2] // 2, target.shape[-1] // 2, dtype=torch.long, device=target.device)
    # Paint low priority first so a higher-priority lesion overwrites it.
    for name in reversed(OVERLAP_PRIORITY):
        index = class_index(name)
        out[pooled[:, index] > 0] = index
    return out


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of per-lesion soft Dice over the mini-batch. Background is excluded."""
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    lesion_probs = probs[:, 1:]
    lesion_true = one_hot[:, 1:]
    dims = (0, 2, 3)
    intersection = (lesion_probs * lesion_true).sum(dim=dims)
    denominator = lesion_probs.sum(dim=dims) + lesion_true.sum(dim=dims)
    score = (2.0 * intersection + DICE_EPS) / (denominator + DICE_EPS)
    return (1.0 - score).sum()


def weighted_mce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over pixels of -w_y log p_y. Weights follow section 3.1."""
    weight = logits.new_tensor(SEGMENTATION_WEIGHTS)
    log_prob = F.log_softmax(logits, dim=1)
    log_prob = log_prob.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    flat = target.long().reshape(-1)
    chosen = log_prob[torch.arange(flat.shape[0], device=flat.device), flat]
    return -(weight[flat] * chosen).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> dict:
    dice = dice_loss(logits, target)
    wmce = weighted_mce(logits, target)
    return {"dice": dice, "wmce": wmce, "segmentation": dice + wmce}


def classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.float())


def total_loss(output: dict, batch: dict) -> dict:
    """Sum the branch losses that this forward actually produced. Coefficients are 1."""
    parts = {}
    total = logits_device_zero(output)
    if "local_logits" in output:
        local = segmentation_loss(output["local_logits"], batch["local_mask"])
        parts["local_dice"] = local["dice"]
        parts["local_wmce"] = local["wmce"]
        total = total + local["segmentation"]
    if "context_logits" in output:
        context_target = downsample_label(batch["context_mask"])
        context = segmentation_loss(output["context_logits"], context_target)
        parts["context_dice"] = context["dice"]
        parts["context_wmce"] = context["wmce"]
        total = total + context["segmentation"]
    if output.get("local_cls_logits") is not None:
        loss = classification_loss(output["local_cls_logits"], batch["local_cls"])
        parts["local_bce"] = loss
        total = total + loss
    if output.get("context_cls_logits") is not None:
        loss = classification_loss(output["context_cls_logits"], batch["context_cls"])
        parts["context_bce"] = loss
        total = total + loss
    parts["total"] = total
    return parts


def logits_device_zero(output: dict) -> torch.Tensor:
    for value in output.values():
        if torch.is_tensor(value):
            return value.sum() * 0.0
    raise RuntimeError("model output is empty")
