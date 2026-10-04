"""Exclusive class indices for softmax heads.

A four-channel target is MA, HE, EX, SE. Where those masks overlap, the pixel
keeps the later name in ``M2MRF_OVERWRITE_ORDER`` (EX, HE, SE, MA), so a
microaneurysm is not replaced by a larger lesion. Empty pixels stay 0.
``class_index`` is the network channel of MA, HE, EX, SE, in that order.
"""

import torch

from bench.common.io import LESION_CLASSES, M2MRF_OVERWRITE_ORDER


def overwrite_removal_counts(target):
    """Positive pixels of each class removed by the exclusive overwrite.

    ``target`` is (N, 4, H, W) or (4, H, W), MA HE EX SE. Later names in
    ``M2MRF_OVERWRITE_ORDER`` replace earlier ones, so MA is kept and EX can
    be removed by HE, SE or MA. The count is the training target. Evaluation
    keeps the original four masks and does not apply this rule.
    """
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if target.ndim != 4 or target.shape[1] != len(LESION_CLASSES):
        raise ValueError(f"expected four lesion masks, got {tuple(target.shape)}")
    plane_of = {name: index for index, name in enumerate(LESION_CLASSES)}
    positive = target > 0.5
    later = torch.zeros(target.shape[0], target.shape[2], target.shape[3], dtype=torch.bool, device=target.device)
    counts = {}
    for name in reversed(M2MRF_OVERWRITE_ORDER):
        plane = positive[:, plane_of[name]]
        counts[name] = {
            "positive": int(plane.sum().item()),
            "removed": int((plane & later).sum().item()),
        }
        later = later | plane
    return counts


def exclusive_label(target, class_index):
    if target.dtype in (torch.int32, torch.int64, torch.long) and target.ndim == 3:
        return target.long()
    class_index = tuple(int(channel) for channel in class_index)
    if target.ndim != 4 or target.shape[1] != len(LESION_CLASSES) or len(class_index) != len(LESION_CLASSES):
        raise ValueError(
            f"expected class indices (N, H, W) or four lesion masks (N, 4, H, W) "
            f"with four class ids, got {tuple(target.shape)} and {class_index}"
        )
    plane_of = {name: index for index, name in enumerate(LESION_CLASSES)}
    label = torch.zeros(target.shape[0], target.shape[2], target.shape[3], dtype=torch.long, device=target.device)
    for name in M2MRF_OVERWRITE_ORDER:
        plane = plane_of[name]
        channel = class_index[plane]
        label = torch.where(target[:, plane] > 0.5, torch.full_like(label, channel), label)
    return label
