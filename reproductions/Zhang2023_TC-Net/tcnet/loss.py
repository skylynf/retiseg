"""Dynamic cyclical focal loss. Equations (9)–(13)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tcnet.assumptions import (
    BETA_LOG_OFFSET,
    CYCLICAL_FACTOR,
    EPOCHS,
    GAMMA_HIGH,
    GAMMA_LOW,
    NUM_CLASSES,
    PT_EPS,
)


def cyclical_indicator(epoch: int, total_epochs: int, factor: float = CYCLICAL_FACTOR) -> float:
    """Equation (11). ``epoch`` is 1-indexed and inclusive of ``total_epochs``."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if epoch < 1 or epoch > total_epochs:
        raise ValueError(f"epoch must be in 1..{total_epochs}, got {epoch}")
    if factor == 1:
        raise ValueError("cyclical factor r must not be 1")
    ratio = epoch / total_epochs
    if factor * epoch <= total_epochs:
        return 1.0 - factor * ratio
    return (factor * ratio - 1.0) / (factor - 1.0)


def class_beta(counts: Tensor, log_offset: float = BETA_LOG_OFFSET) -> Tensor:
    """Equation (10). ``counts`` is a length-c vector of pixel counts."""
    counts = counts.to(dtype=torch.float32)
    frequency = counts / counts.sum().clamp(min=1.0)
    return torch.reciprocal(torch.log(frequency + log_offset))


class DynamicCyclicalFocalLoss(nn.Module):
    """DCFL on mutually exclusive classes.

    ``pt`` is the softmax probability of the ground-truth class. ``beta`` is
    looked up per pixel from that class. Pass epoch-level pixel counts through
    ``set_epoch_class_counts`` when the caller has them; otherwise the current
    target is used (assumption A17).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        total_epochs: int = EPOCHS,
        factor: float = CYCLICAL_FACTOR,
        gamma_high: float = GAMMA_HIGH,
        gamma_low: float = GAMMA_LOW,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.factor = factor
        self.gamma_high = gamma_high
        self.gamma_low = gamma_low
        self.last_beta_source = "unset"
        self._epoch_counts: Tensor | None = None

    def set_epoch_class_counts(self, counts: Tensor) -> None:
        counts = counts.detach().to(dtype=torch.float32).cpu().reshape(-1)
        if counts.numel() != self.num_classes:
            raise ValueError(f"expected {self.num_classes} class counts, got {counts.numel()}")
        self._epoch_counts = counts

    def clear_epoch_class_counts(self) -> None:
        self._epoch_counts = None

    def _counts_from_target(self, target: Tensor) -> Tensor:
        flat = target.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("target is empty")
        if int(flat.min()) < 0 or int(flat.max()) >= self.num_classes:
            raise ValueError(
                f"target labels must be in 0..{self.num_classes - 1}, "
                f"got {int(flat.min())}..{int(flat.max())}"
            )
        return torch.bincount(flat, minlength=self.num_classes).to(dtype=torch.float32)

    def forward(self, logits: Tensor, target: Tensor, epoch: int) -> Tensor:
        if logits.dim() != 4:
            raise ValueError("logits must be NCHW")
        if logits.size(1) != self.num_classes:
            raise ValueError(f"expected {self.num_classes} classes, got {logits.size(1)}")
        if target.shape != logits.shape[-2: ] and target.shape != logits.shape[0:1] + logits.shape[-2:]:
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}"
            )
        target = target.to(dtype=torch.long)
        if self._epoch_counts is None:
            counts = self._counts_from_target(target)
            self.last_beta_source = "batch"
        else:
            counts = self._epoch_counts
            self.last_beta_source = "epoch"
        beta = class_beta(counts).to(device=logits.device)
        probability = torch.softmax(logits, dim=1)
        true_probability = probability.gather(1, target.unsqueeze(1)).squeeze(1)
        log_probability = torch.log(true_probability.clamp(min=PT_EPS))
        high = -torch.pow(1.0 + true_probability, self.gamma_high) * log_probability
        low = -torch.pow(1.0 - true_probability, self.gamma_low) * log_probability
        indicator = cyclical_indicator(epoch, self.total_epochs, self.factor)
        per_pixel = indicator * high + beta[target] * (1.0 - indicator) * low
        return per_pixel.mean()
