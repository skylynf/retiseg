"""ResNeSt stage used by the CNN encoder.

K and R are the paper values. The split topology follows section 3.1 and
Figure 2(b,c). Hidden width and the ReLU between the two fully connected
layers are assumption A2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tcnet.assumptions import SPLIT_MIN_HIDDEN, SPLIT_REDUCTION


class SplitBranch(nn.Module):
    """Two convolutions, each followed by BN and ReLU. Figure 2(b)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SplitAttention(nn.Module):
    """Equations (1) and (2). Softmax is over the radix axis."""

    def __init__(self, channels: int, radix: int):
        super().__init__()
        hidden = max(channels // SPLIT_REDUCTION, SPLIT_MIN_HIDDEN)
        self.radix = radix
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * radix)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, parts: list[Tensor]) -> Tensor:
        if len(parts) != self.radix:
            raise ValueError(f"expected {self.radix} splits, got {len(parts)}")
        pooled = torch.stack(parts, dim=0).sum(dim=0)
        pooled = pooled.mean(dim=(2, 3))
        weights = self.fc2(self.relu(self.fc1(pooled)))
        batch, channels = parts[0].shape[:2]
        weights = weights.view(batch, self.radix, channels, 1, 1)
        weights = torch.softmax(weights, dim=1)
        out = parts[0] * weights[:, 0]
        for index in range(1, self.radix):
            out = out + parts[index] * weights[:, index]
        return out


class Cardinal(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, radix: int):
        super().__init__()
        if in_channels % radix != 0:
            raise ValueError(f"cardinal channels {in_channels} not divisible by R={radix}")
        split_in = in_channels // radix
        self.branches = nn.ModuleList(
            SplitBranch(split_in, out_channels, stride) for _ in range(radix)
        )
        self.attend = SplitAttention(out_channels, radix)

    def forward(self, x: Tensor) -> Tensor:
        parts = list(torch.chunk(x, len(self.branches), dim=1))
        return self.attend([branch(part) for branch, part in zip(self.branches, parts)])


class ResNeStStage(nn.Module):
    """One ResNeSt block. Equation (3): 1×1 projection plus shortcut T."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        cardinality: int,
        radix: int,
    ):
        super().__init__()
        if in_channels % cardinality != 0 or out_channels % cardinality != 0:
            raise ValueError("channels must be divisible by K")
        self.cardinality = cardinality
        card_in = in_channels // cardinality
        card_out = out_channels // cardinality
        self.cards = nn.ModuleList(
            Cardinal(card_in, card_out, stride, radix) for _ in range(cardinality)
        )
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        parts = torch.chunk(x, self.cardinality, dim=1)
        merged = torch.cat([card(part) for card, part in zip(self.cards, parts)], dim=1)
        out = self.proj_bn(self.proj(merged)) + self.shortcut(x)
        return self.relu(out)
