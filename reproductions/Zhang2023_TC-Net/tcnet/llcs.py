"""Locality-aware and long-range dependency concatenation. Section 3.3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tcnet.assumptions import FUSION_HIDDEN_RATIO, SPATIAL_ATTN_SIZE


class SpatialPath(nn.Module):
    """Equation (6). The operating grid is assumption A11."""

    def __init__(self, channels: int, attn_size: int = SPATIAL_ATTN_SIZE):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("spatial path needs an even channel count")
        self.attn_size = attn_size
        reduced = channels // 2
        self.proj_i = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.proj_j = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        pooled = F.adaptive_avg_pool2d(features, self.attn_size)
        left = self.proj_i(pooled).flatten(2).transpose(1, 2)
        right = self.proj_j(pooled).flatten(2).transpose(1, 2)
        energy = torch.bmm(left, right.transpose(1, 2))
        # Equation (6) normalizes over i, the first spatial index.
        attention = torch.softmax(energy, dim=1)
        values = pooled.flatten(2).transpose(1, 2)
        mixed = torch.bmm(attention.transpose(1, 2), values)
        mixed = mixed.transpose(1, 2).reshape(batch, channels, self.attn_size, self.attn_size)
        if (height, width) != (self.attn_size, self.attn_size):
            mixed = F.interpolate(mixed, size=(height, width), mode="bilinear", align_corners=False)
        return mixed


class ChannelPath(nn.Module):
    """Equation (7). The 1×1 expands Ct/2 to Ct. Assumption A12."""

    def __init__(self, channels: int):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channel path needs an even channel count")
        reduced = channels // 2
        self.proj_desc = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.proj_gate = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.expand = nn.Linear(reduced, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, features: Tensor) -> Tensor:
        described = self.proj_desc(features).flatten(2)
        gate = self.proj_gate(features).flatten(2)
        attention = torch.softmax(gate, dim=-1)
        summary = torch.bmm(described, attention.transpose(1, 2)).squeeze(-1)
        weights = torch.sigmoid(self.norm(self.expand(summary)))
        return features * weights[:, :, None, None]


class FusionPath(nn.Module):
    """GAP, two fully connected layers, sigmoid, then channel gating."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(int(channels * FUSION_HIDDEN_RATIO), 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        pooled = features.mean(dim=(2, 3))
        gate = torch.sigmoid(self.fc2(self.relu(self.fc1(pooled))))
        return features * gate[:, :, None, None]


class LLCS(nn.Module):
    """Equation (8): concat(Os, Oc) + Of."""

    def __init__(self, cnn_channels: int, transformer_channels: int):
        super().__init__()
        self.spatial = SpatialPath(cnn_channels)
        self.channel = ChannelPath(transformer_channels)
        self.fusion = FusionPath(cnn_channels + transformer_channels)

    def forward(self, cnn_features: Tensor, transformer_features: Tensor) -> Tensor:
        if cnn_features.shape[-2:] != transformer_features.shape[-2:]:
            raise ValueError(
                "LLCS expects Dc and Dt at the same spatial size, "
                f"got {tuple(cnn_features.shape[-2:])} and {tuple(transformer_features.shape[-2:])}"
            )
        spatial = self.spatial(cnn_features)
        channel = self.channel(transformer_features)
        fused = self.fusion(torch.cat([cnn_features, transformer_features], dim=1))
        return torch.cat([spatial, channel], dim=1) + fused
