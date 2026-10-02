"""Selective-kernel convolution used by CLC-Net.

Section 3.1 names kernels 3×3, 5×5 and 7×7. Assumption A3 follows the
cited SKNet paper: those larger kernels are 3×3 convolutions with
dilation 2 and 3, groups G=32, reduction r=16, minimum bottleneck L=32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from clcnet.assumptions import SK_DILATIONS, SK_GROUPS, SK_MIN_CHANNELS, SK_REDUCTION


class SKConv(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        if channels % SK_GROUPS != 0:
            raise ValueError(
                f"SKConv channels {channels} must be divisible by G={SK_GROUPS}"
            )
        self.channels = channels
        branches = []
        for dilation in SK_DILATIONS:
            padding = dilation  # 3×3 kernel, keeps spatial size when stride is 1
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=SK_GROUPS,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        hidden = max(channels // SK_REDUCTION, SK_MIN_CHANNELS)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.select = nn.ModuleList(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True) for _ in SK_DILATIONS
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([branch(x) for branch in self.branches], dim=1)
        compact = self.fuse(self.gap(feats.sum(dim=1)))
        attention = torch.stack([layer(compact) for layer in self.select], dim=1)
        attention = torch.softmax(attention, dim=1)
        return (feats * attention).sum(dim=1)
