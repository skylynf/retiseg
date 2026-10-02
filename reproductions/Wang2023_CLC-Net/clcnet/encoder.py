"""SE-ResNeXt-50 encoder.

Figure 3 shows one residual block: 32 paths, 4 channels on the grouped
3×3, then a squeeze-and-excitation module. Assumption A1 uses the standard
SE-ResNeXt-50 stage layout that this figure depicts, with random init.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        scale = self.pool(x).view(batch, channels)
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(scale))))
        return x * scale.view(batch, channels, 1, 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, groups, base_width, reduction):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SEModule(planes * self.expansion, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SEResNeXt50(nn.Module):
    def __init__(self, groups: int = 32, base_width: int = 4, reduction: int = 16):
        super().__init__()
        self.groups = groups
        self.base_width = base_width
        self.reduction = reduction
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        outplanes = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        layers = [
            Bottleneck(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.reduction,
            )
        ]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                    1,
                    None,
                    self.groups,
                    self.base_width,
                    self.reduction,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        stem = x
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return stem, c1, c2, c3, c4
