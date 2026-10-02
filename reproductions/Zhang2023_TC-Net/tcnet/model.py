"""TC-Net forward pass. Figure 2, sections 3.1–3.4.

The only input is an RGB image. There is no vessel map, text prompt, or mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tcnet.assumptions import (
    C1_CHANNELS,
    C2_CHANNELS,
    C3_CHANNELS,
    CARDINALITY,
    CNN_OUT_CHANNELS,
    EXTRA_INPUTS,
    FUNDUS_CLASSES,
    HEAD_MID_CHANNELS,
    INPUTS,
    NUM_CLASSES,
    RADIX,
    TF_CHANNELS,
)
from tcnet.llcs import LLCS
from tcnet.mhssa import CrossTransformerBlock, EncoderTransformerBlock
from tcnet.resnest_block import ResNeStStage


def initialize(module: nn.Module) -> None:
    """Kaiming initialization. Assumption A18. Does not load ResNeSt-50."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class CNNEncoder(nn.Module):
    """Stem plus two ResNeSt stages. C1, C2, C3 as in Figure 2 and section 3.1."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C1_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C1_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = ResNeStStage(
            C1_CHANNELS, C2_CHANNELS, stride=2, cardinality=CARDINALITY, radix=RADIX
        )
        self.stage3 = ResNeStStage(
            C2_CHANNELS, C3_CHANNELS, stride=2, cardinality=CARDINALITY, radix=RADIX
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c1 = self.stem(image)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return c1, c2, c3


class DecoderStage(nn.Module):
    """Bilinear ×2, a 3×3 that halves channels, optional skip, then two 3×3 blocks."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.up = ConvBNReLU(in_channels, out_channels)
        self.conv1 = ConvBNReLU(out_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    f"skip spatial size {tuple(skip.shape[-2:])} != {tuple(x.shape[-2:])}"
                )
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = DecoderStage(C3_CHANNELS, C2_CHANNELS, skip_channels=C2_CHANNELS)
        self.stage2 = DecoderStage(C2_CHANNELS, C1_CHANNELS, skip_channels=C1_CHANNELS)
        self.stage3 = DecoderStage(C1_CHANNELS, CNN_OUT_CHANNELS, skip_channels=0)

    def forward(self, c1: Tensor, c2: Tensor, c3: Tensor) -> Tensor:
        x = self.stage1(c3, c2)
        x = self.stage2(x, c1)
        return self.stage3(x)


class TransformerBranch(nn.Module):
    """Basic block, four encoder blocks, four decoder blocks, one full-res conv."""

    def __init__(self):
        super().__init__()
        self.basic = nn.Sequential(
            ConvBNReLU(3, TF_CHANNELS, stride=2),
            ConvBNReLU(TF_CHANNELS, TF_CHANNELS, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.encoder = nn.ModuleList(EncoderTransformerBlock(TF_CHANNELS) for _ in range(4))
        self.cross = CrossTransformerBlock(TF_CHANNELS)
        self.decoder = nn.ModuleList(EncoderTransformerBlock(TF_CHANNELS) for _ in range(3))
        self.out_conv = nn.Conv2d(TF_CHANNELS, TF_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, image: Tensor) -> Tensor:
        x = self.basic(image)
        features = []
        for block in self.encoder:
            x = block(x)
            features.append(x)
        _x2, _x3, x4, x5 = features
        x = self.cross(x4, x5)
        for block in self.decoder:
            x = block(x)
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_act(self.out_conv(x))


class SegmentationHead(nn.Module):
    """Section 3.4. 3×3 + ReLU, then 1×1. Logits, not probabilities."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.relu(self.conv(x)))


class TCNet(nn.Module):
    """Hybrid CNN / Transformer segmentation network.

    Parameters
    ----------
    num_classes:
        Fundus default is 5: background, EX, HE, MA, SE. The skin setting in
        the paper is the same network with two classes and the same RGB input.
    """

    extra_inputs = EXTRA_INPUTS
    inputs = INPUTS

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        self.num_classes = num_classes
        self.cnn_encoder = CNNEncoder()
        self.cnn_decoder = CNNDecoder()
        self.transformer = TransformerBranch()
        self.llcs = LLCS(CNN_OUT_CHANNELS, TF_CHANNELS)
        if HEAD_MID_CHANNELS != CNN_OUT_CHANNELS + TF_CHANNELS:
            raise RuntimeError("segmentation-head width drifted from LLCS output")
        self.head = SegmentationHead(HEAD_MID_CHANNELS, num_classes)
        initialize(self)

    def forward(self, image: Tensor) -> Tensor:
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError(
                "TC-Net takes one RGB tensor of shape (N, 3, H, W). extra_inputs=()."
            )
        if image.size(-2) % 64 != 0 or image.size(-1) % 64 != 0:
            raise ValueError(
                "spatial size must be a multiple of 64; section 4.2 uses 512×512"
            )
        c1, c2, c3 = self.cnn_encoder(image)
        cnn_features = self.cnn_decoder(c1, c2, c3)
        transformer_features = self.transformer(image)
        fused = self.llcs(cnn_features, transformer_features)
        return self.head(fused)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def fundus_class_names() -> tuple[str, ...]:
    return FUNDUS_CLASSES
