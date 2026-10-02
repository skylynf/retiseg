"""CLC-Net: contextual branch, local branch, SKM fusion, auxiliary heads.

The forward pass follows Figure 2. The contextual 512×512 patch is
bilinearly reduced to 256×256 before its encoder (the 2x reduction drawn in
Figure 2(b)). Both branches then share the same stage geometry and
keep separate parameters. Fusion, when enabled, runs from the
contextual decoder into the local decoder at each of the four scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from clcnet.assumptions import (
    CLS_HIDDEN,
    DECODER_CHANNELS,
    LOCAL_SIZE,
    NUM_LESIONS,
    SKIP_CHANNELS,
    UP_IN_CHANNELS,
    Variant,
)
from clcnet.encoder import SEResNeXt50
from clcnet.skconv import SKConv


def initialize(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(layer.bias)


class UpBlock(nn.Module):
    """Conv2d-BN-SKNet-TransConv2d. Assumption A2."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid = in_channels // 2
        self.conv = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.relu = nn.ReLU(inplace=True)
        self.sk = SKConv(mid)
        self.up = nn.ConvTranspose2d(mid, mid, kernel_size=2, stride=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.sk(x)
        return self.up(x)


class ClassificationHead(nn.Module):
    """Two fully connected layers on the encoder output. Assumption A8."""

    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, CLS_HIDDEN)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(CLS_HIDDEN, NUM_LESIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.fc2(self.relu(self.fc1(x)))


class Fusion(nn.Module):
    """Combine one contextual decoder map into the local map. Assumption A7."""

    def __init__(self, channels: int, mode: str):
        super().__init__()
        self.mode = mode
        self.sk = SKConv(channels) if mode == "skm" else None
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, local_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        if context_feat.shape[-2:] != local_feat.shape[-2:]:
            raise RuntimeError(
                f"decoder scales differ: local {tuple(local_feat.shape)} "
                f"context {tuple(context_feat.shape)}"
            )
        other = self.sk(context_feat) if self.sk is not None else context_feat
        return self.proj(torch.cat([local_feat, other], dim=1))


class Branch(nn.Module):
    def __init__(self, use_cls: bool, num_classes: int = 5):
        super().__init__()
        self.encoder = SEResNeXt50()
        self.blocks = nn.ModuleList(UpBlock(channels) for channels in UP_IN_CHANNELS)
        self.side_heads = nn.ModuleList(
            nn.Conv2d(channels, num_classes, kernel_size=1) for channels in DECODER_CHANNELS
        )
        self.fuse = nn.Conv2d(num_classes * len(DECODER_CHANNELS), num_classes, kernel_size=1)
        self.cls_head = ClassificationHead() if use_cls else None

    def forward(self, x: torch.Tensor, context_feats=None, fusions=None):
        if x.shape[-2:] != (LOCAL_SIZE, LOCAL_SIZE):
            raise ValueError(
                f"branch encoder expects {LOCAL_SIZE}×{LOCAL_SIZE}, got {tuple(x.shape[-2:])}"
            )
        stem, c1, c2, c3, c4 = self.encoder(x)
        skips = (c3, c2, c1, stem)
        if tuple(skip.shape[1] for skip in skips) != SKIP_CHANNELS:
            raise RuntimeError("encoder skip channels do not match the locked schedule")
        hidden = c4
        side_logits = []
        decoded = []
        out_size = x.shape[-2:]
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            hidden = torch.cat([hidden, skips[index]], dim=1)
            if fusions is not None:
                hidden = fusions[index](hidden, context_feats[index])
            if hidden.shape[1] != DECODER_CHANNELS[index]:
                raise RuntimeError(
                    f"decoder stage {index} has {hidden.shape[1]} channels, "
                    f"expected {DECODER_CHANNELS[index]}"
                )
            decoded.append(hidden)
            side = self.side_heads[index](hidden)
            side = F.interpolate(side, size=out_size, mode="bilinear", align_corners=False)
            side_logits.append(side)
        logits = self.fuse(torch.cat(side_logits, dim=1))
        cls_logits = self.cls_head(c4) if self.cls_head is not None else None
        return logits, cls_logits, decoded


class CLCNet(nn.Module):
    def __init__(self, variant: Variant, num_classes: int = 5):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.context = Branch(variant.use_cls, num_classes) if variant.use_context else None
        self.local = Branch(variant.use_cls, num_classes) if variant.use_local else None
        if variant.fusion == "none":
            self.fusions = None
        else:
            self.fusions = nn.ModuleList(
                Fusion(channels, variant.fusion) for channels in DECODER_CHANNELS
            )
        initialize(self)

    def forward(self, local: torch.Tensor | None, context: torch.Tensor | None):
        output = {}
        context_decoded = None
        if self.context is not None:
            if context is None:
                raise ValueError("contextual branch requires the 512×512 patch")
            context_in = F.interpolate(
                context,
                size=(LOCAL_SIZE, LOCAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            ctx_logits, ctx_cls, context_decoded = self.context(context_in)
            output["context_logits"] = ctx_logits
            output["context_cls_logits"] = ctx_cls
        if self.local is not None:
            if local is None:
                raise ValueError("local branch requires the 256×256 patch")
            fusions = None
            feats = None
            if self.fusions is not None:
                fusions = self.fusions
                feats = context_decoded
            loc_logits, loc_cls, _ = self.local(local, context_feats=feats, fusions=fusions)
            output["local_logits"] = loc_logits
            output["local_cls_logits"] = loc_cls
        return output
