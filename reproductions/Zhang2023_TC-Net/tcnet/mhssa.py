"""Multi-head squeezable self-attention. Equation (4) and Figure 3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tcnet.assumptions import ATTN_DROPOUT, KV_REDUCE_SIZE, MLP_RATIO, NUM_HEADS


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each spatial position. Assumption A7."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class RelativePositionBias(nn.Module):
    """Swin relative-position bias B, on the reduced key grid. Assumption A9."""

    def __init__(self, num_heads: int, height: int, width: int):
        super().__init__()
        self.num_heads = num_heads
        self.height = height
        self.width = width
        self.table = nn.Parameter(
            torch.randn((2 * height - 1) * (2 * width - 1), num_heads) * 0.02
        )
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        flat = coords.flatten(1)
        relative = flat[:, :, None] - flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += height - 1
        relative[:, :, 1] += width - 1
        relative[:, :, 0] *= 2 * width - 1
        index = relative.sum(-1)
        self.register_buffer("index", index, persistent=False)

    def forward(self, query_h: int, query_w: int) -> Tensor:
        if query_h % self.height != 0 or query_w % self.width != 0:
            raise ValueError(
                f"query {(query_h, query_w)} is not a multiple of "
                f"the reduced key grid {(self.height, self.width)}"
            )
        bias = self.table[self.index.reshape(-1)].view(
            self.height, self.width, self.height * self.width, self.num_heads
        )
        bias = bias.repeat_interleave(query_h // self.height, dim=0)
        bias = bias.repeat_interleave(query_w // self.width, dim=1)
        bias = bias.reshape(query_h * query_w, self.height * self.width, self.num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)


class MHSSA(nn.Module):
    """Equation (4). Q stays at its own resolution. K and V are bilinearly reduced."""

    def __init__(
        self,
        channels: int,
        heads: int = NUM_HEADS,
        reduce_size: int = KV_REDUCE_SIZE,
        dropout: float = ATTN_DROPOUT,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels {channels} not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = channels // heads
        self.reduce_size = reduce_size
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.drop = nn.Dropout(dropout)
        self.relative = RelativePositionBias(heads, reduce_size, reduce_size)

    def _heads(self, tensor: Tensor, height: int, width: int) -> Tensor:
        batch = tensor.shape[0]
        tensor = tensor.view(batch, self.heads, self.head_dim, height * width)
        return tensor.permute(0, 1, 3, 2)

    def forward(self, query: Tensor, context: Tensor | None = None) -> Tensor:
        if context is None:
            context = query
        batch, _, query_h, query_w = query.shape
        _, _, key_h, key_w = context.shape
        q = self._heads(self.q_proj(query), query_h, query_w)
        k = self.k_proj(context)
        v = self.v_proj(context)
        if (key_h, key_w) != (self.reduce_size, self.reduce_size):
            k = F.interpolate(k, size=self.reduce_size, mode="bilinear", align_corners=False)
            v = F.interpolate(v, size=self.reduce_size, mode="bilinear", align_corners=False)
        k = self._heads(k, self.reduce_size, self.reduce_size)
        v = self._heads(v, self.reduce_size, self.reduce_size)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        logits = logits + self.relative(query_h, query_w)
        attention = self.drop(torch.softmax(logits, dim=-1))
        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        return out.view(batch, self.heads * self.head_dim, query_h, query_w)


class ResidualConv(nn.Module):
    """The residual block that maps X to X1. Assumption A6."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class MLP(nn.Module):
    """Position-wise MLP. Assumption A8."""

    def __init__(self, channels: int, ratio: int = MLP_RATIO):
        super().__init__()
        hidden = channels * ratio
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class EncoderTransformerBlock(nn.Module):
    """Residual conv, then pre-norm MHSSA and pre-norm MLP."""

    def __init__(self, channels: int):
        super().__init__()
        self.residual = ResidualConv(channels)
        self.norm1 = ChannelLayerNorm(channels)
        self.attn = MHSSA(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.mlp = MLP(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.residual(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossTransformerBlock(nn.Module):
    """Q from the low-level map, K and V from the high-level map. Section 3.2."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm_q = ChannelLayerNorm(channels)
        self.norm_kv = ChannelLayerNorm(channels)
        self.attn = MHSSA(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.mlp = MLP(channels)

    def forward(self, low: Tensor, high: Tensor) -> Tensor:
        fused = low + self.attn(self.norm_q(low), self.norm_kv(high))
        return fused + self.mlp(self.norm2(fused))
