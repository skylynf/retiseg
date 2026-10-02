#!/usr/bin/env python3
"""Count parameters and multiply-adds for the full CLC-Net.

The paper reports 92.19 M parameters and 20.58e9 FLOPs. Those numbers are
printed beside the count from this implementation. They are not a target
used to change the architecture after the fact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clcnet.assumptions import REPORTED_FLOPS, REPORTED_PARAMS
from clcnet.engine import build_model
from clcnet.losses import total_loss


def main():
    model = build_model("full")
    model.eval()
    params = sum(layer.numel() for layer in model.parameters())
    macs = _macs(model)
    print(f"parameters {params} ({params / 1e6:.2f} M)")
    print(f"paper parameters {REPORTED_PARAMS:.0f} ({REPORTED_PARAMS / 1e6:.2f} M)")
    print(f"multiply_adds {macs:.3e}")
    print(f"multiply_adds_as_2flops {2 * macs:.3e}")
    print(f"paper FLOPs {REPORTED_FLOPS:.3e}")
    local = torch.zeros(2, 3, 256, 256)
    context = torch.zeros(2, 3, 512, 512)
    output = model(local, context)
    print("local_logits", tuple(output["local_logits"].shape))
    print("context_logits", tuple(output["context_logits"].shape))
    print("local_cls", tuple(output["local_cls_logits"].shape))
    print("context_cls", tuple(output["context_cls_logits"].shape))
    batch = {
        "local_mask": torch.zeros(2, 256, 256, dtype=torch.long),
        "context_mask": torch.zeros(2, 512, 512, dtype=torch.long),
        "local_cls": torch.zeros(2, 4),
        "context_cls": torch.zeros(2, 4),
    }
    model.train()
    parts = total_loss(output, batch)
    parts["total"].backward()
    print(f"loss {float(parts['total'].detach()):.4f}")


def _macs(model: torch.nn.Module) -> float:
    total = 0.0

    def conv_hook(layer, inputs, output):
        nonlocal total
        if isinstance(layer, torch.nn.Conv2d):
            out_h, out_w = output.shape[-2:]
            total += output.shape[0] * layer.out_channels * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w
        elif isinstance(layer, torch.nn.ConvTranspose2d):
            out_h, out_w = output.shape[-2:]
            total += output.shape[0] * layer.in_channels * (layer.out_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w
        elif isinstance(layer, torch.nn.Linear):
            total += output.shape[0] * layer.in_features * layer.out_features

    handles = []
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(conv_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 256, 256), torch.zeros(1, 3, 512, 512))
    for handle in handles:
        handle.remove()
    return float(total)


if __name__ == "__main__":
    main()
