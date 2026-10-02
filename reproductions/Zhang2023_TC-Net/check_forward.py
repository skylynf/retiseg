"""Shape check for the TC-Net design. Not a training run and not a table fit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from tcnet import SEED, TCNet, set_seed
from tcnet.assumptions import EPOCHS, POSTHOC
from tcnet.loss import DynamicCyclicalFocalLoss, class_beta, cyclical_indicator
from tcnet.model import parameter_count


def main() -> None:
    set_seed(SEED)
    model = TCNet(num_classes=5)
    model.train()
    image = torch.zeros(1, 3, 128, 128)
    logits = model(image)
    if logits.shape != (1, 5, 128, 128):
        raise SystemExit(f"unexpected logits shape {tuple(logits.shape)}")
    target = torch.zeros(1, 128, 128, dtype=torch.long)
    target[:, 40:50, 40:50] = 1
    target[:, 60:64, 60:64] = 3
    criterion = DynamicCyclicalFocalLoss()
    loss_early = criterion(logits, target, epoch=1)
    loss_mid = criterion(logits, target, epoch=EPOCHS // 2)
    loss_late = criterion(logits, target, epoch=EPOCHS)
    loss_early.backward()
    model.eval()
    with torch.no_grad():
        full = model(torch.zeros(1, 3, 512, 512))
    if full.shape != (1, 5, 512, 512):
        raise SystemExit(f"unexpected 512 logits shape {tuple(full.shape)}")
    counts = torch.tensor([1000, 10, 20, 1, 5], dtype=torch.float32)
    beta = class_beta(counts)
    params = parameter_count(model)
    print(f"seed {SEED}")
    print(f"extra_inputs {model.extra_inputs}")
    print(f"logits {tuple(logits.shape)}")
    print(f"parameters {params}")
    print(f"table6_parameters {POSTHOC['params']}")
    print(f"indicator epoch1 {cyclical_indicator(1, EPOCHS):.6f}")
    print(f"indicator midpoint {cyclical_indicator(EPOCHS // 2, EPOCHS):.6f}")
    print(f"indicator last {cyclical_indicator(EPOCHS, EPOCHS):.6f}")
    print(f"beta {beta.tolist()}")
    print(f"loss {float(loss_early):.6f} {float(loss_mid):.6f} {float(loss_late):.6f}")
    print(f"beta_source {criterion.last_beta_source}")


if __name__ == "__main__":
    main()
