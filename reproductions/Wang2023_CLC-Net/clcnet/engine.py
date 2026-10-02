"""Training and evaluation steps that do not own the data layout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from clcnet.assumptions import BATCH_SIZE, CONTEXT_SIZE, EPOCHS, LOCAL_SIZE, SEED, VARIANTS, learning_rate
from clcnet.losses import total_loss
from clcnet.model import CLCNet
from clcnet.seed import seed_worker, set_seed


def build_model(variant_name: str) -> CLCNet:
    if variant_name not in VARIANTS:
        known = ", ".join(VARIANTS)
        raise KeyError(f"unknown variant {variant_name}. known: {known}")
    return CLCNet(VARIANTS[variant_name])


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    from clcnet.assumptions import ADAM_BETAS, ADAM_EPS, ADAM_WEIGHT_DECAY, LR_INITIAL

    return torch.optim.Adam(
        model.parameters(),
        lr=LR_INITIAL,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
    )


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> dict:
    model.train()
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(epoch)
    totals = {}
    steps = 0
    for batch in loader:
        local = batch["local_image"].to(device, non_blocking=True)
        context = batch["context_image"].to(device, non_blocking=True)
        moved = {
            "local_mask": batch["local_mask"].to(device, non_blocking=True),
            "context_mask": batch["context_mask"].to(device, non_blocking=True),
            "local_cls": batch["local_cls"].to(device, non_blocking=True),
            "context_cls": batch["context_cls"].to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        output = model(local, context)
        parts = total_loss(output, moved)
        parts["total"].backward()
        optimizer.step()
        steps += 1
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    if steps == 0:
        raise RuntimeError("training loader produced no batches; check drop_last and the split size")
    return {key: value / steps for key, value in totals.items()} | {
        "lr": learning_rate(epoch),
        "steps": steps,
    }


def predict_probabilities(model, local, context, head: str) -> torch.Tensor:
    model.eval()
    output = model(local, context)
    if head == "local":
        probabilities = torch.softmax(output["local_logits"], dim=1)
    elif head == "context":
        probabilities = torch.softmax(output["context_logits"], dim=1)
        probabilities = F.interpolate(
            probabilities,
            size=(CONTEXT_SIZE, CONTEXT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    else:
        raise ValueError(head)
    return probabilities


def stitch_image(dataset, model, image_id: str, device, head: str, batch_size: int) -> np.ndarray:
    """Average overlapping window probabilities on the cropped FOV canvas (A15)."""
    target = dataset.full_target(image_id)
    height, width = target["height"], target["width"]
    accumulator = np.zeros((5, height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    indices = [i for i, patch in enumerate(dataset.patches) if patch[0] == image_id]
    model.eval()
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        samples = [dataset[index] for index in chunk]
        local = torch.stack([sample["local_image"] for sample in samples]).to(device)
        context = torch.stack([sample["context_image"] for sample in samples]).to(device)
        with torch.no_grad():
            probabilities = predict_probabilities(model, local, context, head).cpu().numpy()
        for sample, probability in zip(samples, probabilities):
            y, x = sample["y"], sample["x"]
            if head == "local":
                _add_window(accumulator, counts, probability, y, x, LOCAL_SIZE)
            else:
                origin_y, origin_x = y - (CONTEXT_SIZE - LOCAL_SIZE) // 2, x - (CONTEXT_SIZE - LOCAL_SIZE) // 2
                _add_window(accumulator, counts, probability, origin_y, origin_x, CONTEXT_SIZE)
    covered = counts > 0
    probabilities = np.zeros_like(accumulator)
    probabilities[:, covered] = accumulator[:, covered] / counts[covered]
    probabilities[0, ~covered] = 1.0
    return probabilities


def _add_window(accumulator, counts, probability, y, x, size):
    height, width = counts.shape
    src_y0 = max(-y, 0)
    src_x0 = max(-x, 0)
    dst_y0 = max(y, 0)
    dst_x0 = max(x, 0)
    dst_y1 = min(y + size, height)
    dst_x1 = min(x + size, width)
    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    accumulator[:, dst_y0:dst_y1, dst_x0:dst_x1] += probability[:, src_y0:src_y1, src_x0:src_x1]
    counts[dst_y0:dst_y1, dst_x0:dst_x1] += 1.0


def public_run_record(variant_name: str, dataset: str, split: str) -> dict:
    from clcnet.assumptions import ASSUMPTIONS, EXTRA_INPUTS, INPUTS, PAPER

    return {
        "paper": PAPER,
        "source": "本项目重训",
        "extra_inputs": list(EXTRA_INPUTS),
        "inputs": list(INPUTS),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "reported_checkpoint_epoch": EPOCHS,
        "variant": VARIANTS[variant_name].as_dict(),
        "dataset": dataset,
        "split": split,
        "assumptions": [{"id": item["id"], "text": item["text"]} for item in ASSUMPTIONS],
        "optimizer_note": "Adam hyperparameters that the paper does not state are the locked values in clcnet/assumptions.py",
    }


def make_loader(dataset, train: bool, seed: int = SEED, workers: int = 0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=workers,
        drop_last=train,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def prepare_runtime(seed: int = SEED):
    set_seed(seed)
