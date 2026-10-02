#!/usr/bin/env python3
"""Train one CLC-Net variant for 60 epochs.

The reported checkpoint is epoch 60. Validation loss is not used to pick
a different epoch. Run from this directory:

    python train.py --dataset idrid --data-root /path/to/IDRiD --output runs/idrid_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clcnet.assumptions import EPOCHS, SEED, VARIANTS
from clcnet.dataset import LesionPatchDataset
from clcnet.engine import build_model, build_optimizer, make_loader, public_run_record, train_one_epoch
from clcnet.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLC-Net from the locked paper settings")
    parser.add_argument("--dataset", required=True, choices=["idrid", "ddr", "eophtha"])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (output / "cache")
    dataset = LesionPatchDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split="train",
        train=True,
        cache_dir=cache,
        split_file=ROOT / "splits" / f"eophtha_seed{SEED}.json",
        seed=SEED,
    )
    loader = make_loader(dataset, train=True, seed=SEED, workers=args.workers)
    model = build_model(args.variant).to(args.device)
    optimizer = build_optimizer(model)
    record = public_run_record(args.variant, args.dataset, "train")
    record["train_images"] = len(dataset.records)
    record["train_patches"] = len(dataset)
    (output / "run_config.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    history = []
    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch)
        losses = train_one_epoch(model, loader, optimizer, args.device, epoch)
        row = {"epoch": epoch, **losses}
        history.append(row)
        print(
            f"epoch {epoch:02d}/{EPOCHS} lr {losses['lr']:.1e} "
            f"loss {losses['total']:.4f} steps {losses['steps']}",
            flush=True,
        )
        _save(output / "last.pt", model, optimizer, epoch, record)
        if epoch in (20, 50, 60):
            _save(output / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint: {output / 'epoch_060.pt'}")


def _save(path: Path, model, optimizer, epoch: int, record: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "record": record,
        },
        path,
    )


if __name__ == "__main__":
    main()
