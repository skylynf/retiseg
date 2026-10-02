#!/usr/bin/env python3
"""Stitch a checkpoint and write lesion-wise AUPR.

Results are labeled 本项目重训. Paper tables are not copied into this file.
The primary head is the local branch. Pass --head context for the other
output of the same collaborative model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clcnet.assumptions import SEED, VARIANTS
from clcnet.dataset import LesionPatchDataset
from clcnet.engine import build_model, public_run_record, stitch_image
from clcnet.metrics import score_split
from clcnet.seed import set_seed

REPORT_LESIONS = {
    "idrid": ("HE", "MA", "EX", "SE"),
    "ddr": ("HE", "MA", "EX", "SE"),
    "eophtha": ("MA", "EX"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CLC-Net checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(REPORT_LESIONS))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--head", default="local", choices=["local", "context"])
    parser.add_argument("--variant", default=None, choices=sorted(VARIANTS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint["record"]["variant"]["name"]
    model = build_model(variant)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()
    cache = args.cache_dir or (args.checkpoint.parent / "cache")
    dataset = LesionPatchDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        train=False,
        cache_dir=cache,
        split_file=ROOT / "splits" / f"eophtha_seed{SEED}.json",
        seed=SEED,
    )
    probabilities = []
    binaries = []
    for image_id in dataset.records:
        probabilities.append(stitch_image(dataset, model, image_id, args.device, args.head, args.batch_size))
        binaries.append(dataset.full_target(image_id)["planes"])
    metrics = score_split(probabilities, binaries, REPORT_LESIONS[args.dataset])
    payload = public_run_record(variant, args.dataset, args.split)
    payload["checkpoint"] = str(args.checkpoint)
    payload["checkpoint_epoch"] = int(checkpoint["epoch"])
    payload["predict_head"] = args.head
    payload["metrics"] = metrics
    payload["result_source"] = "本项目重训"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    destination = args.output or args.checkpoint.parent / f"metrics_{args.dataset}_{args.split}_{args.head}.json"
    destination.write_text(text, encoding="utf-8")
    mean = metrics["AUPR_mean"]
    print(f"AUPR mean {mean:.4f}  head {args.head}  wrote {destination}")


if __name__ == "__main__":
    main()
