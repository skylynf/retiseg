#!/usr/bin/env python3
"""E10 cost audit: parameters, FLOPs, inference time and peak memory per model.

Builds every registered wrapper the way bench.predict does (the official code,
weights from pretrained/), then measures on one 1440 canvas (the B-std canvas,
a 1440 x 960 field of view as in IDRiD) with the model's own inference geometry
(card.forward_size windows, overlap 0.5, card.window_batch windows per forward):

    params_m             total parameters, in millions
    params_trainable_m   parameters with requires_grad under the model's recipe
    flops_g              GFLOPs of one forward over the whole canvas (all windows),
                         torch.utils.flop_counter; conv and matmul only
    infer_s_per_image    median wall time of one canvas over --repeat passes after
                         two warm-up passes, eval mode, no grad
    peak_infer_mem_gb    torch.cuda.max_memory_allocated during those passes

Training peak memory is not measured here: it is read from the training logs
(torch.cuda.max_memory_allocated written by bench.bstd at the end of a run) and
training time from environment.json; paper/scripts/collect_runs.py reads both.

Output: a TSV in the format of paper/data/cost_measured.tsv (model, train_set,
metric, value, note), one block per model, appended to --out. Run on a GPU node:

    python bench/scripts/cost_audit.py --out paper/data/cost_audit.tsv
    python bench/scripts/cost_audit.py --models U-Net M2MRF --repeat 10

SAM-Adapter (ViT-H, 1024 windows) and MedSAM load 2.5 GB of weights; run them
with the others only on a 40 GB card. Models whose weights are not under
pretrained/ are skipped with a message.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.infer import canvas_logits  # noqa: E402
from bench.models import registry  # noqa: E402

PAPER_NAME = {  # registry name -> name used in paper/data/results.csv
    "DeepLabv3": "DeepLabv3",
    "HRNet": "HRNet-W48",
    "MedSAM": "MedSAM (full FT)",
    "RETFound": "RETFound + UPerNet",
}


def trainable_parameters(model, dataset="IDRiD"):
    """Parameters the B-std recipe trains. Wrappers that freeze parts do it in prepare_for_training when present."""
    if hasattr(model, "prepare_for_training"):
        try:
            model.prepare_for_training(model.author_recipe(dataset))
        except Exception:  # noqa: BLE001 - the count is informative only
            pass
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flops_of(model, tensor, card):
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc, torch.no_grad():
        canvas_logits(model, tensor, card, window=card.forward_size, overlap=0.5, batch=card.window_batch)
    return fc.get_total_flops()


def time_of(model, tensor, card, repeat, device):
    def once():
        with torch.no_grad():
            canvas_logits(model, tensor, card, window=card.forward_size, overlap=0.5, batch=card.window_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(2):
        once()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    return statistics.median(times), peak


def audit(name, repeat, device, canvas):
    model = registry.build_model(name)
    card = model.card
    total = model.n_parameters()
    trainable = trainable_parameters(model)
    model = model.to(device).eval()
    tensor = torch.rand(3, *canvas, device=device)
    rows = []
    paper = PAPER_NAME.get(name, name)
    rows.append((paper, "", "params_m", total / 1e6, "cost_audit.py"))
    rows.append((paper, "", "params_trainable_m", trainable / 1e6, "cost_audit.py"))
    try:
        rows.append((paper, "", "flops_g", flops_of(model, tensor, card) / 1e9, f"cost_audit.py, one {canvas[0]}x{canvas[1]} canvas, windows {card.forward_size}"))
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: FLOP counter failed ({e}); flops_g skipped")
    sec, peak = time_of(model, tensor, card, repeat, device)
    rows.append((paper, "", "infer_s_per_image", sec, f"cost_audit.py, median of {repeat} on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}"))
    if device.type == "cuda":
        rows.append((paper, "", "peak_infer_mem_gb", peak, "cost_audit.py"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(registry.WRAPPERS))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--canvas", type=int, nargs=2, default=(960, 1440), help="height width of the test canvas")
    ap.add_argument("--out", default=str(REPO / "docs" / "paper" / "retiseg_paper" / "paper" / "data" / "cost_audit.tsv"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    new_file = not out.exists()
    with out.open("a") as fh:
        if new_file:
            fh.write("model\ttrain_set\tmetric\tvalue\tnote\n")
        for name in args.models:
            print(f"{name} ...", flush=True)
            try:
                rows = audit(name, args.repeat, device, tuple(args.canvas))
            except FileNotFoundError as e:
                print(f"  skipped: {e}")
                continue
            for r in rows:
                fh.write("\t".join([r[0], r[1], r[2], f"{r[3]:.4g}", r[4]]) + "\n")
                print(f"  {r[2]} = {r[3]:.4g}")
            fh.flush()
    print(f"appended to {out}; merge with: python paper/scripts/collect_runs.py (reads data/cost_measured.tsv and data/cost_audit.tsv)")


if __name__ == "__main__":
    main()
