#!/usr/bin/env python3
"""Write the BF cells and bench/configs/factor_jobs.yaml for E3b, E4, E7, E8 and E9 (docs/paper/16 section 3).

Every cell is a bench/factor.py cell: B-std protocol, the DDR pixel budget and
DDR recipe, one factor changed, seed 0. The paper block of each cell names the
row of paper/data/results.csv that paper/scripts/collect_runs.py fills.

    E3b  U-Net, M2MRF, SAM2-UNet, RETFound x {TJDR_std, FGADR}          8 cells   (nnU-Net: its own bridge, 17 section 9)
    E4   U-Net, M2MRF x {DDR+FGADR 383, DDR+FGADR, DDR+RetinalLesions, RetinalLesions}   8 cells
    E7   M2MRF, RETFound x {512, native}                                  4 cells
    E8   U-Net, M2MRF x {DDR 10%, DDR 50%, fine merged}                   6 cells
    E9   U-Net x {CE, batch Dice}                                          2 cells
    28 cells. est_hours order the queue only (A100 hours; DDR B-std rates from docs/ops/00, E7 native and
    the 1.5-2x sources by pixels).

    python bench/scripts/make_factor_jobs.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from bench.scripts.launch_b1 import _BINDING, _SLUG  # noqa: E402

CELLS = REPO / "bench" / "configs" / "cells"
JOBS = REPO / "bench" / "configs" / "factor_jobs.yaml"
SUBSETS = "bench/data/splits/subsets"
PREP = "dataset/prepared"
# DDR seed-0 hours at the B-std budget (docs/ops/00; SAM2-UNet and RETFound from design/15 estimates).
DDR_HOURS = {"U-Net": 6.0, "M2MRF": 11.6, "SAM2-UNet": 16.0, "RETFound": 22.0}
HEADER = """# Factor experiments E3b, E4, E7, E8, E9 (docs/paper/16 section 3). Written by bench/scripts/make_factor_jobs.py; edit that script.
# Each job trains a bench/factor.py cell (B-std protocol, DDR budget and recipe, one factor changed, seed 0) and scores its
# internal test; independent tests follow through bench/e3a.py. Start: RETISEG_GPUS="..." bash bench/scripts/server_run.sh factor [E3b|E4|E7|E8|E9]
"""


def src(dataset, split="train", subset=None):
    item = {"dataset": f"{PREP}/{dataset}", "split": split}
    if subset:
        item["subset"] = f"{SUBSETS}/{subset}"
    return item


def cell(factor, model, train, evaluation, paper, fov=1440, loss_override=None):
    body = {
        "experiment": "BF",
        "factor": factor,
        "model": model,
        "dataset": f"{PREP}/{evaluation}",
        "split_val": "val",
        "split_test": "test",
        "fov_diameter": int(fov),
        "train": train,
        "budget_reference": "DDR",
        "seed": 0,
        "num_workers": 4,
        "paper": paper,
    }
    if loss_override:
        body["loss_override"] = loss_override
    return body


def jobs():
    out = []

    def add(name, factor, model, body, hours):
        env, script, _predict = _BINDING[model]
        path = CELLS / f"{name}.yaml"
        path.write_text(yaml.safe_dump(body, sort_keys=False))
        out.append({"name": name, "experiment": factor, "model": model, "dataset": Path(body["dataset"]).name, "seed": 0,
                    "env": env, "config": f"bench/configs/cells/{name}.yaml", "script": script,
                    "run_dir": f"runs/BF_{name}", "est_hours": round(hours, 1)})

    # E3b: train on another source, evaluate on it (independent tests through E3a).
    for model in ("U-Net", "M2MRF", "SAM2-UNet", "RETFound"):
        for source, label in (("TJDR_std", "TJDR-std"), ("FGADR", "FGADR")):
            name = f"e3b_{_SLUG[model]}_{source.lower()}_seed0"
            add(name, "E3b", model, cell("E3b", model, [src(source)], source, {"experiment": "E3b", "variant": "bstd", "train_set": label}), DDR_HOURS[model])
    # E4: granularity of the training labels, evaluated on DDR (fine) and, through E3a, on IDRiD and TJDR_std.
    e4 = [
        ("ddr_fgadr383", [src("DDR"), src("FGADR", subset="FGADR_train_383.json")], "DDR", "DDR + FGADR (383)"),
        ("ddr_fgadr", [src("DDR"), src("FGADR")], "DDR", "DDR + FGADR"),
        ("ddr_retles", [src("DDR"), src("RetinalLesions")], "DDR", "DDR + Retinal-Lesions"),
        ("retles", [src("RetinalLesions")], "RetinalLesions", "Retinal-Lesions"),
    ]
    for model in ("U-Net", "M2MRF"):
        for tag, train, evaluation, label in e4:
            name = f"e4_{_SLUG[model]}_{tag}_seed0"
            add(name, "E4", model, cell("E4", model, train, evaluation, {"experiment": "E4", "variant": label, "train_set": label}), DDR_HOURS[model])
    # E7: canvas size, DDR.
    for model in ("M2MRF", "RETFound"):
        for fov, tag, label, factor_h in ((512, "512", "512", 1.0), (0, "native", "native patches", 1.6)):
            name = f"e7_{_SLUG[model]}_ddr_{tag}_seed0"
            add(name, "E7", model, cell("E7", model, [src("DDR")], "DDR", {"experiment": "E7", "variant": f"res:{label}", "train_set": "DDR"}, fov=fov), DDR_HOURS[model] * factor_h)
    # E8: amount of DDR, and the fine-grained sources merged; same pixel budget in every row.
    e8 = [
        ("ddr10", [src("DDR", subset="DDR_train_10pct.json")], "size:DDR 10%"),
        ("ddr50", [src("DDR", subset="DDR_train_50pct.json")], "size:DDR 50%"),
        ("fine", [src("IDRiD"), src("DDR"), src("TJDR_std")], "size:fine merged"),
    ]
    for model in ("U-Net", "M2MRF"):
        for tag, train, label in e8:
            name = f"e8_{_SLUG[model]}_{tag}_seed0"
            add(name, "E8", model, cell("E8", model, train, "DDR", {"experiment": "E8", "variant": label, "train_set": "DDR"}), DDR_HOURS[model])
    # E9: U-Net loss terms, DDR. CE + batch Dice is the main table; CE + per-image Dice is the registered run.
    e9 = [
        ("ce", {"loss": "bce_with_logits", "loss_params": {}}, "loss:CE"),
        ("dice", {"loss": "dice", "loss_params": {"batch_dice": True, "smooth": 1.0e-5}}, "loss:Dice"),
    ]
    for tag, override, label in e9:
        name = f"e9_unet_ddr_{tag}_seed0"
        add(name, "E9", "U-Net", cell("E9", "U-Net", [src("DDR")], "DDR", {"experiment": "E9", "variant": label, "train_set": "DDR"}, loss_override=override), DDR_HOURS["U-Net"])
    return out


def main():
    CELLS.mkdir(parents=True, exist_ok=True)
    table = jobs()
    JOBS.write_text(HEADER + yaml.safe_dump({"jobs": table}, sort_keys=False))
    hours = sum(job["est_hours"] for job in table)
    print(f"wrote {len(table)} jobs to {JOBS.relative_to(REPO)}, about {hours:.0f} A100 hours")


if __name__ == "__main__":
    main()
