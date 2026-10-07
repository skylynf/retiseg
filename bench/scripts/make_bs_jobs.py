"""Write the B-std cell configs, the budget-probe cells and bench/configs/bs_jobs.yaml.

One cell per model, dataset and seed in launch_b1._PAIRS["BS"]. Cells carry
no budget keys; bs_frozen.yaml sets the budget. est_hours only orders the
queue: rough A100 hours at the provisional budget, larger for slower nets.

    python bench/scripts/make_bs_jobs.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from bench.scripts.launch_b1 import _BINDING, _PAIRS, _SLUG, active_models

EST_HOURS = {
    "U-Net": (3, 5),
    "DeepLabv3": (4, 7),
    "HRNet": (4, 7),
    "Swin-Unet": (2, 4),
    "FCT": (3, 5),
    "H2Former": (5, 9),
    "M2MRF": (7, 12),
    "HACDR-Net": (6, 10),
}
# Budget probe: seed 0 at twice the frozen amount, train and validation only.
# Not in bs_jobs.yaml; bench/scripts/server_run.sh probe runs them.
PROBE_MODELS = ("U-Net", "M2MRF")
CELLS = _ROOT / "bench" / "configs" / "cells"
JOBS = _ROOT / "bench" / "configs" / "bs_jobs.yaml"
HEADER = """# B-std table. Written by bench/scripts/make_bs_jobs.py; edit that script.
# Budget, augmentation and selection are in bench/configs/bs_frozen.yaml.
# Order: --smoke, --seed0 (train and validation only), read the curves,
# --score-seed0, then --rest. Every command takes --jobs bench/configs/bs_jobs.yaml.
"""


def cell(model, dataset, seed):
    return {
        "experiment": "BS",
        "model": model,
        "dataset": f"dataset/prepared/{dataset}",
        "split_train": "train",
        "split_val": "val",
        "split_test": "test",
        "fov_diameter": 1440,
        "seed": int(seed),
        "num_workers": 4,
    }


def job(model, dataset, seed, smoke=False):
    slug = _SLUG[model]
    env, script, _predict = _BINDING[model]
    row = {
        "name": f"{'smoke_' if smoke else ''}bs_{slug}_{dataset.lower()}_seed{seed}",
        "experiment": "BS",
        "model": model,
        "dataset": dataset,
        "seed": int(seed),
        "gpu": 0,
        "env": env,
        "config": f"bench/configs/cells/bs_{slug}_{dataset.lower()}_seed{seed}.yaml",
        "script": script,
    }
    if smoke:
        row["max_steps"] = 20
    else:
        row["est_hours"] = EST_HOURS[model][0 if dataset == "IDRiD" else 1]
    return row


def main():
    models = active_models()
    jobs, smoke = [], []
    for model in models:
        for dataset, seed in _PAIRS["BS"]:
            path = CELLS / f"bs_{_SLUG[model]}_{dataset.lower()}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(cell(model, dataset, seed), sort_keys=False))
            jobs.append(job(model, dataset, seed))
        for dataset in ("IDRiD", "DDR"):
            smoke.append(job(model, dataset, 0, smoke=True))
    for index, row in enumerate(jobs + smoke):
        row["gpu"] = index % 8
    document = {"experiment": "BS", "jobs": jobs, "smoke": smoke}
    JOBS.write_text(HEADER + yaml.safe_dump(document, sort_keys=False))
    for model in PROBE_MODELS:
        for dataset in ("IDRiD", "DDR"):
            probe = dict(cell(model, dataset, 0), budget_probe=2)
            path = CELLS / f"bs_probe_{_SLUG[model]}_{dataset.lower()}_seed0.yaml"
            path.write_text(yaml.safe_dump(probe, sort_keys=False))
    print(
        f"wrote {len(jobs)} jobs, {len(smoke)} smoke jobs, {len(jobs)} cells "
        f"and {2 * len(PROBE_MODELS)} probe cells",
        flush=True,
    )


if __name__ == "__main__":
    main()
