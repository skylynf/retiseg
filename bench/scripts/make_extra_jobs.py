"""Write the cells and bench/configs/extra_jobs.yaml for runs outside the B-std table.

Now: E2c, the learning-rate check (05 E2c; 13 section 4 step 4). One model per
family that is trained under B-std: HRNet for the general networks (the best
of them on both sources) and M2MRF for the fundus-specific ones; nnU-Net keeps
its own schedule and the foundation models get theirs in E5's encoder
comparison. Learning rate (and the poly floor) times 1/3 and 3, seed 0, both
sources, validation only; times 1 is the B-std seed-0 run. 8 jobs, about 58
A100 hours at the 10-08 rates.

    python bench/scripts/make_extra_jobs.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from bench.scripts.launch_b1 import _BINDING, _SLUG
from bench.scripts.make_bs_jobs import EST_HOURS, cell

E2C_MODELS = ("HRNet", "M2MRF")
E2C_FACTORS = {"lr1o3": 1.0 / 3.0, "lr3": 3.0}
CELLS = _ROOT / "bench" / "configs" / "cells"
JOBS = _ROOT / "bench" / "configs" / "extra_jobs.yaml"
HEADER = """# Runs outside the B-std main table. Written by bench/scripts/make_extra_jobs.py; edit that script.
# Every job here is train and validation only (--diagnostic); none reads a test split.
# Start: RETISEG_GPUS="..." bash bench/scripts/server_run.sh extra E2c
"""


def e2c_jobs():
    jobs = []
    for model in E2C_MODELS:
        slug = _SLUG[model]
        env, script, _predict = _BINDING[model]
        for dataset in ("IDRiD", "DDR"):
            for tag, factor in E2C_FACTORS.items():
                name = f"e2c_{slug}_{dataset.lower()}_seed0_{tag}"
                body = dict(cell(model, dataset, 0), lr_scale=float(factor))
                path = CELLS / f"{name}.yaml"
                path.write_text(yaml.safe_dump(body, sort_keys=False))
                jobs.append(
                    {
                        "name": name,
                        "experiment": "E2c",
                        "model": model,
                        "dataset": dataset,
                        "seed": 0,
                        "env": env,
                        "config": f"bench/configs/cells/{name}.yaml",
                        "script": script,
                        "run_dir": f"runs/E2c_{slug}_{dataset.lower()}_seed0_{tag}",
                        "est_hours": EST_HOURS[model][0 if dataset == "IDRiD" else 1],
                    }
                )
    return jobs


def main():
    jobs = e2c_jobs()
    JOBS.write_text(HEADER + yaml.safe_dump({"jobs": jobs}, sort_keys=False))
    print(f"wrote {len(jobs)} jobs to {JOBS.relative_to(_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
