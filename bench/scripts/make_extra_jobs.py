"""Write the cells and bench/configs/extra_jobs.yaml for runs outside the B-std table.

E5tf32: SAM2-UNet IDRiD seed 0 with full-fp32 matrix multiplies, to compare
with the TF32 seed 0 of the main table (design/15 section 16 item 1).
E5enc: the encoder comparison's learning-rate sweep on DDR seed 0 (design/15
section 8.3): RETFound's UPerNet head and recipe on five ViT-L encoders, learning rate
x 1/3, 1, 3 (RETFound x 1 is the main-table DDR seed 0). The chosen rate is
written to e5_frozen.yaml from validation before the other seeds run.
E2c, the learning-rate check (05 E2c; 13 section 4 step 4). One model per
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
ENCODERS = ("RETFound", "ViT-UPerNet MAE-IN1k", "ViT-UPerNet RETFound-DINOv2", "ViT-UPerNet DINOv2", "ViT-UPerNet augreg")
ENC_FACTORS = {"lr1o3": 1.0 / 3.0, "lr1": 1.0, "lr3": 3.0}
ENC_HOURS = 21.0  # RETFound DDR estimate (design/15 section 9); ViT-L/14 has 256 tokens per 224 window instead of 196
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


def _job(name, experiment, model, dataset, body, hours):
    slug = _SLUG[model]
    env, script, _predict = _BINDING[model]
    path = CELLS / f"{name}.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return {
        "name": name,
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "seed": int(body["seed"]),
        "env": env,
        "config": f"bench/configs/cells/{name}.yaml",
        "script": script,
        "run_dir": f"runs/{experiment}_{name.split('_', 1)[1]}",
        "est_hours": float(hours),
    }


def e5_jobs():
    jobs = [
        _job(
            "e5tf32_sam2_unet_idrid_seed0_fp32",
            "E5tf32",
            "SAM2-UNet",
            "IDRiD",
            dict(cell("SAM2-UNet", "IDRiD", 0), matmul_tf32=False),
            EST_HOURS["SAM2-UNet"][0] * 3,
        )
    ]
    for model in ENCODERS:
        for tag, factor in ENC_FACTORS.items():
            if model == "RETFound" and tag == "lr1":
                continue  # the main-table BS_retfound_ddr_seed0
            body = cell(model, "DDR", 0)
            if factor != 1.0:
                body["lr_scale"] = float(factor)
            jobs.append(_job(f"e5enc_{_SLUG[model]}_ddr_seed0_{tag}", "E5enc", model, "DDR", body, ENC_HOURS))
    # IDRiD column of the encoder block (table 6 / figure 5d; docs/paper/14 note 59, decided 2026-10-23 after C1 on DDR):
    # the two general encoders at the main-table learning rate, same head and recipe as BS_retfound_idrid_seed0;
    # a supplementary row, never a main-table one (the 19 main-table rows are registered)
    for model in ("ViT-UPerNet MAE-IN1k", "ViT-UPerNet DINOv2"):
        jobs.append(_job(f"e5enc_{_SLUG[model]}_idrid_seed0_lr1", "E5enc", model, "IDRiD", cell(model, "IDRiD", 0), EST_HOURS["RETFound"][0]))
    # the second pair's retinal encoder on IDRiD (14 note 62, 2026-10-25, decided after the two IDRiD cells above had
    # been read): with it the IDRiD column carries both pairs, as the DDR column does; RETFound-MAE on IDRiD is the
    # main-table BS_retfound_idrid_seed0
    jobs.append(_job(f"e5enc_{_SLUG['ViT-UPerNet RETFound-DINOv2']}_idrid_seed0_lr1", "E5enc", "ViT-UPerNet RETFound-DINOv2", "IDRiD",
                     cell("ViT-UPerNet RETFound-DINOv2", "IDRiD", 0), EST_HOURS["RETFound"][0]))
    return jobs


def main():
    jobs = e5_jobs() + e2c_jobs()
    JOBS.write_text(HEADER + yaml.safe_dump({"jobs": jobs}, sort_keys=False))
    print(f"wrote {len(jobs)} jobs to {JOBS.relative_to(_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
