"""Write data/results.csv: every number the figures, tables and text macros read.

Two kinds of rows, told apart by the ``source`` column.

measured     Already in the repository (docs/ops/00, docs/paper/14, audit JSON).
placeholder  Simulated here so that figures and tables can be laid out before
             the runs finish. These are NOT results. The generator is a fixed
             seeded model (class difficulty + model offset + domain shift +
             seed noise); its only job is to have the right shape and a
             plausible range.

Real runs replace placeholder rows by key with scripts/collect_runs.py.
Key = (experiment, model, variant, train_set, seed, test_set, cls, metric).

    python paper/scripts/build_placeholder.py            # refuses to overwrite an existing table
    python paper/scripts/build_placeholder.py --force    # start again from placeholders only
"""

import sys

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "results.csv"
RNG_SEED = 20261009

CLASSES = ["MA", "HE", "EX", "SE"]
TRAIN_MAIN = ["IDRiD", "DDR"]
TESTS = ["IDRiD", "DDR", "TJDR-std", "TJDR-uwf", "FGADR", "Retinal-Lesions", "DiaRetDB1"]
INDEPENDENT = ["TJDR-std", "TJDR-uwf", "FGADR", "Retinal-Lesions", "DiaRetDB1"]

# name, family, experiment that produces it, logit offset, MA handicap
# (models with a coarse stride or 16-pixel tokens lose the smallest lesions).
MAIN_MODELS = [
    ("U-Net", "general", "E2", -0.30, 0.00),
    ("DeepLabv3", "general", "E2", -0.45, -0.35),
    ("HRNet-W48", "general", "E2", 0.05, 0.05),
    ("Swin-Unet", "general", "E2", -0.60, -0.25),
    ("FCT", "general", "E2", -0.72, -0.15),
    ("nnU-Net", "auto", "E2", 0.10, 0.10),
    ("nnU-Net ResEnc-M", "auto", "E2", 0.16, 0.10),
    ("M2MRF", "fundus", "E2", 0.00, 0.10),
    ("HACDR-Net", "fundus", "E2", -0.25, -0.05),
    ("SAMed", "foundation", "E5", -0.50, -0.40),
    ("SAM-Adapter", "foundation", "E5", -0.55, -0.40),
    ("SAM2-UNet", "foundation", "E5", -0.06, -0.10),
    ("MedSAM (full FT)", "foundation", "E5", -0.45, -0.45),
    ("RETFound + UPerNet", "foundation", "E5", -0.28, -0.30),
]
E5_MODE = {
    "SAMed": "peft",
    "SAM-Adapter": "peft",
    "SAM2-UNet": "peft",
    "MedSAM (full FT)": "encoder_decoder",
    "RETFound + UPerNet": "encoder_decoder",
}
REPRESENTATIVE = ["U-Net", "nnU-Net", "M2MRF", "SAM2-UNet", "RETFound + UPerNet"]
# Scope after docs/paper/16 (2026-10-11): factor experiments use seed 0 and one or two models.
FACTOR_SEEDS = 1
E4_MODELS = ["U-Net", "M2MRF"]
E7_MODELS = ["M2MRF", "RETFound + UPerNet"]
E8_MODELS = ["U-Net", "M2MRF"]
E9_MODELS = ["U-Net"]

# Difficulty of each test set per class: the AUPR a reference model would reach
# when trained on the same source. IDRiD follows the M2MRF author checkpoint
# under this project's protocol; DDR is scaled to its measured mean of 0.401;
# the other five are guesses.
BASE = {
    "IDRiD": {"MA": 0.41, "HE": 0.65, "EX": 0.83, "SE": 0.66},
    "DDR": {"MA": 0.21, "HE": 0.46, "EX": 0.56, "SE": 0.37},
    "TJDR-std": {"MA": 0.30, "HE": 0.58, "EX": 0.70, "SE": 0.52},
    "TJDR-uwf": {"MA": 0.22, "HE": 0.50, "EX": 0.60, "SE": 0.45},
    "FGADR": {"MA": 0.20, "HE": 0.55, "EX": 0.62, "SE": 0.45},
    "Retinal-Lesions": {"MA": 0.15, "HE": 0.45, "EX": 0.50, "SE": 0.40},
    "DiaRetDB1": {"MA": 0.12, "HE": 0.35, "EX": 0.40, "SE": 0.30},
}
# Logit shift, relative to in-domain training, when a model trained on `train`
# is tested on `test`.
SHIFT = {
    ("IDRiD", "IDRiD"): 0.0,
    ("IDRiD", "DDR"): -0.75,
    ("IDRiD", "TJDR-std"): -0.85,
    ("IDRiD", "TJDR-uwf"): -1.50,
    ("IDRiD", "FGADR"): -1.10,
    ("IDRiD", "Retinal-Lesions"): -1.20,
    ("IDRiD", "DiaRetDB1"): -0.90,
    ("DDR", "DDR"): 0.0,
    ("DDR", "IDRiD"): -0.50,
    ("DDR", "TJDR-std"): -0.65,
    ("DDR", "TJDR-uwf"): -1.30,
    ("DDR", "FGADR"): -0.90,
    ("DDR", "Retinal-Lesions"): -1.05,
    ("DDR", "DiaRetDB1"): -0.75,
    ("TJDR-std", "TJDR-std"): 0.0,
    ("TJDR-std", "IDRiD"): -0.60,
    ("TJDR-std", "DDR"): -0.65,
    ("TJDR-std", "TJDR-uwf"): -0.70,
    ("TJDR-std", "FGADR"): -1.00,
    ("TJDR-std", "Retinal-Lesions"): -1.15,
    ("TJDR-std", "DiaRetDB1"): -0.85,
    ("FGADR", "FGADR"): 0.0,
    ("FGADR", "IDRiD"): -1.00,
    ("FGADR", "DDR"): -0.95,
    ("FGADR", "TJDR-std"): -0.90,
    ("FGADR", "TJDR-uwf"): -1.60,
    ("FGADR", "Retinal-Lesions"): -0.70,
    ("FGADR", "DiaRetDB1"): -0.60,
}
# Class-specific part of the shift: small lesions travel worst.
SHIFT_CLASS = {"MA": -0.35, "HE": 0.05, "EX": 0.15, "SE": -0.10}


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


class Sim:
    def __init__(self):
        self.rng = np.random.default_rng(RNG_SEED)
        self.rows = []
        self._model_test = {}

    def add(self, experiment, model, variant, train, seed, test, cls, metric, value, source="placeholder"):
        self.rows.append(
            dict(
                experiment=experiment,
                model=model,
                variant=variant,
                train_set=train,
                seed=seed,
                test_set=test,
                cls=cls,
                metric=metric,
                value=float(value),
                source=source,
            )
        )

    def model_test_effect(self, model, family, test):
        key = (model, test)
        if key not in self._model_test:
            bias = {"foundation": 0.12, "fundus": -0.08}.get(family, 0.0)
            self._model_test[key] = self.rng.normal(bias, 0.14)
        return self._model_test[key]

    def internal_base(self, train, test, cls):
        """Reference logit of class `cls` on `test` for a model trained on `train`."""
        shift = SHIFT[(train, test)]
        return logit(BASE[test][cls]) + shift + SHIFT_CLASS[cls] * (shift < 0)

    def aupr(self, model, family, offset, ma_handicap, train, test, cls, seed_noise):
        x = self.internal_base(train, test, cls) + offset
        if cls == "MA":
            x += ma_handicap
        if train != test:
            x += self.model_test_effect(model, family, test)
        return expit(x + seed_noise)

    def companion_metrics(self, experiment, model, variant, train, seed, test, cls, ap, calib):
        """Metrics derived from one AUPR value, with the right ordering between them."""
        r = self.rng
        dice_opt = np.clip(0.10 + 0.78 * ap + r.normal(0, 0.012), 0.0, 0.97)
        dice_fix = np.clip(dice_opt - abs(r.normal(0.035, 0.02)) - 0.05 * (train != test), 0.0, 0.97)
        self.add(experiment, model, variant, train, seed, test, cls, "dice_valopt", dice_opt)
        self.add(experiment, model, variant, train, seed, test, cls, "dice_fixed", dice_fix)
        self.add(experiment, model, variant, train, seed, test, cls, "iou_valopt", dice_opt / (2 - dice_opt))
        # The 11-threshold implementation counts every probability above 0 at
        # threshold 0, so it adds an amount that depends on calibration.
        self.add(experiment, model, variant, train, seed, test, cls, "aupr_11pt", min(ap + calib + r.normal(0, 0.003), 0.99))
        half = 0.035 if train == "DDR" or test != "IDRiD" else 0.07
        self.add(experiment, model, variant, train, seed, test, cls, "aupr_ci_lo", max(ap - half * (0.8 + 0.4 * r.random()), 0))
        self.add(experiment, model, variant, train, seed, test, cls, "aupr_ci_hi", min(ap + half * (0.8 + 0.4 * r.random()), 1))
        size_off = {"small": -1.1, "medium": 0.2, "large": 1.3}
        for size, off in size_off.items():
            rec = expit(logit(np.clip(ap, 0.02, 0.98)) + 0.5 + off + r.normal(0, 0.08))
            self.add(experiment, model, variant, train, seed, test, cls, f"lesion_recall_{size}", rec)
        self.add(experiment, model, variant, train, seed, test, cls, "lesion_precision", np.clip(0.25 + 0.6 * ap + r.normal(0, 0.03), 0, 1))
        if test == "DiaRetDB1":
            self.add(experiment, model, variant, train, seed, test, cls, "img_auc", np.clip(0.70 + 0.9 * ap + r.normal(0, 0.02), 0.5, 0.99))


def main_tables(sim):
    """E2 (B-std + nnU-Net) and the trained E5 models: 14 models x 2 sources x 3 seeds x 7 tests."""
    r = sim.rng
    for model, family, exp, off, ma in MAIN_MODELS:
        calib = abs(r.normal(0.018, 0.012))
        variant = E5_MODE.get(model, "bstd")
        for train in TRAIN_MAIN:
            sd = 0.10 if train == "IDRiD" else 0.06
            for seed in range(3):
                run_noise = r.normal(0, sd)
                for test in TESTS:
                    for cls in CLASSES:
                        extra = 0.0
                        # U-Net rare-class collapse under patch training with a
                        # per-image Dice (docs/paper/14, note 21).
                        if model == "U-Net":
                            extra = -1.6 if cls == "SE" else (-0.8 if (train == "DDR" and cls in ("MA", "HE")) else 0.0)
                        ap = sim.aupr(model, family, off + extra, ma, train, test, cls, run_noise + r.normal(0, sd / 2))
                        sim.add(exp, model, variant, train, seed, test, cls, "aupr", ap)
                        sim.companion_metrics(exp, model, variant, train, seed, test, cls, ap, calib)


def original_models(sim):
    """E1 author weights and E1r recipe retraining. M2MRF numbers are measured."""
    r = sim.rng
    m = "measured"
    # --- M2MRF, IDRiD checkpoint (iteration 40000) on the official test set.
    sim.add("E1", "M2MRF", "paper", "IDRiD", -1, "IDRiD", "mean", "aupr_author_impl", 0.6724, m)
    sim.add("E1", "M2MRF", "author_function", "IDRiD", -1, "IDRiD", "mean", "aupr_author_impl", 0.6755, m)
    for cls, v in dict(EX=0.8203, HE=0.6829, SE=0.7084, MA=0.4905).items():
        sim.add("E1", "M2MRF", "author_function", "IDRiD", -1, "IDRiD", cls, "aupr_author_impl", v, m)
    for cls, v in dict(MA=0.408, HE=0.653, EX=0.832, SE=0.664).items():
        sim.add("E1", "M2MRF", "this_protocol", "IDRiD", -1, "IDRiD", cls, "aupr", v, m)
    sim.add("E1", "M2MRF", "this_protocol", "IDRiD", -1, "IDRiD", "mean", "aupr", 0.639, m)
    sim.add("E1", "M2MRF", "this_protocol", "IDRiD", -1, "IDRiD", "mean", "aupr_11pt", 0.6446, m)
    sim.add("E1", "M2MRF", "this_protocol", "IDRiD", -1, "IDRiD", "mean", "dice_fixed", 0.661, m)
    # --- M2MRF, DDR checkpoint (DDR-C, iteration 60000).
    sim.add("E1", "M2MRF", "paper", "DDR", -1, "DDR", "mean", "aupr_author_impl", 0.4894, m)
    sim.add("E1", "M2MRF", "author_function", "DDR", -1, "DDR", "mean", "aupr_author_impl", 0.4920, m)
    for cls, v in dict(EX=0.6380, HE=0.5443, SE=0.4928, MA=0.2928).items():
        sim.add("E1", "M2MRF", "author_function", "DDR", -1, "DDR", cls, "aupr_author_impl", v, m)
    sim.add("E1", "M2MRF", "this_protocol", "DDR", -1, "DDR", "mean", "aupr", 0.4005, m)
    sim.add("E1", "M2MRF", "this_protocol", "DDR", -1, "DDR", "mean", "aupr_11pt", 0.426, m)
    # Per-class exact AUPR on DDR is not written in the progress log: placeholder
    # values rescaled to the measured mean.
    ddr = np.array([0.215, 0.455, 0.555, 0.377])
    ddr *= 0.4005 / ddr.mean()
    for cls, v in zip(CLASSES, ddr):
        sim.add("E1", "M2MRF", "this_protocol", "DDR", -1, "DDR", cls, "aupr", v)
    # --- M2MRF author weights on the remaining test sets (not run yet).
    for train in TRAIN_MAIN:
        for test in TESTS:
            if test == train:
                continue
            for cls in CLASSES:
                ap = sim.aupr("M2MRF-author", "fundus", 0.12, 0.10, train, test, cls, r.normal(0, 0.05))
                sim.add("E1", "M2MRF", "this_protocol", train, -1, test, cls, "aupr", ap)
    # --- WSRFNet: weights requested by e-mail, row reserved.
    paper = {"IDRiD": 0.69, "DDR": 0.50}
    for train in TRAIN_MAIN:
        sim.add("E1", "WSRFNet", "paper", train, -1, train, "mean", "aupr_author_impl", paper[train])
        sim.add("E1", "WSRFNet", "author_function", train, -1, train, "mean", "aupr_author_impl", paper[train] - abs(r.normal(0.01, 0.006)))
        vals = []
        for test in TESTS:
            for cls in CLASSES:
                ap = sim.aupr("WSRFNet-author", "fundus", 0.05, 0.05, train, test, cls, r.normal(0, 0.05))
                sim.add("E1", "WSRFNet", "this_protocol", train, -1, test, cls, "aupr", ap)
                if test == train:
                    vals.append(ap)
        sim.add("E1", "WSRFNet", "this_protocol", train, -1, train, "mean", "aupr", np.mean(vals))
        sim.add("E1", "WSRFNet", "this_protocol", train, -1, train, "mean", "aupr_11pt", np.mean(vals) + 0.02)

    # --- E1r. First M2MRF run (effective batch 1, workstation) is measured.
    sim.add("E1r", "M2MRF", "bs1_first_run", "IDRiD", 0, "IDRiD", "mean", "aupr_author_impl", 0.5763, m)
    sim.add("E1r", "M2MRF", "bs1_first_run", "IDRiD", 0, "IDRiD", "mean", "aupr", 0.493, m)
    for cls, v in dict(MA=0.282, HE=0.449, EX=0.746, SE=0.495).items():
        sim.add("E1r", "M2MRF", "bs1_first_run", "IDRiD", 0, "IDRiD", cls, "aupr", v, m)
    reported = {("M2MRF", "IDRiD"): (0.6724, m), ("M2MRF", "DDR"): (0.4894, m), ("HACDR-Net", "IDRiD"): (0.70, "placeholder"), ("HACDR-Net", "DDR"): (0.52, "placeholder")}
    gap = {("M2MRF", "IDRiD"): 0.045, ("M2MRF", "DDR"): 0.03, ("HACDR-Net", "IDRiD"): 0.11, ("HACDR-Net", "DDR"): 0.09}
    for (model, train), (val, src) in reported.items():
        sim.add("E1r", model, "paper", train, -1, train, "mean", "aupr_author_impl", val, src)
        for seed in range(3):
            a = val - gap[(model, train)] + r.normal(0, 0.012)
            sim.add("E1r", model, "author_recipe", train, seed, train, "mean", "aupr_author_impl", a)
            sim.add("E1r", model, "author_recipe", train, seed, train, "mean", "aupr", a - abs(r.normal(0.035, 0.008)))


def b1_and_tuning(sim):
    """E2b source recipes (seed 0) and E2c learning-rate sweep (validation only)."""
    r = sim.rng
    b1_delta = {"U-Net": 0.25, "DeepLabv3": -0.20, "HRNet-W48": -0.15, "Swin-Unet": -1.10, "FCT": -0.55, "M2MRF": -0.10, "HACDR-Net": -0.20}
    for model, family, exp, off, ma in MAIN_MODELS:
        if model not in b1_delta:
            continue
        for train in TRAIN_MAIN:
            noise = r.normal(0, 0.06)
            for cls in CLASSES:
                extra = -0.9 if (model in ("Swin-Unet", "FCT") and cls in ("MA", "EX")) else 0.0
                ap = sim.aupr(model, family, off + b1_delta[model] + extra, ma, train, train, cls, noise)
                sim.add("E2b", model, "b1", train, 0, train, cls, "aupr", ap)
    sweep = {"U-Net": (-0.10, 0.0, -0.25), "nnU-Net": (-0.04, 0.0, -0.06), "M2MRF": (-0.06, 0.0, -0.12), "SAM2-UNet": (0.03, 0.0, -0.20), "RETFound + UPerNet": (0.05, 0.0, -0.30)}
    lookup = {m[0]: m for m in MAIN_MODELS}
    for model, deltas in sweep.items():
        _, family, _, off, ma = lookup[model]
        for train in TRAIN_MAIN:
            for mult, d in zip(("1/3", "1", "3"), deltas):
                vals = [sim.aupr(model, family, off + d - (1.6 if model == "U-Net" and c == "SE" else 0), ma, train, train, c, r.normal(0, 0.03)) for c in CLASSES]
                sim.add("E2c", model, f"lr_x{mult}", train, 0, f"{train}-val", "mean", "aupr", np.mean(vals))


def cross_source(sim):
    """E3b: representative models trained on TJDR-std and FGADR, seed 0, all test sets."""
    r = sim.rng
    lookup = {m[0]: m for m in MAIN_MODELS}
    for model in REPRESENTATIVE:
        _, family, _, off, ma = lookup[model]
        for train in ("TJDR-std", "FGADR"):
            for seed in range(FACTOR_SEEDS):
                noise = r.normal(0, 0.06)
                for test in TESTS:
                    for cls in CLASSES:
                        extra = -1.6 if (model == "U-Net" and cls == "SE") else 0.0
                        ap = sim.aupr(model, family, off + extra, ma, train, test, cls, noise + r.normal(0, 0.03))
                        sim.add("E3b", model, "bstd", train, seed, test, cls, "aupr", ap)


def granularity(sim):
    """E4: six training sets at equal image passes, tested on fine-grained sets."""
    r = sim.rng
    lookup = {m[0]: m for m in MAIN_MODELS}
    configs = {
        "DDR": 0.0,
        "FGADR": -0.55,
        "Retinal-Lesions": -1.15,
        "DDR + FGADR": 0.10,
        "DDR + Retinal-Lesions": -0.22,
        "DDR + FGADR (383)": 0.02,
    }
    for model in E4_MODELS:
        _, family, _, off, ma = lookup[model]
        for cfg, d in configs.items():
            for seed in range(FACTOR_SEEDS):
                noise = r.normal(0, 0.05)
                for test in ("IDRiD", "TJDR-std", "DiaRetDB1"):
                    for cls in CLASSES:
                        extra = -1.6 if (model == "U-Net" and cls == "SE") else 0.0
                        cls_d = d * (1.5 if cls == "MA" else 0.8)
                        ap = sim.aupr(model, family, off + extra + cls_d, ma, "DDR", test, cls, noise + r.normal(0, 0.03))
                        sim.add("E4", model, cfg, cfg, seed, test, cls, "aupr", ap)


def foundation(sim):
    """E5: zero-shot, oracle prompts, wording sensitivity, encoder controls."""
    r = sim.rng
    zero = {
        ("SAM", "auto_any_lesion"): 0.035,
        ("SAM", "auto_oracle_class"): 0.11,
        ("SAM 2", "auto_any_lesion"): 0.04,
        ("SAM 2", "auto_oracle_class"): 0.13,
    }
    cls_scale = {"MA": 0.25, "HE": 1.0, "EX": 1.3, "SE": 1.5}
    for (model, variant), base in zero.items():
        for test in TESTS:
            for cls in CLASSES:
                v = np.clip(base * cls_scale[cls] * r.uniform(0.7, 1.3), 0.001, 0.9)
                sim.add("E5", model, f"zero_shot:{variant}", "none", -1, test, cls, "aupr", v)
    wording = {
        "MA": ["microaneurysm", "red dot", "small red lesion"],
        "HE": ["retinal hemorrhage", "blot hemorrhage", "dark red patch"],
        "EX": ["hard exudate", "yellow deposit", "bright lesion"],
        "SE": ["cotton wool spot", "soft exudate", "white fluffy patch"],
    }
    sam3 = {"MA": 0.03, "HE": 0.16, "EX": 0.24, "SE": 0.20}
    for test in TESTS:
        for cls in CLASSES:
            for i, w in enumerate(wording[cls]):
                v = np.clip(sam3[cls] * (1.0 - 0.28 * i) * r.uniform(0.75, 1.25) * (0.7 if test not in ("IDRiD", "DDR") else 1), 0.002, 0.9)
                sim.add("E5", "SAM 3", f"text:{w}", "none", -1, test, cls, "aupr", v)
    # Oracle prompts: performance against prompts per image and class.
    ceiling = {
        ("SAM", "box"): {"MA": 0.30, "HE": 0.62, "EX": 0.66, "SE": 0.72},
        ("SAM", "point"): {"MA": 0.18, "HE": 0.45, "EX": 0.48, "SE": 0.60},
        ("SAM 2", "box"): {"MA": 0.33, "HE": 0.66, "EX": 0.69, "SE": 0.75},
        ("SAM 2", "point"): {"MA": 0.20, "HE": 0.50, "EX": 0.52, "SE": 0.63},
        ("MedSAM", "box"): {"MA": 0.22, "HE": 0.58, "EX": 0.57, "SE": 0.70},
    }
    # IDRiD training images average about 44 MA and 142 EX components, so a few
    # prompts cover a small part of them.
    half = {"MA": 30, "HE": 14, "EX": 60, "SE": 2}
    for (model, kind), ceil in ceiling.items():
        for test in ("IDRiD", "DDR", "TJDR-std"):
            scale = {"IDRiD": 1.0, "DDR": 0.88, "TJDR-std": 0.92}[test]
            for k in (1, 5, 10, 20, "all"):
                for cls in CLASSES:
                    frac = 1.0 if k == "all" else k / (k + half[cls])
                    v = np.clip(ceil[cls] * scale * frac + r.normal(0, 0.008), 0.001, 0.95)
                    sim.add("E5", model, f"oracle:{kind}:{k}", "none", -1, test, cls, "aupr", v)
    # Encoder controls under one UPerNet head and one training amount.
    enc = {
        "RETFound-MAE": (-0.28, -0.30, 0.20),
        "RETFound-DINOv2": (-0.20, -0.28, 0.24),
        "DINOv2 ViT-L": (-0.24, -0.30, 0.12),
        "ImageNet-sup. ViT-L": (-0.42, -0.34, -0.02),
    }
    for model, (off, ma, robust) in enc.items():
        for train in TRAIN_MAIN:
            for seed in range(3):
                noise = r.normal(0, 0.06)
                for test in TESTS:
                    for cls in CLASSES:
                        x = sim.internal_base(train, test, cls) + off + (ma if cls == "MA" else 0)
                        if test != train:
                            x += robust + sim.model_test_effect(model, "encoder", test)
                        sim.add("E5", model, "encoder_control", train, seed, test, cls, "aupr", expit(x + noise + r.normal(0, 0.03)))


def factors(sim):
    """E7 resolution (M2MRF, RETFound), E8 training amount (U-Net, M2MRF), E9 loss (U-Net). Seed 0 (docs/paper/16)."""
    r = sim.rng
    lookup = {m[0]: m for m in MAIN_MODELS}
    res_levels = {"512": (-0.85, -1.4), "1440": (0.0, 0.0), "native patches": (0.06, 0.22)}
    size_levels = {"DDR 10%": -0.95, "DDR 50%": -0.24, "DDR 100%": 0.0, "fine merged": 0.16}
    # CE + Dice (batch) is the main-table recipe; the per-image Dice of the registered recipe collapses rare classes.
    loss_levels = {"CE": -0.12, "Dice": -0.05, "CE + Dice": 0.0, "CE + per-image Dice": -0.10}
    for model in E7_MODELS:
        _, family, _, off, ma = lookup[model]
        token = model in ("RETFound + UPerNet",)
        for level, (d_all, d_small) in res_levels.items():
            for seed in range(FACTOR_SEEDS):
                noise = r.normal(0, 0.05)
                aps = []
                for cls in CLASSES:
                    d = d_all * (1.6 if cls in ("MA", "EX") else 0.6) * (0.6 if token else 1.0)
                    ap = sim.aupr(model, family, off + d, ma, "DDR", "DDR", cls, noise)
                    sim.add("E7", model, f"res:{level}", "DDR", seed, "DDR", cls, "aupr", ap)
                    aps.append(ap)
                base = logit(np.clip(np.mean(aps), 0.02, 0.98))
                sim.add("E7", model, f"res:{level}", "DDR", seed, "DDR", "mean", "lesion_recall_small", expit(base - 0.3 + d_small * (0.5 if token else 1.0) + r.normal(0, 0.05)))
                sim.add("E7", model, f"res:{level}", "DDR", seed, "DDR", "mean", "lesion_recall_large", expit(base + 1.6 + 0.15 * d_small + r.normal(0, 0.05)))
    for model in E8_MODELS:
        _, family, _, off, ma = lookup[model]
        for level, d in size_levels.items():
            gain = d * (0.75 if family == "foundation" else 1.0)
            for seed in range(FACTOR_SEEDS):
                noise = r.normal(0, 0.05)
                for test in ("DDR", "TJDR-std", "FGADR"):
                    for cls in CLASSES:
                        bonus = 0.55 if (test == "TJDR-std" and level == "fine merged") else 0.0
                        ap = sim.aupr(model, family, off + gain + bonus, ma, "DDR", test, cls, noise + r.normal(0, 0.03))
                        sim.add("E8", model, f"size:{level}", "DDR", seed, test, cls, "aupr", ap)
    for model in E9_MODELS:
        _, family, _, off, ma = lookup[model]
        for level, d in loss_levels.items():
            for seed in range(FACTOR_SEEDS):
                noise = r.normal(0, 0.05)
                for cls in CLASSES:
                    extra = -2.4 if (cls in ("SE", "HE", "MA") and level == "CE + per-image Dice") else 0.0
                    cls_d = d * (1.8 if cls in ("MA", "SE") else 0.6)
                    ap = sim.aupr(model, family, off + extra + cls_d, ma, "DDR", "DDR", cls, noise)
                    sim.add("E9", model, f"loss:{level}", "DDR", seed, "DDR", cls, "aupr", ap)


def cost(sim):
    """E10. Parameter counts written in docs are measured; GPU hours follow docs/ops/00 estimates."""
    m = "measured"
    #              params(M)      FLOPs(G) mem(GB) h IDRiD        h DDR  s/img
    table = {
        "U-Net": ((7.8, m), 248, 9.8, (1.4, m), 6.0, 0.21),
        "DeepLabv3": ((60.9, "placeholder"), 610, 21.5, (4.0, m), 15.0, 0.34),
        "HRNet-W48": ((65.8, "placeholder"), 372, 18.9, (2.6, m), 10.5, 0.38),
        "Swin-Unet": ((27.2, "placeholder"), 6.2, 6.4, (9.5, "placeholder"), 38.0, 2.9),
        "FCT": ((51.4, "placeholder"), 30.5, 5.1, (24.0, m), 96.0, 3.6),
        "nnU-Net": ((33.4, "placeholder"), 180, 9.2, (10.0, "placeholder"), 14.0, 1.9),
        "nnU-Net ResEnc-M": ((57.5, "placeholder"), 260, 11.6, (14.0, "placeholder"), 20.0, 2.6),
        "M2MRF": ((70.30, m), 412, 25.3, (3.5, m), 12.5, 0.44),
        "HACDR-Net": ((49.6, "placeholder"), 285, 30.1, (6.0, m), 24.0, 0.52),
        "SAMed": ((91.2, "placeholder"), 371, 24.0, (7.0, "placeholder"), 27.0, 1.2),
        "SAM-Adapter": ((93.7, "placeholder"), 380, 26.5, (8.0, "placeholder"), 30.0, 1.3),
        "SAM2-UNet": ((216.0, "placeholder"), 432, 31.0, (9.0, "placeholder"), 34.0, 1.1),
        "MedSAM (full FT)": ((93.7, "placeholder"), 371, 36.0, (11.0, "placeholder"), 42.0, 1.2),
        "RETFound + UPerNet": ((338.0, "placeholder"), 520, 33.0, (13.0, "placeholder"), 50.0, 4.8),
    }
    for model, ((params, psrc), flops, mem, (h_idrid, hsrc), h_ddr, sec) in table.items():
        sim.add("E10", model, "", "", -1, "", "", "params_m", params, psrc)
        sim.add("E10", model, "", "", -1, "", "", "flops_g", flops)
        sim.add("E10", model, "", "", -1, "", "", "peak_mem_gb", mem)
        sim.add("E10", model, "", "IDRiD", -1, "", "", "train_gpu_h", h_idrid, hsrc)
        sim.add("E10", model, "", "DDR", -1, "", "", "train_gpu_h", h_ddr)  # extrapolated in docs/ops/00, not measured
        sim.add("E10", model, "", "", -1, "", "", "infer_s_per_image", sec)


def literature(sim):
    """L1 paper audit (placeholder counts) and reported ranges per method."""
    r = sim.rng
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    n_papers = [2, 2, 4, 6, 9, 13, 5, 4]  # 45 method papers in the ledger; per-year split is a placeholder
    for year, n in zip(years, n_papers):
        for ds in ("IDRiD", "DDR"):
            k = n if ds == "IDRiD" else int(round(n * 0.6))
            off = int(round(k * r.uniform(0.35, 0.6)))
            own = int(round((k - off) * r.uniform(0.5, 0.8)))
            for name, v in (("official", off), ("own split", own), ("not stated", k - off - own)):
                sim.add("L1", "papers", f"split:{name}", ds, year, "", "", "n_papers", v)
        impl = r.multinomial(n, [0.18, 0.34, 0.20, 0.28])
        for name, v in zip(("exact AP", "11-threshold", "per-image mean", "not stated"), impl):
            sim.add("L1", "papers", f"impl:{name}", "", year, "", "", "n_papers", v)
        src = r.multinomial(n * 6, [0.38, 0.34, 0.18, 0.10])
        for name, v in zip(("re-run by the authors", "copied, 1st generation", "copied, 2nd generation or later", "source not stated"), src):
            sim.add("L1", "papers", f"provenance:{name}", "", year, "", "", "n_baseline_numbers", v)
    methods = {"U-Net": 0.50, "DeepLabv3+": 0.56, "HED": 0.52, "HRNet": 0.60, "L-Seg": 0.65, "CARNet": 0.66, "RTNet": 0.66, "M2MRF": 0.67, "PMCNet": 0.68, "HACDR-Net": 0.70}
    for method, centre in methods.items():
        k = int(r.integers(3, 9))
        spread = r.uniform(0.02, 0.09) if method not in ("M2MRF", "HACDR-Net") else 0.012
        vals = np.clip(centre + r.normal(0, spread, k), 0.2, 0.9)
        for i, v in enumerate(vals):
            sim.add("L1", method, f"paper_{i}", "IDRiD", -1, "IDRiD", "mean", "reported_maupr", v)
    # Published rank on IDRiD (from each paper's own table) for the Kendall tau panel.
    published = {"M2MRF": 0.6724, "HACDR-Net": 0.70, "HRNet-W48": 0.60, "DeepLabv3": 0.56, "U-Net": 0.50, "Swin-Unet": 0.58, "FCT": 0.57}
    for model, v in published.items():
        src = "measured" if model == "M2MRF" else "placeholder"
        sim.add("L1", model, "published_idrid", "IDRiD", -1, "IDRiD", "mean", "reported_maupr", v, src)
    sim.add("L2", "funnel", "reproduced within 0.02 mAUPR", "", -1, "", "", "n_models", 1)
    sim.add("L2", "funnel", "authors contacted", "", -1, "", "", "n_models", 14)
    sim.add("L2", "funnel", "authors replied", "", -1, "", "", "n_models", 3)


def label_audit(sim):
    """Lesions per image and mean relative area per dataset and class.

    Counts for IDRiD (train 44) and DDR (train 383) come from
    bench/eval/lesion_size_freeze.json; every other number is a placeholder.
    """
    r = sim.rng
    measured_n = {"IDRiD": dict(MA=1931 / 44, HE=1125 / 44, EX=6269 / 44, SE=86 / 44)}
    placeholder_n = {
        "DDR": dict(MA=15.1, HE=12.4, EX=35.0, SE=1.9),
        "TJDR-std": dict(MA=18.0, HE=15.0, EX=40.0, SE=2.4),
        "TJDR-uwf": dict(MA=26.0, HE=24.0, EX=55.0, SE=3.5),
        "FGADR": dict(MA=9.0, HE=6.5, EX=11.0, SE=1.6),
        "Retinal-Lesions": dict(MA=4.2, HE=3.6, EX=3.0, SE=1.3),
        "DiaRetDB1": dict(MA=5.0, HE=4.4, EX=2.6, SE=1.2),
    }
    diam = {"IDRiD": dict(MA=0.0053, HE=0.0133, EX=0.0050, SE=0.0318)}  # medians, measured
    ph_diam = {
        "DDR": dict(MA=0.0044, HE=0.0102, EX=0.0047, SE=0.0190),
        "TJDR-std": dict(MA=0.0050, HE=0.0120, EX=0.0052, SE=0.0240),
        "TJDR-uwf": dict(MA=0.0026, HE=0.0065, EX=0.0030, SE=0.0110),
        "FGADR": dict(MA=0.0075, HE=0.0220, EX=0.0130, SE=0.0330),
        "Retinal-Lesions": dict(MA=0.0160, HE=0.0460, EX=0.0520, SE=0.0500),
        "DiaRetDB1": dict(MA=0.0300, HE=0.0700, EX=0.0900, SE=0.0800),
    }
    for ds, d in measured_n.items():
        for cls, v in d.items():
            sim.add("audit", "labels", "", ds, -1, "", cls, "lesions_per_image", v, "measured")
            sim.add("audit", "labels", "", ds, -1, "", cls, "median_rel_diameter", diam[ds][cls], "measured")
    for ds, d in placeholder_n.items():
        for cls, v in d.items():
            sim.add("audit", "labels", "", ds, -1, "", cls, "lesions_per_image", v * r.uniform(0.95, 1.05))
            sim.add("audit", "labels", "", ds, -1, "", cls, "median_rel_diameter", ph_diam[ds][cls])


def main():
    if OUT.exists() and "--force" not in sys.argv:
        old = pd.read_csv(OUT, keep_default_na=False)
        n = int(((old["source"] == "measured") & old["experiment"].isin(["E2", "E2b", "E3b", "E4", "E5", "E7", "E8", "E9"])).sum())
        raise SystemExit(f"{OUT} exists ({n} measured run rows). Rebuilding would discard them; pass --force to do it anyway.")
    sim = Sim()
    main_tables(sim)
    original_models(sim)
    b1_and_tuning(sim)
    cross_source(sim)
    granularity(sim)
    foundation(sim)
    factors(sim)
    cost(sim)
    literature(sim)
    label_audit(sim)
    df = pd.DataFrame(sim.rows)
    df["value"] = df["value"].round(5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    by = df.groupby(["experiment", "source"]).size().unstack(fill_value=0)
    print(by.to_string())
    print(f"wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
