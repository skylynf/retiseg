"""Read data/results.csv and the frozen audit files; shared aggregations.

Every accessor returns values together with a flag saying whether any row it
used is a placeholder, so a panel can tag itself.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PAPER = Path(__file__).resolve().parents[2]


def _find_repo(start):
    """Nearest ancestor holding bench/configs/models.yaml (the paper may live under docs/paper/)."""
    for p in [start, *start.parents]:
        if (p / "bench" / "configs" / "models.yaml").exists():
            return p
    return start.parent


REPO = _find_repo(PAPER)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.eval import stats as bstats  # noqa: E402  (the benchmark's own statistics)

CLASSES = ["MA", "HE", "EX", "SE"]
TRAIN_MAIN = ["IDRiD", "DDR"]
TESTS = ["IDRiD", "DDR", "TJDR-std", "TJDR-uwf", "FGADR", "Retinal-Lesions", "DiaRetDB1"]
INDEPENDENT = ["TJDR-std", "TJDR-uwf", "FGADR", "Retinal-Lesions", "DiaRetDB1"]
FINE_INDEPENDENT = ["TJDR-std", "TJDR-uwf"]

FAMILY_OF = {
    "U-Net": "general",
    "DeepLabv3": "general",
    "HRNet-W48": "general",
    "Swin-Unet": "general",
    "FCT": "general",
    "nnU-Net": "auto",
    "nnU-Net ResEnc-M": "auto",
    "M2MRF": "fundus",
    "HACDR-Net": "fundus",
    "HRDecoder": "fundus",
    "PMCNet": "fundus",
    "WSRFNet": "fundus",
    "CLC-Net": "fundus",
    "TC-Net": "fundus",
    "SAMed": "foundation",
    "SAM-Adapter": "foundation",
    "SAM2-UNet": "foundation",
    "MedSAM (full FT)": "foundation",
    "RETFound + UPerNet": "foundation",
}
MAIN_MODELS = list(FAMILY_OF)
TRACK_C = ["CLC-Net", "TC-Net"]   # re-implemented from the paper (docs/paper/18); marked in the tables
SAME_BACKBONE = ["HRNet-W48", "M2MRF", "HRDecoder"]   # HRNet-W48 backbone: the backbone-controlled comparison
MAIN_VARIANTS = ["bstd", "peft", "encoder_decoder"]
REPRESENTATIVE = ["U-Net", "nnU-Net", "M2MRF", "SAM2-UNet", "RETFound + UPerNet"]


@lru_cache(maxsize=1)
def results():
    df = pd.read_csv(PAPER / "data" / "results.csv", keep_default_na=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # NaN written by the evaluator (empty size stratum) reads back as NaN
    return df


def sel(**kw):
    """Rows of the long table matching every keyword; a list value means isin. The any-lesion rows
    (cls "any", written by collect_runs from the protocol's merged AUPR) are left out unless cls is asked for,
    so that class means stay means over the four classes."""
    df = results()
    mask = np.ones(len(df), bool)
    if "cls" not in kw:
        mask &= (df["cls"] != "any").to_numpy()
    for key, val in kw.items():
        if isinstance(val, (list, tuple, set)):
            mask &= df[key].isin(list(val)).to_numpy()
        else:
            mask &= (df[key] == val).to_numpy()
    return df[mask]


def simulated(df):
    return bool((df["source"] != "measured").any())


def main_runs(metric="aupr"):
    """Unified-retraining rows of the 14 main-table models."""
    return sel(experiment=["E2", "E5"], variant=MAIN_VARIANTS, model=MAIN_MODELS, metric=metric)


def seed_table(metric="aupr", cls="mean"):
    """One value per (model, train, seed, test). cls='mean' averages the four classes."""
    df = main_runs(metric)
    if cls == "mean":
        out = df.groupby(["model", "train_set", "seed", "test_set"], sort=False)["value"].mean()
    else:
        out = df[df["cls"] == cls].set_index(["model", "train_set", "seed", "test_set"])["value"]
    return out.reset_index()


def mean_sd(metric="aupr", cls="mean"):
    """Seed mean and s.d. per (model, train, test)."""
    t = seed_table(metric, cls)
    g = t.groupby(["model", "train_set", "test_set"], sort=False)["value"]
    return g.mean().rename("mean").to_frame().join(g.std(ddof=1).rename("sd")).reset_index()


def internal(metric="aupr"):
    """model x class table per training source, on its own official test set."""
    df = main_runs(metric)
    df = df[df["train_set"] == df["test_set"]]
    return df.groupby(["model", "train_set", "cls"], sort=False)["value"].mean().unstack("cls")[CLASSES]


def score_matrix(train, metric="aupr", models=None):
    """model x test-set seed-mean score for one training source."""
    t = mean_sd(metric)
    t = t[t["train_set"] == train].pivot(index="model", columns="test_set", values="mean")
    return t.loc[models or MAIN_MODELS, TESTS]


def block_scores(models=None):
    """model x 7 test-set blocks for the Friedman test and the critical-difference diagram.

    Internal blocks use the checkpoint trained on that source. Independent
    blocks average the IDRiD-trained and DDR-trained checkpoints. DiaRetDB1
    uses the official image-level ROC AUC (ranks are computed within a block,
    so blocks need not share a metric). See STRUCTURE.md, section 5.
    """
    models = models or MAIN_MODELS
    a, b = score_matrix("IDRiD", models=models), score_matrix("DDR", models=models)
    out = pd.DataFrame(index=models, columns=TESTS, dtype=float)
    out["IDRiD"], out["DDR"] = a["IDRiD"], b["DDR"]
    for test in INDEPENDENT:
        out[test] = (a[test] + b[test]) / 2
    auc = main_runs("img_auc")
    if len(auc):
        out["DiaRetDB1"] = auc.groupby("model")["value"].mean().reindex(models)
    return out


def internal_vs_independent(train, models=None, tests=None):
    """Seed-mean mAUPR on the internal test and averaged over independent tests."""
    m = score_matrix(train, models=models)
    tests = tests or INDEPENDENT
    return m[train], m[tests].mean(axis=1)


def friedman(scores):
    """scores: DataFrame model x block. Returns the bench.eval.stats dictionary."""
    return bstats.friedman_nemenyi(scores.to_numpy(dtype=float))


@lru_cache(maxsize=1)
def registry():
    """bench/configs/models.yaml flattened to one row per variant."""
    doc = yaml.safe_load((REPO / "bench" / "configs" / "models.yaml").read_text())
    rows = []
    for family, entries in doc.items():
        for e in entries:
            rows.append(
                dict(
                    family=family,
                    name=e["name"],
                    repro_level=e["repro_level"],
                    tier=e["tier"],
                    tracks="".join(e.get("tracks", [])),
                    modes=",".join(e.get("modes", [])),
                    repo=e.get("repo") or "",
                    notes=e.get("notes", ""),
                )
            )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def lesion_size_freeze():
    return json.loads((REPO / "bench" / "eval" / "lesion_size_freeze.json").read_text())


@lru_cache(maxsize=1)
def input_label_audit():
    return json.loads((REPO / "bench" / "eval" / "input_label_audit.json").read_text())


# Frozen facts used by Fig. 1 and Table 1 (docs/paper/14, M1 and M3).
# Field of view of IDRiD (50°) and DDR (45°) is from the dataset papers, not from
# docs/; check both against the PDFs before submission. n.s. = not stated.
DATASETS = [
    # name, year, images, train, val, test, role, tier, resolution, field of view, annotation, public maps
    dict(name="IDRiD", ref="Porwal 2018", n=81, train=44, val=10, test=27, role="train", tier="fine", res="4288×2848", fov="50°", ann="pixel contours, expert review", public=True),
    dict(name="DDR", ref="Li 2019", n=757, train=383, val=149, test=225, role="train", tier="fine", res="variable", fov="45°", ann="pixel masks", public=True),
    dict(name="TJDR-std", ref="Mao 2023", n=257, train=172, val=30, test=55, role="independent", tier="fine", res="2048×2048", fov="35–50°", ann="polygons, 2 graders + adjudicator", public=True),
    dict(name="TJDR-uwf", ref="Mao 2023", n=304, train=209, val=37, test=58, role="independent", tier="fine", res="3912×3912", fov="133°", ann="polygons, 2 graders + adjudicator", public=True),
    dict(name="FGADR", ref="Zhou 2021", n=1842, train=1096, val=193, test=553, role="independent", tier="mixed", res="1280×1280", fov="n.s.", ann="pixel masks, 2 residents + senior", public=False),
    dict(name="Retinal-Lesions", ref="Wei 2020", n=1593, train=951, val=164, test=478, role="independent", tier="coarse", res="896×896", fov="n.s.", ann="ellipses and polygons, ≥3 of 45 graders", public=False),
    dict(name="DiaRetDB1", ref="Kauppi 2007", n=89, train=0, val=28, test=61, role="independent", tier="region", res="1500×1152", fov="50°", ann="regions with confidence, 4 experts", public=None),
]

# literature/06 and 08: fixed search of 2026-10-03, frozen at 2026-09-30.
SEARCH_FLOW = [
    ("Records identified", 1389, "PubMed 240 · Europe PMC 250 · OpenAlex 899"),
    ("Passed the Boolean gate", 948, "three concepts in title or abstract"),
    ("After de-duplication", 764, "by normalized title"),
    ("Full-text queue", 112, "title names lesion segmentation"),
    ("Priority list screened", 29, "27 DOIs · TJDR lead · 1 corrected DOI"),
    ("Newly included", 26, "24 methods · 2 datasets"),
]
