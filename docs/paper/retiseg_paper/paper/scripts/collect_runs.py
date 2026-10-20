"""Replace placeholder rows of data/results.csv with measured values from runs/.

Reads the files that bench.eval.cli evaluate writes (bench/eval/evaluate.py):

    runs/BS_<slug>_<dataset>_seed<k>/metrics_test.json          internal test
    runs/BS_<slug>_<dataset>_seed<k>/metrics_<testset>.json     any other test set
    runs/B1_<slug>_<dataset>_seed<k>/metrics_test.json          source recipe (E2b)
    runs/BS_<slug>_<dataset>_seed0/environment.json             E10: parameters, wall time
    runs/BS_<slug>_<dataset>_seed<k>_registered/metrics_test.json  retired recipe (variant "registered"; U-Net DDR seed 0 also E9)
    runs/NN_<default|resencm>_<dataset>_fold0/metrics_test.json    nnU-Net (E2, seed 0)
    runs/E1r_<slug>_<dataset>_seed<k>_bs4*/metrics_test.json       author recipe (E1r), plus author_*_test.json for the authors' metric
    runs/E2c_<slug>_<dataset>_seed0_<lr1o3|lr3>/history.jsonl       learning-rate check (best validation mAUPR; needs timing.json)
    runs/E3a/<source>/<set>/metrics.json + done.json               independent tests (bench/e3a.py), main variant only
    runs/BF_<cell>/metrics_test.json                               factor cells (bench/factor.py); the row key is the cell's paper block
    runs/E5/<tier>/<source>/<set>[__variant]/metrics.json + done.json   E5 inference tiers (bench/e5), main canvas variant only
    data/cost_measured.tsv                                      E10 numbers recorded by hand (docs/ops/00)
    data/cost_audit.tsv                                         E10 numbers from bench/scripts/cost_audit.py

A row is replaced only when its key (experiment, model, variant, train_set,
seed, test_set, cls, metric) matches; every other row is left as it is, and
replaced rows get source = measured. Nothing is averaged or filled in.

    python paper/scripts/collect_runs.py --runs runs            # write
    python paper/scripts/collect_runs.py --runs runs --dry-run  # only report

The reader was written against the schema in bench/eval/evaluate.py and tested
end to end on a toy dataset scored by the real bench.eval.evaluate
(paper/scripts/test_collect_runs.py). It has not yet seen a report produced
on the cluster, and the file name for independent test sets
(metrics_<testset>.json) is this script's assumption, because that inference
step is not wired in bench/ yet: adjust TESTSET when it is.
"""

import argparse
import json
import sys
import re
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]

def _find_repo(start):
    """Nearest ancestor holding bench/configs/models.yaml (the paper lives under docs/paper/)."""
    for p in [start, *start.parents]:
        if (p / "bench" / "configs" / "models.yaml").exists():
            return p
    return start.parent

REPO = _find_repo(PAPER)
CSV = PAPER / "data" / "results.csv"
ANY_LESION_METRIC = "aupr_any_lesion"
KEY = ["experiment", "model", "variant", "train_set", "seed", "test_set", "cls", "metric"]

SLUG = {  # bench/scripts/launch_b1.py::_SLUG -> names used in the paper
    "unet": "U-Net",
    "deeplabv3": "DeepLabv3",
    "hrnet": "HRNet-W48",
    "swin_unet": "Swin-Unet",
    "fct": "FCT",
    "h2former": "H2Former",
    "m2mrf": "M2MRF",
    "hacdr": "HACDR-Net",
    "hrdecoder": "HRDecoder",
    "pmcnet": "PMCNet",
    "wsrfnet": "WSRFNet",
    "clcnet": "CLC-Net",
    "tcnet": "TC-Net",
    "sam2_unet": "SAM2-UNet",
    "samed": "SAMed",
    "sam_adapter": "SAM-Adapter",
    "medsam": "MedSAM (full FT)",
    "vit_upernet": "RETFound + UPerNet",
    "retfound_upernet": "RETFound + UPerNet",
    "retfound": "RETFound + UPerNet",   # launch_b1's cell slug (bs_retfound_<lib>_seed<k>)
}
# The E5 training tiers are B-std runs (BS_ directories) that the paper keys as experiment E5 with the
# tier as the variant (build_placeholder.foundation_trained; data.MAIN_VARIANTS), not as E2 / bstd.
E5_TRAINED = {
    "SAMed": ("E5", "peft"),
    "SAM-Adapter": ("E5", "peft"),
    "SAM2-UNet": ("E5", "peft"),
    "MedSAM (full FT)": ("E5", "encoder_decoder"),
    "RETFound + UPerNet": ("E5", "encoder_decoder"),
}


def run_key(m):
    """(experiment, variant, model, train, seed) of a BS_/B1_ run directory match."""
    experiment, variant = EXPERIMENT[m.group(1)]
    model, train, seed = SLUG[m["slug"]], DATASET[m["dataset"]], int(m["seed"])
    if experiment == "E2" and model in E5_TRAINED:
        experiment, variant = E5_TRAINED[model]
    if m["registered"]:
        # runs of a recipe that was later revised (bs_frozen.yaml recipe_revisions): kept as a
        # separate variant for the supplementary table; the U-Net DDR seed 0 is also the
        # "CE + per-image Dice" row of E9.
        variant = "registered"
    return experiment, variant, model, train, seed
DATASET = {"idrid": "IDRiD", "ddr": "DDR"}
TESTSET = {  # suffix of metrics_<suffix>.json -> test set name
    "test": None,  # the training source itself
    "idrid": "IDRiD",
    "ddr": "DDR",
    "tjdr_std": "TJDR-std",
    "tjdr_uwf": "TJDR-uwf",
    "fgadr": "FGADR",
    "retinallesions": "Retinal-Lesions",
    "diaretdb1": "DiaRetDB1",
    "maples-dr": "MAPLES-DR",
}
EXPERIMENT = {"BS": ("E2", "bstd"), "B1": ("E2b", "b1")}
RUN = re.compile(r"^(BS|B1)_(?P<slug>[a-z0-9_]+?)_(?P<dataset>idrid|ddr)_seed(?P<seed>\d+)(?P<registered>_registered)?(?:_runner)?$")
NN = re.compile(r"^NN_(?P<cfg>default|resencm)_(?P<dataset>idrid|ddr|tjdr_std|fgadr)_fold0(?:_rep(?P<rep>\d+))?$")
NN_MODEL = {"default": "nnU-Net", "resencm": "nnU-Net ResEnc-M"}
NN_TRAIN = {"idrid": "IDRiD", "ddr": "DDR", "tjdr_std": "TJDR-std", "fgadr": "FGADR"}


def nn_key(match):
    """(experiment, variant, model, train_set, seed) of an NN_* run: E2 for the main-table sources, E3b for a
    run trained on TJDR_std or FGADR (the nnU-Net row of the training-source matrix, docs/paper/16)."""
    train = NN_TRAIN[match["dataset"]]
    experiment = "E2" if train in ("IDRiD", "DDR") else "E3b"
    return experiment, "bstd", NN_MODEL[match["cfg"]], train, int(match["rep"] or 0)
E1R = re.compile(r"^E1r_(?P<slug>[a-z0-9]+)_(?P<dataset>idrid|ddr)_seed(?P<seed>\d+)_bs4(?:_[a-z0-9]+)?$")
E2C = re.compile(r"^E2c_(?P<slug>[a-z0-9_]+?)_(?P<dataset>idrid|ddr)_seed0_(?P<lr>lr1o3|lr3)$")
E2C_LEVEL = {"lr1o3": "1/3", "lr3": "3"}
E3A_SET = {"IDRiD": "IDRiD", "DDR": "DDR", "TJDR_std": "TJDR-std", "TJDR_uwf": "TJDR-uwf", "FGADR": "FGADR", "Retinal-Lesions": "Retinal-Lesions", "DiaRetDB1": "DiaRetDB1", "MAPLES-DR": "MAPLES-DR"}
E1_AUTHOR = re.compile(r"^E1_(?P<slug>[a-z0-9]+)_(?P<dataset>idrid|ddr)$")
PREPARED = {"IDRiD": "IDRiD", "DDR": "DDR", "TJDR_std": "TJDR-std", "TJDR_uwf": "TJDR-uwf", "FGADR": "FGADR", "RetinalLesions": "Retinal-Lesions", "DiaRetDB1": "DiaRetDB1", "MAPLES-DR": "MAPLES-DR"}
MODEL_NAME = {"RETFound": "RETFound + UPerNet", "HRNet": "HRNet-W48", "MedSAM": "MedSAM (full FT)"}  # cell model name -> paper name


def factor_key(run):
    """(experiment, model, variant, train_set, seed, internal test set) of a BF run from its cell (bench/factor.py)."""
    cfg = run / "config.yaml"
    if not cfg.exists():
        return None
    import yaml

    config = yaml.safe_load(cfg.read_text())
    if config.get("experiment") != "BF" or "paper" not in config:
        return None
    paper = config["paper"]
    model = MODEL_NAME.get(config["model"], config["model"])
    test = PREPARED[Path(config["dataset"]).name]
    return paper["experiment"], model, paper["variant"], paper["train_set"], int(config["seed"]), test


UNALIGNED = []


def load_report(path):
    """An evaluate() report, or None (and a line in UNALIGNED) when its excluded_ids are not the list its training
    sources require (bench/data/splits/leak_exclusions.json): such a number never reaches results.csv (17 section 14)."""
    path = Path(path)
    report = json.loads(path.read_text())
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from bench.scripts.pack_handoff import expected_exclusions

        expected = expected_exclusions(path)
    except Exception as exc:  # the rule itself failing is reported, the report is kept
        print(f"warning: exclusion check failed for {path}: {exc}")
        return report
    if expected is not None and sorted(report.get("excluded_ids") or []) != expected:
        UNALIGNED.append(f"{path}: excluded {len(report.get('excluded_ids') or [])}, expected {len(expected)}")
        return None
    return report


def rows_from_report(report, experiment, variant, model, train, seed, test):
    """Long-table rows for one evaluate() report."""
    out = []

    def add(cls, metric, value):
        # A None in the report is a NaN the evaluator wrote on purpose (for example a
        # size stratum without lesions). It replaces the placeholder as NaN, so that
        # no simulated value survives next to measured ones.
        value = float("nan") if value is None else value
        out.append(dict(experiment=experiment, model=model, variant=variant, train_set=train, seed=seed, test_set=test,
                        cls=cls, metric=metric, value=float(value), source="measured"))

    for cls, entry in report["per_class"].items():
        add(cls, "aupr", entry.get("aupr"))
        add(cls, "aupr_11pt", entry.get("aupr_m2mrf_impl"))
        ci = entry.get("aupr_ci") or [None, None]
        add(cls, "aupr_ci_lo", ci[0])
        add(cls, "aupr_ci_hi", ci[1])
        th = entry.get("thresholds", {})
        if "fixed" in th:
            add(cls, "dice_fixed", th["fixed"]["pooled"].get("dice"))
        if "val_opt" in th:
            pooled = th["val_opt"]["pooled"]
            add(cls, "dice_valopt", pooled.get("dice"))
            add(cls, "iou_valopt", pooled.get("iou"))
            lesion = th["val_opt"].get("lesion") or {}
            add(cls, "lesion_precision", lesion.get("precision"))
            for size, part in (lesion.get("by_size") or {}).items():
                add(cls, f"lesion_recall_{size}", part.get("recall"))
        image = entry.get("image_level") or {}
        if image.get("max"):
            # image-level detection (DiaRetDB1 protocol): the maximum pixel score is the main score,
            # the mean of the top 100 the check (bench/eval/protocol.yaml image_level)
            add(cls, "img_auc", image["max"].get("auc"))
            if image.get("top100_mean"):
                add(cls, "img_auc_top100", image["top100_mean"].get("auc"))
    # the class-agnostic "any lesion" AUPR (protocol.yaml any_lesion): the main zero-shot metric and a
    # reference line for every model. It has a metric name of its own: every reader of metric "aupr"
    # takes those rows as the four classes (retifig.data.seed_table averages them into mAUPR).
    if report.get("any_lesion"):
        add("any", ANY_LESION_METRIC, report["any_lesion"].get("aupr"))
    return out


def rows_from_environment(env, model, train):
    """E10 rows from bench.runtime.write_environment: parameter count and wall time of the seed-0 run."""
    out = []
    if env.get("parameters"):
        out.append(dict(experiment="E10", model=model, variant="", train_set="", seed=-1, test_set="", cls="", metric="params_m", value=env["parameters"] / 1e6, source="measured"))
    if env.get("seconds"):
        out.append(dict(experiment="E10", model=model, variant="", train_set=train, seed=-1, test_set="", cls="", metric="train_gpu_h", value=env["seconds"] / 3600.0, source="measured"))
    return out


def collect_sheet(path):
    """Measured cost numbers recorded by hand in data/cost_measured.tsv (from docs/ops/00 and wrapper tests)."""
    if not Path(path).exists():
        return pd.DataFrame()
    sheet = pd.read_csv(path, sep="\t", comment="#")
    return pd.DataFrame([dict(experiment="E10", model=r.model, variant="", train_set="" if pd.isna(r.train_set) else r.train_set, seed=-1, test_set="", cls="", metric=r.metric, value=float(r.value), source="measured") for r in sheet.itertuples()])


def best_val_maupr(run):
    """Highest validation mAUPR in history.jsonl (the checkpoint-selection criterion of B-std)."""
    hist = run / "history.jsonl"
    if not hist.exists():
        return None
    vals = [json.loads(line).get("val_maupr") for line in hist.read_text().splitlines() if line.strip()]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def collect(runs_dir):
    rows, seen = [], []
    runs = sorted(Path(runs_dir).iterdir()) if Path(runs_dir).is_dir() else []
    if not runs:
        print(f"no runs directory at {runs_dir}; only the cost sheet is collected")
    for run in runs:
        m = RUN.match(run.name)
        if m and m["slug"] in SLUG:
            experiment, variant, model, train, seed = run_key(m)
            env = run / "environment.json"
            if experiment in ("E2", "E5") and seed == 0 and variant != "registered" and env.exists():
                rows += rows_from_environment(json.loads(env.read_text()), model, train)
                if train == "DDR" and model in ("HRNet-W48", "M2MRF"):
                    v = best_val_maupr(run)
                    if v is not None:
                        rows.append(dict(experiment="E2c", model=model, variant="lr_x1", train_set=train, seed=0, test_set=f"{train}-val", cls="mean", metric="aupr", value=v, source="measured"))
            for path in sorted(run.glob("metrics_*.json")):
                suffix = path.stem[len("metrics_"):].lower()
                if suffix not in TESTSET:
                    continue
                test = TESTSET[suffix] or train
                report = load_report(path)
                if report is None:
                    continue
                rows += rows_from_report(report, experiment, variant, model, train, seed, test)
                if variant == "registered" and model == "U-Net" and train == "DDR" and seed == 0 and test == "DDR":
                    rows += rows_from_report(report, "E9", "loss:CE + per-image Dice", model, train, seed, test)
                seen.append((run.name, path.name, report.get("summary", {}).get("mAUPR")))
            continue
        m = NN.match(run.name)
        if m:
            experiment, variant, model, train, seed = nn_key(m)  # nnU-Net fixes no seed; rep1, rep2 are seeds 1, 2
            for path in sorted(run.glob("metrics_*.json")):
                suffix = path.stem[len("metrics_"):].lower()
                if suffix not in TESTSET:
                    continue
                test = TESTSET[suffix] or train
                report = load_report(path)
                if report is None:
                    continue
                rows += rows_from_report(report, experiment, variant, model, train, seed, test)
                seen.append((run.name, path.name, report.get("summary", {}).get("mAUPR")))
            continue
        m = E1R.match(run.name)
        if m and m["slug"] in SLUG:
            model, train, seed = SLUG[m["slug"]], DATASET[m["dataset"]], int(m["seed"])
            test_json = run / "metrics_test.json"
            if test_json.exists():
                report = json.loads(test_json.read_text())
                rows += rows_from_report(report, "E1r", "author_recipe", model, train, seed, train)
                summ = report.get("summary", {})
                for metric, key in (("aupr", "mAUPR"), ("aupr_11pt", "mAUPR_m2mrf_impl")):
                    if summ.get(key) is not None:
                        rows.append(dict(experiment="E1r", model=model, variant="author_recipe", train_set=train, seed=seed, test_set=train, cls="mean", metric=metric, value=float(summ[key]), source="measured"))
                seen.append((run.name, test_json.name, report.get("summary", {}).get("mAUPR")))
            author = next(iter(sorted(run.glob("author_*_test.json"))), None)
            if author is not None:
                a = json.loads(author.read_text())["author_metric"]
                rows.append(dict(experiment="E1r", model=model, variant="author_recipe", train_set=train, seed=seed, test_set=train, cls="mean", metric="aupr_author_impl", value=float(a["mAUPR"]), source="measured"))
                for cls, entry in a["per_class"].items():
                    if cls in ("MA", "HE", "EX", "SE"):
                        rows.append(dict(experiment="E1r", model=model, variant="author_recipe", train_set=train, seed=seed, test_set=train, cls=cls, metric="aupr_author_impl", value=float(entry["aupr"]), source="measured"))
            continue
        if run.name.startswith("BF_"):
            key = factor_key(run)
            path = run / "metrics_test.json"
            if key is None or not path.exists():
                continue
            experiment, model, variant, train, seed, test = key
            report = load_report(path)
            if report is None:
                continue
            rows += rows_from_report(report, experiment, variant, model, train, seed, test)
            seen.append((run.name, path.name, report.get("summary", {}).get("mAUPR")))
            continue
        m = E2C.match(run.name)
        if m and m["slug"] in SLUG:
            if not (run / "timing.json").exists():
                print(f"{run.name}: training not finished (no timing.json); skipped")
                continue
            v = best_val_maupr(run)
            if v is not None:
                model, train = SLUG[m["slug"]], DATASET[m["dataset"]]
                rows.append(dict(experiment="E2c", model=model, variant=f"lr_x{E2C_LEVEL[m['lr']]}", train_set=train, seed=0, test_set=f"{train}-val", cls="mean", metric="aupr", value=v, source="measured"))
                seen.append((run.name, "history.jsonl", v))
    return pd.DataFrame(rows), seen


GROUP = ["experiment", "model", "variant", "train_set", "test_set"]


def collect_e3a(runs_dir):
    """runs/E3a/<source>/<set>[__variant]/metrics.json (bench/e3a.py). The main variant only; __fov and
    other sensitivity variants go to the supplement later. Sources are B-std/nnU-Net run names or
    E1_<model>_<train> for author weights (experiment E1, variant this_protocol, seed -1: the E1 table's key)."""
    rows, seen = [], []
    root = Path(runs_dir) / "E3a"
    if not root.is_dir():
        return pd.DataFrame(rows), seen
    for src in sorted(root.iterdir()):
        m = RUN.match(src.name)
        nn = NN.match(src.name)
        au = E1_AUTHOR.match(src.name)
        if m and m["slug"] in SLUG and not m["registered"]:
            experiment, variant, model, train, seed = run_key(m)
        elif nn:
            experiment, variant, model, train, seed = nn_key(nn)
        elif au and au["slug"] in SLUG:
            # the E1 table's key for the released weights under this protocol (build_placeholder.original_models)
            experiment, variant, model, train, seed = "E1", "this_protocol", SLUG[au["slug"]], DATASET[au["dataset"]], -1
        elif src.name.startswith("BF_") and factor_key(Path(runs_dir) / src.name) is not None:
            experiment, model, variant, train, seed, _test = factor_key(Path(runs_dir) / src.name)
        else:
            continue
        for setdir in sorted(src.iterdir()):
            if "__" in setdir.name or setdir.name not in E3A_SET or not (setdir / "done.json").exists():
                continue
            path = setdir / "metrics.json"
            if not path.exists():
                continue
            report = load_report(path)
            if report is None:
                continue
            rows += rows_from_report(report, experiment, variant, model, train, seed, E3A_SET[setdir.name])
            seen.append((f"E3a/{src.name}", setdir.name, report.get("summary", {}).get("mAUPR")))
    return pd.DataFrame(rows), seen


E1_RUN = re.compile(r"^E1_(?P<slug>[a-z0-9]+)$")   # runs/E1_<slug>: the released weights on their own test split


def collect_e1(runs_dir):
    """runs/E1_<slug>/author_<idrid|ddr>_test.json (the author's own metric, bench/predict_<slug>.py) and
    metrics_<idrid|ddr>.json (bench.eval on the same maps under this protocol) -> the E1 table's rows
    (variant author_function / this_protocol, seed -1, test set = training source). M2MRF's were typed
    into build_placeholder before this collector existed; HRDecoder's arrive through it."""
    rows, seen = [], []
    root = Path(runs_dir)
    if not root.is_dir():
        return pd.DataFrame(rows), seen
    for run in sorted(root.iterdir()):
        m = E1_RUN.match(run.name)
        if not m or m["slug"] not in SLUG:
            continue
        model = SLUG[m["slug"]]
        for dataset, train in DATASET.items():
            author = run / f"author_{dataset}_test.json"
            if author.exists():
                a = json.loads(author.read_text())["author_metric"]
                rows.append(dict(experiment="E1", model=model, variant="author_function", train_set=train, seed=-1, test_set=train, cls="mean", metric="aupr_author_impl", value=float(a["mAUPR"]), source="measured"))
                for cls, entry in a["per_class"].items():
                    if cls in ("MA", "HE", "EX", "SE"):
                        rows.append(dict(experiment="E1", model=model, variant="author_function", train_set=train, seed=-1, test_set=train, cls=cls, metric="aupr_author_impl", value=float(entry["aupr"]), source="measured"))
                seen.append((run.name, author.name, float(a["mAUPR"])))
            metrics = run / f"metrics_{dataset}.json"
            if metrics.exists():
                report = load_report(metrics)
                if report is None:
                    continue
                rows += rows_from_report(report, "E1", "this_protocol", model, train, -1, train)
                summ = report.get("summary", {})
                for metric, key in (("aupr", "mAUPR"), ("aupr_11pt", "mAUPR_m2mrf_impl"), ("dice_fixed", "mDice_pooled_fixed")):
                    if summ.get(key) is not None:
                        rows.append(dict(experiment="E1", model=model, variant="this_protocol", train_set=train, seed=-1, test_set=train, cls="mean", metric=metric, value=float(summ[key]), source="measured"))
                seen.append((run.name, metrics.name, summ.get("mAUPR")))
    return pd.DataFrame(rows), seen


# ---- E5 inference tiers (bench/e5): runs/E5/<a|b>/<source>/<set>[-val][__token]... ----------------
E5_SOURCE = re.compile(r"^(?P<tier>[ab])_(?P<slug>sam3|sam2|sam|medsam|m2mrf|nnunet)_(?P<rest>.+)$")
E5_MODEL = {"sam": "SAM", "sam2": "SAM 2", "sam3": "SAM 3", "medsam": "MedSAM", "m2mrf": "M2MRF", "nnunet": "nnU-Net"}


def e5_set(dirname):
    """(test set, pool run, variant tokens) of a directory name <set>[-val][__token]..."""
    base, *tokens = dirname.split("__")
    pool = base.endswith("-val")
    return (base[:-4] if pool else base), pool, tokens


def e5_rows(runs_dir):
    """Rows of the E5 inference tiers, main canvas variant only (TJDR_uwf __fov goes to the supplement later),
    except where the registered budget rule left no main variant: see below.

    a_<sam|sam2>_<backbone>_p<density>/<set>[__cap<f>][__oracle_class|__stability]
        model SAM / SAM 2, variant zero_shot:auto_<any_lesion|oracle_class|stability>, plus
        :p<density>:<cap<f>|nocap> for a pool run (test set <set>-val); a test run also carries
        its selected density, cap and canvas as rows (metric points_per_side, area_cap_fraction,
        canvas_long_side). When a model's largest density is selected, e5_frozen.yaml's budget rule
        skips its TJDR_uwf per-degree main variant (4256 canvas) and runs the fov variant (1440) only;
        that fov run records stands_in_for_main = true (bench/e5/amg.py evaluates the registered rule),
        and only such a directory, with no done.json in its main counterpart, stands in for the main
        result; its canvas_long_side row (1440 instead of 4256) marks the cell (review fix 8, B4, and
        review N1: a fov run of a density the rule does not cover never stands in, however the main
        variant fares). Any other __fov directory is skipped as before.
    a_sam3_ph<i>/<set>      model SAM 3, per class the variant text:<that class's phrase> (figure 5c)
    a_sam3_selected/<set>   model SAM 3, variant text:selected
    b_<sam|sam2|medsam>_<backbone>_<box|point>/<set>__k<k>   variant oracle:<box|point>:<k>
    b_<m2mrf|nnunet>_ref_<tokens>/<set>__k<k>                the same-information reference,
        variant oracle_ref:box:<k>, train_set from the tokens, seed 0
    Inference-tier rows have train_set "none" and seed -1, as the placeholder rows do.
    """
    rows, seen = [], []
    root = Path(runs_dir) / "E5"
    if not root.is_dir():
        return rows, seen
    for tier_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for src in sorted(p for p in tier_dir.iterdir() if p.is_dir()):
            m = E5_SOURCE.match(src.name)
            if not m or m["tier"] != tier_dir.name:
                continue
            tier, slug, rest = m["tier"], m["slug"], m["rest"]
            for setdir in sorted(p for p in src.iterdir() if p.is_dir()):
                if not (setdir / "done.json").exists() or not (setdir / "metrics.json").exists():
                    continue
                set_name, pool, tokens = e5_set(setdir.name)
                if set_name not in E3A_SET:
                    continue
                done = json.loads((setdir / "done.json").read_text())
                if "fov" in tokens:
                    main_tokens = [t for t in tokens if t != "fov"]
                    main_dir = src / "".join([setdir.name.split("__")[0], *(f"__{t}" for t in main_tokens)])
                    stands_in = tier == "a" and slug in ("sam", "sam2") and not pool and bool(done.get("stands_in_for_main"))
                    if not stands_in or (main_dir / "done.json").exists():
                        continue  # a canvas variant for the supplement; the main variant, where it exists, is the result
                    tokens = main_tokens  # the registered budget rule left no main variant: this fov run stands in for it
                test = E3A_SET[set_name] + ("-val" if pool else "")
                report = load_report(setdir / "metrics.json")  # the same exclusion check as every other collector
                if report is None:
                    continue
                k = next((t[1:] for t in tokens if re.fullmatch(r"k(\d+|all)", t)), None)
                if tier == "a" and slug in ("sam", "sam2"):
                    density = int(rest.rsplit("_p", 1)[1])
                    kind = next((t for t in tokens if t in ("oracle_class", "stability")), "any_lesion")
                    cap = next((t for t in tokens if t.startswith("cap")), "nocap")
                    variant = f"zero_shot:auto_{kind}" + (f":p{density}:{cap}" if pool else "")
                    new = rows_from_report(report, "E5", variant, E5_MODEL[slug], "none", -1, test)
                    if not pool:
                        canvas = done.get("canvas_long_side")
                        canvas = None if canvas in (None, "native") else canvas
                        for metric, value in (("points_per_side", density), ("area_cap_fraction", done.get("area_cap_fraction")), ("canvas_long_side", canvas)):
                            if value is not None:
                                new.append(dict(experiment="E5", model=E5_MODEL[slug], variant=variant, train_set="none", seed=-1,
                                                test_set=test, cls="", metric=metric, value=float(value), source="measured"))
                elif tier == "a" and slug == "sam3":
                    phrases = done.get("phrases") or {}
                    new = rows_from_report(report, "E5", "text:selected", "SAM 3", "none", -1, test)
                    if rest != "selected":  # a phrase run: per class its own phrase; the class-agnostic row means nothing here
                        new = [row for row in new if row["cls"] in phrases]
                        for row in new:
                            row["variant"] = f"text:{phrases[row['cls']]}"
                elif tier == "b" and "_ref_" in src.name and k is not None:
                    ref_tokens = rest.split("_")[1:]
                    train = next((DATASET[t] for t in ref_tokens if t in DATASET), "")
                    new = rows_from_report(report, "E5", f"oracle_ref:box:{k}", E5_MODEL[slug], train, 0, test)
                elif tier == "b" and slug in ("sam", "sam2", "medsam") and k is not None:
                    kind = rest.rsplit("_", 1)[1]
                    new = rows_from_report(report, "E5", f"oracle:{kind}:{k}", E5_MODEL[slug], "none", -1, test)
                else:
                    continue
                rows += new
                seen.append((f"E5/{tier}/{src.name}", setdir.name, report.get("summary", {}).get("mAUPR")))
    return rows, seen


def collect_e5(runs_dir):
    rows, seen = e5_rows(runs_dir)
    return pd.DataFrame(rows), seen


def merge(table, new):
    """Replace placeholder rows by key; then drop the remaining placeholder rows of every
    (experiment, model, variant, train_set, test_set) group that has at least one measured row,
    so that seed means and intervals are computed over measured seeds only (a model with one
    measured seed is reported with one seed, not with one measured and two simulated ones)."""
    if new.empty:
        return table, 0, 0, 0
    table = table.copy()
    table["seed"] = table["seed"].astype(int)
    idx = table.set_index(KEY).index
    new_idx = new.set_index(KEY).index
    replace = idx.isin(new_idx)
    kept = table[~replace]
    added = int((~new_idx.isin(idx)).sum())
    merged = pd.concat([kept, new], ignore_index=True)
    measured_groups = merged[merged.source == "measured"].set_index(GROUP).index.unique()
    in_group = merged.set_index(GROUP).index.isin(measured_groups)
    stale = (merged.source != "measured") & in_group
    return merged[~stale].reset_index(drop=True), int(replace.sum()), added, int(stale.sum())


# The reference row of each factor block is a main-table or E3b run, not a run of its own: E4 "DDR" and E7
# "res:1440", E8 "size:DDR 100%", E9 "loss:CE + Dice" are the model's B-std DDR run (seed 0), E4 "FGADR" is its
# E3b run trained on FGADR. build_placeholder writes them as placeholders; nothing under runs/ carries their key,
# so they stayed simulated after every collect and every delta of table 7 / figures 6e, 7d, 7e was taken against
# a simulated reference (noticed 2026-10-20 when RETFound's E7 512 row read "+0.09"). Copy them from the
# measured source rows (same model, test set, class and metric; seed 0; the main variants of E2/E5, or E3b).
REFERENCE_SOURCES = {  # (experiment, variant) -> (source experiments, source variants, source train_set)
    ("E4", "DDR"): (["E2", "E5"], ["bstd", "peft", "encoder_decoder"], "DDR"),
    ("E7", "res:1440"): (["E2", "E5"], ["bstd", "peft", "encoder_decoder"], "DDR"),
    ("E8", "size:DDR 100%"): (["E2", "E5"], ["bstd", "peft", "encoder_decoder"], "DDR"),
    ("E9", "loss:CE + Dice"): (["E2", "E5"], ["bstd", "peft", "encoder_decoder"], "DDR"),
    ("E4", "FGADR"): (["E3b"], ["bstd"], "FGADR"),
}


def derive_references(table):
    """Replace the factor blocks' placeholder reference rows by copies of the measured rows they stand for.

    For every (experiment, variant, model, test set) of REFERENCE_SOURCES that still holds placeholder rows, the
    placeholder rows are dropped and every measured seed-0 row of the source run on that test set (all classes
    and metrics, so the reference carries the same per-class lesion-level rows as the other levels of the block)
    is inserted under the factor's key. Returns (table, n_rows_inserted).
    """
    table = table.copy()
    table["seed"] = table["seed"].astype(int)
    inserted = []
    drop = []
    for (exp, variant), (src_exps, src_variants, src_train) in REFERENCE_SOURCES.items():
        targets = table[(table.experiment == exp) & (table.variant == variant) & (table.source != "measured")]
        if targets.empty:
            continue
        src = table[table.experiment.isin(src_exps) & table.variant.isin(src_variants) & (table.train_set == src_train)
                    & (table.seed == 0) & (table.source == "measured")]
        for (model, test), group in targets.groupby(["model", "test_set"]):
            rows = src[(src.model == model) & (src.test_set == test)].drop_duplicates(subset=["cls", "metric"], keep="first")
            if rows.empty:
                continue
            rows = rows.assign(experiment=exp, variant=variant, train_set=group.train_set.iloc[0], seed=0, source="measured")
            inserted.append(rows)
            drop.extend(group.index.tolist())
    if not inserted:
        return table, 0
    table = pd.concat([table.drop(index=drop), *inserted], ignore_index=True)
    return table, int(sum(len(r) for r in inserted))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    new, seen = collect(args.runs)
    e3a, seen_e3a = collect_e3a(args.runs)
    if len(e3a):
        new = pd.concat([new, e3a], ignore_index=True)
        seen += seen_e3a
    e5, seen_e5 = collect_e5(args.runs)
    if len(e5):
        new = pd.concat([new, e5], ignore_index=True)
        seen += seen_e5
    e1, seen_e1 = collect_e1(args.runs)
    if len(e1):
        new = pd.concat([new, e1], ignore_index=True)
        seen += seen_e1
    sheets = [collect_sheet(PAPER / "data" / f) for f in ("cost_measured.tsv", "cost_audit.tsv")]
    sheet = pd.concat([x for x in sheets if len(x)], ignore_index=True) if any(len(x) for x in sheets) else pd.DataFrame()
    new = pd.concat([sheet, new], ignore_index=True) if len(new) else sheet
    if len(new):
        new = new.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)  # environment.json beats the hand sheet
    if len(sheet):
        print(f"{len(sheet)} cost rows from data/cost_measured.tsv and data/cost_audit.tsv")
    for run, name, maupr in seen:
        print(f"{run}/{name}  mAUPR {maupr if maupr is None else round(maupr, 4)}")
    for line in UNALIGNED:
        print(f"left out (exclusions not aligned): {line}")
    table = pd.read_csv(args.csv, keep_default_na=False)
    merged, replaced, added, stale = merge(table, new)
    print(f"{len(seen)} reports, {len(new)} measured rows: {replaced} placeholder rows replaced, {added} rows with no placeholder counterpart added, {stale} placeholder seeds of measured groups dropped")
    merged, filled = derive_references(merged)
    print(f"{filled} factor reference rows copied from their main-table / E3b runs (seed 0, all classes and metrics)")
    left = merged[(merged.experiment.isin(["E2", "E2b"])) & (merged.source != "measured")]
    print(f"placeholder rows left in E2/E2b: {len(left)}")
    if not args.dry_run:
        merged.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}; now run: make -C paper")


if __name__ == "__main__":
    main()
