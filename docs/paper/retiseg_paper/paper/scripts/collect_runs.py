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
    "sam2_unet": "SAM2-UNet",
    "samed": "SAMed",
    "sam_adapter": "SAM-Adapter",
    "medsam": "MedSAM (full FT)",
    "vit_upernet": "RETFound + UPerNet",
    "retfound_upernet": "RETFound + UPerNet",
}
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
}
EXPERIMENT = {"BS": ("E2", "bstd"), "B1": ("E2b", "b1")}
RUN = re.compile(r"^(BS|B1)_(?P<slug>[a-z0-9_]+?)_(?P<dataset>idrid|ddr)_seed(?P<seed>\d+)(?P<registered>_registered)?(?:_runner)?$")
NN = re.compile(r"^NN_(?P<cfg>default|resencm)_(?P<dataset>idrid|ddr)_fold0(?:_rep(?P<rep>\d+))?$")
NN_MODEL = {"default": "nnU-Net", "resencm": "nnU-Net ResEnc-M"}
E1R = re.compile(r"^E1r_(?P<slug>[a-z0-9]+)_(?P<dataset>idrid|ddr)_seed(?P<seed>\d+)_bs4(?:_[a-z0-9]+)?$")
E2C = re.compile(r"^E2c_(?P<slug>[a-z0-9_]+?)_(?P<dataset>idrid|ddr)_seed0_(?P<lr>lr1o3|lr3)$")
E2C_LEVEL = {"lr1o3": "1/3", "lr3": "3"}
E3A_SET = {"IDRiD": "IDRiD", "DDR": "DDR", "TJDR_std": "TJDR-std", "TJDR_uwf": "TJDR-uwf", "FGADR": "FGADR", "Retinal-Lesions": "Retinal-Lesions", "DiaRetDB1": "DiaRetDB1"}
E1_AUTHOR = re.compile(r"^E1_(?P<slug>[a-z0-9]+)_(?P<dataset>idrid|ddr)$")
PREPARED = {"IDRiD": "IDRiD", "DDR": "DDR", "TJDR_std": "TJDR-std", "TJDR_uwf": "TJDR-uwf", "FGADR": "FGADR", "RetinalLesions": "Retinal-Lesions", "DiaRetDB1": "DiaRetDB1"}
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
            experiment, variant = EXPERIMENT[m.group(1)]
            model, train, seed = SLUG[m["slug"]], DATASET[m["dataset"]], int(m["seed"])
            if m["registered"]:
                # runs of a recipe that was later revised (bs_frozen.yaml recipe_revisions): kept as a
                # separate variant for the supplementary table; the U-Net DDR seed 0 is also the
                # "CE + per-image Dice" row of E9.
                variant = "registered"
            env = run / "environment.json"
            if experiment == "E2" and seed == 0 and variant == "bstd" and env.exists():
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
                report = json.loads(path.read_text())
                rows += rows_from_report(report, experiment, variant, model, train, seed, test)
                if variant == "registered" and model == "U-Net" and train == "DDR" and seed == 0 and test == "DDR":
                    rows += rows_from_report(report, "E9", "loss:CE + per-image Dice", model, train, seed, test)
                seen.append((run.name, path.name, report.get("summary", {}).get("mAUPR")))
            continue
        m = NN.match(run.name)
        if m:
            model, train = NN_MODEL[m["cfg"]], DATASET[m["dataset"]]
            seed = int(m["rep"] or 0)  # nnU-Net fixes no seed; the first run is "seed 0", repeats rep1, rep2 are seeds 1, 2
            for path in sorted(run.glob("metrics_*.json")):
                suffix = path.stem[len("metrics_"):].lower()
                if suffix not in TESTSET:
                    continue
                test = TESTSET[suffix] or train
                report = json.loads(path.read_text())
                rows += rows_from_report(report, "E2", "bstd", model, train, seed, test)
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
            report = json.loads(path.read_text())
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
    E1_<model>_<train> for author weights (experiment E1, variant author_weights)."""
    rows, seen = [], []
    root = Path(runs_dir) / "E3a"
    if not root.is_dir():
        return pd.DataFrame(rows), seen
    for src in sorted(root.iterdir()):
        m = RUN.match(src.name)
        nn = NN.match(src.name)
        au = E1_AUTHOR.match(src.name)
        if m and m["slug"] in SLUG and not m["registered"]:
            experiment, variant = EXPERIMENT[m.group(1)]
            model, train, seed = SLUG[m["slug"]], DATASET[m["dataset"]], int(m["seed"])
        elif nn:
            experiment, variant, model, train, seed = "E2", "bstd", NN_MODEL[nn["cfg"]], DATASET[nn["dataset"]], int(nn["rep"] or 0)
        elif au and au["slug"] in SLUG:
            experiment, variant, model, train, seed = "E1", "author_weights", SLUG[au["slug"]], DATASET[au["dataset"]], 0
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
            report = json.loads(path.read_text())
            rows += rows_from_report(report, experiment, variant, model, train, seed, E3A_SET[setdir.name])
            seen.append((f"E3a/{src.name}", setdir.name, report.get("summary", {}).get("mAUPR")))
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
    sheets = [collect_sheet(PAPER / "data" / f) for f in ("cost_measured.tsv", "cost_audit.tsv")]
    sheet = pd.concat([x for x in sheets if len(x)], ignore_index=True) if any(len(x) for x in sheets) else pd.DataFrame()
    new = pd.concat([sheet, new], ignore_index=True) if len(new) else sheet
    if len(new):
        new = new.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)  # environment.json beats the hand sheet
    if len(sheet):
        print(f"{len(sheet)} cost rows from data/cost_measured.tsv and data/cost_audit.tsv")
    for run, name, maupr in seen:
        print(f"{run}/{name}  mAUPR {maupr if maupr is None else round(maupr, 4)}")
    table = pd.read_csv(args.csv, keep_default_na=False)
    merged, replaced, added, stale = merge(table, new)
    print(f"{len(seen)} reports, {len(new)} measured rows: {replaced} placeholder rows replaced, {added} rows with no placeholder counterpart added, {stale} placeholder seeds of measured groups dropped")
    left = merged[(merged.experiment.isin(["E2", "E2b"])) & (merged.source != "measured")]
    print(f"placeholder rows left in E2/E2b: {len(left)}")
    if not args.dry_run:
        merged.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}; now run: make -C paper")


if __name__ == "__main__":
    main()
