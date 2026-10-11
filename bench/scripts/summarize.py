"""Collect internal and independent-test metrics into one table, and rank the models.

Reads only small files, so it runs on the workstation after the json files are
copied back (11 section 6, item 5): runs/<BS_*|B1_*|NN_*>/metrics_test.json,
the E1 author-weight metrics, and runs/E3a/<source>/<set>/{done,metrics}.json.
Never reads maps or labels.

    python bench/scripts/summarize.py [--runs runs] [--out runs/summary] [--seed0-only]

Writes <out>/results.tsv (one row per source and test set), <out>/table_<train>.tsv
(internal against independent, as in the draft's Table 5) and <out>/ranking.json
(Friedman test and Nemenyi critical difference over the seven test sets, the
fine-only version and the version without Retinal-Lesions). In an independent
block a model's score is the mean over its IDRiD- and DDR-trained checkpoints;
the internal blocks are the IDRiD-trained models on IDRiD and the DDR-trained
models on DDR; DiaRetDB1 uses the image-level ROC AUC (mean over the four
classes). Seeds are averaged unless --seed0-only.
"""

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml

CLASSES = ("MA", "HE", "EX", "SE")
TEST_ORDER = ("IDRiD", "DDR", "TJDR_std", "TJDR_uwf", "FGADR", "Retinal-Lesions", "DiaRetDB1")
TIERS = {
    "IDRiD": "fine", "DDR": "fine", "TJDR_std": "fine", "TJDR_uwf": "fine",
    "FGADR": "mixed", "Retinal-Lesions": "coarse", "DiaRetDB1": "region",
}
FAMILY = {
    "U-Net": "general", "DeepLabv3": "general", "HRNet": "general", "Swin-Unet": "general",
    "FCT": "general", "H2Former": "general",
    "nnU-Net": "self-configuring", "nnU-Net ResEnc-M": "self-configuring",
    "M2MRF": "fundus-specific", "HACDR-Net": "fundus-specific",
}
# Independent sets whose images a model's pretraining has seen (design/15 section 10, 14 note 9;
# FGADR's _3 batch is Kaggle EyePACS, leak audit 2026-10-12): left out of that model's blocks in
# the "pretraining_overlap_removed" ranking, which then needs complete blocks from the rest.
SEEN_IN_PRETRAINING = {"RETFound": {"Retinal-Lesions", "FGADR"}}


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _row(metrics, **fields):
    summary = metrics["summary"]
    row = dict(fields)
    row["mAUPR"] = summary.get("mAUPR")
    row["mAUPR_11thr"] = summary.get("mAUPR_m2mrf_impl")
    row["AUPR_any"] = summary.get("AUPR_any_lesion")
    row["mDice_val_opt"] = summary.get("mDice_pooled_val_opt")
    row["mDice_fixed"] = summary.get("mDice_pooled_fixed")
    row["mAUC_image"] = summary.get("mAUC_image_max")
    row["n_images"] = metrics.get("n_images")
    for cls in CLASSES:
        entry = metrics.get("per_class", {}).get(cls) or {}
        row[f"AUPR_{cls}"] = entry.get("aupr")
        ci = entry.get("aupr_ci") or [None, None]
        row[f"AUPR_{cls}_lo"], row[f"AUPR_{cls}_hi"] = ci[0], ci[1]
    return row


def internal_rows(runs):
    rows = []
    for prefix in ("BS_", "B1_"):
        for run in sorted(runs.glob(f"{prefix}*")):
            metrics = _json(run / "metrics_test.json")
            if metrics is None or not (run / "config.yaml").is_file():
                continue
            config = yaml.safe_load((run / "config.yaml").read_text())
            if config.get("diagnostic_loss_params") or config.get("budget_probe") or run.name == "B1_unet_idrid_seed0":
                continue
            train = Path(config["dataset"]).name
            model = config["model"] + (" (registered recipe)" if run.name.endswith("_registered") else "")
            rows.append(_row(metrics, source=run.name, kind=config["experiment"], model=model,
                             seed=int(config["seed"]), train=train, test=train, variant="main", role="internal"))
    for run in sorted(runs.glob("NN_*_fold0")):
        metrics = _json(run / "metrics_test.json")
        if metrics is None:
            continue
        _nn, plan, train_lower, _fold = run.name.split("_")
        train = "IDRiD" if train_lower == "idrid" else "DDR"
        model = "nnU-Net" if plan == "default" else "nnU-Net ResEnc-M"
        rows.append(_row(metrics, source=run.name, kind="nnU-Net", model=model, seed=0, train=train,
                         test=train, variant="main", role="internal"))
    for train, name in (("IDRiD", "idrid"), ("DDR", "ddr")):
        metrics = _json(runs / "E1_m2mrf" / "metrics" / f"{name}_test_protocol_luma.json")
        if metrics is not None:
            rows.append(_row(metrics, source=f"E1_m2mrf_{name}", kind="E1", model="M2MRF", seed=0, train=train,
                             test=train, variant="main", role="internal"))
    return rows


def e3a_rows(runs):
    rows = []
    base = runs / "E3a"
    if not base.is_dir():
        return rows
    for source in sorted(p for p in base.iterdir() if p.is_dir()):
        for directory in sorted(p for p in source.iterdir() if p.is_dir()):
            done = _json(directory / "done.json")
            metrics = _json(directory / "metrics.json")
            if done is None or metrics is None:
                continue
            model = done["model"]
            if model == "nnU-Net default":
                model = "nnU-Net"
            elif model == "nnU-Net resencm":
                model = "nnU-Net ResEnc-M"
            kind = done.get("kind")
            role = "cross" if done["test_set"] in ("IDRiD", "DDR") else "independent"
            rows.append(_row(metrics, source=source.name, kind=kind, model=model, seed=int(done.get("seed", 0)),
                             train=done["train_source"], test=done["test_set"], variant=done["variant"], role=role))
    return rows


def label(row):
    if row["kind"] == "E1":
        return f"{row['model']} (author weights)"
    if row["kind"] == "B1":
        return f"{row['model']} (B1)"
    return row["model"]


def aggregate(rows, seed0_only=False):
    """{(label, train, test, variant): mean over seeds of the metrics}, with the seeds used."""
    groups = {}
    for row in rows:
        if seed0_only and row["seed"] != 0:
            continue
        key = (label(row), row["train"], row["test"], row["variant"])
        groups.setdefault(key, []).append(row)
    out = {}
    for key, members in groups.items():
        merged = {"seeds": sorted(r["seed"] for r in members), "kind": members[0]["kind"], "model": members[0]["model"]}
        for metric in ("mAUPR", "mAUPR_11thr", "AUPR_any", "mDice_val_opt", "mAUC_image") + tuple(f"AUPR_{c}" for c in CLASSES):
            values = [r[metric] for r in members if r.get(metric) is not None]
            merged[metric] = float(np.mean(values)) if values else None
            merged[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        out[key] = merged
    return out


def block_scores(agg, labels, blocks, clean=False):
    """(n_models, n_blocks) scores; NaN where a model lacks a block."""
    table = np.full((len(labels), len(blocks)), np.nan)
    for i, name in enumerate(labels):
        model = name.split(" (")[0]
        for j, block in enumerate(blocks):
            if clean and block in SEEN_IN_PRETRAINING.get(model, ()):
                continue
            if block in ("IDRiD", "DDR"):
                entry = agg.get((name, block, block, "main"))
                table[i, j] = np.nan if entry is None or entry["mAUPR"] is None else entry["mAUPR"]
                continue
            metric = "mAUC_image" if block == "DiaRetDB1" else "mAUPR"
            values = [
                agg[(name, train, block, "main")][metric]
                for train in ("IDRiD", "DDR")
                if (name, train, block, "main") in agg and agg[(name, train, block, "main")][metric] is not None
            ]
            table[i, j] = np.mean(values) if len(values) == 2 else np.nan
    return table


def ranking(agg, blocks, clean=False):
    from bench.eval import stats

    side_rows = ("(B1)", "(author weights)", "(registered recipe)")
    labels = sorted({key[0] for key in agg if not key[0].endswith(side_rows)})
    table = block_scores(agg, labels, blocks, clean=clean)
    complete = [i for i in range(len(labels)) if not np.isnan(table[i]).any()]
    if len(complete) < 3:
        return {"blocks": list(blocks), "models": [labels[i] for i in complete], "note": "fewer than 3 complete models"}
    result = stats.friedman_nemenyi(table[complete])
    order = np.argsort(result["avg_rank"])
    return {
        "blocks": list(blocks),
        "n_models": len(complete),
        "friedman_chi2": result["statistic"],
        "friedman_p": result["p"],
        "critical_difference": result["critical_difference"],
        "average_rank": {labels[complete[k]]: result["avg_rank"][k] for k in order},
        "incomplete": [labels[i] for i in range(len(labels)) if i not in complete],
    }


def train_table(agg, train):
    """Rows: models trained on ``train``; columns: internal, cross, independent sets, fine mean and its gap."""
    other = "DDR" if train == "IDRiD" else "IDRiD"
    columns = ["Int", "Cross", "TJDR_std", "TJDR_uwf", "TJDR_uwf__fov", "FGADR", "Retinal-Lesions", "DiaRetDB1_AUC",
               "fine_indep_mean", "delta_fine_minus_int", "any_lesion_int"]
    lines = ["model\t" + "\t".join(columns)]
    for name in sorted({key[0] for key in agg if key[1] == train}):
        def get(test, variant="main", metric="mAUPR"):
            entry = agg.get((name, train, test, variant))
            return None if entry is None else entry.get(metric)
        internal = get(train)
        fine = [v for v in (get("TJDR_std"), get("TJDR_uwf")) if v is not None]
        fine_mean = float(np.mean(fine)) if len(fine) == 2 else None
        values = [
            internal, get(other), get("TJDR_std"), get("TJDR_uwf"), get("TJDR_uwf", "fov"), get("FGADR"),
            get("Retinal-Lesions"), get("DiaRetDB1", metric="mAUC_image"), fine_mean,
            None if fine_mean is None or internal is None else fine_mean - internal, get(train, metric="AUPR_any"),
        ]
        lines.append(name + "\t" + "\t".join("" if v is None else f"{v:.4f}" for v in values))
    return "\n".join(lines) + "\n"


def implementation_rank_change(agg, train="IDRiD"):
    """Kendall tau between ranks by exact mAUPR and by the 11-threshold implementation (internal test)."""
    from bench.eval import stats

    pairs = [
        (key[0], entry["mAUPR"], entry["mAUPR_11thr"])
        for key, entry in agg.items()
        if key[1] == train and key[2] == train and entry["mAUPR"] is not None and entry["mAUPR_11thr"] is not None
        and entry["kind"] in ("BS", "nnU-Net")
    ]
    if len(pairs) < 3:
        return None
    exact = [p[1] for p in pairs]
    eleven = [p[2] for p in pairs]
    tau = stats.kendall_tau(exact, eleven)
    rank_exact = np.argsort(np.argsort(-np.array(exact)))
    rank_eleven = np.argsort(np.argsort(-np.array(eleven)))
    return {
        "train": train,
        "n_models": len(pairs),
        "kendall_tau": tau,
        "models_changing_rank": int((rank_exact != rank_eleven).sum()),
        "mean_lift": float(np.mean(np.array(eleven) - np.array(exact))),
        "max_lift": float(np.max(np.array(eleven) - np.array(exact))),
    }


def _best_val(history):
    rows = []
    if history.is_file():
        for line in history.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    values = [row["val_maupr"] for row in rows if row.get("val_maupr") is not None]
    return max(values) if values else None


def e2c_table(runs):
    """Best validation mAUPR at learning rate x 1/3, 1, 3 (E2c; x 1 is the B-std seed-0 run)."""
    lines = ["model\tdataset\tx1/3\tx1\tx3\tbest"]
    for run in sorted(runs.glob("E2c_*_seed0_lr1o3")):
        stem = run.name[len("E2c_") : -len("_seed0_lr1o3")]
        slug, dataset = stem.rsplit("_", 1)
        values = {
            "x1/3": _best_val(run / "history.jsonl"),
            "x1": _best_val(runs / f"BS_{slug}_{dataset}_seed0" / "history.jsonl"),
            "x3": _best_val(runs / f"E2c_{slug}_{dataset}_seed0_lr3" / "history.jsonl"),
        }
        known = {k: v for k, v in values.items() if v is not None}
        best = max(known, key=known.get) if known else ""
        lines.append(
            f"{slug}\t{dataset}\t" + "\t".join("" if values[k] is None else f"{values[k]:.4f}" for k in ("x1/3", "x1", "x3"))
            + f"\t{best}"
        )
    return "\n".join(lines) + "\n"


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(_ROOT / "runs"))
    parser.add_argument("--out", default=str(_ROOT / "runs" / "summary"))
    parser.add_argument("--seed0-only", action="store_true")
    args = parser.parse_args(argv)
    runs = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = internal_rows(runs) + e3a_rows(runs)
    columns = ["source", "kind", "model", "seed", "train", "test", "variant", "role", "n_images", "mAUPR", "mAUPR_11thr",
               "AUPR_any", "mDice_val_opt", "mDice_fixed", "mAUC_image"]
    columns += [f"AUPR_{c}{s}" for c in CLASSES for s in ("", "_lo", "_hi")]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join("" if row.get(c) is None else (f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])) for c in columns))
    (out / "results.tsv").write_text("\n".join(lines) + "\n")
    agg = aggregate(rows, seed0_only=args.seed0_only)
    for train in ("IDRiD", "DDR"):
        (out / f"table_{train}.tsv").write_text(train_table(agg, train))
    report = {
        "seed0_only": bool(args.seed0_only),
        "n_rows": len(rows),
        "all_seven": ranking(agg, TEST_ORDER),
        "fine_only": ranking(agg, tuple(t for t in TEST_ORDER if TIERS[t] == "fine")),
        "without_retinal_lesions": ranking(agg, tuple(t for t in TEST_ORDER if t != "Retinal-Lesions")),
        "pretraining_overlap_removed": ranking(agg, TEST_ORDER, clean=True),
        "implementation_rank_change": {t: implementation_rank_change(agg, t) for t in ("IDRiD", "DDR")},
    }
    (out / "ranking.json").write_text(json.dumps(_clean(report), indent=1) + "\n")
    (out / "e2c.tsv").write_text(e2c_table(runs))
    print(f"{len(rows)} rows -> {out}/results.tsv", flush=True)
    for train in ("IDRiD", "DDR"):
        print(f"\n== trained on {train}\n" + (out / f"table_{train}.tsv").read_text(), flush=True)
    print(json.dumps(_clean({k: v for k, v in report.items() if k in ("all_seven", "implementation_rank_change")}), indent=1))


if __name__ == "__main__":
    main()
