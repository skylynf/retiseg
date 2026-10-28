"""95% bootstrap interval of mAUPR (the mean of the four class AUPRs) for every scored report (27 section 6
item 2; 2026-10-28), from the per-image histograms bench.eval stores beside each metrics.json.

The per-class intervals of the protocol resample test images with replacement (2,000 replicates, seed 0,
percentile interval; bench/eval/stats.py). The same replicates, drawn once per report, give the four class
AUPRs of each replicate; their mean is the replicate's mAUPR. A replicate in which a class has no positive
pixel is dropped, as the per-class rule drops it, so the interval is over the replicates that have every class.
The per-class intervals are recomputed on the way and must match the stored ones, which checks that the
histograms and the bootstrap parameters are the ones the report was written with.

The result goes into the report's metrics.json as summary.mAUPR_ci = [low, high] and
summary.mAUPR_ci_n_replicates; docs/paper/retiseg_paper/paper/scripts/collect_runs.py reads the first as the
rows (cls mean, aupr_ci_lo / aupr_ci_hi). A report already carrying mAUPR_ci is skipped unless --force.

    python bench/scripts/maupr_ci.py [--runs runs] [--only SUBSTRING] [--force] [--dry-run]

CPU only: a few seconds per report; about an hour for every report of runs/.
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from bench.eval import stats

CLASSES = ("MA", "HE", "EX", "SE")


def reports(runs_root):
    """(metrics.json, metrics.npz) pairs below runs/, every report that has its histograms."""
    out = []
    for path in sorted(Path(runs_root).rglob("metrics*.json")):
        npz = path.with_suffix(".npz")
        if npz.is_file() and _has_per_class(path):
            out.append((path, npz))
    return out


def _has_per_class(path):
    try:
        return "per_class" in json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def maupr_interval(arrays, boot, classes=CLASSES):
    """(low, high, n_replicates, per-class intervals) from <cls>_pos / <cls>_neg histograms."""
    pos = {cls: np.asarray(arrays[f"{cls}_pos"]) for cls in classes}
    neg = {cls: np.asarray(arrays[f"{cls}_neg"]) for cls in classes}
    n_images = pos[classes[0]].shape[0]
    w = stats._bootstrap_weights(n_images, int(boot["n"]), int(boot["seed"]))
    ap = {}
    per_class = {}
    a = (1 - float(boot["ci"])) / 2
    for cls in classes:
        if pos[cls].sum() == 0:
            ap[cls] = np.full(w.shape[0], np.nan)
            per_class[cls] = (float("nan"), float("nan"))
            continue
        values = stats._boot_ap(w, pos[cls], neg[cls])
        ap[cls] = values
        kept = values[~np.isnan(values)]
        per_class[cls] = (float(np.quantile(kept, a)), float(np.quantile(kept, 1 - a)))
    stacked = np.stack([ap[cls] for cls in classes])
    replicate_mean = stacked.mean(axis=0)
    kept = replicate_mean[~np.isnan(replicate_mean)]
    if kept.size == 0:
        return float("nan"), float("nan"), 0, per_class
    return float(np.quantile(kept, a)), float(np.quantile(kept, 1 - a)), int(kept.size), per_class


def _close(stored, computed, tol=1e-9):
    if stored is None or any(v is None for v in stored):
        return all(np.isnan(v) for v in computed)
    if any(isinstance(v, float) and np.isnan(v) for v in stored):
        return all(np.isnan(v) for v in computed)
    return abs(stored[0] - computed[0]) <= tol and abs(stored[1] - computed[1]) <= tol


def process(path, npz, force=False, dry_run=False):
    """Returns "written", "skipped" (already there), "mismatch" (per-class intervals differ; nothing written)
    or "no-histograms"."""
    report = json.loads(path.read_text())
    summary = report.get("summary") or {}
    if "mAUPR_ci" in summary and not force:
        return "skipped"
    boot = (report.get("protocol") or {}).get("bootstrap") or {"n": 2000, "seed": 0, "ci": 0.95}
    with np.load(npz) as arrays:
        if any(f"{cls}_pos" not in arrays for cls in CLASSES):
            return "no-histograms"
        low, high, n_kept, per_class = maupr_interval(arrays, boot)
    for cls in CLASSES:
        stored = (report["per_class"].get(cls) or {}).get("aupr_ci")
        if not _close(stored, per_class[cls]):
            print(f"{path}: per-class interval of {cls} differs from the stored one ({stored} vs {per_class[cls]}); not written", flush=True)
            return "mismatch"
    if dry_run:
        print(f"{path}: mAUPR {summary.get('mAUPR')} [{low:.4f}, {high:.4f}] ({n_kept} replicates)", flush=True)
        return "written"
    summary["mAUPR_ci"] = [None if np.isnan(low) else low, None if np.isnan(high) else high]
    summary["mAUPR_ci_n_replicates"] = n_kept
    report["summary"] = summary
    path.write_text(json.dumps(report, indent=1))
    return "written"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default=str(_ROOT / "runs"))
    parser.add_argument("--only", default="", help="substring of the report path")
    parser.add_argument("--force", action="store_true", help="recompute reports that already carry mAUPR_ci")
    parser.add_argument("--dry-run", action="store_true", help="print the intervals, write nothing")
    args = parser.parse_args(argv)
    counts = {}
    for path, npz in reports(args.runs):
        if args.only and args.only not in str(path):
            continue
        status = process(path, npz, force=args.force, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
    print("maupr_ci: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) if counts else "maupr_ci: no report with histograms", flush=True)


if __name__ == "__main__":
    main()
