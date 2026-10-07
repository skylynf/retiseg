"""Summarize run directories without loading checkpoints or reading the test split.

status: one row per runs/<name> with a recipe.json. State is one of running
(resume.pt present), failed (console.log ends in a nonzero exit), trained
(diagnostic.json, test not read), scored (metrics_test.json) or started.
Test mAUPR is printed only when metrics_test.json already exists. The table
is also written to runs/status.tsv.

curves: validation history of the matched runs, and where the best
validation value sits. A best value in the last quarter of the run means the
curve was still rising, which is the budget probe's question.

    python bench/scripts/run_status.py status
    python bench/scripts/run_status.py curves --glob 'BSprobe_*'
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
RUNS = _ROOT / "runs"


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _history(run):
    path = run / "history.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def _last_exit(run):
    path = run / "console.log"
    if not path.is_file():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith("== exit "):
            return int(line.split()[2])
        if line.startswith("== start "):
            return None
    return None


def _state(run):
    if (run / "resume.pt").is_file():
        return "running"
    code = _last_exit(run)
    if code not in (None, 0):
        return f"failed({code})"
    if (run / "metrics_test.json").is_file():
        return "scored"
    if (run / "diagnostic.json").is_file():
        return "trained"
    return "started"


def _selection(rows):
    if rows and "val_maupr" in rows[-1]:
        return "val_maupr", max
    return "val_loss", min


def status_rows(pattern="*"):
    out = []
    for run in sorted(RUNS.glob(pattern)):
        recipe = _json(run / "recipe.json")
        if not run.is_dir() or recipe is None:
            continue
        rows = _history(run)
        key, pick = _selection(rows)
        values = [row.get(key) for row in rows if row.get(key) is not None]
        best = pick(values) if values else None
        metrics = _json(run / "metrics_test.json")
        environment = _json(run / "environment.json") or {}
        timing = _json(run / "timing.json") or {}
        out.append(
            {
                "run": run.name,
                "state": _state(run),
                "iteration": rows[-1]["iteration"] if rows else 0,
                "budget": recipe.get("iterations"),
                "select_by": key,
                "best_val": None if best is None else round(float(best), 4),
                "test_mAUPR": None if not metrics else round(float(metrics["summary"]["mAUPR"]), 4),
                "hours": None if "train_seconds_total" not in timing else round(timing["train_seconds_total"] / 3600, 2),
                "git": str(environment.get("git", ""))[:8],
            }
        )
    return out


def status(pattern):
    rows = status_rows(pattern)
    columns = ["run", "state", "iteration", "budget", "select_by", "best_val", "test_mAUPR", "hours", "git"]
    lines = ["\t".join(columns)] + ["\t".join("" if row[c] is None else str(row[c]) for c in columns) for row in rows]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if pattern == "*":
        (RUNS / "status.tsv").write_text(text)


def curves(pattern):
    for run in sorted(RUNS.glob(pattern)):
        rows = _history(run)
        if not rows:
            continue
        key, pick = _selection(rows)
        values = [(row["iteration"], row.get(key)) for row in rows if row.get(key) is not None]
        if not values:
            continue
        best_iter, best = pick(values, key=lambda item: item[1])
        last_iter = rows[-1]["iteration"]
        budget = (_json(run / "recipe.json") or {}).get("iterations") or last_iter
        position = best_iter / float(budget)
        print(f"== {run.name}  {key}  best {best:.4f} at {best_iter}/{budget} ({position:.0%})")
        for row in rows:
            if row.get(key) is None:
                continue
            classes = " ".join(
                f"{cls} {row[f'val_aupr_{cls}']:.3f}"
                for cls in ("MA", "HE", "EX", "SE")
                if row.get(f"val_aupr_{cls}") is not None
            )
            print(f"   {row['iteration']:>7}  {row[key]:.4f}  {classes}")
        if last_iter >= budget and position > 0.75:
            print("   best is in the last quarter: still rising at this budget")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "curves"))
    parser.add_argument("--glob", default="*")
    args = parser.parse_args(argv)
    if args.command == "status":
        status(args.glob)
    else:
        curves(args.glob)


if __name__ == "__main__":
    main()
