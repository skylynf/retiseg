"""Re-score the factor cells whose primary checkpoint was the last step (docs/ops/21 #97, 2026-10-28).

bench/train.py scored a BF cell the B1 way when the model's author recipe sets a step budget: last.pt
as the primary checkpoint and best.pt as the sensitivity one. The factor experiments follow the B-std
protocol (docs/paper/16 section 3; bench/factor.py; the paper's "Checkpoint selection"), which scores
the checkpoint with the highest validation mAUPR. The U-Net cells (no author budget) were scored from
best.pt, the M2MRF, RETFound, SAMed, SAM2-UNet and HRDecoder cells from last.pt. score_run predicted
both checkpoints, so the maps of best.pt already sit in prob_val_sensitivity / prob_test_sensitivity.

For every BF cell whose checkpoint_roles.json says ``primary: last.pt`` this script

  1. swaps the map directories (prob_val <-> prob_val_sensitivity, prob_test <-> prob_test_sensitivity),
  2. rewrites checkpoint_roles.json (primary best.pt, sensitivity last.pt) and keeps the old file as
     checkpoint_roles_laststep.json,
  3. keeps metrics_test.json (and .npz) as metrics_test_laststep.* and evaluates the swapped prob_test
     against prob_val with the command the factor launcher uses (bench.eval, the cell's leak exclusions),
  4. moves runs/E3a/<cell> to runs/E3a_laststep/<cell>, so bench.e3a (launch_e3a.py --only ...) predicts
     the independent tests again from best.pt. Nothing is deleted.

    python bench/scripts/rescore_factor_roles.py plan                 # what would change; nothing written
    python bench/scripts/rescore_factor_roles.py apply [--cells a,b] [--no-evaluate]

A cell whose sensitivity directories are missing or shorter than the primary ones is reported and left
alone (its best.pt maps would have to be predicted first: bench/train.py --score-only after removing the
primary directories). Everything runs on the CPU; one evaluation takes a few minutes.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

PRIMARY_DIRS = ("prob_val", "prob_test")
SENSITIVITY_DIRS = ("prob_val_sensitivity", "prob_test_sensitivity")
LASTSTEP_E3A = "E3a_laststep"


def _count(directory):
    """Map files below a prediction directory (<cls>/<id>.png), ignoring temporary files."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.rglob("*.png") if not path.name.startswith("."))


def affected_cells(runs_root):
    """BF run directories scored from last.pt, with what the swap needs. Each item is
    (run_dir, status) where status is "ready", or the reason the cell cannot be swapped."""
    items = []
    for run in sorted(Path(runs_root).glob("BF_*")):
        roles_path = run / "checkpoint_roles.json"
        config_path = run / "config.yaml"
        if not run.is_dir() or not roles_path.is_file() or not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text())
        if config.get("experiment") != "BF":
            continue
        roles = json.loads(roles_path.read_text())
        if roles.get("primary") != "last.pt":
            continue
        missing = [name for name in ("best.pt", "last.pt") if not (run / name).is_file()]
        if missing:
            items.append((run, f"checkpoint missing: {missing}"))
            continue
        short = []
        for primary, sensitivity in zip(PRIMARY_DIRS, SENSITIVITY_DIRS):
            n_primary, n_sensitivity = _count(run / primary), _count(run / sensitivity)
            if n_sensitivity == 0 or n_sensitivity < n_primary:
                short.append(f"{sensitivity}: {n_sensitivity} maps against {n_primary} in {primary}")
        if short:
            items.append((run, "sensitivity maps incomplete; " + "; ".join(short)))
            continue
        if not (run / "metrics_test.json").is_file():
            items.append((run, "metrics_test.json missing; the cell is not scored"))
            continue
        items.append((run, "ready"))
    return items


def swap_directories(run):
    """prob_val <-> prob_val_sensitivity and prob_test <-> prob_test_sensitivity, through a temporary name."""
    run = Path(run)
    for primary, sensitivity in zip(PRIMARY_DIRS, SENSITIVITY_DIRS):
        a, b, tmp = run / primary, run / sensitivity, run / f".{primary}.swap"
        if tmp.exists():
            raise FileExistsError(f"{tmp} exists; a previous swap was interrupted, sort it out by hand")
        a.rename(tmp)
        b.rename(a)
        tmp.rename(b)


def rewrite_roles(run):
    run = Path(run)
    old = run / "checkpoint_roles.json"
    kept = run / "checkpoint_roles_laststep.json"
    if not kept.exists():
        shutil.copy2(old, kept)
    old.write_text(
        json.dumps(
            {
                "primary": "best.pt",
                "sensitivity": "last.pt",
                "primary_directories": list(PRIMARY_DIRS),
                "sensitivity_directories": list(SENSITIVITY_DIRS),
                "rescored": "2026-10-28 bench/scripts/rescore_factor_roles.py (docs/ops/21 #97): maps swapped, last-step scores kept as *_laststep",
            },
            indent=2,
        )
        + "\n"
    )


def keep_laststep_metrics(run):
    run = Path(run)
    for suffix in (".json", ".npz"):
        src = run / f"metrics_test{suffix}"
        dst = run / f"metrics_test_laststep{suffix}"
        if src.is_file() and not dst.exists():
            src.rename(dst)
    full = run / "metrics_test_full.json"
    if full.is_file() and not (run / "metrics_test_full_laststep.json").exists():
        full.rename(run / "metrics_test_full_laststep.json")


def evaluate_argv(run, python=None):
    """The factor launcher's evaluation of the primary maps (bench.eval with the cell's leak exclusions)."""
    from bench.scripts.launch_factor import internal_exclusions

    run = Path(run)
    cell = yaml.safe_load((run / "config.yaml").read_text())
    rel = run.relative_to(_ROOT) if run.is_absolute() else run
    argv = [python or sys.executable, "-m", "bench.eval.cli", "evaluate", "--data", str(cell["dataset"]), "--split",
            cell["split_test"], "--pred", f"{rel}/prob_test", "--val-pred", f"{rel}/prob_val", "--out", f"{rel}/metrics_test"]
    exclude = internal_exclusions(cell)
    if exclude:
        argv += ["--exclude", ",".join(exclude)]
    return argv


def move_e3a(run, runs_root):
    """runs/E3a/<cell> -> runs/E3a_laststep/<cell>; None when the cell had no independent tests."""
    runs_root = Path(runs_root)
    src = runs_root / "E3a" / Path(run).name
    if not src.is_dir():
        return None
    dst = runs_root / LASTSTEP_E3A / Path(run).name
    if dst.exists():
        raise FileExistsError(f"{dst} exists; the cell was moved before")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return dst


def apply(run, runs_root, evaluate=True, python=None):
    swap_directories(run)
    rewrite_roles(run)
    keep_laststep_metrics(run)
    moved = move_e3a(run, runs_root)
    if evaluate:
        argv = evaluate_argv(run, python)
        print("  " + " ".join(argv), flush=True)
        subprocess.run(argv, check=True, cwd=_ROOT)
    return moved


def summary_line(run):
    """mAUPR before (laststep) and after, for docs/ops/00."""
    run = Path(run)
    before = json.loads((run / "metrics_test_laststep.json").read_text())["summary"]["mAUPR"] if (run / "metrics_test_laststep.json").is_file() else float("nan")
    after_path = run / "metrics_test.json"
    after = json.loads(after_path.read_text())["summary"]["mAUPR"] if after_path.is_file() else float("nan")
    return f"{run.name}: last step {before:.4f} -> best validation {after:.4f} ({after - before:+.4f})"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("plan", "apply"))
    parser.add_argument("--runs", default=str(_ROOT / "runs"))
    parser.add_argument("--cells", default="", help="comma-separated run names; default every affected BF cell")
    parser.add_argument("--no-evaluate", action="store_true", help="swap and move only; evaluate by hand")
    parser.add_argument("--python", default=None, help="interpreter for bench.eval (default: this one)")
    args = parser.parse_args(argv)
    chosen = {name.strip() for name in args.cells.split(",") if name.strip()}
    items = [(run, status) for run, status in affected_cells(args.runs) if not chosen or run.name in chosen]
    if not items:
        print("no BF cell is scored from last.pt", flush=True)
        return
    for run, status in items:
        e3a_dir = Path(args.runs) / "E3a" / run.name
        print(f"{run.name}: {status}; E3a {'present (moved aside and rerun)' if e3a_dir.is_dir() else 'none'}", flush=True)
    if args.action == "plan":
        ready = [run for run, status in items if status == "ready"]
        print(f"{len(ready)} of {len(items)} cells ready; nothing written", flush=True)
        return
    done = []
    for run, status in items:
        if status != "ready":
            print(f"skip {run.name}: {status}", flush=True)
            continue
        print(f"apply {run.name}", flush=True)
        apply(run, args.runs, evaluate=not args.no_evaluate, python=args.python)
        done.append(run)
    for run in done:
        print(summary_line(run), flush=True)
    if done:
        names = ",".join(run.name for run in done if (Path(args.runs) / LASTSTEP_E3A / run.name).is_dir())
        print("independent tests to rerun from best.pt (one queue, FGADR / Retinal-Lesions maps dropped after scoring):", flush=True)
        if names:
            print(f'  RETISEG_GPUS="..." python bench/scripts/launch_e3a.py --submit --only {names} --drop-maps FGADR,Retinal-Lesions', flush=True)


if __name__ == "__main__":
    main()
