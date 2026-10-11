"""Move aside the B-std runs of one model that were trained under a superseded recipe.

After a recipe revision (bench/configs/bs_frozen.yaml, recipe_revisions), the
model's old run directories still hold diagnostic.json and metrics_test.json,
so the launcher would skip them as finished. This renames every formal run of
the model whose recipe.json signature differs from the current one to
<name>_<suffix> (default _registered), keeping it for the supplementary
table; a run whose console.log shows a live process is left alone and
reported. Nothing is deleted. Without --apply it only prints.

    python bench/scripts/retire_runs.py --model U-Net [--apply]
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.scripts.launch_b1 import (
    check_job,
    fresh_signature,
    in_progress,
    load_document,
    select_jobs,
    active_models,
    signature_matches,
)

BS_JOBS = _ROOT / "bench" / "configs" / "bs_jobs.yaml"


def stale_runs(model, jobs_path=BS_JOBS):
    """(run_dir, reason) for every formal run of ``model`` that exists and is not current."""
    document = load_document(jobs_path)
    out = []
    for job in select_jobs(document, False, active_models()):
        if job["model"] != model:
            continue
        config, run_dir = check_job(job, False)
        if not (_ROOT / run_dir).is_dir():
            continue
        same, reason = signature_matches(run_dir, fresh_signature(job, config))
        if not same:
            out.append((run_dir, reason))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suffix", default="registered")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    stale = stale_runs(args.model)
    if not stale:
        print(f"{args.model}: every existing run matches the current recipe", flush=True)
        return
    blocked = []
    for run_dir, reason in stale:
        target = Path(f"{run_dir}_{args.suffix}")
        running = in_progress(run_dir)
        if running is not None:
            blocked.append(f"{run_dir}: still running ({running})")
            continue
        if (_ROOT / target).exists():
            blocked.append(f"{run_dir}: {target} already exists")
            continue
        print(f"{'move' if args.apply else 'would move'} {run_dir} -> {target} ({reason})", flush=True)
        if args.apply:
            os.rename(_ROOT / run_dir, _ROOT / target)
    if blocked:
        raise SystemExit("not moved:\n" + "\n".join(blocked))


if __name__ == "__main__":
    main()
