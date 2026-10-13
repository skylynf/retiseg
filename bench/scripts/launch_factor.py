"""Print or queue the factor jobs in bench/configs/factor_jobs.yaml (E3b, E4, E7, E8, E9).

Each job runs `<script> --config <cell> --run-dir <run_dir> --resume` (training,
then the test maps through score_run) followed by `bench.eval.cli evaluate` on
the primary maps, which writes metrics_test.json; one process per card in
RETISEG_GPUS, longest first. A job whose run directory holds metrics_test.json
is finished and skipped; one whose console.log shows a live process is skipped
too. A run that finished training (best.pt and timing.json) without
metrics_test.json is scored (--score-only predicts from the saved checkpoints)
and evaluated instead of retrained; --score-only restricts the queue to those.
The internal test excludes the same-eye images of the cell's training sources
(leak_exclusions.json, --exclude); a metrics_test.json written without them is
re-evaluated (evaluate only, no new maps). Jobs of a factor whose weights or subsets are missing
fail at their first step and are reported, not retried.

    python bench/scripts/launch_factor.py [--experiment E4] [--only NAME,...]
    RETISEG_GPUS="2 3" python bench/scripts/launch_factor.py --submit --experiment E4
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from bench.scripts.launch_b1 import _DIST_VARS, in_progress, python_argv, run_queue

JOBS = _ROOT / "bench" / "configs" / "factor_jobs.yaml"
REQUIRED = ("name", "experiment", "model", "dataset", "seed", "env", "config", "script", "run_dir")


def load_jobs(path=JOBS):
    jobs = yaml.safe_load(Path(path).read_text())["jobs"]
    for job in jobs:
        missing = [key for key in REQUIRED if key not in job]
        if missing:
            raise ValueError(f"{job.get('name')} is missing {missing}")
        if Path(job["run_dir"]).parts[:1] != ("runs",):
            raise ValueError(f"{job['name']}: run_dir must be under runs/")
        cell = yaml.safe_load((_ROOT / job["config"]).read_text())
        if cell.get("experiment") != "BF" or cell.get("factor") != job["experiment"] or cell.get("model") != job["model"]:
            raise ValueError(f"{job['name']}: cell {job['config']} does not match the job")
        for item in cell["train"]:
            subset = item.get("subset")
            if subset and not (_ROOT / subset).is_file():
                raise FileNotFoundError(f"{job['name']}: subset list {subset} is missing; run bench/scripts/make_subsets.py first")
    return jobs


def internal_exclusions(cell):
    """Internal-test images that show an eye one of the cell's training sources has seen
    (bench/data/splits/leak_exclusions.json, the same list bench/e3a.py applies): FGADR's 106 same-eye test
    images for a run trained on FGADR, Retinal-Lesions' 1, TJDR_uwf's 2; empty for IDRiD and DDR."""
    from bench.e3a import PREPARED_TO_SET, leak_exclusions, train_sources_of

    test_name = PREPARED_TO_SET.get(Path(cell["dataset"]).name, Path(cell["dataset"]).name)
    return sorted(leak_exclusions(train_sources_of(cell), test_name))


def evaluate_argv(job, py_argv):
    """bench.eval on the primary test maps of a finished run: the step that writes metrics_test.json
    (the training script only writes the maps, as in launch_b1's formal wave). Same-eye test images of a
    training source are excluded (--exclude), as E3a does for every other test set."""
    cell = yaml.safe_load((_ROOT / job["config"]).read_text())
    run_dir = job["run_dir"]
    argv = list(py_argv) + ["-m", "bench.eval.cli", "evaluate", "--data", str(cell["dataset"]), "--split", cell["split_test"],
                            "--pred", f"{run_dir}/prob_test", "--val-pred", f"{run_dir}/prob_val", "--out", f"{run_dir}/metrics_test"]
    exclude = internal_exclusions(cell)
    if exclude:
        argv += ["--exclude", ",".join(exclude)]
    return argv


def needs_reevaluation(job):
    """metrics_test.json written without the exclusions the cell needs (runs scored by hand on 2026-10-13)."""
    path = _ROOT / job["run_dir"] / "metrics_test.json"
    if not path.is_file():
        return False
    cell = yaml.safe_load((_ROOT / job["config"]).read_text())
    expected = internal_exclusions(cell)
    if not expected:
        return False
    import json

    written = sorted(json.loads(path.read_text()).get("excluded_ids") or [])
    return written != expected


def maps_written(run_dir):
    """score_run already wrote the primary maps (checkpoint_roles.json and a non-empty prob_test)."""
    run_dir = _ROOT / run_dir
    return (run_dir / "checkpoint_roles.json").is_file() and any((run_dir / "prob_test").glob("*"))


def command(job, py_argv=None, score_only=False):
    """Train, then evaluate. With score_only: predict from the saved checkpoints when the maps are
    missing (score_run refuses to overwrite a finished prediction directory), then evaluate."""
    py_argv = python_argv(job["env"]) if py_argv is None else py_argv
    run_dir = job["run_dir"]
    steps = []
    if not score_only:
        steps.append(list(py_argv) + [job["script"], "--config", job["config"], "--run-dir", run_dir, "--resume"])
    elif not maps_written(run_dir):
        steps.append(list(py_argv) + [job["script"], "--config", job["config"], "--run-dir", run_dir, "--score-only"])
    steps.append(evaluate_argv(job, py_argv))
    chain = "unset " + " ".join(_DIST_VARS) + "; " + " && ".join(shlex.join(step) for step in steps)
    log = shlex.quote(str(Path(run_dir) / "console.log"))
    return (
        f"mkdir -p {shlex.quote(run_dir)} && "
        f"{{ echo \"== start {job['experiment']} $(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) pid $$\"; {chain}; rc=$?; "
        f"echo \"== exit $rc $(date -Is)\"; exit $rc; }} >> {log} 2>&1"
    )


def plan(experiment=None, only=(), score_only=False):
    """The jobs this launcher would start now, in order, each with _score_only set (launch_queue reads this too)."""
    only = set(only or ())
    chosen = []
    for job in sorted(load_jobs(), key=lambda job: -float(job.get("est_hours", 1.0))):
        if experiment and job["experiment"] != experiment:
            continue
        if only and job["name"] not in only:
            continue
        if (_ROOT / job["run_dir"] / "metrics_test.json").is_file():
            if needs_reevaluation(job):
                print(f"queued {job['name']} (re-evaluating with the same-eye exclusions) -> {job['run_dir']}", flush=True)
                chosen.append(dict(job, _score_only=True, est_hours=0.3))
            else:
                print(f"skip {job['name']}: finished", flush=True)
            continue
        running = in_progress(Path(job["run_dir"]))
        if running is not None:
            print(f"skip {job['name']}: still running ({running})", flush=True)
            continue
        trained = (_ROOT / job["run_dir"] / "best.pt").is_file() and (_ROOT / job["run_dir"] / "timing.json").is_file()
        if score_only and not trained:
            print(f"skip {job['name']}: not trained yet", flush=True)
            continue
        if trained and not score_only:
            # trained before this launcher evaluated (2026-10-13): score and evaluate instead of retraining
            print(f"queued {job['name']} (trained; predicting from best.pt and evaluating) -> {job['run_dir']}", flush=True)
            chosen.append(dict(job, _score_only=True, est_hours=0.5))
            continue
        print(f"queued {job['name']} est {job.get('est_hours', '?')} h -> {job['run_dir']}", flush=True)
        chosen.append(dict(job, _score_only=score_only))
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--experiment")
    parser.add_argument("--only")
    parser.add_argument("--score-only", action="store_true", help="only runs with best.pt and prob_test: predict again and evaluate")
    args = parser.parse_args(argv)
    only = {item.strip() for item in (args.only or "").split(",") if item.strip()}
    chosen = plan(args.experiment, only, args.score_only)
    if not args.submit:
        for job in chosen[:2]:
            print("command " + command(job, py_argv=["python"], score_only=job.get("_score_only", False)), flush=True)
        print(f"{len(chosen)} jobs; not submitted, pass --submit with RETISEG_GPUS", flush=True)
        return
    gpus = [int(gpu) for gpu in os.environ.get("RETISEG_GPUS", "").replace(",", " ").split()]
    if not gpus:
        raise SystemExit("--submit needs RETISEG_GPUS")
    errors = run_queue([(job["name"], command(job, score_only=job.get("_score_only", False))) for job in chosen], gpus, wait_free=True)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
