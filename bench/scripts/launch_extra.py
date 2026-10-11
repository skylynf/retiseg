"""Print or queue the diagnostic jobs in bench/configs/extra_jobs.yaml (E2c now).

Each job runs `<script> --config <cell> --run-dir <run_dir> --diagnostic --resume`
in its environment, one process per card in RETISEG_GPUS, longest first. A job
whose run directory has diagnostic.json is finished and skipped, and so is one
whose console.log shows a live process. Nothing here reads a test split.

    python bench/scripts/launch_extra.py [--experiment E2c] [--only NAME,...]
    RETISEG_GPUS="2 3" python bench/scripts/launch_extra.py --submit --experiment E2c
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

JOBS = _ROOT / "bench" / "configs" / "extra_jobs.yaml"
REQUIRED = ("name", "experiment", "model", "dataset", "seed", "env", "config", "script", "run_dir")


def load_jobs(path=JOBS):
    jobs = yaml.safe_load(Path(path).read_text())["jobs"]
    for job in jobs:
        missing = [key for key in REQUIRED if key not in job]
        if missing:
            raise ValueError(f"{job.get('name')} is missing {missing}")
        if not Path(job["run_dir"]).parts[:1] == ("runs",) or "smoke" in Path(job["run_dir"]).parts:
            raise ValueError(f"{job['name']}: run_dir must be under runs/ and not a smoke directory")
        cell = yaml.safe_load((_ROOT / job["config"]).read_text())
        if cell.get("model") != job["model"] or Path(cell["dataset"]).name != job["dataset"]:
            raise ValueError(f"{job['name']}: cell {job['config']} does not match the job")
    return jobs


def command(job, py_argv=None):
    py_argv = python_argv(job["env"]) if py_argv is None else py_argv
    run_dir = job["run_dir"]
    train = list(py_argv) + [job["script"], "--config", job["config"], "--run-dir", run_dir, "--diagnostic", "--resume"]
    chain = "unset " + " ".join(_DIST_VARS) + "; " + shlex.join(train)
    log = shlex.quote(str(Path(run_dir) / "console.log"))
    return (
        f"mkdir -p {shlex.quote(run_dir)} && "
        f"{{ echo \"== start {job['experiment']} $(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) pid $$\"; {chain}; rc=$?; "
        f"echo \"== exit $rc $(date -Is)\"; exit $rc; }} >> {log} 2>&1"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--experiment")
    parser.add_argument("--only")
    args = parser.parse_args(argv)
    only = {item.strip() for item in (args.only or "").split(",") if item.strip()}
    chosen = []
    for job in sorted(load_jobs(), key=lambda job: -float(job.get("est_hours", 1.0))):
        if args.experiment and job["experiment"] != args.experiment:
            continue
        if only and job["name"] not in only:
            continue
        if (_ROOT / job["run_dir"] / "diagnostic.json").is_file():
            print(f"skip {job['name']}: finished", flush=True)
            continue
        running = in_progress(Path(job["run_dir"]))
        if running is not None:
            print(f"skip {job['name']}: still running ({running})", flush=True)
            continue
        print(f"queued {job['name']} est {job.get('est_hours', '?')} h -> {job['run_dir']}", flush=True)
        chosen.append(job)
    if not args.submit:
        for job in chosen[:2]:
            print("command " + command(job, py_argv=["python"]), flush=True)
        print(f"{len(chosen)} jobs; not submitted, pass --submit with RETISEG_GPUS", flush=True)
        return
    gpus = [int(gpu) for gpu in os.environ.get("RETISEG_GPUS", "").replace(",", " ").split()]
    if not gpus:
        raise SystemExit("--submit needs RETISEG_GPUS")
    errors = run_queue([(job["name"], command(job)) for job in chosen], gpus, wait_free=True)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
