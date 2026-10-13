"""One launcher, one queue file: bench/configs/queue.yaml (docs/ops/17 section 14).

    python bench/scripts/launch_queue.py                 # print the merged, ordered plan; nothing starts
    RETISEG_GPUS="3 4 5" python bench/scripts/launch_queue.py --submit

The queue file lists stages with a priority. Every time one of this launcher's
cards is free, the file is read again, every stage is asked for the items it
would start now (finished and running items are left out by the stage's own
rules), the items are ordered by (priority, stage order), and the first one
not already taken by this launcher starts on that card. So the priority
order can be changed by editing and committing the file while the launcher
runs; nothing has to be killed, and no process has to "discover" new work:
the next free card simply reads the file.

Stage kinds (all planning is done by the existing launchers, which keep
working on their own as before):
    e3a      bench/scripts/launch_e3a.py plan()        keys: seed0_only, only, drop_maps
    bs       bench/scripts/launch_b1.py plan_queue_all keys: jobs (yaml), only (models), seed0_only, datasets
    factor   bench/scripts/launch_factor.py plan()     keys: experiment, only, score_only
    nnunet   bench/scripts/nnunet_run.sh               keys: runs: [{train, plan}]; needs the converted raw data
    e5_amg   python -m bench.e5 amg                    keys: models, generators, sets (each "<set>" or "<set>:<split>"), geometry
    e5_prompted  python -m bench.e5 prompted           keys: models, kinds, sets, k
    shell    literal items                             keys: env, items: [{label, command, done, run_dir}]

An item is "done" when its stage says so (marker files); "running" when its
console.log has an open "== start" of a live process (launch_b1.in_progress),
whichever launcher started it. A stage that cannot plan (missing data, a
missing environment) is reported and skipped, never fatal. --submit needs
RETISEG_GPUS; the launcher never touches other cards.
"""

import argparse
import contextlib
import io
import os
import shlex
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from bench.scripts.launch_b1 import _DIST_VARS, _start, gpu_is_free, in_progress, python_argv

QUEUE = _ROOT / "bench" / "configs" / "queue.yaml"


def load_queue(path=QUEUE):
    document = yaml.safe_load(Path(path).read_text())
    stages = document.get("stages") or []
    for index, stage in enumerate(stages):
        for key in ("name", "priority", "kind"):
            if key not in stage:
                raise ValueError(f"stage {index} lacks {key}")
    return sorted(stages, key=lambda stage: (float(stage["priority"]), stages.index(stage)))


def _wrap(label_tag, run_dir, py_argv, argv):
    """Shell body in the launchers' format: console.log with == start / == exit lines (in_progress reads them)."""
    chain = "unset " + " ".join(_DIST_VARS) + "; " + shlex.join(list(py_argv) + list(argv))
    log = shlex.quote(str(Path(run_dir) / "console.log"))
    return (
        f"mkdir -p {shlex.quote(str(run_dir))} && "
        f"{{ echo \"== start {label_tag} $(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) pid $$\"; {chain}; rc=$?; "
        f"echo \"== exit $rc $(date -Is)\"; exit $rc; }} >> {log} 2>&1"
    )


def _quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- stage planners: each returns [(label, command, gate)]


def plan_e3a(stage, dry):
    from bench.scripts import launch_e3a

    drop = tuple(stage.get("drop_maps") or ())
    items = _quiet(launch_e3a.plan, bool(stage.get("seed0_only")), set(stage.get("only") or ()))
    out = []
    for _tier, _cost, label, spec in items:
        command = launch_e3a.command(*spec, py_argv=["python"] if dry else None, drop_maps=drop)
        out.append((f"e3a:{label}", command, None))
    return out


def plan_bs(stage, dry):
    from bench.scripts import launch_b1

    document = launch_b1.load_document(stage.get("jobs") or str(_ROOT / "bench" / "configs" / "bs_jobs.yaml"))
    models = launch_b1.active_models(document.get("experiment", "B1"))
    jobs = launch_b1.select_jobs(document, False, models)
    only = set(stage.get("only") or ())
    if only:
        jobs = [job for job in jobs if job["model"] in only]
    datasets = set(stage.get("datasets") or ())
    if datasets:
        jobs = [job for job in jobs if str(job["dataset"]) in datasets]
    if stage.get("seed0_only"):
        jobs = [job for job in jobs if int(job["seed"]) == 0]
    if dry:
        # python_argv resolves conda; a dry run on a machine without it still lists the jobs
        with _patched(launch_b1, "python_argv", lambda env: ["python"]):
            plan = _quiet(launch_b1.plan_queue_all, jobs, 0)
    else:
        plan = _quiet(launch_b1.plan_queue_all, jobs, 0)
    # plan_queue_all lists a job up to three times (seed-0 training, its gated test maps, the other seeds):
    # the gated one is the score step, named so that the two never share a label
    return [(f"bs:{item[0]}" + (":score" if len(item) > 2 and item[2] is not None else ""), item[1],
             item[2] if len(item) > 2 else None) for item in plan]


def plan_factor(stage, dry):
    from bench.scripts import launch_factor

    chosen = _quiet(launch_factor.plan, stage.get("experiment"), set(stage.get("only") or ()), bool(stage.get("score_only")))
    out = []
    for job in chosen:
        py = ["python"] if dry else None
        command = launch_factor.command(job, py_argv=py, score_only=job.get("_score_only", False))
        out.append((f"factor:{job['name']}", command, None))
    return out


def plan_nnunet(stage, dry):
    from bench.scripts.nnunet_bridge import DATASET_IDS, RAW

    out = []
    for spec in stage.get("runs") or ():
        train, plan = spec["train"], spec.get("plan", "default")
        rep = spec.get("rep")
        if train not in DATASET_IDS:
            raise ValueError(f"nnunet stage: unknown training source {train}")
        run_dir = Path("runs") / f"NN_{plan}_{train.lower()}_fold0{('_' + rep) if rep else ''}"
        if (_ROOT / run_dir / "metrics_test.json").is_file():
            continue
        if in_progress(run_dir) is not None:
            continue
        raw = list(RAW.glob(f"Dataset{DATASET_IDS[train]:03d}_*")) if RAW.is_dir() else []
        if not raw:
            raise FileNotFoundError(f"nnunet stage: dataset/nnunet/raw/Dataset{DATASET_IDS[train]:03d}_* missing; convert + plan_and_preprocess + splits first")
        env = f"RETISEG_NN_REP={rep} " if rep else ""
        command = f"{env}bash bench/scripts/nnunet_run.sh {shlex.quote(train)} {shlex.quote(plan)}"
        out.append((f"nnunet:{run_dir.name}", command, None))
    return out


def _e5_sets(stage):
    for item in stage.get("sets") or ():
        name, _, split = str(item).partition(":")
        yield name, split or "test"


def _e5_py(stage, dry):
    return ["python"] if dry else python_argv(stage.get("env", "retiseg"))


def plan_e5_amg(stage, dry):
    from bench.e5 import amg, common

    out = []
    geometry = stage.get("geometry", "G1")
    for model in stage.get("models") or ("sam", "sam2"):
        for generator in stage.get("generators") or ("default",):
            for test_set, split in _e5_sets(stage):
                directory = amg.out_dir(model, generator, test_set, split, geometry)
                if common.is_done(directory):
                    continue
                log_dir = directory.relative_to(_ROOT)
                if in_progress(log_dir) is not None:
                    continue
                argv = ["-m", "bench.e5", "amg", "--model", model, "--generator", generator, "--set", test_set, "--split", split,
                        "--geometry", geometry]
                out.append((f"e5_amg:{model}/{generator}/{test_set}_{split}", _wrap("e5_amg", log_dir, _e5_py(stage, dry), argv), None))
    return out


def plan_e5_prompted(stage, dry):
    from bench.e5 import common, prompted

    out = []
    ks = prompted.parse_k(stage.get("k", "1,5,10,20,all"))
    for model in stage.get("models") or ("sam2",):
        for kind in stage.get("kinds") or ("box",):
            for test_set, split in _e5_sets(stage):
                dirs = {k: prompted.out_dir(model, kind, k, test_set, split) for k in ks}
                if all(common.is_done(d) for d in dirs.values()):
                    continue
                log_dir = dirs[ks[-1]].relative_to(_ROOT)
                if in_progress(log_dir) is not None:
                    continue
                argv = ["-m", "bench.e5", "prompted", "--model", model, "--kind", kind, "--set", test_set, "--split", split,
                        "--k", ",".join(str(k) for k in ks)]
                out.append((f"e5_prompted:{model}/{kind}/{test_set}_{split}", _wrap("e5_prompted", log_dir, _e5_py(stage, dry), argv), None))
    return out


def plan_shell(stage, dry):
    out = []
    for item in stage.get("items") or ():
        done = item.get("done")
        if done and (_ROOT / done).is_file():
            continue
        run_dir = item.get("run_dir") or (str(Path(done).parent) if done else f"runs/logs/queue/{item['label']}")
        if in_progress(Path(run_dir)) is not None:
            continue
        py = ["python"] if dry else python_argv(stage.get("env", "retiseg"))
        argv = shlex.split(item["command"])
        if argv and argv[0] == "python":
            argv = argv[1:]
            command = _wrap("shell", run_dir, py, argv)
        else:
            command = _wrap("shell", run_dir, [], argv)
        out.append((f"shell:{item['label']}", command, None))
    return out


PLANNERS = {"e3a": plan_e3a, "bs": plan_bs, "factor": plan_factor, "nnunet": plan_nnunet, "e5_amg": plan_e5_amg,
            "e5_prompted": plan_e5_prompted, "shell": plan_shell}


@contextlib.contextmanager
def _patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def gather(path=QUEUE, dry=False, report=None):
    """Every stage's pending items in queue order: [(priority, stage, label, command, gate)]."""
    out = []
    for stage in load_queue(path):
        if not stage.get("enabled", True):
            continue
        planner = PLANNERS.get(stage["kind"])
        if planner is None:
            raise ValueError(f"stage {stage['name']}: unknown kind {stage['kind']}")
        try:
            items = planner(stage, dry)
        except Exception as exc:  # a stage that cannot plan is reported, the queue goes on
            if report is not None:
                report.append(f"{stage['name']}: {type(exc).__name__}: {exc}")
            continue
        limit = stage.get("max_items")
        if limit:
            items = items[: int(limit)]
        for label, command, gate in items:
            out.append((float(stage["priority"]), stage["name"], label, command, gate))
    return out


def run(gpus, path=QUEUE, poll_seconds=60, idle_exit_minutes=None):
    """One worker per card; a free card reads the queue again and takes the first item nobody holds."""
    lock = threading.Lock()
    taken = set()
    errors = []
    idle_since = {gpu: None for gpu in gpus}

    def _next():
        report = []
        items = gather(path, dry=False, report=report)
        for line in report:
            print(f"stage skipped: {line}", flush=True)
        with lock:
            for _priority, stage, label, command, gate in items:
                if label in taken:
                    continue
                state = gate() if gate is not None else "open"
                if state == "never":
                    taken.add(label)
                    errors.append(f"not started: {label}: its gate closed")
                    continue
                if state != "open":
                    continue
                taken.add(label)
                return stage, label, command
        return None

    def _worker(gpu):
        while True:
            if not gpu_is_free(gpu):
                time.sleep(poll_seconds)
                continue
            item = _next()
            if item is None:
                if idle_exit_minutes is not None:
                    idle_since[gpu] = idle_since[gpu] or time.time()
                    if time.time() - idle_since[gpu] > 60 * idle_exit_minutes:
                        print(f"gpu {gpu}: nothing to do for {idle_exit_minutes} min, worker exits", flush=True)
                        return
                time.sleep(poll_seconds)
                continue
            idle_since[gpu] = None
            stage, label, command = item
            print(f"gpu {gpu} <- {stage} {label}", flush=True)
            process = _start(command, gpu)
            code = process.wait()
            if code != 0:
                with lock:
                    errors.append(f"gpu {gpu} exited {code}: {label}")
                print(f"gpu {gpu}: {label} exited {code}", flush=True)
            with lock:
                taken.discard(label)   # the stage's own done/running rules decide whether it comes back

    threads = [threading.Thread(target=_worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--queue", default=str(QUEUE))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-exit-minutes", type=float, default=None, help="a worker exits after this long with nothing to do (default: wait for ever)")
    args = parser.parse_args(argv)
    if not args.submit:
        report = []
        items = gather(args.queue, dry=True, report=report)
        for line in report:
            print(f"stage skipped: {line}", flush=True)
        for priority, stage, label, command, gate in items:
            print(f"{priority:>5g} {stage:<14s} {label}{'  [gated]' if gate is not None else ''}", flush=True)
        print(f"{len(items)} items; not submitted (pass --submit with RETISEG_GPUS)", flush=True)
        if items:
            print("first command: " + items[0][3], flush=True)
        return
    gpus = [int(gpu) for gpu in os.environ.get("RETISEG_GPUS", "").replace(",", " ").split()]
    if not gpus:
        raise SystemExit("--submit needs RETISEG_GPUS, the cards this launcher may use")
    errors = run(gpus, args.queue, args.poll_seconds, args.idle_exit_minutes)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
