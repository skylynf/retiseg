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
    factor   bench/scripts/launch_factor.py plan()     keys: experiment, only, models, score_only
    nnunet   bench/scripts/nnunet_run.sh               keys: runs: [{train, plan}]; needs the converted raw data
    e5_infer bench/scripts/launch_e5_infer.py plan()   keys: phase (pool | test), only (sources), drop_maps
    shell    literal items                             keys: env, items: [{label, command, done, run_dir}]

An item is "done" when its stage says so (marker files); "running" when its
console.log has an open "== start" of a live process (launch_b1.in_progress),
whichever launcher started it. A stage that cannot plan (missing data, a
missing environment) is reported and skipped, never fatal. A stage may set
max_concurrent (how many of its items this launcher runs at once; the next
stage gets the card when the cap is reached) and max_items. --submit needs
RETISEG_GPUS; the launcher never touches other cards.

One launcher per host is the rule (docs/ops/17 section 1). When there are
several anyway, each claims an item's directory (claim.json, created
atomically) before starting it: the "== start" line only appears once the
process runs, and a second launcher reading the queue in between would start
the same item (2026-10-15: three launchers opened
runs/E3a/BS_fct_idrid_seed1/FGADR at the same minute). A claim whose launcher
is dead on this host is stale and taken over; one from another host stands
until RETISEG_IGNORE_RUNNING=1, like an open "== start".
"""

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import socket
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from bench.scripts.launch_b1 import _DIST_VARS, _pid_alive, _start, gpu_is_free, in_progress, python_argv

QUEUE = _ROOT / "bench" / "configs" / "queue.yaml"
CLAIM = "claim.json"
_LOG_RE = re.compile(r">> *'?([^'\s]+)/console\.log'?")


def claim_dir(label, command):
    """The directory a queue item writes its console.log to, or None when the command keeps no log of its own."""
    match = _LOG_RE.search(command)
    if match:
        return Path(match.group(1))
    if label.startswith("nnunet:"):   # nnunet_run.sh writes runs/<name>/console.log itself
        return Path("runs") / label.split(":", 1)[1]
    return None


def claim(label, run_dir):
    """Hold ``run_dir`` for this launcher: True when claim.json is now ours.

    The file is created with O_EXCL, so of two launchers reading the queue at the
    same time only one gets the item. A claim whose pid is dead on this host is
    replaced; one of a live pid, or of another host, is left alone (RETISEG_IGNORE_RUNNING=1
    treats every claim as stale, as in_progress does for "== start" lines).
    """
    path = _ROOT / run_dir / CLAIM
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps({"label": label, "host": socket.gethostname(), "pid": os.getpid(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    for _attempt in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if not _claim_is_stale(path):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(record)
        return True
    return False


def _claim_is_stale(path):
    if os.environ.get("RETISEG_IGNORE_RUNNING") == "1":
        return True
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return True   # unreadable or half-written: nobody can hold it
    if record.get("host") != socket.gethostname():
        return False
    pid = record.get("pid")
    return not (isinstance(pid, int) and _pid_alive(pid))


def release(run_dir):
    """Drop this launcher's claim (another launcher's claim is left alone)."""
    path = _ROOT / run_dir / CLAIM
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if record.get("host") == socket.gethostname() and record.get("pid") == os.getpid():
        try:
            path.unlink()
        except FileNotFoundError:
            pass


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

    chosen = _quiet(launch_factor.plan, stage.get("experiment"), set(stage.get("only") or ()), bool(stage.get("score_only")),
                    set(stage.get("models") or ()))
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


_PROMPT_LOCK = threading.Lock()


def plan_e5_infer(stage, dry):
    """The E5 inference tiers through bench/scripts/launch_e5_infer.py (docs/ops/25, the albert merge).

    ``phase: pool`` lists the validation-pool runs (the grid, the SAM 3 phrases, the oracle tier on
    IDRiD and DDR val); ``phase: test`` lists the test runs, which appear only once e5_frozen.yaml's
    selections are filled, each gated on its own source's pool directories (three-state gate:
    "never" when nothing queued writes them). Items of a set whose prepared directory is not on this
    machine are left out until it exists (MAPLES-DR before its export). Missing oracle prompt files are
    written here, on the CPU, before the first item is returned (not in a dry plan).
    """
    from bench.e5 import frozen as fz
    from bench.e5 import geometry as geo
    from bench.scripts import launch_e5_infer as le

    phase = stage.get("phase", "pool")
    if phase not in ("pool", "test"):
        raise ValueError(f"e5_infer stage: phase must be pool or test, got {phase!r}")
    frozen = fz.load_frozen()
    only = set(stage.get("only") or ())
    items = _quiet(le.plan, frozen, only, phase == "pool", phase == "test")
    items = [item for item in items if geo.data_dir(item.set_name).is_dir()]
    if dry:
        missing = le.missing_prompt_files(frozen, items)
        if missing:
            print("e5_infer: prompt files to write first: " + ", ".join(f"{n}/{s}/{v}" for n, s, v in missing), flush=True)
    else:
        with _PROMPT_LOCK:  # every worker thread calls gather; one of them writes a missing file
            missing = le.missing_prompt_files(frozen, items)
            if missing:
                le.write_prompt_files(frozen, missing)
    drop = tuple(stage.get("drop_maps") or ())
    producers = None
    if not dry:
        running = [it for it in _quiet(le.discover, frozen) if it.running() is not None]
        producers = le.producers_of(running + _quiet(le.plan, frozen, only, False, False))
    out = []
    for item in items:
        command = le.command(item, py_argv=["python"] if dry else None, drop_maps=drop)
        out.append((f"e5_infer:{item.label}", command, le.gate_for(item, producers)))
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


PLANNERS = {"e3a": plan_e3a, "bs": plan_bs, "factor": plan_factor, "nnunet": plan_nnunet, "e5_infer": plan_e5_infer,
            "shell": plan_shell}


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
        try:
            if planner is None:
                raise ValueError(f"unknown kind {stage['kind']} (a launcher older than the queue file: restart it)")
            items = planner(stage, dry)
        except (Exception, SystemExit) as exc:  # a stage that cannot plan is reported, the queue goes on
            # SystemExit too: python_argv exits when a stage's conda env is missing (retiseg-sam3 before
            # cluster_env.sh sam3), which must not end every worker thread (merge review 2026-10-14)
            if report is not None:
                report.append(f"{stage['name']}: {type(exc).__name__}: {exc}")
            continue
        limit = stage.get("max_items")
        if limit:
            items = items[: int(limit)]
        for label, command, gate in items:
            out.append((float(stage["priority"]), stage["name"], label, command, gate))
    return out


RETRY_MINUTES = (10, 30, 60, 180, 360)


def retry_delay(count):
    """Seconds an item that exited non-zero ``count`` times in a row waits before this launcher starts it
    again: 10, 30, 60, 180 minutes, then six hours (2026-10-16: a scoring item that failed in seconds was
    restarted about 1,200 times in a night and held three cards in front of the rest of the queue)."""
    return 60 * RETRY_MINUTES[min(max(int(count), 1), len(RETRY_MINUTES)) - 1]


def run(gpus, path=QUEUE, poll_seconds=60, idle_exit_minutes=None):
    """One worker per card; a free card reads the queue again and takes the first item nobody holds.

    A stage's ``max_concurrent`` caps how many of its items this launcher runs
    at once (E3a keeps two cards, the rest go to training); items of a stage at
    its cap are passed over for the next stage, not waited for. An item whose
    process exits non-zero is passed over for retry_delay(count) before this
    launcher starts it again (the stage's own markers still decide whether it
    is listed at all).
    """
    lock = threading.Lock()
    taken = set()
    running = {}   # label -> stage of the items this launcher has on a card
    failures = {}  # label -> (count, time of the last non-zero exit): retried after a growing pause
    errors = []
    idle_since = {gpu: None for gpu in gpus}

    def _next():
        report = []
        items = gather(path, dry=False, report=report)
        for line in report:
            print(f"stage skipped: {line}", flush=True)
        caps = {stage["name"]: stage.get("max_concurrent") for stage in load_queue(path)}
        with lock:
            for _priority, stage, label, command, gate in items:
                if label in taken:
                    continue
                if label in failures and time.time() < failures[label][1] + retry_delay(failures[label][0]):
                    continue   # it failed a moment ago; the queue goes on to the next item instead of looping on it
                cap = caps.get(stage)
                if cap and sum(1 for held in running.values() if held == stage) >= int(cap):
                    continue
                state = gate() if gate is not None else "open"
                if state == "never":
                    taken.add(label)
                    errors.append(f"not started: {label}: its gate closed")
                    continue
                if state != "open":
                    continue
                run_dir = claim_dir(label, command)
                if run_dir is not None and not claim(label, run_dir):
                    continue   # another launcher holds it; its markers hide the item once it runs
                taken.add(label)
                running[label] = stage
                return stage, label, command, run_dir
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
            stage, label, command, run_dir = item
            print(f"gpu {gpu} <- {stage} {label}", flush=True)
            process = _start(command, gpu)
            code = process.wait()
            if code != 0:
                with lock:
                    count = failures.get(label, (0, 0.0))[0] + 1
                    failures[label] = (count, time.time())
                    errors.append(f"gpu {gpu} exited {code}: {label} (failure {count})")
                print(f"gpu {gpu}: {label} exited {code} (failure {count}; not retried for {retry_delay(count) / 60:.0f} min)", flush=True)
            else:
                with lock:
                    failures.pop(label, None)
            if run_dir is not None:
                release(run_dir)
            with lock:
                taken.discard(label)   # the stage's own done/running rules decide whether it comes back
                running.pop(label, None)

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
