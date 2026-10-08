"""Print or start the B1 or B-std jobs listed in a jobs file.

bench/configs/b1_jobs.yaml is the default. bench/configs/bs_jobs.yaml is the
B-std table (experiment: BS, three seeds per dataset); pass it with --jobs.
The waves and their order are the same for both.

Without --submit the command only prints the table and the commands it would
run. --smoke prints or submits the short runs. --submit is what starts a
process. Each process gets one GPU through CUDA_VISIBLE_DEVICES. The gpu
column in the table is a placeholder in 0-7. --submit puts the wave on one
shared queue, longest estimated job first, and each free card takes the next
job when its process exits. Training commands pass --resume, so submitting
the same wave again continues interrupted runs. Distributed launch is not used.

    python bench/scripts/launch_b1.py
    python bench/scripts/launch_b1.py --smoke
    python bench/scripts/launch_b1.py --submit --smoke
    python bench/scripts/launch_b1.py --submit --seed0
    python bench/scripts/launch_b1.py --submit --score-seed0
    python bench/scripts/launch_b1.py --jobs bench/configs/bs_jobs.yaml --submit --seed0

--submit alone does not launch the 64 formal jobs. The current batch is one
seed: seed 0 on every model and both datasets. That wave is training and
validation only; --score-seed0 writes its test maps afterwards. --rest is
the other seeds and is not this batch. Changing the budget means rerunning
seed 0.
"""

import argparse
import ast
import importlib.util
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

# Order is the B1 table order. H2Former is inserted only when WRAPPERS names it.
_REQUIRED = ("U-Net", "DeepLabv3", "HRNet", "Swin-Unet", "FCT", "M2MRF", "HACDR-Net")
_H2FORMER = "H2Former"
_SLUG = {
    "U-Net": "unet",
    "DeepLabv3": "deeplabv3",
    "HRNet": "hrnet",
    "Swin-Unet": "swin_unet",
    "FCT": "fct",
    "H2Former": "h2former",
    "M2MRF": "m2mrf",
    "HACDR-Net": "hacdr",
}
_BINDING = {
    "U-Net": ("retiseg", "bench/train.py", "bench/predict.py"),
    "DeepLabv3": ("retiseg", "bench/train.py", "bench/predict.py"),
    "HRNet": ("retiseg", "bench/train.py", "bench/predict.py"),
    "Swin-Unet": ("retiseg", "bench/train.py", "bench/predict.py"),
    "FCT": ("retiseg", "bench/train.py", "bench/predict.py"),
    "H2Former": ("retiseg", "bench/train.py", "bench/predict.py"),
    # M2MRF imports bench/compat/mmcv ahead of site-packages, so it runs on
    # PyTorch 2.x. PyTorch 1.6 with CUDA 10.2 has no sm_80 kernels for A100.
    "M2MRF": ("retiseg", "bench/scripts/train_m2mrf_b1.py", "bench/scripts/predict_m2mrf_b1.py"),
    "HACDR-Net": ("retiseg-hacdr", "bench/scripts/train_hacdr_b1.py", "bench/scripts/predict_hacdr_b1.py"),
}
_PAIRS = {
    "B1": [("IDRiD", seed) for seed in range(5)] + [("DDR", seed) for seed in range(3)],
    "BS": [("IDRiD", seed) for seed in range(3)] + [("DDR", seed) for seed in range(3)],
}
_FORMAL_PAIRS = _PAIRS["B1"]
_CELL_KEYS = {
    "experiment",
    "model",
    "dataset",
    "split_train",
    "split_val",
    "split_test",
    "fov_diameter",
    "seed",
    "num_workers",
    "batch_size",
}
_SMOKE_STEPS = 20
_DIST_VARS = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK", "MASTER_ADDR", "MASTER_PORT")
_PROTECTED_EXACT = {"B1_unet_idrid_seed0"}
_PROTECTED_PREFIX = ("E0_", "E1_", "E1r_")


def _string_key(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise SystemExit("WRAPPERS keys in bench/models/registry.py must be strings")


def registered_wrappers():
    """Read WRAPPERS without importing the model modules.

    Importing the registry pulls in torch. Printing the table does not need a
    GPU stack, so the names are taken from the source assignment.
    """
    path = _ROOT / "bench" / "models" / "registry.py"
    tree = ast.parse(path.read_text())
    imported = set()
    found = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("bench.models."):
            for alias in node.names:
                imported.add(alias.name)
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "WRAPPERS":
                found = node.value
    if not isinstance(found, ast.Dict):
        raise SystemExit("bench/models/registry.py has no WRAPPERS dict")
    names = {}
    for key, value in zip(found.keys, found.values):
        label = _string_key(key)
        if not isinstance(value, ast.Name) or value.id not in imported:
            raise SystemExit(f"{label} is not bound to an imported wrapper class")
        names[label] = value.id
    return names


def active_models():
    wrapped = registered_wrappers()
    missing = [name for name in _REQUIRED if name not in wrapped]
    if missing:
        raise SystemExit(
            "missing registered models: " + ", ".join(missing) + ". No substitute model is added."
        )
    models = list(_REQUIRED)
    if _H2FORMER in wrapped:
        models.insert(5, _H2FORMER)
    else:
        print(
            "blocked: H2Former is not in WRAPPERS, so it stays out of the table. "
            "The other registered models are unchanged.",
            flush=True,
        )
    return models


def load_document(path):
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path} must be a mapping with jobs and smoke")
    return document


def existing_file(path_str):
    path = Path(path_str)
    if path.is_file():
        return path
    rooted = _ROOT / path_str
    if rooted.is_file():
        return rooted
    raise FileNotFoundError(path_str)


def experiment_of(job):
    experiment = job.get("experiment", "B1")
    if experiment not in _PAIRS:
        raise ValueError(f"experiment must be one of {sorted(_PAIRS)}, got {experiment!r}")
    return experiment


def run_dir_for(job, smoke):
    slug = _SLUG[job["model"]]
    dataset = str(job["dataset"]).lower()
    seed = int(job["seed"])
    experiment = experiment_of(job)
    if smoke:
        prefix = "" if experiment == "B1" else experiment.lower() + "_"
        path = Path("runs") / "smoke" / f"{prefix}{slug}_{dataset}_seed{seed}"
    elif experiment == "B1" and job["model"] == "U-Net" and job["dataset"] == "IDRiD" and seed == 0:
        path = Path("runs") / "B1_unet_idrid_seed0_runner"
    else:
        path = Path("runs") / f"{experiment}_{slug}_{dataset}_seed{seed}"
    name = path.name
    if name in _PROTECTED_EXACT or name.startswith(_PROTECTED_PREFIX):
        raise SystemExit(f"refusing run directory {path}")
    if smoke:
        if path.parts[:2] != ("runs", "smoke") or name.startswith(tuple(f"{key}_" for key in _PAIRS)):
            raise SystemExit(f"smoke run directory must stay under runs/smoke, got {path}")
    elif "smoke" in path.parts:
        raise SystemExit(f"formal run directory must not use runs/smoke, got {path}")
    return path


def check_job(job, smoke, placeholder=True):
    required = ("name", "model", "dataset", "seed", "gpu", "env", "config", "script")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"job is missing {missing}")
    if job["model"] not in _BINDING:
        raise ValueError(f"unknown model {job['model']!r}")
    env_name, script, predict_script = _BINDING[job["model"]]
    if job["env"] != env_name:
        raise ValueError(f"{job['model']} env is {job['env']!r}, expected {env_name}")
    if job["script"] != script:
        raise ValueError(f"{job['model']} script is {job['script']!r}, expected {script}")
    if not existing_file(job["script"]).is_file():
        raise FileNotFoundError(job["script"])
    if not existing_file(predict_script).is_file():
        raise FileNotFoundError(predict_script)
    config_path = existing_file(job["config"])
    config = yaml.safe_load(config_path.read_text())
    experiment = experiment_of(job)
    allowed = set(_CELL_KEYS)
    # The U-Net card leaves both epochs and iterations at 0. The B1 cell is
    # the cap. Loss, learning rate and the patience of 20 stay on the card.
    # B-std takes every budget from bs_frozen.yaml.
    if experiment == "B1" and job["model"] == "U-Net":
        allowed.add("epochs")
    extra = set(config) - allowed
    if extra:
        raise ValueError(f"{config_path} has extra keys {sorted(extra)}")
    if experiment == "B1" and job["model"] == "U-Net" and int(config.get("epochs") or 0) != 200:
        raise ValueError(f"{config_path} must set epochs: 200; the U-Net card does not set a step count")
    if config.get("experiment") != experiment:
        raise ValueError(f"{config_path} is not a {experiment} cell")
    if config["model"] != job["model"]:
        raise ValueError(f"{config_path} model is {config['model']!r}, job says {job['model']!r}")
    if Path(config["dataset"]).name != job["dataset"]:
        raise ValueError(f"{config_path} dataset does not match job dataset {job['dataset']!r}")
    if int(config["seed"]) != int(job["seed"]):
        raise ValueError(f"{config_path} seed is {config['seed']}, job says {job['seed']}")
    if int(config["fov_diameter"]) != 1440:
        raise ValueError(f"{config_path} fov_diameter must be 1440")
    for key, expected in (("split_train", "train"), ("split_val", "val"), ("split_test", "test")):
        if config[key] != expected:
            raise ValueError(f"{config_path} {key} must be {expected!r}")
    if int(config["num_workers"]) != 4:
        raise ValueError(f"{config_path} num_workers must be 4")
    gpu = int(job["gpu"])
    if gpu < 0 or (placeholder and gpu > 7):
        raise ValueError(f"gpu placeholder must be 0-7, got {gpu}")
    if smoke:
        if int(job.get("max_steps", -1)) != _SMOKE_STEPS:
            raise ValueError(f"smoke max_steps must be {_SMOKE_STEPS}")
        if job["dataset"] not in ("IDRiD", "DDR") or int(job["seed"]) != 0:
            raise ValueError("smoke is seed 0 on IDRiD and DDR")
    elif "max_steps" in job:
        raise ValueError("formal jobs do not set max_steps")
    return config, run_dir_for(job, smoke)


def select_jobs(document, smoke, models):
    key = "smoke" if smoke else "jobs"
    rows = document.get(key)
    if not rows:
        raise ValueError(f"the jobs file has no {key}")
    experiment = document.get("experiment", "B1")
    pairs = _PAIRS[experiment]
    chosen = []
    for job in rows:
        if experiment_of(job) != experiment:
            raise ValueError(f"{job['name']} is {experiment_of(job)}, the jobs file is {experiment}")
        if job["model"] == _H2FORMER and _H2FORMER not in models:
            continue
        if job["model"] not in models:
            raise ValueError(f"{job['model']} is not an active B1 model")
        chosen.append(job)
    if smoke:
        expected_smoke = {(model, dataset) for model in models for dataset in ("IDRiD", "DDR")}
        found_smoke = {(job["model"], job["dataset"]) for job in chosen}
        if found_smoke != expected_smoke:
            missing = sorted(expected_smoke - found_smoke)
            extra = sorted(found_smoke - expected_smoke)
            raise ValueError(f"smoke must list each model on IDRiD and DDR at seed 0; missing={missing} extra={extra}")
        if any(int(job["seed"]) != 0 for job in chosen):
            raise ValueError("smoke seed must be 0")
    else:
        expected = {(model, dataset, seed) for model in models for dataset, seed in pairs}
        found = {(job["model"], job["dataset"], int(job["seed"])) for job in chosen}
        if found != expected:
            missing = sorted(expected - found)
            extra = sorted(found - expected)
            raise ValueError(f"formal table mismatch missing={missing} extra={extra}")
        if len(chosen) != len(models) * len(pairs):
            raise ValueError(f"formal task count does not match registered models times {len(pairs)}")
    return chosen


def python_argv(env_name):
    """Interpreter for a conda env name, or for a path to a virtualenv."""
    candidate = Path(env_name)
    if candidate.is_dir() and (candidate / "bin" / "python").is_file():
        return [str(candidate / "bin" / "python")]
    for tool in ("micromamba", "mamba", "conda"):
        exe = shutil.which(tool)
        if exe is None:
            continue
        try:
            listed = subprocess.check_output([exe, "env", "list"], text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in listed.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.split()[0] == env_name:
                return [exe, "run", "-n", env_name, "--no-capture-output", "python"]
    for path in (
        Path.home() / ".conda" / "envs" / env_name / "bin" / "python",
        Path("/opt/conda/envs") / env_name / "bin" / "python",
    ):
        if path.is_file():
            return [str(path)]
    raise SystemExit(
        f"could not find environment {env_name!r}. Use a conda env of that name, or set env to a virtualenv path."
    )


def _chain(py_argv, segments):
    pieces = [shlex.join(list(py_argv) + list(segment)) for segment in segments]
    prefix = "unset " + " ".join(_DIST_VARS) + ";"
    return prefix + " " + " && ".join(pieces)


def command_segments(job, config, run_dir, smoke, wave="formal"):
    run_text = str(run_dir)
    if smoke:
        train = [
            "bench/scripts/launch_b1.py",
            "--capped-train",
            "--script",
            job["script"],
            "--config",
            job["config"],
            "--run-dir",
            run_text,
            "--max-steps",
            str(_SMOKE_STEPS),
        ]
    elif wave == "score-seed0":
        train = [job["script"], "--config", job["config"], "--run-dir", run_text, "--score-only"]
    elif wave == "seed0":
        train = [job["script"], "--config", job["config"], "--run-dir", run_text, "--diagnostic", "--resume"]
    else:
        train = [job["script"], "--config", job["config"], "--run-dir", run_text, "--resume"]
    # A full train writes prob_val, prob_test and the two sensitivity directories.
    # This command only evaluates the primary maps. Predicting again would
    # score every job twice. Seed 0 diagnostic does not read the test split.
    segments = [train]
    if not smoke and wave in ("formal", "rest", "score-seed0"):
        segments.append(
            [
                "-m",
                "bench.eval.cli",
                "evaluate",
                "--data",
                str(config["dataset"]),
                "--split",
                "test",
                "--pred",
                f"{run_text}/prob_test",
                "--val-pred",
                f"{run_text}/prob_val",
                "--out",
                f"{run_text}/metrics_test",
            ]
        )
    return segments


def format_task(job, run_dir):
    return (
        f"task name={job['name']} model={job['model']} dataset={job['dataset']} "
        f"seed={int(job['seed'])} gpu={int(job['gpu'])} env={job['env']} "
        f"config={job['config']} script={job['script']} run={run_dir}"
    )


def format_command(job, config, run_dir, smoke, py_argv=None, wave="formal"):
    gpu = int(job["gpu"])
    body = _chain(py_argv or ["python"], command_segments(job, config, run_dir, smoke, wave))
    return f"export CUDA_VISIBLE_DEVICES={gpu}; {body}"


def _busy_limit_mib():
    return float(os.environ.get("RETISEG_BUSY_MIB", "2048"))


def _own_process(pid):
    """True when pid belongs to this user; a process of ours of any size makes the card busy."""
    try:
        return os.stat(f"/proc/{int(pid)}").st_uid == os.getuid()
    except (OSError, ValueError):
        return False


def gpu_memory_held():
    """{GPU index: MiB held by compute processes}, or None when nvidia-smi is absent.

    A card with any process of this user counts as fully held, because a small
    job of ours (FCT holds about 1.4 GB) must not get a second job beside it.
    """
    try:
        listing = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        busy_text = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    by_uuid = {}
    for line in busy_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        uuid = parts[0]
        if not uuid or uuid.lower() == "gpu_uuid":
            continue
        try:
            mib = float(parts[2])
        except (IndexError, ValueError):
            mib = float("inf")
        if len(parts) > 1 and _own_process(parts[1]):
            mib = float("inf")
        by_uuid[uuid] = by_uuid.get(uuid, 0.0) + mib
    held = {}
    for line in listing.splitlines():
        if not line.strip():
            continue
        index, uuid = [part.strip() for part in line.split(",", 1)]
        held[int(index)] = by_uuid.get(uuid, 0.0)
    return held


def gpu_is_free(gpu):
    held = gpu_memory_held()
    if held is None:
        return True
    return held.get(int(gpu), float("inf")) < _busy_limit_mib()


def free_gpu_ids():
    """GPU indexes with no process of ours and less than RETISEG_BUSY_MIB held by others.

    The default 2048 MiB lets a card through when another user only keeps an idle
    CUDA context on it (about 0.5 GB), and still refuses cards running a training
    or a model server. Falls back to 0-7 when nvidia-smi is absent.
    """
    held = gpu_memory_held()
    if held is None:
        print("nvidia-smi unavailable; --submit cycles GPU 0-7", flush=True)
        return list(range(8))
    allowed = os.environ.get("RETISEG_GPUS", "").replace(",", " ").split()
    free = []
    for index in sorted(held):
        if allowed and str(index) not in allowed:
            continue
        if held[index] < _busy_limit_mib():
            free.append(index)
    if not free:
        raise SystemExit("no free GPU; refusing to place a second process on a busy card")
    print("free gpus: " + ",".join(str(gpu) for gpu in free), flush=True)
    return free


def _prepare_env(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for key in _DIST_VARS:
        env.pop(key, None)
    return env


def _start(command, gpu):
    print(f"start gpu {gpu}: {command}", flush=True)
    return subprocess.Popen(command, cwd=_ROOT, env=_prepare_env(gpu), shell=True, executable="/bin/bash")


# Rough A100 GPU-hours of one formal B1 job, used only to order the queue.
# Extrapolated from E1r on a 4060 Ti; a job may set est_hours to override.
_EST_HOURS = {
    ("U-Net", "IDRiD"): 0.5,
    ("U-Net", "DDR"): 2,
    ("DeepLabv3", "IDRiD"): 50,
    ("DeepLabv3", "DDR"): 55,
    ("HRNet", "IDRiD"): 1.5,
    ("HRNet", "DDR"): 12,
    ("Swin-Unet", "IDRiD"): 0.2,
    ("Swin-Unet", "DDR"): 0.5,
    ("FCT", "IDRiD"): 0.5,
    ("FCT", "DDR"): 2,
    ("H2Former", "IDRiD"): 1,
    ("H2Former", "DDR"): 3,
    ("M2MRF", "IDRiD"): 10,
    ("M2MRF", "DDR"): 20,
    ("HACDR-Net", "IDRiD"): 4,
    ("HACDR-Net", "DDR"): 5,
}


def estimated_hours(job):
    if "est_hours" in job:
        return float(job["est_hours"])
    return float(_EST_HOURS.get((job["model"], job["dataset"]), 1.0))


def longest_first(jobs):
    """Stable sort by estimated hours, longest first."""
    return sorted(jobs, key=lambda job: -estimated_hours(job))


def finished_marker(run_dir, smoke, wave):
    """File whose presence means this wave already completed for the run."""
    if smoke:
        return None
    if wave == "seed0":
        return Path(run_dir) / "diagnostic.json"
    return Path(run_dir) / "metrics_test.json"


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def in_progress(run_dir):
    """Why the last "== start" in console.log looks unfinished, or None.

    A start line with host and pid counts as running only while that process
    is alive on this host. Older start lines have no pid, and a start on
    another host cannot be checked, so both count as running until
    RETISEG_IGNORE_RUNNING=1.
    """
    log = _ROOT / run_dir / "console.log"
    if not log.is_file():
        return None
    start = None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("== start "):
            start = line
        elif line.startswith("== exit ") and start is not None:
            start = None
    if start is None or os.environ.get("RETISEG_IGNORE_RUNNING") == "1":
        return None
    fields = start.split()
    host = fields[fields.index("host") + 1] if "host" in fields[:-1] else None
    pid = fields[fields.index("pid") + 1] if "pid" in fields[:-1] else None
    if host == socket.gethostname() and pid is not None and pid.isdigit():
        return start if _pid_alive(int(pid)) else None
    return start


def run_queue(commands, gpus, start=None, wait_free=False, poll_seconds=60):
    """Run (label, command[, gate]) items on a shared queue, one process per GPU at a time.

    Each GPU thread takes the first item whose gate is "open" when its previous
    process exits, so a long job does not hold back the jobs queued behind it on
    that card. A gate returns "open", "wait" or "never"; "never" drops the item
    and records it as a failure. With wait_free, a thread also waits until its
    card is free (see gpu_is_free) before it takes an item, so cards still
    held by another launcher's jobs join the queue when those jobs end.
    Returns the list of failure messages.
    """
    start = start or _start
    pending = list(commands)
    lock = threading.Lock()
    errors = []

    def _take():
        with lock:
            for item in list(pending):
                gate = item[2] if len(item) > 2 else None
                state = gate() if gate is not None else "open"
                if state == "never":
                    pending.remove(item)
                    errors.append(f"not started: {item[0]}: its gate closed (seed 0 missing, failed or stale)")
                elif state == "open":
                    pending.remove(item)
                    return item, False
            return None, not pending

    def _worker(gpu):
        while True:
            with lock:
                if not pending:
                    return
            if wait_free and not gpu_is_free(gpu):
                time.sleep(poll_seconds)
                continue
            item, done = _take()
            if done:
                return
            if item is None:
                time.sleep(poll_seconds)
                continue
            label, command = item[0], item[1]
            process = start(command, gpu)
            code = process.wait()
            if code != 0:
                with lock:
                    errors.append(f"gpu {gpu} exited {code}: {label}: {command}")

    threads = [threading.Thread(target=_worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def submit(jobs, smoke, wave="formal"):
    missing = [
        str(path)
        for path in (_ROOT / "dataset" / "prepared" / "IDRiD", _ROOT / "dataset" / "prepared" / "DDR")
        if not path.is_dir()
    ]
    if missing:
        raise SystemExit("prepared dataset missing: " + ", ".join(missing))
    free = free_gpu_ids()
    commands = []
    for job in longest_first(jobs):
        copied = dict(job)
        copied["gpu"] = free[0]
        config, run_dir = check_job(copied, smoke, placeholder=False)
        marker = finished_marker(run_dir, smoke, wave)
        if marker is not None and (_ROOT / marker).is_file():
            print(f"skip {job['name']}: {marker} exists", flush=True)
            continue
        running = in_progress(run_dir)
        if running is not None:
            print(f"skip {job['name']}: still running ({running})", flush=True)
            continue
        # CUDA_VISIBLE_DEVICES comes from the environment of the worker that runs it.
        body = _chain(python_argv(job["env"]), command_segments(copied, config, run_dir, smoke, wave))
        log = shlex.quote(str(Path(run_dir) / "console.log"))
        body = (
            f"mkdir -p {shlex.quote(str(run_dir))} && "
            f"{{ echo \"== start {wave} $(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) pid $$\"; {body}; rc=$?; "
            f"echo \"== exit $rc $(date -Is)\"; exit $rc; }} >> {log} 2>&1"
        )
        commands.append((job["name"], body))
        print(f"queued {job['name']} est {estimated_hours(job):g} h log {run_dir}/console.log", flush=True)
    errors = run_queue(commands, free)
    if errors:
        raise SystemExit("\n".join(errors))


def last_exit_code(run_dir):
    """Exit code of the last finished start in console.log; None if never started or still open."""
    log = _ROOT / run_dir / "console.log"
    if not log.is_file():
        return None
    code = None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("== start "):
            code = None
        elif line.startswith("== exit "):
            fields = line.split()
            code = int(fields[2]) if len(fields) > 2 and fields[2].lstrip("-").isdigit() else -1
    return code


def seed0_gate(seed0_jobs):
    """Gate that opens once every listed seed-0 job has a current, finished diagnostic."""
    state = {"open": False}

    def gate():
        if state["open"]:
            return "open"
        for job in seed0_jobs:
            _config, run_dir = check_job(job, False)
            if (_ROOT / run_dir / "diagnostic.json").is_file():
                continue
            code = last_exit_code(run_dir)
            return "never" if code not in (None, 0) else "wait"
        try:
            assert_seed0_current(seed0_jobs)
        except SystemExit as exc:
            print(f"gate closed: {exc}", flush=True)
            return "never"
        state["open"] = True
        return "open"

    return gate


def _queue_item(job, wave, gate=None):
    config, run_dir = check_job(job, False, placeholder=False)
    body = _chain(python_argv(job["env"]), command_segments(job, config, run_dir, False, wave))
    log = shlex.quote(str(Path(run_dir) / "console.log"))
    body = (
        f"mkdir -p {shlex.quote(str(run_dir))} && "
        f"{{ echo \"== start {wave} $(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) pid $$\"; {body}; rc=$?; "
        f"echo \"== exit $rc $(date -Is)\"; exit $rc; }} >> {log} 2>&1"
    )
    return (job["name"], body, gate) if gate is not None else (job["name"], body)


def queue_all(formal, gpus):
    """One queue for seed 0, the seed-0 test maps and seeds 1+, on fixed cards.

    Seed 0 goes first. The test maps of seed 0 and the other seeds of one model
    on one dataset wait until that model's seed-0 job on that dataset has a
    finished diagnostic on the current budget, so one slow model does not hold
    back the others. Jobs already finished or still running are skipped.
    """
    missing = [
        str(path)
        for path in (_ROOT / "dataset" / "prepared" / "IDRiD", _ROOT / "dataset" / "prepared" / "DDR")
        if not path.is_dir()
    ]
    if missing:
        raise SystemExit("prepared dataset missing: " + ", ".join(missing))
    seed0 = [job for job in formal if int(job["seed"]) == 0]
    gates = {(job["model"], job["dataset"]): seed0_gate([job]) for job in seed0}
    plan = []
    for wave, jobs in (
        ("seed0", longest_first(seed0)),
        ("score-seed0", longest_first(seed0)),
        ("rest", longest_first([job for job in formal if int(job["seed"]) != 0])),
    ):
        for job in jobs:
            placed = dict(job)
            placed["gpu"] = gpus[0]
            _config, run_dir = check_job(placed, False, placeholder=False)
            marker = finished_marker(run_dir, False, wave)
            if (_ROOT / marker).is_file():
                print(f"skip {wave} {job['name']}: {marker} exists", flush=True)
                continue
            running = in_progress(run_dir)
            # A seed-0 training still running leaves its score item queued behind the gate;
            # a score already running is skipped like any other running job.
            if running is not None and (wave != "score-seed0" or running.startswith("== start score-seed0 ")):
                print(f"skip {wave} {job['name']}: still running ({running})", flush=True)
                continue
            key = (job["model"], job["dataset"])
            gate = None if wave == "seed0" else gates.get(key)
            if wave != "seed0" and gate is None:
                raise SystemExit(f"{job['name']}: no seed-0 job of {key[0]} on {key[1]} in this queue")
            plan.append(_queue_item(placed, wave, gate))
            print(f"queued {wave} {job['name']} est {estimated_hours(job):g} h", flush=True)
    print(f"queue: {len(plan)} items on gpus {','.join(str(gpu) for gpu in gpus)}", flush=True)
    errors = run_queue(plan, gpus, wait_free=True)
    if errors:
        raise SystemExit("\n".join(errors))


def _load_train_module(script):
    path = existing_file(script)
    spec = importlib.util.spec_from_file_location("retiseg_b1_train_entry", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cap_resolve(original):
    def wrapped(recipe, config, n_train):
        resolved, deviations, per_epoch = original(recipe, config, n_train)
        if int(resolved.iterations) > _SMOKE_STEPS:
            deviations = list(deviations)
            deviations.append(
                f"smoke stops after {_SMOKE_STEPS} optimizer steps; "
                f"the resolved step count {int(resolved.iterations)} is not run"
            )
            resolved = replace(resolved, iterations=_SMOKE_STEPS)
        return resolved, deviations, per_epoch

    return wrapped


def capped_train(script, config, run_dir, max_steps):
    """Train one smoke job and stop after 20 optimizer steps.

    Formal recipes are unchanged. The cap is applied only in this process,
    after the author recipe has been resolved, and only for a runs/smoke directory.
    """
    if int(max_steps) != _SMOKE_STEPS:
        raise SystemExit(f"smoke max steps must be {_SMOKE_STEPS}")
    path = Path(run_dir)
    if path.parts[:2] != ("runs", "smoke"):
        raise SystemExit(f"capped train only writes under runs/smoke, got {run_dir}")
    os.chdir(_ROOT)
    for key in _DIST_VARS:
        os.environ.pop(key, None)
    module = _load_train_module(script)
    if not hasattr(module, "resolve_recipe") or not hasattr(module, "train"):
        raise SystemExit(f"{script} does not expose resolve_recipe and train")
    module.resolve_recipe = _cap_resolve(module.resolve_recipe)
    import bench.bstd

    bench.bstd.MAX_STEPS = _SMOKE_STEPS
    module.train(config, run_dir, diagnostic=True)


def assert_seed0_current(seed0_jobs):
    """Refuse later seeds when seed 0 is missing or was trained on another budget."""
    import json

    from bench.data.b1_input import EXPECTED_COUNTS
    from bench.models.registry import build_model
    from bench.runtime import budget_signature, micro_batch_and_deviations, optimizer_step_sizes, resolve_recipe

    for job in seed0_jobs:
        config, run_dir = check_job(job, False)
        directory = _ROOT / run_dir
        recipe_path = directory / "recipe.json"
        history_path = directory / "history.jsonl"
        diagnostic_path = directory / "diagnostic.json"
        if not recipe_path.is_file() or not history_path.is_file() or not diagnostic_path.is_file():
            raise SystemExit(
                f"{run_dir} has no finished seed-0 diagnostic. "
                "Run --submit --seed0 and read training and validation before the other seeds."
            )
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic.get("test_split_read") is not False:
            raise SystemExit(f"{run_dir} read the test split during the diagnostic")
        saved = json.loads(recipe_path.read_text())
        model = build_model(job["model"])
        n_train = EXPECTED_COUNTS[job["dataset"]]["train"]
        declared = model.author_recipe(job["dataset"])
        if experiment_of(job) == "BS":
            from bench import bstd
            from bench.common.io import load_split

            frozen = bstd.load_frozen()
            patch, _window = bstd.geometry(model.card, frozen)
            dataset_dir = _ROOT / config["dataset"]
            ids = load_split(dataset_dir, config["split_train"])
            pixels = bstd.canvas_pixels(dataset_dir, ids, int(config["fov_diameter"]))
            resolved, _deviations, _per_epoch, n_train, _eval_every = bstd.budget(
                declared, job["dataset"], frozen, pixels, patch
            )
        else:
            resolved, _deviations, _per_epoch = resolve_recipe(declared, config, n_train)
        micro, _batch_deviations = micro_batch_and_deviations(config, resolved)
        fresh = {
            "iterations": int(resolved.iterations),
            "epochs": int(resolved.epochs),
            "loss": str(resolved.loss),
            "optimizer": str(resolved.optimizer),
            "lr": float(resolved.lr),
            "weight_decay": float(resolved.weight_decay),
            "schedule": str(resolved.schedule),
            "batch_size": int(micro),
            "nominal_effective_batch_size": int(resolved.effective_batch_size),
            "step_image_counts": optimizer_step_sizes(n_train, micro, resolved.effective_batch_size),
        }
        try:
            current = budget_signature(saved)
        except KeyError as exc:
            raise SystemExit(f"{run_dir} {exc}. Rerun seed 0.") from exc
        if current != budget_signature(fresh):
            raise SystemExit(
                f"{run_dir} was trained under a different budget. Rerun seed 0 before the other seeds."
            )
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        if not rows:
            raise SystemExit(f"{run_dir} has an empty training history")
        early_path = directory / "early_stop.json"
        stopped = early_path.is_file() and bool(json.loads(early_path.read_text()).get("stopped_early"))
        if not stopped and int(rows[-1]["iteration"]) != int(fresh["iterations"]):
            raise SystemExit(
                f"{run_dir} stopped at iteration {rows[-1]['iteration']}, "
                f"budget is {fresh['iterations']}. Rerun seed 0."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print B1 jobs, or start one wave with --submit.")
    parser.add_argument("--jobs", default=str(_ROOT / "bench" / "configs" / "b1_jobs.yaml"))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed0", action="store_true")
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--score-seed0", action="store_true")
    parser.add_argument("--capped-train", action="store_true")
    parser.add_argument("--script", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=_SMOKE_STEPS)
    parser.add_argument("--only", default=None, help="comma-separated model names; other jobs of the wave are left out")
    parser.add_argument("--datasets", default=None, help="comma-separated datasets, e.g. IDRiD; other jobs of the wave are left out")
    parser.add_argument(
        "--queue-all",
        action="store_true",
        help="with --submit: seed 0, its test maps, then the other seeds, on the cards in RETISEG_GPUS",
    )
    args = parser.parse_args(argv)
    if args.capped_train:
        if not args.script or not args.config or not args.run_dir:
            raise SystemExit("--capped-train needs --script, --config and --run-dir")
        capped_train(args.script, args.config, args.run_dir, args.max_steps)
        return
    selected = []
    if args.smoke:
        selected.append("smoke")
    if args.seed0:
        selected.append("seed0")
    if args.rest:
        selected.append("rest")
    if args.score_seed0:
        selected.append("score-seed0")
    if len(selected) > 1:
        raise SystemExit("pass only one of --smoke, --seed0, --rest, --score-seed0")
    if args.queue_all and selected:
        raise SystemExit("--queue-all is its own wave; do not combine it with --smoke, --seed0, --rest or --score-seed0")
    models = active_models()
    document = load_document(args.jobs)
    formal = select_jobs(document, False, models)
    if args.submit and not selected and not args.queue_all:
        raise SystemExit(
            f"refusing to submit all {len(formal)} formal jobs at once. "
            "The current batch is seed 0. "
            "Run --submit --smoke, then --submit --seed0, read training and validation only, "
            "then --submit --score-seed0. Do not use --rest in this batch. "
            "If the budget changes, rerun seed 0."
        )
    wave = selected[0] if selected else "formal"
    if wave == "smoke":
        jobs = select_jobs(document, True, models)
    elif wave in ("seed0", "score-seed0"):
        jobs = [job for job in formal if int(job["seed"]) == 0]
    elif wave == "rest":
        jobs = [job for job in formal if int(job["seed"]) != 0]
    else:
        jobs = formal
    if args.only:
        only = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = sorted(only - set(models))
        if unknown:
            raise SystemExit(f"--only names unknown models: {unknown}")
        jobs = [job for job in jobs if job["model"] in only]
    if args.datasets:
        wanted = {name.strip() for name in args.datasets.split(",") if name.strip()}
        known = {str(job["dataset"]) for job in formal}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"--datasets names unknown datasets: {unknown}; known: {sorted(known)}")
        jobs = [job for job in jobs if str(job["dataset"]) in wanted]
    if args.queue_all:
        gpus = [int(gpu) for gpu in os.environ.get("RETISEG_GPUS", "").replace(",", " ").split()]
        if not gpus:
            raise SystemExit("--queue-all needs RETISEG_GPUS, the cards this queue may use")
        print(f"queue-all over {len(jobs)} registered jobs", flush=True)
        if not args.submit:
            print("not submitted; pass --submit to start the queue", flush=True)
            return
        queue_all(jobs, gpus)
        return
    if args.submit and wave in ("rest", "score-seed0"):
        assert_seed0_current([job for job in formal if int(job["seed"]) == 0])
    for job in jobs:
        config, run_dir = check_job(job, wave == "smoke")
        print(format_task(job, run_dir), flush=True)
        print("command " + format_command(job, config, run_dir, wave == "smoke", wave=wave), flush=True)
    print(f"{wave} tasks: {len(jobs)}", flush=True)
    if not args.submit:
        if wave == "formal":
            print(
                f"not submitted. Do not launch all {len(formal)} at once; "
                "the current batch is seed 0: "
                "--submit --smoke, then --submit --seed0, then --submit --score-seed0. "
                "Do not use --rest in this batch.",
                flush=True,
            )
        else:
            print("not submitted; pass --submit to launch this wave", flush=True)
        return
    submit(jobs, wave == "smoke", wave)


if __name__ == "__main__":
    main()
