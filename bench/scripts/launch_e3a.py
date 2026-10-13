"""Queue E3a (independent tests) for every finished source on the cards in RETISEG_GPUS.

One queue item is one source on one test set (with its variants), so the
cards share a slow source. Sources, in this order:
  1. seed 0 of B-std, both nnU-Net configurations, the M2MRF author weights;
  2. B1 seed 0 runs that are scored;
  3. B-std seeds 1 and 2 that are scored.
Inside a tier the items with the most canvas pixels go first. A B-std or B1
run is a source once its metrics_test.json exists (it has its test maps,
prob_val and checkpoint_roles.json). An item whose sets all have done.json is
skipped, and so is one whose console.log shows a live process. One process
per card; a card with any process of this user, or another user's processes
holding RETISEG_BUSY_MIB or more, is waited for (bench/scripts/launch_b1.py
run_queue). Each item writes runs/E3a/<source>/<set>/console.log.

    python bench/scripts/launch_e3a.py                      # print the plan
    RETISEG_GPUS="2 3" python bench/scripts/launch_e3a.py --submit [--seed0-only] [--only BS_hrnet_idrid_seed0,...]
"""

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from bench import e3a
from bench.scripts.launch_b1 import _BINDING, _DIST_VARS, in_progress, python_argv, run_queue

NNUNET_ENV = "retiseg-nnunet"
NNUNET_PLANS = ("default", "resencm")
_SKIP_EXACT = {"B1_unet_idrid_seed0"}
# Formal B-std and B1 run names; a directory renamed aside (e.g. ..._registered) is not a source.
RUN_NAME = re.compile(r"(BS|B1)_[a-z0-9_]+_(idrid|ddr)_seed\d+(_runner)?")
# Factor runs that get independent tests: E3b (other training sources), E4 (merged labels), E8 (amount).
# E7 changes the canvas and E9 is read on DDR only (docs/paper/16).
FACTOR_RUN_NAME = re.compile(r"BF_e(3b|4|8)_[a-z0-9_]+_seed\d+")


def _scored_run(run):
    return all((run / name).is_file() for name in ("config.yaml", "checkpoint_roles.json", "metrics_test.json"))


def discover(seed0_only=False):
    """(tier, source name, env, argv, train sources) for every source that can run."""
    frozen = e3a.load_frozen()
    runs = _ROOT / "runs"
    found = []
    for prefix in ("BS_", "B1_"):
        for run in sorted(runs.glob(f"{prefix}*")):
            if run.name in _SKIP_EXACT or not RUN_NAME.fullmatch(run.name) or not run.is_dir() or not _scored_run(run):
                continue
            config = yaml.safe_load((run / "config.yaml").read_text())
            if config.get("diagnostic_loss_params") or config.get("budget_probe"):
                continue
            seed = int(config["seed"])
            if seed0_only and seed != 0:
                continue
            tier = 1 if prefix == "BS_" and seed == 0 else (2 if prefix == "B1_" else 3)
            env = _BINDING[config["model"]][0]
            argv = ["-m", "bench.e3a", "run", "--run", f"runs/{run.name}"]
            found.append((tier, run.name, env, argv, [e3a.train_source_of(config)]))
    for run in sorted(runs.glob("BF_*")):
        if not FACTOR_RUN_NAME.fullmatch(run.name) or not run.is_dir() or not _scored_run(run):
            continue
        config = yaml.safe_load((run / "config.yaml").read_text())
        if int(config["fov_diameter"]) != int(frozen["canvas_reference"]):
            continue
        env = _BINDING[config["model"]][0]
        argv = ["-m", "bench.e3a", "run", "--run", f"runs/{run.name}"]
        found.append((2, run.name, env, argv, e3a.train_sources_of(config)))
    from bench.scripts.nnunet_bridge import DATASET_IDS

    for train, dataset_id in DATASET_IDS.items():
        for plan in NNUNET_PLANS:
            name = f"NN_{plan}_{train.lower()}_fold0"
            # the raw folder is Dataset<id>_<meta name>; the meta name equals the test-set name for the sources used so far
            folders = list((runs / "nnunet").glob(f"Dataset{dataset_id:03d}_*")) if (runs / "nnunet").is_dir() else []
            ckpts = [
                folder / f"nnUNetTrainer__{'nnUNetPlans' if plan == 'default' else 'nnUNetResEncUNetMPlans'}__2d" / "fold_0" / "checkpoint_final.pth"
                for folder in folders
            ]
            if (runs / name / "metrics_test.json").is_file() and any(ckpt.is_file() for ckpt in ckpts):
                argv = ["bench/scripts/nnunet_e3a.py", "--train", train, "--plan", plan]
                tier = 1 if train in ("IDRiD", "DDR") else 2
                found.append((tier, name, NNUNET_ENV, argv, [train]))
    for model, by_train in frozen.get("author_weights", {}).items():
        for train, spec in by_train.items():
            if (_ROOT / spec["checkpoint"]).is_file():
                argv = ["-m", "bench.e3a", "author", "--model", model, "--train", train]
                found.append((1, e3a.author_source(model, train), "retiseg", argv, list(spec["trained_on"])))
    return sorted(found, key=lambda item: (item[0], item[1]))


def pending_sets(source, trained_on, author=False):
    """Sets a source still has to run. A BF source whose training field of view is not frozen
    (TJDR_std, FGADR, Retinal-Lesions) skips the per-degree variant of the ultra-widefield set, as bench.e3a does."""
    frozen = e3a.load_frozen()
    no_degrees = (source.startswith("BF_") and e3a.scale_source_of(yaml.safe_load((_ROOT / "runs" / source / "config.yaml").read_text())) is None) or (
        source.startswith("NN_") and any(name not in frozen["train_field_degrees"] for name in trained_on)
    )
    return [
        (test, variant)
        for test, variant in e3a.targets(trained_on, frozen)
        if not (author and variant != e3a.MAIN)
        and not (no_degrees and test in frozen.get("per_degree_sets", ()) and variant == e3a.MAIN)
        and not e3a.is_done(source, test, variant)
    ]


# Rough relative cost of one canvas pixel, to order the queue only.
_COST = {"nnunet": 4.0, "run": 1.0, "author": 0.5}


def item_cost(kind, source, test, train_names, frozen):
    """Images times canvas pixels of every pending variant of ``test`` (queue order only)."""
    from bench.common.io import load_split

    n = len(load_split(_ROOT / frozen["test_sets"][test]["data"], frozen["split"])) if (
        _ROOT / frozen["test_sets"][test]["data"]
    ).is_dir() else 100
    reference = {"IDRiD": 3413, "DDR": 1658}.get(train_names[0], 1440) if kind == "nnunet" else frozen["canvas_reference"]
    total = 0.0
    for name, variant in e3a.targets(train_names, frozen, [test]):
        if kind == "author" and variant != e3a.MAIN:
            continue
        if name in frozen.get("per_degree_sets", ()) and variant == e3a.MAIN and train_names[0] not in frozen["train_field_degrees"]:
            continue   # no per-degree scale for a source without a frozen field of view (E3b sources)
        side = e3a.target_long_side(name, variant, train_names[0], reference, frozen)
        total += n * (side / 1440.0) ** 2
    return _COST[kind] * total


def command(source, env, argv, test=None, py_argv=None, drop_maps=()):
    """Shell body of one queue item. ``py_argv`` None resolves the conda env (only when submitting)."""
    log_dir = Path("runs") / "E3a" / source
    if test is not None:
        log_dir = log_dir / test
        argv = list(argv) + ["--sets", test]
        if test in drop_maps:
            argv += ["--drop-maps", test]
    py_argv = python_argv(env) if py_argv is None else py_argv
    chain = "unset " + " ".join(_DIST_VARS) + "; " + shlex.join(list(py_argv) + list(argv))
    log = shlex.quote(str(log_dir / "console.log"))
    return (
        f"mkdir -p {shlex.quote(str(log_dir))} && "
        f"{{ echo \"== start e3a $(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) pid $$\"; {chain}; rc=$?; "
        f"echo \"== exit $rc $(date -Is)\"; exit $rc; }} >> {log} 2>&1"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print the E3a queue, or start it with --submit.")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--seed0-only", action="store_true", help="leave out B-std seeds 1 and 2")
    parser.add_argument("--only", help="comma-separated source names")
    parser.add_argument(
        "--drop-maps",
        default="",
        help="comma-separated test sets whose maps are removed after evaluation (e.g. FGADR,Retinal-Lesions)",
    )
    args = parser.parse_args(argv)
    drop = tuple(item.strip() for item in args.drop_maps.split(",") if item.strip())
    only = {item.strip() for item in (args.only or "").split(",") if item.strip()}
    items = plan(args.seed0_only, only)
    for tier, negative_cost, label, spec in items:
        print(f"tier {tier} {label} env {spec[1]} cost {-negative_cost:.0f}", flush=True)
    print(f"E3a queue: {len(items)} items", flush=True)
    if not args.submit:
        for _tier, _cost, _label, spec in items[:3]:
            print("command " + command(*spec, py_argv=["python"], drop_maps=drop), flush=True)
        print("not submitted; pass --submit (with RETISEG_GPUS) to start", flush=True)
        return
    plan_items = [(label, command(*spec, drop_maps=drop)) for _tier, _cost, label, spec in items]
    gpus = [int(gpu) for gpu in os.environ.get("RETISEG_GPUS", "").replace(",", " ").split()]
    if not gpus:
        raise SystemExit("--submit needs RETISEG_GPUS, the cards this queue may use")
    missing = [name for name, spec in e3a.load_frozen()["test_sets"].items() if not (_ROOT / spec["data"]).is_dir()]
    if missing:
        raise SystemExit("prepared test sets missing on this machine: " + ", ".join(missing))
    errors = run_queue(plan_items, gpus, wait_free=True)
    if errors:
        raise SystemExit("\n".join(errors))


def plan(seed0_only=False, only=()):
    """Pending (tier, -cost, label, (source, env, argv, test)) items, ordered; launch_queue reads this too."""
    only = set(only or ())
    frozen = e3a.load_frozen()
    items = []
    for tier, source, env, run_argv, trained_on in discover(seed0_only):
        if only and source not in only:
            continue
        kind = "nnunet" if source.startswith("NN_") else ("author" if source.startswith("E1_") else "run")
        todo = pending_sets(source, trained_on, author=kind == "author")
        if not todo:
            print(f"skip {source}: every set is done", flush=True)
            continue
        for test in sorted({name for name, _variant in todo}):
            running = in_progress(Path("runs") / "E3a" / source / test)
            if running is not None:
                print(f"skip {source} {test}: still running ({running})", flush=True)
                continue
            cost = item_cost(kind, source, test, trained_on, frozen)
            items.append((tier, -cost, f"{source}:{test}", (source, env, run_argv, test)))
    items.sort(key=lambda item: (item[0], item[1]))
    return items


if __name__ == "__main__":
    main()
