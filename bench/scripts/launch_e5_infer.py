"""Queue the E5 inference tiers (a zero-shot, b oracle prompts) on the cards in RETISEG_GPUS.

Items, in order:
  1. pool runs, the val split of IDRiD and DDR (decision D2): every (model, density) of the
     zero-shot grid, every SAM 3 phrase index, every (model, kind) of the oracle tier; they
     feed the two selections in e5_frozen.yaml and the Dice thresholds;
  2. test runs on the seven sets (TJDR_uwf main and fov): SAM / SAM 2 only once
     zero_shot.grid.selected is filled, SAM 3 only once zero_shot.sam3.selected is filled
     (then a gated assembly of a_sam3_selected per set: the phrase runs take
     --no-assemble and keep their maps, the assembly copies them and, on a --drop-maps
     set, removes them once it is evaluated), the oracle tier and the
     same-information references always; larger canvases first. A test item waits
     (gate) for the pool directories of its own source, which give its Dice thresholds;
     a gate closes ("never") when nothing in the queue will write them.
Each item is one process on one card running python -m bench.e5.<module>; SAM 3 runs in
retiseg-sam3 (RETISEG_SAM3_ENV overrides), everything else in retiseg. The oracle prompt
files of every (set, split, variant) are written first, in this process (CPU). An item
whose directories all have done.json is skipped (a smoke run's done.json, written with
--limit, does not count for a full queue), and so is one whose console.log shows a live
process (bench/scripts/launch_b1.py in_progress). Items write their log to the first of
their directories. Only the prepared sets and prompt files of the planned items are
required, so a --pool-only smoke runs on a machine that holds IDRiD and DDR alone.

    python bench/scripts/launch_e5_infer.py                                   # print the plan
    RETISEG_GPUS="2 3" python bench/scripts/launch_e5_infer.py --submit \\
        [--pool-only | --test-only] [--only a_sam_vith_p64,b_sam2_box] [--drop-maps FGADR,Retinal-Lesions] [--limit 2]
"""

import argparse
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.common.io import load_split
from bench.e5 import amg, prompted, prompts, same_info, sam3_text
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.e5 import outputs as out
from bench.scripts.launch_b1 import _DIST_VARS, in_progress, python_argv, run_queue

ENV = {"SAM": "retiseg", "SAM 2": "retiseg", "MedSAM": "retiseg", "SAM 3": os.environ.get("RETISEG_SAM3_ENV", "retiseg-sam3"), "ref": "retiseg"}
# relative cost per canvas pixel, queue order only: ViT-H / Hiera-L / SAM 3 generators 1, the ViT-B
# oracle decoders and MedSAM 0.3, the CPU reference 0.05
COST_A = {"SAM": 1.0, "SAM 2": 1.0, "SAM 3": 1.0}
COST_B = {"SAM": 0.3, "SAM 2": 1.0, "MedSAM": 0.3}
COST_REF = 0.05
# the same-information references of docs/paper/16 on the two internal test sets (maps on the server)
REFERENCES = (
    ("M2MRF", ("{src}", "seed0"), "runs/BS_m2mrf_{src}_seed0/prob_test"),
    ("nnU-Net", ("default", "{src}"), "runs/NN_default_{src}_fold0/prob_test"),
)
POOL, TEST = 1, 2


@dataclass
class Item:
    label: str
    phase: int
    cost: float
    env: str
    argv: list
    log_dir: Path
    dirs: list
    set_name: str
    gate_dirs: list = field(default_factory=list)  # an item waits until these have done.json
    split: str = "test"
    variant: str = geo.MAIN
    needs_prompts: bool = False                    # the oracle tier and the references read the prompt file of (set, split, variant)
    prerequisite: bool = False                     # a pool-phase item the test runs need (the pool assembly of a_sam3_selected): kept by --test-only
    owns_drop: bool = True                         # False: another item reads this item's maps and removes them (a SAM 3 phrase run; the assembly drops)

    def done(self, limit=None):
        """All directories evaluated for this request; a smoke run's done.json does not answer a full request (outputs.is_done)."""
        return bool(self.dirs) and all(out.is_done(path, limit) for path in self.dirs)

    def running(self):
        return in_progress(self.log_dir)  # bench.scripts.launch_b1.in_progress keeps an absolute path as is


def _log_dir(paths):
    """The main directory of an item (the fewest variant tokens): where its console.log goes."""
    return min(paths, key=lambda path: (len(path.name), path.name))


def _filled(reader, frozen):
    try:
        reader(frozen)
    except ValueError:
        return False
    return True


def n_images(set_name, split):
    """Images of a split, or 100 where the prepared set is not on this machine (queue order only)."""
    try:
        return len(load_split(geo.data_dir(set_name), split))
    except FileNotFoundError:
        return 100


def cost(weight, set_name, split, variant, frozen):
    side = geo.long_side_for(set_name, variant, frozen)
    return weight * n_images(set_name, split) * (side / 1440.0) ** 2


def targets(frozen):
    """[(set, split, variant)]: the pool runs, then every test set with its canvas variants."""
    pool = [(name, split, geo.MAIN) for name, _dir, split in fz.pool_sets(frozen)]
    test = [(name, "test", variant) for name in geo.test_sets() for variant in geo.variants_for(name, frozen)]
    return pool + test


def discover(frozen):
    """Every item the current protocol allows; done and running items are filtered by the caller.

    A test item whose evaluation takes its Dice thresholds from the pool (D2) is
    gated on the pool directories of its own source, so it waits for them instead of
    being evaluated without (review fix 3, A3). The same-information references use
    the fixed threshold only and have no gate.
    """
    items = []
    pool_sets = [(name, split) for name, _dir, split in fz.pool_sets(frozen)]
    grid_selected = _filled(fz.selected_pair, frozen)
    sam3_selected = _filled(fz.selected_phrases, frozen)
    densities = [int(d) for d in frozen["zero_shot"]["grid"]["points_per_side"]]
    for set_name, split, variant in targets(frozen):
        pool_run = fz.is_pool_split(split)
        phase = POOL if pool_run else TEST
        tag = f"{set_name}:{split}:{variant}"
        # a: automatic masks
        if pool_run or grid_selected:
            for model in amg.MODELS:
                for density in densities if pool_run else [fz.selected_pair(frozen)[0]]:
                    if amg.budget_rule_skips(frozen, split, density, set_name, variant):
                        continue
                    source, _directories, dirs = amg.planned(frozen, model, density, set_name, split, variant)
                    argv = ["-m", "bench.e5.amg", "--model", model, "--density", str(density), "--set", set_name, "--split", split, "--variant", variant]
                    paths = list(dirs.values())
                    gates = [] if pool_run else [path for name, sp in pool_sets for path in amg.planned(frozen, model, density, name, sp)[2].values()]
                    items.append(Item(f"{source}:{tag}", phase, cost(COST_A[model], set_name, split, variant, frozen), ENV[model], argv, _log_dir(paths), paths, set_name,
                                      gate_dirs=gates, split=split, variant=variant))
        # a: SAM 3 phrases, then the gated assembly of the selected phrases
        if pool_run or sam3_selected:
            phrase_dirs = []
            for index in sam3_text.PHRASE_INDICES:
                source, directory = sam3_text.planned(frozen, index, set_name, split, variant)
                # one writer per directory: the phrase run leaves the assembly, and the removal of its maps, to the gated item
                argv = ["-m", "bench.e5.sam3_text", "--phrase", str(index), "--set", set_name, "--split", split, "--variant", variant, "--no-assemble"]
                gates = [] if pool_run else [sam3_text.planned(frozen, index, name, sp)[1] for name, sp in pool_sets]
                items.append(Item(f"{source}:{tag}", phase, cost(COST_A["SAM 3"], set_name, split, variant, frozen), ENV["SAM 3"], argv, directory, [directory], set_name,
                                  gate_dirs=gates, split=split, variant=variant, owns_drop=False))
                phrase_dirs.append(directory)
            if sam3_selected:
                selected = sam3_text.planned_selected(set_name, split, variant)
                argv = ["-m", "bench.e5.sam3_text", "--assemble", "--set", set_name, "--split", split, "--variant", variant]
                # a test assembly also waits for the pool assembly, which gives its thresholds; the pool assembly is a prerequisite of the test phase
                gates = phrase_dirs + ([] if pool_run else [sam3_text.planned_selected(name, sp) for name, sp in pool_sets])
                items.append(Item(f"{sam3_text.selected_source()}:{tag}", phase, 0.0, ENV["ref"], argv, selected, [selected], set_name, gate_dirs=gates,
                                  split=split, variant=variant, prerequisite=pool_run))
        # b: oracle prompts
        for model, kinds in fz.oracle_models(frozen).items():
            for kind in kinds:
                source, dirs = prompted.planned(frozen, model, kind, set_name, split, variant)
                argv = ["-m", "bench.e5.prompted", "--model", model, "--kind", kind, "--set", set_name, "--split", split, "--variant", variant]
                paths = list(dirs.values())
                gates = [] if pool_run else [path for name, sp in pool_sets for path in prompted.planned(frozen, model, kind, name, sp)[1].values()]
                items.append(Item(f"{source}:{tag}", phase, cost(COST_B[model], set_name, split, variant, frozen), ENV[model], argv, _log_dir(paths), paths, set_name,
                                  gate_dirs=gates, split=split, variant=variant, needs_prompts=True))
        # b: same-information references, internal test sets only, where the reference maps exist
        if not pool_run and set_name in ("IDRiD", "DDR"):
            src = set_name.lower()
            for model, tokens, pred in REFERENCES:
                pred_dir = _ROOT / pred.format(src=src)
                if not pred_dir.is_dir():
                    continue
                tokens = [token.format(src=src) for token in tokens]
                source, dirs = same_info.planned(frozen, model, tokens, set_name, split, variant)
                argv = ["-m", "bench.e5.same_info", "--model", model, "--tokens", *tokens, "--pred-dir", pred.format(src=src), "--set", set_name, "--split", split, "--variant", variant]
                paths = list(dirs.values())
                items.append(Item(f"{source}:{tag}", phase, cost(COST_REF, set_name, split, variant, frozen), ENV["ref"], argv, _log_dir(paths), paths, set_name,
                                  split=split, variant=variant, needs_prompts=True))
    return items


def missing_prompt_files(frozen, items=None):
    """(set, split, variant) whose oracle prompt file does not exist yet: of every target, or only of the
    planned ``items`` that read one (the oracle tier and the references), so a plan limited by --only,
    --pool-only or the selections asks for nothing beyond its own runs."""
    wanted = targets(frozen) if items is None else sorted({(it.set_name, it.split, it.variant) for it in items if it.needs_prompts})
    return [(name, split, variant) for name, split, variant in wanted if not fz.prompt_file(name, split, variant=variant).is_file()]


def absent_sets(items):
    """Planned sets whose prepared directory is missing on this machine (review fix 2: a --pool-only smoke on a
    machine holding IDRiD and DDR only must not be stopped by the five sets it does not run)."""
    needed = {it.set_name for it in items}
    return [name for name in geo.test_sets() if name in needed and not geo.data_dir(name).is_dir()]


def write_prompt_files(frozen, missing):
    for name, split, variant in missing:
        path = prompts.build(name, split, variant, frozen=frozen)
        print(f"prompts {path}", flush=True)


def command(item, py_argv=None, drop_maps=(), limit=None):
    """Shell body of one queue item. ``py_argv`` None resolves the conda env (only when submitting)."""
    argv = list(item.argv)
    if item.phase == TEST and item.owns_drop and item.set_name in set(drop_maps):
        argv += ["--drop-maps", item.set_name]
    if limit is not None:
        argv += ["--limit", str(int(limit))]
    py_argv = python_argv(item.env) if py_argv is None else py_argv
    chain = "unset " + " ".join(_DIST_VARS) + "; " + shlex.join(list(py_argv) + argv)
    log_dir = item.log_dir
    log = shlex.quote(str(log_dir / "console.log"))
    return (
        f"mkdir -p {shlex.quote(str(log_dir))} && "
        f"{{ echo \"== start e5 $(date -Is) gpu $CUDA_VISIBLE_DEVICES host $(hostname) pid $$\"; {chain}; rc=$?; "
        f"echo \"== exit $rc $(date -Is)\"; exit $rc; }} >> {log} 2>&1"
    )


def completed_runs(log_dir):
    """How many runs an item's console.log has finished ("== exit" lines); 0 without a log."""
    log = Path(log_dir) / "console.log"
    if not log.is_file():
        return 0
    return sum(1 for line in log.read_text(errors="replace").splitlines() if line.startswith("== exit "))


def producers_of(items):
    """{directory: (log directory, runs completed so far)} of the items that write each directory.

    Taken once, before the queue starts: a gate then knows that a producer has run
    since (its completed count grew) and did not write the directory, whatever an
    older failure in the same log says.
    """
    return {path: (it.log_dir, completed_runs(it.log_dir)) for it in items for path in it.dirs}


def gate_for(item, producers=None, limit=None):
    """A run_queue gate over the item's gate directories.

    "open" once every one has done.json for this request (outputs.is_done); "never"
    when one cannot arrive any more: no item of this queue (nor one still running
    from another launcher) writes it, or the item that writes it has finished a run
    since the queue started without producing it; otherwise "wait". ``producers`` is
    producers_of(...) over the queued and the running items; None means every
    missing directory is still expected (a plan that is only printed).
    """
    if not item.gate_dirs:
        return None

    def gate():
        missing = [path for path in item.gate_dirs if not out.is_done(path, limit)]
        if not missing:
            return "open"
        if producers is not None:
            for path in missing:
                if path not in producers:
                    return "never"
                log_dir, before = producers[path]
                if completed_runs(log_dir) > before:
                    return "never"  # the producer has run and left no done.json; run_queue records its own failure
        return "wait"

    return gate


def plan(frozen, only=(), pool_only=False, test_only=False, limit=None):
    """The items to run, in queue order, with done and running ones left out.

    Without ``limit`` an item whose directories hold a smoke run only is not done (outputs.is_done).
    --test-only keeps the pool-phase prerequisites of the test runs (the pool assembly of
    a_sam3_selected, listed once the phrases are selected): the documented order is
    pool runs -> selections -> --test-only, and the test assembly needs its thresholds.
    """
    chosen = []
    for item in discover(frozen):
        source = item.label.split(":", 1)[0]
        if only and source not in only:
            continue
        if (pool_only and item.phase != POOL) or (test_only and item.phase != TEST and not item.prerequisite):
            continue
        if item.done(limit):
            print(f"skip {item.label}: done", flush=True)
            continue
        running = item.running()
        if running is not None:
            print(f"skip {item.label}: still running ({running})", flush=True)
            continue
        chosen.append(item)
    chosen.sort(key=lambda it: (it.phase, -it.cost, it.label))
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print the E5 inference queue, or start it with --submit.")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--pool-only", action="store_true", help="the validation-pool runs only")
    parser.add_argument("--test-only", action="store_true", help="the test-set runs only")
    parser.add_argument("--only", help="comma-separated source names, e.g. a_sam_vith_p64,b_sam2_box")
    parser.add_argument("--drop-maps", default="", help="comma-separated test sets whose maps are removed after evaluation (e.g. FGADR,Retinal-Lesions)")
    parser.add_argument("--limit", type=int, help="first N images of every item (smoke)")
    args = parser.parse_args(argv)
    drop = tuple(item.strip() for item in args.drop_maps.split(",") if item.strip())
    only = {item.strip() for item in (args.only or "").split(",") if item.strip()}
    frozen = fz.load_frozen()
    items = plan(frozen, only, args.pool_only, args.test_only, args.limit)
    for item in items:
        print(f"phase {item.phase} {item.label} env {item.env} cost {item.cost:.0f}" + (" gated" if item.gate_dirs else ""), flush=True)
    print(f"E5 inference queue: {len(items)} items", flush=True)
    missing = missing_prompt_files(frozen, items)
    if missing:
        print("prompt files to write first: " + ", ".join(f"{n}/{s}/{v}" for n, s, v in missing), flush=True)
    if not args.submit:
        for item in items[:3]:
            print("command " + command(item, py_argv=["python"], drop_maps=drop, limit=args.limit), flush=True)
        print("not submitted; pass --submit (with RETISEG_GPUS) to start", flush=True)
        return
    gpus = [int(gpu) for gpu in os.environ.get("RETISEG_GPUS", "").replace(",", " ").split()]
    if not gpus:
        raise SystemExit("--submit needs RETISEG_GPUS, the cards this queue may use")
    absent = absent_sets(items)
    if absent:
        raise SystemExit("prepared sets this plan needs are missing on this machine: " + ", ".join(absent))
    write_prompt_files(frozen, missing)
    # who writes each gate directory: the items of this queue and the items another launcher is still running
    running = [it for it in discover(frozen) if it.running() is not None]
    producers = producers_of(running + items)
    queue = []
    for item in items:
        entry = (item.label, command(item, drop_maps=drop, limit=args.limit))
        gate = gate_for(item, producers, args.limit)
        queue.append(entry + ((gate,) if gate is not None else ()))
    errors = run_queue(queue, gpus, wait_free=True)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
