"""Queue the E5 inference tiers (a zero-shot, b oracle prompts) on the cards in RETISEG_GPUS.

Items, in order:
  1. pool runs, the val split of IDRiD and DDR (decision D2): every (model, density) of the
     zero-shot grid, every SAM 3 phrase index, every (model, kind) of the oracle tier; they
     feed the two selections in e5_frozen.yaml and the Dice thresholds;
  2. test runs on the seven sets (TJDR_uwf main and fov): SAM / SAM 2 only once
     zero_shot.grid.selected is filled, SAM 3 only once zero_shot.sam3.selected is filled
     (then a gated assembly of a_sam3_selected per set), the oracle tier and the
     same-information references always; larger canvases first.
Each item is one process on one card running python -m bench.e5.<module>; SAM 3 runs in
retiseg-sam3 (RETISEG_SAM3_ENV overrides), everything else in retiseg. The oracle prompt
files of every (set, split, variant) are written first, in this process (CPU). An item
whose directories all have done.json is skipped, and so is one whose console.log shows a
live process (bench/scripts/launch_b1.py in_progress). Items write their log to the
first of their directories.

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

    def done(self):
        return bool(self.dirs) and all(out.is_done(path) for path in self.dirs)

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
    """Every item the current protocol allows; done and running items are filtered by the caller."""
    items = []
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
                    items.append(Item(f"{source}:{tag}", phase, cost(COST_A[model], set_name, split, variant, frozen), ENV[model], argv, _log_dir(paths), paths, set_name))
        # a: SAM 3 phrases, then the gated assembly of the selected phrases
        if pool_run or sam3_selected:
            phrase_dirs = []
            for index in sam3_text.PHRASE_INDICES:
                source, directory = sam3_text.planned(frozen, index, set_name, split, variant)
                argv = ["-m", "bench.e5.sam3_text", "--phrase", str(index), "--set", set_name, "--split", split, "--variant", variant]
                items.append(Item(f"{source}:{tag}", phase, cost(COST_A["SAM 3"], set_name, split, variant, frozen), ENV["SAM 3"], argv, directory, [directory], set_name))
                phrase_dirs.append(directory)
            if sam3_selected:
                selected = sam3_text.planned_selected(set_name, split, variant)
                argv = ["-m", "bench.e5.sam3_text", "--assemble", "--set", set_name, "--split", split, "--variant", variant]
                items.append(Item(f"{sam3_text.selected_source()}:{tag}", phase, 0.0, ENV["ref"], argv, selected, [selected], set_name, gate_dirs=phrase_dirs))
        # b: oracle prompts
        for model, kinds in fz.oracle_models(frozen).items():
            for kind in kinds:
                source, dirs = prompted.planned(frozen, model, kind, set_name, split, variant)
                argv = ["-m", "bench.e5.prompted", "--model", model, "--kind", kind, "--set", set_name, "--split", split, "--variant", variant]
                paths = list(dirs.values())
                items.append(Item(f"{source}:{tag}", phase, cost(COST_B[model], set_name, split, variant, frozen), ENV[model], argv, _log_dir(paths), paths, set_name))
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
                items.append(Item(f"{source}:{tag}", phase, cost(COST_REF, set_name, split, variant, frozen), ENV["ref"], argv, _log_dir(paths), paths, set_name))
    return items


def missing_prompt_files(frozen):
    """(set, split, variant) whose oracle prompt file does not exist yet."""
    return [(name, split, variant) for name, split, variant in targets(frozen) if not fz.prompt_file(name, split, variant=variant).is_file()]


def write_prompt_files(frozen, missing):
    for name, split, variant in missing:
        path = prompts.build(name, split, variant, frozen=frozen)
        print(f"prompts {path}", flush=True)


def command(item, py_argv=None, drop_maps=(), limit=None):
    """Shell body of one queue item. ``py_argv`` None resolves the conda env (only when submitting)."""
    argv = list(item.argv)
    if item.phase == TEST and item.set_name in set(drop_maps):
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


def gate_for(item):
    """A run_queue gate: open once every gate directory has done.json, otherwise wait."""
    if not item.gate_dirs:
        return None

    def gate():
        return "open" if all(out.is_done(path) for path in item.gate_dirs) else "wait"

    return gate


def plan(frozen, only=(), pool_only=False, test_only=False):
    """The items to run, in queue order, with done and running ones left out."""
    chosen = []
    for item in discover(frozen):
        source = item.label.split(":", 1)[0]
        if only and source not in only:
            continue
        if (pool_only and item.phase != POOL) or (test_only and item.phase != TEST):
            continue
        if item.done():
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
    items = plan(frozen, only, args.pool_only, args.test_only)
    for item in items:
        print(f"phase {item.phase} {item.label} env {item.env} cost {item.cost:.0f}" + (" gated" if item.gate_dirs else ""), flush=True)
    print(f"E5 inference queue: {len(items)} items", flush=True)
    missing = missing_prompt_files(frozen)
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
    absent = [name for name in geo.test_sets() if not geo.data_dir(name).is_dir()]
    if absent:
        raise SystemExit("prepared test sets missing on this machine: " + ", ".join(absent))
    write_prompt_files(frozen, missing)
    queue = []
    for item in items:
        entry = (item.label, command(item, drop_maps=drop, limit=args.limit))
        gate = gate_for(item)
        queue.append(entry + ((gate,) if gate is not None else ()))
    errors = run_queue(queue, gpus, wait_free=True)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
