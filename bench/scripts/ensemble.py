"""Ensembles of the main-table probability maps (docs/paper/26 N2; docs/paper/14 M15; registered 2026-10-19).

An ensemble is the pixelwise arithmetic mean of its members' 16-bit probability maps (probability
space, equal weights), scored by the same bench.eval protocol as every single model. Nothing is
trained. The members are the seed-0 main-table runs of one training source (B-std runs, the E5
training tiers, the two nnU-Net plans); a `_registered` run (a retired recipe) is never a member.

Compositions, fixed on 2026-10-19 before any ensemble was scored; they use validation data only:
    all          every main-table member of the training source
    fam:<name>   one family: general | auto | fundus | foundation
    top3, top5   the 3 / 5 members with the highest validation mAUPR
    1perfam      the best member by validation mAUPR of each family (4 members)
The validation mAUPR of a member is bench.eval on its prob_val (the training source's validation
split, the maps of the checkpoint the model is scored with) -- one definition for every member,
nnU-Net included (its training-time validation number is not comparable with history.jsonl of the
B-std runs). Ties are broken by the member order of MEMBERS. The table is written once to
runs/ENS/<train>/members.json and reused; delete the file to recompute.

Maps: the internal test and the validation split come from the run directory (prob_test, prob_val);
every other test set from runs/E3a/<run name>/<set>/prob_test (the main variant). FGADR and
Retinal-Lesions were run with --drop-maps, so their member maps have to be regenerated first:
    python -m bench.e3a run --run runs/BS_hrnet_ddr_seed0 --sets FGADR,Retinal-Lesions --regenerate-maps
(docs/ops/17 section 4 item 23). `plan` lists what is missing.

    python bench/scripts/ensemble.py members --train DDR                  # rank table (runs/ENS/DDR/members.json)
    python bench/scripts/ensemble.py plan --train DDR --composition top3   # members and missing maps
    python bench/scripts/ensemble.py run --train DDR --composition top3 [--sets DDR,IDRiD,...] [--drop-maps]
    python bench/scripts/ensemble.py run --train DDR --composition all,fam:general,fam:auto,fam:fundus,fam:foundation,top3,top5,1perfam

Output: runs/ENS/<train>/<composition>/prob_val (the ensemble's validation maps, used for the val_opt
thresholds exactly as a single model's are) and <set>/{prob_test, metrics.json, metrics.npz, done.json}.
The paper reads these through docs/paper/retiseg_paper/paper/scripts/collect_runs.py (experiment ENS).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.common import io  # noqa: E402
from bench import e3a  # noqa: E402

REPO = _ROOT
FAMILIES = ("general", "auto", "fundus", "foundation")

# (paper name, family, run-directory template). The order is the tie-break order of the rank table and the
# member order of every ensemble; it follows the main table (docs/paper/retiseg_paper/paper/figures/retifig/data.py).
MEMBERS = [
    ("U-Net", "general", "BS_unet_{t}_seed0"),
    ("DeepLabv3", "general", "BS_deeplabv3_{t}_seed0"),
    ("HRNet-W48", "general", "BS_hrnet_{t}_seed0"),
    ("Swin-Unet", "general", "BS_swin_unet_{t}_seed0"),
    ("FCT", "general", "BS_fct_{t}_seed0"),
    ("nnU-Net", "auto", "NN_default_{t}_fold0"),
    ("nnU-Net ResEnc-M", "auto", "NN_resencm_{t}_fold0"),
    ("M2MRF", "fundus", "BS_m2mrf_{t}_seed0"),
    ("HACDR-Net", "fundus", "BS_hacdr_{t}_seed0"),
    ("HRDecoder", "fundus", "BS_hrdecoder_{t}_seed0"),
    ("PMCNet", "fundus", "BS_pmcnet_{t}_seed0"),
    ("WSRFNet", "fundus", "BS_wsrfnet_{t}_seed0"),
    ("CLC-Net", "fundus", "BS_clcnet_{t}_seed0"),
    ("TC-Net", "fundus", "BS_tcnet_{t}_seed0"),
    ("SAMed", "foundation", "BS_samed_{t}_seed0"),
    ("SAM-Adapter", "foundation", "BS_sam_adapter_{t}_seed0"),
    ("SAM2-UNet", "foundation", "BS_sam2_unet_{t}_seed0"),
    ("MedSAM (full FT)", "foundation", "BS_medsam_{t}_seed0"),
    ("RETFound + UPerNet", "foundation", "BS_retfound_{t}_seed0"),
]
COMPOSITIONS = ("all", "top3", "top5", "1perfam") + tuple(f"fam:{f}" for f in FAMILIES)


def train_data_dir(train, frozen):
    return e3a.test_data_dir(train, frozen)


def runs_dir(runs_root=None):
    return Path(runs_root) if runs_root is not None else REPO / "runs"


def ens_root(runs_root=None):
    return runs_dir(runs_root) / "ENS"


def ensemble_dir(train, composition, runs_root=None):
    return ens_root(runs_root) / train / composition.replace(":", "_")


def member_runs(train, runs_root=None):
    """[(name, family, run_dir)] of the members whose seed-0 run is scored (metrics_test.json exists)."""
    out = []
    for name, family, template in MEMBERS:
        run_dir = runs_dir(runs_root) / template.format(t=train.lower())
        if (run_dir / "metrics_test.json").is_file():
            out.append((name, family, run_dir))
    return out


def map_dir(run_dir, train, test_name, e3a_root=None):
    """Where a member's maps for a test set live: prob_test of the run for the internal test, E3a otherwise."""
    if test_name == train:
        return Path(run_dir) / "prob_test"
    e3a_root = Path(e3a_root) if e3a_root is not None else Path(run_dir).parent / "E3a"
    return e3a_root / Path(run_dir).name / e3a.set_dirname(test_name, e3a.MAIN) / "prob_test"


def registered_validation_maupr(run_dir):
    """The number the run's checkpoint was selected with (docs/paper/14 M15: history.jsonl's best validation mAUPR,
    which environment.json stores as best_val_maupr), or None for runs that have no such number (nnU-Net)."""
    env = Path(run_dir) / "environment.json"
    if env.is_file():
        value = json.loads(env.read_text()).get("best_val_maupr")
        if value is not None:
            return float(value)
    history = Path(run_dir) / "history.jsonl"
    if history.is_file():
        values = [json.loads(line).get("val_maupr") for line in history.read_text().splitlines() if line.strip()]
        values = [float(v) for v in values if v is not None]
        if values:
            return max(values)
    return None


def validation_maupr(run_dir, train, frozen, proto):
    """bench.eval mAUPR of the run's prob_val on the training source's validation split (the fallback of M15 for a
    member without a checkpoint-selection number: nnU-Net's own predictor wrote its validation maps)."""
    from bench.eval.evaluate import evaluate

    result = evaluate(train_data_dir(train, frozen), "val", Path(run_dir) / "prob_val", proto)
    return float(result["summary"]["mAUPR"])


def members_table(train, frozen=None, proto=None, runs_root=None, force=False):
    """{name: {family, run, val_maupr}} of every scored member, cached in runs/ENS/<train>/members.json."""
    frozen = frozen or e3a.load_frozen()
    if proto is None:
        from bench.eval.evaluate import load_protocol

        proto = load_protocol(REPO / frozen["eval_protocol"])
    path = ens_root(runs_root) / train / "members.json"
    table = json.loads(path.read_text()) if path.is_file() and not force else {}
    changed = False
    for name, family, run_dir in member_runs(train, runs_root):
        if name in table and table[name].get("run") == run_dir.name:
            continue
        if not (run_dir / "prob_val").is_dir():
            print(f"{name}: {run_dir.name} has no prob_val; left out of every composition", flush=True)
            continue
        started = time.perf_counter()
        value = registered_validation_maupr(run_dir)
        source = "checkpoint selection (environment.json best_val_maupr / history.jsonl)"
        if value is None:
            value = validation_maupr(run_dir, train, frozen, proto)
            source = "bench.eval mAUPR on prob_val (no selection number: M15 fallback)"
        table[name] = {"family": family, "run": run_dir.name, "val_maupr": value, "source": source, "seconds": round(time.perf_counter() - started, 1)}
        changed = True
        print(f"{name}: validation mAUPR {value:.4f} ({run_dir.name}; {source.split(' (')[0]})", flush=True)
    if changed or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"train": train, "rule": "the validation mAUPR the checkpoint was selected with (14 M15); "
                                    "bench.eval on prob_val for members without one; ties by MEMBERS order", "members": table}, indent=1) + "\n")
    return table


def compose(composition, table):
    """Member names of a composition, in MEMBERS order (docs/paper/14 M15)."""
    order = [name for name, _f, _t in MEMBERS if name in table]
    ranked = sorted(order, key=lambda n: (-table[n]["val_maupr"], order.index(n)))
    if composition == "all":
        chosen = order
    elif composition.startswith("fam:"):
        family = composition.split(":", 1)[1]
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; one of {FAMILIES}")
        chosen = [n for n in order if table[n]["family"] == family]
    elif composition in ("top3", "top5"):
        chosen = ranked[: int(composition[3:])]
    elif composition == "1perfam":
        chosen = []
        for family in FAMILIES:
            best = [n for n in ranked if table[n]["family"] == family]
            if best:
                chosen.append(best[0])
    else:
        raise ValueError(f"unknown composition {composition!r}; one of {COMPOSITIONS}")
    chosen = [n for n in order if n in chosen]
    if len(chosen) < 2:
        raise ValueError(f"{composition} has {len(chosen)} member(s); an ensemble needs at least two")
    return chosen


def average_maps(member_dirs, ids, classes, out_dir):
    """Write the mean of the members' quantized maps for every id and class; returns the number of images."""
    out_dir = Path(out_dir)
    for image_id in ids:
        for cls in classes:
            total = None
            for directory in member_dirs:
                q = io.read_prob_quantized(directory, cls, image_id).astype(np.float32)
                total = q if total is None else total + q
            io.write_prob(out_dir, cls, image_id, np.clip(total / (len(member_dirs) * io.PROB_LEVELS), 0.0, 1.0))
    return len(ids)


def missing_maps(member_dirs, ids, classes):
    """{member directory: first missing (cls, id)} -- one entry per member directory with a gap."""
    out = {}
    for directory in member_dirs:
        for cls in classes:
            for image_id in ids:
                if not (Path(directory) / cls / f"{image_id}.png").is_file():
                    out[str(directory)] = (cls, image_id)
                    break
            if str(directory) in out:
                break
    return out


def run(train, composition, sets=None, drop_maps=False, runs_root=None, e3a_root=None):
    from bench.eval.evaluate import evaluate, load_protocol

    frozen = e3a.load_frozen()
    proto = load_protocol(REPO / frozen["eval_protocol"])
    table = members_table(train, frozen, proto, runs_root)
    names = compose(composition, table)
    run_dirs = [runs_dir(runs_root) / table[n]["run"] for n in names]
    out = ensemble_dir(train, composition, runs_root)
    data_train = train_data_dir(train, frozen)
    classes = [c for c in proto["classes"] if c in io.load_meta(data_train)["classes"]]
    # the ensemble's validation maps, for its val_opt thresholds (the same rule as for a single model)
    val_dir = out / "prob_val"
    if not (val_dir / "done.json").is_file():
        val_ids = io.load_split(data_train, "val")
        missing = missing_maps([d / "prob_val" for d in run_dirs], val_ids, classes)
        if missing:
            raise FileNotFoundError(f"validation maps missing: {missing}")
        average_maps([d / "prob_val" for d in run_dirs], val_ids, classes, val_dir)
        (val_dir / "done.json").write_text(json.dumps({"members": names, "n_images": len(val_ids)}) + "\n")
    test_names = sets or [train] + [t for t in frozen["test_sets"] if t != train]
    results = {}
    for test_name in test_names:
        directory = out / test_name
        if (directory / "done.json").is_file():
            print(f"{train}/{composition} {test_name}: done", flush=True)
            continue
        if test_name == train:
            ids, excluded = io.load_split(data_train, frozen["split"]), []
        else:
            ids, excluded = e3a.test_ids(test_name, frozen, [train])
        member_dirs = [map_dir(d, train, test_name, e3a_root) for d in run_dirs]
        missing = missing_maps(member_dirs, ids, classes)
        if missing:
            raise FileNotFoundError(f"{test_name}: member maps missing (regenerate with bench.e3a --regenerate-maps): {missing}")
        started = time.perf_counter()
        n = average_maps(member_dirs, ids, classes, directory / "prob_test")
        result = evaluate(
            e3a.test_data_dir(test_name, frozen), frozen["split"], directory / "prob_test", proto,
            val_pred_dir=val_dir, val_split="val", out_prefix=directory / "metrics", val_dataset_dir=data_train, exclude=excluded,
        )
        done = {
            "experiment": "ENS", "composition": composition, "train_source": train, "train_sources": [train], "test_set": test_name,
            "variant": e3a.MAIN, "members": names, "member_runs": [d.name for d in run_dirs], "rule": "pixelwise mean of 16-bit maps",
            "n_images": n, "excluded": excluded, "seconds": round(time.perf_counter() - started, 1),
            "summary": result["summary"], "maps_dropped": bool(drop_maps),
        }
        (directory / "done.json").write_text(json.dumps(done, indent=1) + "\n")
        if drop_maps:
            import shutil

            shutil.rmtree(directory / "prob_test")
        results[test_name] = result["summary"]["mAUPR"]
        print(f"{train}/{composition} {test_name} mAUPR {results[test_name]:.4f} ({len(names)} members)", flush=True)
    return results


def plan(train, compositions, runs_root=None, e3a_root=None):
    frozen = e3a.load_frozen()
    table = members_table(train, frozen, None, runs_root)
    data_train = train_data_dir(train, frozen)
    classes = io.load_meta(data_train)["classes"]
    for name in table:
        print(f"  {name:22s} val mAUPR {table[name]['val_maupr']:.4f}  {table[name]['run']}")
    for composition in compositions:
        names = compose(composition, table)
        print(f"{composition}: {', '.join(names)}")
        for test_name in [train] + [t for t in frozen["test_sets"] if t != train]:
            ids = io.load_split(data_train, frozen["split"]) if test_name == train else e3a.test_ids(test_name, frozen, [train])[0]
            member_dirs = [map_dir(runs_dir(runs_root) / table[n]["run"], train, test_name, e3a_root) for n in names]
            missing = sorted({Path(d).parent.name if test_name == train else Path(d).parents[1].name for d in missing_maps(member_dirs, ids, classes)})
            state = "done" if (ensemble_dir(train, composition, runs_root) / test_name / "done.json").is_file() else ("ready" if not missing else f"missing maps: {missing}")
            print(f"  {test_name:16s} {state}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("members", "plan", "run"):
        p = sub.add_parser(command)
        p.add_argument("--train", required=True, choices=["IDRiD", "DDR"])
        p.add_argument("--runs", default=None, help="runs root (default runs/)")
        if command != "members":
            p.add_argument("--composition", default="all", help="comma-separated: " + ", ".join(COMPOSITIONS))
            p.add_argument("--e3a", default=None, help="E3a root (default runs/E3a)")
        if command == "run":
            p.add_argument("--sets", default=None, help="comma-separated test sets (default: all of e3a_frozen.yaml)")
            p.add_argument("--drop-maps", action="store_true", help="remove the ensemble's prob_test after scoring")
        if command == "members":
            p.add_argument("--force", action="store_true", help="recompute the validation table")
    args = parser.parse_args(argv)
    if args.command == "members":
        members_table(args.train, runs_root=args.runs, force=args.force)
        return
    compositions = [c.strip() for c in args.composition.split(",") if c.strip()]
    if args.command == "plan":
        plan(args.train, compositions, args.runs, args.e3a)
        return
    sets = e3a._sets(args.sets)
    for composition in compositions:
        run(args.train, composition, sets, args.drop_maps, args.runs, args.e3a)


if __name__ == "__main__":
    main()
