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
    python bench/scripts/ensemble.py run --train DDR --composition all,fam:fundus --sets FGADR,Retinal-Lesions --without HACDR-Net   # a member whose maps cannot be made (21 item 87); recorded in done.json
    python bench/scripts/ensemble.py run --train DDR --composition all,fam:general,fam:auto,fam:fundus,fam:foundation,top3,top5,1perfam

Output: runs/ENS/<train>/<composition>/prob_val (the ensemble's validation maps, used for the val_opt
thresholds exactly as a single model's are) and <set>/{prob_test, metrics.json, metrics.npz, done.json}.
The paper reads these through docs/paper/retiseg_paper/paper/scripts/collect_runs.py (experiment ENS).

The reference model (docs/paper/26 section 5 item 2; 14 note 58) is a two-member ensemble of named run
directories, the REF cells trained on IDRiD + DDR + TJDR_std, with no registered composition:
    python bench/scripts/ensemble.py reference --members runs/BF_ref_retfound_fine_seed0,runs/BF_ref_samed_fine_seed0 [--sets ...] [--drop-maps]
Output runs/ENS/REF/reference/<set>/...; done.json carries the paper key (N8, reference:ensemble).
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
# The list is the one registered on 2026-10-19/20 (docs/paper/14 M15, note 47: compositions are fixed after
# registration) and holds the 18 main-table rows scored at that time. MedSAM, scored on 2026-10-21 and revised on
# 2026-10-22, is not a member; a members.json that lists it (the DDR recompute of 2026-10-23) is pruned on read
# (docs/ops/21 item 88).
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
    ("RETFound + UPerNet", "foundation", "BS_retfound_{t}_seed0"),
]
REGISTERED_MEMBERS = [name for name, _family, _template in MEMBERS]
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
    # The file is {"train", "rule", "members": {name: row}}. An earlier read treated that whole
    # object as the table and wrote it back nested; keep only rows that actually have a score.
    blob = json.loads(path.read_text()) if path.is_file() and not force else {}
    stored = blob.get("members", {}) if isinstance(blob, dict) else {}
    table = {k: v for k, v in stored.items() if isinstance(v, dict) and "val_maupr" in v and k in REGISTERED_MEMBERS}
    dropped = sorted(k for k in stored if k not in table)
    if dropped:
        print(f"{train}: {dropped} left out of the member table (not in the registered list; 14 note 47)", flush=True)
    changed = bool(stored) and set(stored) != set(table)
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


def run(train, composition, sets=None, drop_maps=False, runs_root=None, e3a_root=None, without=()):
    """``without``: members left out of this run on the named test sets only, when their maps cannot be produced
    (HACDR-Net's deformable convolution failed when its FGADR / Retinal-Lesions maps were regenerated, docs/ops/21
    item 87). The composition stays as registered: the member is dropped for these sets, done.json records
    members_missing, and collect_runs marks the row (a 17-member E-all is reported as such, never as the full one).
    The ensemble's validation maps keep every member, so the thresholds are those of the registered composition."""
    from bench.eval.evaluate import evaluate, load_protocol

    frozen = e3a.load_frozen()
    proto = load_protocol(REPO / frozen["eval_protocol"])
    table = members_table(train, frozen, proto, runs_root)
    names = compose(composition, table)
    run_dirs = [runs_dir(runs_root) / table[n]["run"] for n in names]
    without = [w for w in without if w in names]
    if without and not sets:
        raise ValueError("--without needs --sets: a member is left out on named test sets only, never on the whole composition")
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
        kept = [(n_, d) for n_, d in zip(names, run_dirs) if n_ not in without]
        if len(kept) < 2:
            raise ValueError(f"{test_name}: fewer than two members left after --without {without}")
        member_dirs = [map_dir(d, train, test_name, e3a_root) for _, d in kept]
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
            "variant": e3a.MAIN, "members": [n_ for n_, _ in kept], "member_runs": [d.name for _, d in kept], "rule": "pixelwise mean of 16-bit maps",
            "n_images": n, "excluded": excluded, "seconds": round(time.perf_counter() - started, 1),
            "summary": result["summary"], "maps_dropped": bool(drop_maps),
        }
        if without:
            done["members_missing"] = list(without)
            done["members_registered"] = names
        (directory / "done.json").write_text(json.dumps(done, indent=1) + "\n")
        if drop_maps:
            import shutil

            shutil.rmtree(directory / "prob_test")
        results[test_name] = result["summary"]["mAUPR"]
        print(f"{train}/{composition} {test_name} mAUPR {results[test_name]:.4f} ({len(kept)} members" + (f"; without {without}" if without else "") + ")", flush=True)
    return results


REFERENCE_NAME = "RetiSeg-Ref"
REFERENCE_PAPER = {"experiment": "N8", "variant": "reference:ensemble"}
SET_LABEL = {"TJDR_std": "TJDR-std", "TJDR_uwf": "TJDR-uwf"}   # the paper's names (collect_runs E3A_SET), for done.json's paper key


def reference_members(run_dirs):
    """(validation / internal-test source, merged training sources) shared by the reference cells, from their configs."""
    import yaml

    seen = None
    for run_dir in run_dirs:
        cfg = Path(run_dir) / "config.yaml"
        config = yaml.safe_load(cfg.read_text()) if cfg.is_file() else {}
        if config.get("experiment") != "BF" or config.get("factor") != "REF":
            raise ValueError(f"{run_dir}: not a REF cell (bench/factor.py; 14 note 58)")
        test_source = e3a.PREPARED_TO_SET.get(Path(config["dataset"]).name, Path(config["dataset"]).name)
        train_names = tuple(e3a.train_sources_of(config))
        scale_known = e3a.scale_source_of(config) is not None
        if seen is not None and seen != (test_source, train_names, scale_known):
            raise ValueError(f"the reference cells disagree on their sources: {seen} vs {(test_source, train_names, scale_known)}")
        seen = (test_source, train_names, scale_known)
    if seen is None:
        raise ValueError("no member run directories")
    return seen


def reference_map_dir(run_dir, test_name, test_source, train_names, scale_known, frozen, e3a_root=None):
    """(maps directory, e3a variant) of a REF member on one test set: prob_test for the validation source's own
    test split, insource_<set>/prob for the other merged sources (17 section 4 item 27), the E3a directory otherwise;
    a per-degree set uses its fov variant, as e3a does for a run whose sources set no single scale."""
    run_dir = Path(run_dir)
    if test_name == test_source:
        return run_dir / "prob_test", e3a.MAIN
    if test_name in train_names:
        return run_dir / f"insource_{test_name}" / "prob", e3a.MAIN
    variant = e3a.MAIN
    if test_name in frozen.get("per_degree_sets", ()) and not scale_known:
        variant = frozen["comparison_variants"][test_name]
    e3a_root = Path(e3a_root) if e3a_root is not None else run_dir.parent / "E3a"
    return e3a_root / run_dir.name / e3a.set_dirname(test_name, variant) / "prob_test", variant


def run_reference(member_run_dirs, sets=None, drop_maps=False, runs_root=None, e3a_root=None):
    """The reference model's ensemble (26 section 5 item 2; 14 note 58): the pixel mean of the maps of the named
    runs, the two REF cells, scored like every other ensemble. The members are run directories, not a registered
    composition; their configs name the merged training sources and the validation source. Thresholds come from
    the mean of the members' prob_val (the validation source's split); the test sets are the validation source's test
    split, the in-source test splits of the other merged sources and every E3a set the members were tested on.
    Output runs/ENS/REF/reference/<set>/{metrics.json, done.json}; done.json carries the paper key
    (experiment N8, variant reference:ensemble, train_set the merged sources) that collect_runs reads."""
    from bench.eval.evaluate import evaluate, load_protocol

    frozen = e3a.load_frozen()
    proto = load_protocol(REPO / frozen["eval_protocol"])
    run_dirs = [Path(d) if Path(d).is_absolute() else REPO / d for d in member_run_dirs]
    for d in run_dirs:
        if not (d / "metrics_test.json").is_file():
            raise FileNotFoundError(f"{d}: not scored yet (metrics_test.json)")
    test_source, train_names, scale_known = reference_members(run_dirs)
    out = ens_root(runs_root) / "REF" / "reference"
    data_val = e3a.test_data_dir(test_source, frozen)
    classes = [c for c in proto["classes"] if c in io.load_meta(data_val)["classes"]]
    val_dir = out / "prob_val"
    if not (val_dir / "done.json").is_file():
        val_ids = io.load_split(data_val, "val")
        missing = missing_maps([d / "prob_val" for d in run_dirs], val_ids, classes)
        if missing:
            raise FileNotFoundError(f"validation maps missing: {missing}")
        average_maps([d / "prob_val" for d in run_dirs], val_ids, classes, val_dir)
        (val_dir / "done.json").write_text(json.dumps({"members": [d.name for d in run_dirs], "n_images": len(val_ids), "validation_source": test_source}) + "\n")
    others = [t for t in train_names if t != test_source]
    default_sets = [test_source] + others + [t for t in frozen["test_sets"] if t not in train_names]
    test_names = [t for t in (sets or default_sets)]
    results = {}
    for test_name in test_names:
        directory = out / test_name
        if (directory / "done.json").is_file():
            print(f"REF/reference {test_name}: done", flush=True)
            continue
        data_test = e3a.test_data_dir(test_name, frozen)
        if test_name in train_names:
            ids, excluded = io.load_split(data_test, frozen["split"]), []
        else:
            ids, excluded = e3a.test_ids(test_name, frozen, list(train_names))
        located = [reference_map_dir(d, test_name, test_source, train_names, scale_known, frozen, e3a_root) for d in run_dirs]
        member_dirs = [m for m, _v in located]
        variant = located[0][1]
        missing = missing_maps(member_dirs, ids, classes)
        if missing:
            raise FileNotFoundError(f"{test_name}: member maps missing (bench.e3a run / insource prediction, 17 section 4 item 47): {missing}")
        started = time.perf_counter()
        n = average_maps(member_dirs, ids, classes, directory / "prob_test")
        result = evaluate(
            data_test, frozen["split"], directory / "prob_test", proto,
            val_pred_dir=val_dir, val_split="val", out_prefix=directory / "metrics", val_dataset_dir=data_val, exclude=excluded,
        )
        done = {
            "experiment": "ENS", "composition": "reference", "name": REFERENCE_NAME, "train_source": "+".join(train_names), "train_sources": list(train_names),
            "validation_source": test_source, "test_set": test_name, "variant": variant,
            "members": [d.name for d in run_dirs], "member_runs": [d.name for d in run_dirs], "rule": "pixelwise mean of 16-bit maps",
            "in_source": test_name in train_names, "n_images": n, "excluded": excluded, "seconds": round(time.perf_counter() - started, 1),
            "summary": result["summary"], "maps_dropped": bool(drop_maps),
            "paper": dict(REFERENCE_PAPER, train_set=" + ".join(SET_LABEL.get(t, t) for t in train_names)),
        }
        (directory / "done.json").write_text(json.dumps(done, indent=1) + "\n")
        if drop_maps:
            import shutil

            shutil.rmtree(directory / "prob_test")
        results[test_name] = result["summary"]["mAUPR"]
        print(f"REF/reference {test_name} mAUPR {results[test_name]:.4f} ({len(run_dirs)} members; variant {variant})", flush=True)
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
    ref = sub.add_parser("reference", help="the reference model's two-member ensemble (14 note 58): explicit run directories")
    ref.add_argument("--members", required=True, help="comma-separated run directories, e.g. runs/BF_ref_retfound_fine_seed0,runs/BF_ref_samed_fine_seed0")
    ref.add_argument("--sets", default=None, help="comma-separated test sets (default: validation source's test, in-source sets, then every E3a set)")
    ref.add_argument("--drop-maps", action="store_true")
    ref.add_argument("--runs", default=None)
    ref.add_argument("--e3a", default=None)
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
            p.add_argument("--without", default="", help="comma-separated members left out on the named --sets only (maps that cannot be produced; recorded in done.json)")
        if command == "members":
            p.add_argument("--force", action="store_true", help="recompute the validation table")
    args = parser.parse_args(argv)
    if args.command == "reference":
        run_reference([m.strip() for m in args.members.split(",") if m.strip()], e3a._sets(args.sets), args.drop_maps, args.runs, args.e3a)
        return
    if args.command == "members":
        members_table(args.train, runs_root=args.runs, force=args.force)
        return
    compositions = [c.strip() for c in args.composition.split(",") if c.strip()]
    if args.command == "plan":
        plan(args.train, compositions, args.runs, args.e3a)
        return
    sets = e3a._sets(args.sets)
    without = tuple(w.strip() for w in args.without.split(",") if w.strip())
    for composition in compositions:
        run(args.train, composition, sets, args.drop_maps, args.runs, args.e3a, without=without)


if __name__ == "__main__":
    main()
