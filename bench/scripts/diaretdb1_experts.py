"""Leave-one-expert-out human reference on DiaRetDB1 (docs/paper/26 N3a; docs/paper/14 M16a).

Each of the four DiaRetDB1 experts in turn plays the model: that expert's own
markings under the frozen confidence mapping (High 1.0, Medium 0.75, Low 0.25,
nothing 0; bench/data/diaretdb.py) are the probability map, and the consensus
of the OTHER three experts under the same mapping (three-expert mean >= 0.75,
the closest to the official four-expert 0.75 rule) is the ground truth. The
maps then go through the protocol evaluator exactly as a model's do: pooled
AUPR inside the FOV, Dice at 0.5 and at the validation-optimal threshold (the
official training list of 28 images, which is every model's DiaRetDB1
validation split), lesion-level precision / recall, image-level AUC. The four
values of each metric are the "expert level" band next to the DiaRetDB1 column
of table 5 / figure 7c. No model output is read; only prepared/DiaRetDB1.

An expert's map takes four values, so its PR curve has four points; the
protocol's exact pooled AUPR is the area of the step curve through them, which
is the number reported (a model's map has 65,535 levels: the comparison favours
the model on this axis, and the text says so). The image-level AUC uses the
protocol's maximum-in-FOV score, i.e. the highest confidence the expert gave.

Output, runs/HUMAN/DiaRetDB1/ (no model output; packs with the small files):
    expert_<e>/metrics.json, metrics.npz      the protocol report of expert e against the other three
    expert_<e>/truth/, expert_<e>/prob/       the consensus masks and the expert's maps (--drop-maps removes them)
    summary.json                              per expert and across experts: min / median / max / mean

    python bench/scripts/diaretdb1_experts.py [--out runs/HUMAN/DiaRetDB1] [--drop-maps]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.common import io  # noqa: E402
from bench.data.diaretdb import CONFIDENCE, CONSENSUS_AT, EXPERTS, LEVELS, unpack_levels  # noqa: E402

PREPARED = REPO / "dataset" / "prepared" / "DiaRetDB1"
DEFAULT_OUT = REPO / "runs" / "HUMAN" / "DiaRetDB1"
SUMMARY_KEYS = ("mAUPR", "mDice_pooled_fixed", "mDice_pooled_val_opt", "mAUC_image_max", "AUPR_any_lesion")


def confidence_values():
    """level 0-3 -> score; the frozen mapping of bench/data/diaretdb.py."""
    value = np.zeros(4, dtype=np.float64)
    for name, level in LEVELS.items():
        value[level] = CONFIDENCE[name]
    return value


def read_levels(prepared, cls, image_id):
    """{expert: level map 0-3} of one class from the packed experts layer that prepare_diaretdb1 wrote.

    prepare.py omits a class png when the packed map is all zeros (no expert marked that class).
    That is the same as four level-0 maps, so a missing png is read as zeros at the FOV size.
    """
    path = Path(prepared) / "experts" / cls / f"{image_id}.png"
    if path.is_file():
        return unpack_levels(np.asarray(Image.open(path)))
    fov = Path(prepared) / "fov" / f"{image_id}.png"
    if not fov.is_file():
        raise FileNotFoundError(f"{path}: prepare_datasets.py wrote no experts layer for DiaRetDB1")
    shape = np.asarray(Image.open(fov)).shape[:2]
    return {expert: np.zeros(shape, dtype=np.uint8) for expert in EXPERTS}


def expert_map(levels, expert, values=None):
    values = confidence_values() if values is None else values
    return values[levels[expert]]


def others_consensus(levels, held_out, values=None, at=CONSENSUS_AT):
    """Positive where the mean confidence of the three experts other than ``held_out`` is >= ``at``."""
    values = confidence_values() if values is None else values
    others = [e for e in EXPERTS if e != held_out]
    mean = sum(values[levels[e]] for e in others) / len(others)
    return mean >= at - 1e-9


def _link_or_copy(source, target):
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(source, target, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def build_pair(prepared, out, expert, ids, classes, log=print):
    """truth/ (consensus of the other three, a prepared-like directory) and prob/ (the expert's maps) for ``ids``."""
    truth = out / "truth"
    prob = out / "prob"
    truth.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(prepared) / "meta.json").read_text())
    meta = {**meta, "name": f"DiaRetDB1 without expert {expert}", "classes": list(classes),
            "layers": {"masks": f"mean confidence of experts {[e for e in EXPERTS if e != expert]} >= {CONSENSUS_AT}"},
            "derived_from": str(Path(prepared).relative_to(REPO)) if str(prepared).startswith(str(REPO)) else str(prepared),
            "held_out_expert": expert}
    (truth / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    shutil.copyfile(Path(prepared) / "splits.json", truth / "splits.json")
    _link_or_copy(Path(prepared) / "fov", truth / "fov")
    values = confidence_values()
    for image_id in ids:
        for cls in classes:
            levels = read_levels(prepared, cls, image_id)
            mask = others_consensus(levels, expert, values)
            path = truth / "masks" / cls / f"{image_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            io.write_prob(prob, cls, image_id, expert_map(levels, expert, values))
    log(f"expert {expert}: {len(ids)} images x {len(classes)} classes written under {out}")
    return truth, prob


def score_expert(prepared, out, expert, proto, log=print):
    from bench.eval.evaluate import evaluate

    classes = [c for c in proto["classes"] if c in io.load_meta(prepared)["classes"]]
    splits = json.loads((Path(prepared) / "splits.json").read_text())
    ids = list(splits["val"]) + list(splits["test"])
    truth, prob = build_pair(prepared, out, expert, ids, classes, log)
    report = evaluate(truth, "test", prob, proto, val_pred_dir=prob, val_split="val", out_prefix=out / "metrics")
    report_path = out / "metrics.json"
    stored = json.loads(report_path.read_text())
    stored["human_reference"] = {
        "held_out_expert": expert, "prediction": "the expert's markings, confidence High 1.0 / Medium 0.75 / Low 0.25",
        "ground_truth": f"mean confidence of the other three experts >= {CONSENSUS_AT}",
        "val_thresholds": "chosen on the official training list (val) of the same expert against the same consensus",
        "source": "docs/paper/14 M16a; bench/scripts/diaretdb1_experts.py",
    }
    report_path.write_text(json.dumps(stored, indent=1) + "\n")
    return report


def summarise(reports):
    """{metric: {expert: value}} plus min / median / max / mean over the experts, per class and overall."""
    experts = sorted(reports)
    per_expert = {}
    for expert in experts:
        rep = reports[expert]
        entry = {key: rep["summary"].get(key) for key in SUMMARY_KEYS}
        entry["per_class"] = {}
        for cls, item in rep["per_class"].items():
            th = item.get("thresholds", {})
            lesion = (th.get("val_opt") or {}).get("lesion") or {}
            entry["per_class"][cls] = {
                "aupr": item.get("aupr"),
                "dice_fixed": (th.get("fixed") or {}).get("pooled", {}).get("dice"),
                "dice_valopt": (th.get("val_opt") or {}).get("pooled", {}).get("dice"),
                "lesion_precision": lesion.get("precision"),
                "lesion_recall": lesion.get("recall"),
                "image_auc_max": ((item.get("image_level") or {}).get("max") or {}).get("auc"),
            }
        per_expert[expert] = entry

    def band(values):
        vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return None
        return {"min": min(vals), "median": float(np.median(vals)), "max": max(vals), "mean": float(np.mean(vals)), "n": len(vals)}

    across = {key: band(per_expert[e][key] for e in experts) for key in SUMMARY_KEYS}
    classes = sorted({cls for e in experts for cls in per_expert[e]["per_class"]})
    across["per_class"] = {cls: {metric: band(per_expert[e]["per_class"].get(cls, {}).get(metric) for e in experts)
                                 for metric in ("aupr", "dice_fixed", "dice_valopt", "lesion_precision", "lesion_recall", "image_auc_max")}
                           for cls in classes}
    return {"experts": experts, "per_expert": per_expert, "across_experts": across}


def main(argv=None):
    from bench.eval.evaluate import DEFAULT_PROTOCOL, load_protocol

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared", default=str(PREPARED))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--experts", default=",".join(EXPERTS), help="comma-separated subset (default all four)")
    parser.add_argument("--drop-maps", action="store_true", help="remove truth/ and prob/ after scoring")
    args = parser.parse_args(argv)
    prepared = Path(args.prepared)
    out = Path(args.out)
    proto = load_protocol(args.protocol)
    experts = [e for e in args.experts.split(",") if e]
    unknown = sorted(set(experts) - set(EXPERTS))
    if unknown:
        raise ValueError(f"unknown experts {unknown}; DiaRetDB1 has {EXPERTS}")
    reports = {}
    for expert in experts:
        expert_out = out / f"expert_{expert}"
        reports[expert] = score_expert(prepared, expert_out, expert, proto)
        print(json.dumps({"expert": expert, **{k: reports[expert]["summary"].get(k) for k in SUMMARY_KEYS}}), flush=True)
        if args.drop_maps:
            shutil.rmtree(expert_out / "truth", ignore_errors=True)
            shutil.rmtree(expert_out / "prob", ignore_errors=True)
    summary = summarise(reports)
    summary["protocol_sha256"] = proto["_sha256"]
    summary["n_test_images"] = int(next(iter(reports.values()))["n_images"])
    summary["rule"] = "leave-one-expert-out: the held-out expert's confidence map against the >= 0.75 mean of the other three (14 M16a)"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["across_experts"], indent=1))


if __name__ == "__main__":
    main()
