"""Evaluate a scored E5 directory with bench.eval, the protocol every other model is scored with.

    python -m bench.e5 evaluate --dir runs/E5/a/sam/default/DDR_test/any [--dir ...]
    python -m bench.e5 evaluate --all            # every directory with paper.json and without metrics_test.json

Thresholds (design/15 section 4): the inference tiers have no training source, so the per-class Dice
threshold comes from the same configuration's maps on the DDR validation split
(the directory with DDR_val in place of <set>_<split>); fixed 0.5 is reported as
well. When the DDR_val maps are missing the fixed threshold is the only one and
done.json says so. Exclusions: the frozen E3a list for the test set (none today).

Writes <dir>/metrics_test.json and metrics_test.npz (bench.eval.evaluate), and
adds the summary to <dir>/done.json.
"""

from pathlib import Path

from bench.e5 import common


def val_dir_of(directory, source=common.THRESHOLD_SOURCE):
    """The DDR_val twin of a scored directory: the <set>_<split> path component replaced."""
    directory = Path(directory)
    parts = list(directory.parts)
    for index, part in enumerate(parts):
        try:
            common.parse_set_dirname(part)
        except ValueError:
            continue
        parts[index] = common.set_dirname(*source)
        return Path(*parts)
    raise ValueError(f"{directory} has no <set>_<split> component")


def set_of(directory):
    for part in Path(directory).parts:
        try:
            return common.parse_set_dirname(part)
        except ValueError:
            continue
    raise ValueError(f"{directory} has no <set>_<split> component")


def evaluate_dir(directory, force=False):
    from bench.eval.evaluate import evaluate, load_protocol

    directory = Path(directory)
    out = directory / "metrics_test"
    if out.with_suffix(".json").is_file() and not force:
        print(f"{out}.json exists", flush=True)
        return common.read_json(out.with_suffix(".json"))
    if not (directory / "paper.json").is_file():
        raise FileNotFoundError(f"{directory} has no paper.json; it is not a scored E5 directory")
    test_set, split = set_of(directory)
    val_dir = val_dir_of(directory)
    use_val = (val_dir / "paper.json").is_file() and val_dir.is_dir() and any(val_dir.glob("MA/*.png"))
    exclude = sorted(set((common.load_e3a().get("exclude") or {}).get(test_set) or ()))
    proto = load_protocol(common.PROTOCOL_PATH)
    result = evaluate(
        common.data_dir(test_set), split, directory, proto,
        val_pred_dir=val_dir if use_val else None, val_split=common.THRESHOLD_SOURCE[1], out_prefix=out,
        val_dataset_dir=common.data_dir(common.THRESHOLD_SOURCE[0]) if use_val else None, exclude=exclude,
    )
    marker = common.done_marker(directory)
    done = common.read_json(marker) if marker.is_file() else {}
    done.update({"thresholds": "fixed and DDR val_opt" if use_val else "fixed only", "val_pred_dir": str(val_dir) if use_val else None,
                 "summary": result["summary"]})
    common.write_json(marker, done)
    summary = result["summary"]
    print(f"{directory}: mAUPR {summary['mAUPR']:.4f} any-lesion AUPR {summary.get('AUPR_any_lesion', float('nan')):.4f}", flush=True)
    return result


def pending(root=common.OUT_ROOT):
    """Scored directories (paper.json) without metrics_test.json, DDR_val first so the thresholds exist."""
    found = [p.parent for p in Path(root).rglob("paper.json") if not (p.parent / "metrics_test.json").is_file()]
    return sorted(found, key=lambda p: (set_of(p) != common.THRESHOLD_SOURCE, str(p)))


def evaluate_all(root=common.OUT_ROOT, force=False):
    results = {}
    for directory in pending(root) if not force else sorted(p.parent for p in Path(root).rglob("paper.json")):
        results[str(directory)] = evaluate_dir(directory, force=force)["summary"]
    return results
