"""Results by DR grade: the clinical strata of docs/paper/26 N5a (docs/paper/14 M17a).

Screening cares most about mild disease, where lesions are few and small. For
every finished independent test (runs/E3a/<run>/<set>) and every DDR internal
test (runs/<run>/metrics_test.npz) on a set whose images carry a DR grade
(DDR, FGADR, Retinal-Lesions, MAPLES-DR), the per-image arrays the evaluator
stored in metrics.npz are split into two strata, grade 0-1 and grade 2-4, and
scored again inside each: pooled AUPR per class from the per-image histograms
(the 4,096-bin coarsening the bootstrap also uses), the mean per-image Dice at
the validation-optimal threshold, and the image-level AUC from the stored
max-in-FOV scores. No probability map is read, so this runs wherever runs/
holds the npz files (server or workstation; the handoff packages leave the npz
out). Lesion-level recall per stratum needs the maps and is not produced here.

Grades come from the prepared directories (FGADR, Retinal-Lesions: stats.grade
in export_cases.jsonl; MAPLES-DR: stats.dr, "R2" -> 2) and, for DDR, from the
official DR_grading/<split>.txt lists of the release (bench/data/ddr.py).
Ungradable images and images without a grade are left out of both strata and
counted.

    python bench/scripts/clinical_strata.py grades            # -> runs/CLIN/grades.json
    python bench/scripts/clinical_strata.py strata            # -> runs/CLIN/strata/<run>/<set>/strata.json
    python bench/scripts/clinical_strata.py summary           # -> runs/CLIN/strata_summary.json (+ a printed table)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.common.io import LESION_CLASSES  # noqa: E402
from bench.eval import imagelevel, pixel  # noqa: E402

GRADED_SETS = ("DDR", "FGADR", "Retinal-Lesions", "MAPLES-DR")
PREPARED_TO_SET = {"RetinalLesions": "Retinal-Lesions"}      # prepared directory name -> set name (bench/data/prepare.PREPARED)
PREPARED_DIR = {"DDR": "DDR", "FGADR": "FGADR", "Retinal-Lesions": "RetinalLesions", "MAPLES-DR": "MAPLES-DR"}
STRATA = {"grade_0_1": (0, 1), "grade_2_4": (2, 3, 4)}
CLIN_ROOT = REPO / "runs" / "CLIN"
E3A_ROOT = REPO / "runs" / "E3a"
PREPARED = REPO / "dataset" / "prepared"


# ---------------------------------------------------------------------------------------------------------------- grades


def _grade_of(stats):
    """The grade recorded by prepare: {'grade': {'g2': 1}} (FGADR, Retinal-Lesions) or {'dr': {'R2': 1}} (MAPLES-DR)."""
    for key in ("grade", "dr"):
        value = stats.get(key)
        if isinstance(value, dict) and value:
            label = next(iter(value))
            m = re.search(r"\d", str(label))
            if m:
                g = int(m.group(0))
                return g if g in (0, 1, 2, 3, 4) else None
    return None


def grades_from_prepared(prepared_dir):
    """{image id: grade} from export_cases.jsonl of one prepared directory (ids without a grade are absent)."""
    path = Path(prepared_dir) / "export_cases.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        grade = _grade_of(case)
        if grade is not None:
            out[case["id"]] = grade
    return out


def grades_ddr(repo_root=REPO):
    from bench.data.ddr import open_ddr

    data = open_ddr(repo_root)
    out = {}
    for split in ("train", "valid", "test"):
        out.update(data.grades(split))
    return out


def build_grades(prepared_root=PREPARED, repo_root=REPO, sets=GRADED_SETS):
    grades = {}
    for name in sets:
        if name == "DDR":
            grades[name] = grades_ddr(repo_root)
        else:
            grades[name] = grades_from_prepared(Path(prepared_root) / PREPARED_DIR[name])
    return grades


# ---------------------------------------------------------------------------------------------------------------- strata


def stratum_scores(npz, classes, keep):
    """Per-class pooled AUPR, mean per-image Dice (val_opt, else fixed) and image-level AUC on the images in ``keep``."""
    out = {}
    for cls in classes:
        if f"{cls}_pos" not in npz:
            continue
        pos = npz[f"{cls}_pos"][keep].sum(axis=0)
        neg = npz[f"{cls}_neg"][keep].sum(axis=0)
        entry = {"n_positive_pixels": int(pos.sum()), "aupr": float(pixel.average_precision(pos, neg)) if pos.sum() > 0 else None}
        for tname in ("val_opt", "fixed"):
            key = f"{cls}_dice_{tname}"
            if key in npz:
                d = npz[key][keep]
                d = d[~np.isnan(d)]
                entry[f"dice_{tname}_mean"] = float(d.mean()) if d.size else None
        if f"{cls}_img_label" in npz:
            labels = npz[f"{cls}_img_label"][keep].astype(bool)
            scores = npz[f"{cls}_img_max"][keep]
            entry["n_images_with_lesion"] = int(labels.sum())
            if labels.any() and (~labels).any():
                fpr, tpr, _ = imagelevel.roc_points(scores, labels)
                entry["img_auc"] = float(imagelevel.auc(fpr, tpr))
            else:
                entry["img_auc"] = None
        out[cls] = entry
    means = {}
    for metric, key in (("aupr", "mAUPR"), ("img_auc", "mean_img_auc"), ("dice_val_opt_mean", "mean_dice_val_opt")):
        values = [out[c][metric] for c in out if out[c].get(metric) is not None]
        means[key] = float(np.mean(values)) if values else None
        means[f"{key}_n_classes"] = len(values)
    return {"per_class": out, "summary": means}


def stratify(npz_path, grades_of_set, classes=LESION_CLASSES):
    """The two strata of one metrics.npz; ids without a grade are counted, not scored."""
    npz = np.load(npz_path, allow_pickle=False)
    ids = [str(i) for i in npz["image_ids"]]
    grade = np.array([grades_of_set.get(i, -1) for i in ids])
    result = {"n_images": len(ids), "n_without_grade": int((grade < 0).sum()), "strata": {}}
    for name, levels in STRATA.items():
        keep = np.isin(grade, levels)
        if keep.sum() == 0:
            result["strata"][name] = {"n_images": 0}
            continue
        scores = stratum_scores(npz, classes, keep)
        result["strata"][name] = {"n_images": int(keep.sum()), "grades": sorted(int(g) for g in set(grade[keep])), **scores}
    return result


def _internal_dataset(run):
    """The evaluation set of a run's internal test, as metrics_test.json names it (prepared name -> set name)."""
    path = run / "metrics_test.json"
    if not path.is_file():
        return None
    try:
        name = json.loads(path.read_text()).get("dataset")
    except ValueError:
        return None
    return PREPARED_TO_SET.get(name, name)


def targets(runs_root=REPO / "runs"):
    """(run name, set, npz path, kind) for every finished independent test on a graded set (runs/E3a/<run>/<set>)
    and every internal test whose evaluation set is graded (runs/<run>/metrics_test.npz: the DDR rows of the main
    table, and the BF cells evaluated on DDR, FGADR or Retinal-Lesions)."""
    found = []
    e3a = Path(runs_root) / "E3a"
    if e3a.is_dir():
        for run in sorted(p for p in e3a.iterdir() if p.is_dir()):
            for setdir in sorted(p for p in run.iterdir() if p.is_dir()):
                if setdir.name in GRADED_SETS and (setdir / "done.json").is_file() and (setdir / "metrics.npz").is_file():
                    found.append((run.name, setdir.name, setdir / "metrics.npz", "independent"))
    for run in sorted(p for p in Path(runs_root).iterdir() if p.is_dir()):
        npz = run / "metrics_test.npz"
        if not npz.is_file():
            continue
        dataset = _internal_dataset(run)
        if dataset in GRADED_SETS:
            found.append((run.name, dataset, npz, "internal"))
    return found


def run_strata(grades, runs_root=REPO / "runs", out_root=CLIN_ROOT / "strata", force=False, log=print):
    written = 0
    for run, set_name, npz_path, kind in targets(runs_root):
        out = Path(out_root) / run / set_name / "strata.json"
        if out.is_file() and not force:
            continue
        result = stratify(npz_path, grades.get(set_name, {}))
        result.update({"run": run, "test_set": set_name, "kind": kind, "source_npz": str(npz_path.relative_to(REPO)) if str(npz_path).startswith(str(REPO)) else str(npz_path)})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
        written += 1
        lo = result["strata"]["grade_0_1"].get("summary", {}).get("mAUPR")
        hi = result["strata"]["grade_2_4"].get("summary", {}).get("mAUPR")
        log(f"{run} {set_name}: grade 0-1 mAUPR {lo if lo is None else round(lo, 4)} (n={result['strata']['grade_0_1']['n_images']}), "
            f"grade 2-4 {hi if hi is None else round(hi, 4)} (n={result['strata']['grade_2_4']['n_images']})")
    return written


def summarise(out_root=CLIN_ROOT / "strata"):
    rows = []
    for path in sorted(Path(out_root).glob("*/*/strata.json")):
        r = json.loads(path.read_text())
        row = {"run": r["run"], "test_set": r["test_set"], "kind": r["kind"], "n_without_grade": r["n_without_grade"]}
        for name in STRATA:
            s = r["strata"].get(name, {})
            row[f"{name}_n"] = s.get("n_images", 0)
            row[f"{name}_mAUPR"] = (s.get("summary") or {}).get("mAUPR")
            row[f"{name}_img_auc"] = (s.get("summary") or {}).get("mean_img_auc")
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grades", help="collect the DR grade of every graded test image -> runs/CLIN/grades.json")
    g.add_argument("--out", default=str(CLIN_ROOT / "grades.json"))
    s = sub.add_parser("strata", help="score the two grade strata of every finished test on a graded set")
    s.add_argument("--grades", default=str(CLIN_ROOT / "grades.json"))
    s.add_argument("--runs", default=str(REPO / "runs"))
    s.add_argument("--out", default=str(CLIN_ROOT / "strata"))
    s.add_argument("--force", action="store_true")
    m = sub.add_parser("summary", help="one table of the strata written so far")
    m.add_argument("--out", default=str(CLIN_ROOT / "strata"))
    args = parser.parse_args(argv)
    if args.command == "grades":
        grades = build_grades()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(grades, indent=1, sort_keys=True) + "\n")
        for name, table in grades.items():
            counts = {g: sum(v == g for v in table.values()) for g in range(5)}
            print(f"{name}: {len(table)} graded images {counts}")
    elif args.command == "strata":
        grades = json.loads(Path(args.grades).read_text())
        n = run_strata(grades, Path(args.runs), Path(args.out), args.force)
        print(f"wrote {n} strata files under {args.out}")
    else:
        rows = summarise(Path(args.out))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        (Path(args.out).parent / "strata_summary.json").write_text(json.dumps(rows, indent=1) + "\n")
        for row in rows:
            print(f"{row['run']:40s} {row['test_set']:16s} 0-1: {row['grade_0_1_mAUPR']} (n={row['grade_0_1_n']})  2-4: {row['grade_2_4_mAUPR']} (n={row['grade_2_4_n']})")


if __name__ == "__main__":
    main()
