"""Pack the server's small result files for the workstation (docs/ops/17 section 1).

    python bench/scripts/pack_handoff.py                      # -> retiseg_runs_handoff_<date>.tar.gz next to the repo
    python bench/scripts/pack_handoff.py --since runs/handoff_manifest_20261014.tsv   # "new since" against an earlier pack
    python bench/scripts/pack_handoff.py --list                # what would go in, nothing written

What goes in: every file under runs/ with one of the SMALL suffixes (metrics,
recipes, histories, console logs, markers, done.json, paper.json, prompts),
each at most --max-mb (default 8). What stays out: *.npz, *.pt, *.pth, *.png,
prob_val / prob_test / masks directories, nnU-Net's raw and preprocessed
trees, and anything else. The pack also carries summary/scored_metrics.tsv
(one row per metrics file: run, test set, n_images, excluded, mAUPR, per-class
AUPR, any-lesion AUPR, git), summary/new_since.tsv when --since is given,
snapshot/ (git head, log, status, nvidia-smi, date) and docs/ops/00_进度.txt.
A manifest (path, size, sha256) is written next to the pack and inside it,
for the next --since.

Metrics written without the exclusions their training sources require
(bench/data/splits/leak_exclusions.json) are listed in summary/unaligned.tsv
and are NOT packed: the number must not reach the paper (17 section 14).
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SMALL = {".json", ".jsonl", ".yaml", ".yml", ".log", ".txt", ".tsv", ".csv", ".md"}
SKIP_DIRS = {"prob_val", "prob_test", "masks", "nnunet", "nnunet_rep1", "nnunet_rep2", "nnunet_val", "nnunet_test", "__pycache__"}
MARKERS = {"done.json", "paper.json", "predicted.json", "timing.json"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def small_files(runs=_ROOT / "runs", max_bytes=8 << 20):
    out = []
    for path in sorted(runs.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or any(part.startswith("nnunet") for part in rel.parts[1:2]):
            continue
        if path.suffix.lower() not in SMALL and path.name not in MARKERS and path.suffix != "":
            continue
        if path.suffix == "" and path.name.startswith(".") is False and path.name not in MARKERS:
            continue   # unnamed files: only dot-markers such as .prob_test_done
        if path.stat().st_size > max_bytes:
            continue
        out.append(rel)
    return out


def expected_exclusions(metrics_path):
    """The exclusion list a metrics file should carry, or None when the rule does not apply."""
    from bench.e3a import PREPARED_TO_SET, leak_exclusions, load_frozen, train_sources_of

    import yaml

    run = metrics_path.parent
    if metrics_path.name == "metrics.json" and run.parent.parent.name == "E3a":    # runs/E3a/<source>/<set>/metrics.json
        source = run.parent
        config = source and (_ROOT / "runs" / source.name / "config.yaml")
        test = run.name.split("__")[0]
        if config.is_file():
            trained = train_sources_of(yaml.safe_load(config.read_text()))
        elif source.name.startswith("NN_"):
            from bench.scripts.nnunet_bridge import DATASET_IDS

            lower = {k.lower(): k for k in DATASET_IDS}
            trained = [lower[source.name.split("_")[2]]] if source.name.split("_")[2] in lower else []
            if source.name.count("_") >= 4 and "_".join(source.name.split("_")[2:4]) in lower:
                trained = [lower["_".join(source.name.split("_")[2:4])]]
        else:
            return None
        return sorted(leak_exclusions(trained, test)) if test in load_frozen()["test_sets"] else None
    if metrics_path.name == "metrics_test.json":
        config = run / "config.yaml"
        if config.is_file():
            cell = yaml.safe_load(config.read_text())
            if cell.get("experiment") != "BF":
                return []
            test = PREPARED_TO_SET.get(Path(cell["dataset"]).name, Path(cell["dataset"]).name)
            return sorted(leak_exclusions(train_sources_of(cell), test))
        if run.name.startswith("NN_"):
            from bench.scripts.nnunet_bridge import DATASET_IDS

            lower = {k.lower(): k for k in DATASET_IDS}
            parts = run.name.split("_")
            train = lower.get("_".join(parts[2:4])) or lower.get(parts[2])
            return sorted(leak_exclusions([train], train)) if train else None
    return None


def metrics_rows(files):
    rows, unaligned = [], []
    head = git(["rev-parse", "--short", "HEAD"])
    for rel in files:
        if rel.name not in ("metrics_test.json", "metrics.json"):
            continue
        path = _ROOT / rel
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        summary = report.get("summary") or {}
        if "mAUPR" not in summary:
            continue
        excluded = sorted(report.get("excluded_ids") or [])
        expected = expected_exclusions(path)
        aligned = expected is None or excluded == expected
        row = {
            "metrics": str(rel), "run": rel.parts[1] if rel.parts[0] == "runs" else "", "n_images": report.get("n_images"),
            "n_excluded": len(excluded), "exclusions_aligned": "yes" if aligned else f"no (expected {len(expected)})",
            "mAUPR": summary.get("mAUPR"), "mAUPR_11thr": summary.get("mAUPR_m2mrf_impl"),
            "mDice_fixed": summary.get("mDice_pooled_fixed"), "mDice_val_opt": summary.get("mDice_pooled_val_opt"),
            "AUPR_any_lesion": summary.get("AUPR_any_lesion"),
            **{f"AUPR_{cls}": (report.get("per_class", {}).get(cls) or {}).get("aupr") for cls in ("MA", "HE", "EX", "SE")},
            "git": head,
        }
        (rows if aligned else unaligned).append(row)
    return rows, unaligned


def git(args):
    try:
        return subprocess.check_output(["git", *args], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    lines = ["\t".join(keys)] + ["\t".join("" if row.get(k) is None else str(row.get(k)) for k in keys) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def read_manifest(path):
    out = {}
    for line in Path(path).read_text().splitlines()[1:]:
        rel, size, digest = line.split("\t")
        out[rel] = digest
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="tar.gz path (default: ../retiseg_runs_handoff_<date>.tar.gz)")
    parser.add_argument("--since", help="manifest.tsv of an earlier pack; summary/new_since.tsv lists changed metrics files")
    parser.add_argument("--max-mb", type=float, default=8.0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    stamp = datetime.date.today().strftime("%Y%m%d")
    files = small_files(max_bytes=int(args.max_mb * (1 << 20)))
    rows, unaligned = metrics_rows(files)
    bad = {row["metrics"] for row in unaligned}
    files = [rel for rel in files if str(rel) not in bad]
    if args.list:
        for rel in files:
            print(rel)
        print(f"{len(files)} files, {len(rows)} metrics rows, {len(unaligned)} unaligned metrics files left out", flush=True)
        for row in unaligned:
            print(f"  unaligned: {row['metrics']} ({row['exclusions_aligned']})")
        return
    name = f"retiseg_runs_handoff_{stamp}"
    out = Path(args.out) if args.out else _ROOT.parent / f"{name}.tar.gz"
    manifest = [(str(rel), (_ROOT / rel).stat().st_size, sha256_of(_ROOT / rel)) for rel in files]
    work = _ROOT / "runs" / "logs" / "handoff" / name
    work.mkdir(parents=True, exist_ok=True)
    write_tsv(work / "summary" / "scored_metrics.tsv", rows)
    write_tsv(work / "summary" / "unaligned.tsv", unaligned)
    if args.since:
        before = read_manifest(args.since)
        changed = {rel for rel, _size, digest in manifest if before.get(rel) != digest}
        write_tsv(work / "summary" / "new_since.tsv", [row for row in rows if row["metrics"] in changed])
    (work / "manifest.tsv").write_text("path\tbytes\tsha256\n" + "".join(f"{p}\t{s}\t{d}\n" for p, s, d in manifest))
    snap = work / "snapshot"
    snap.mkdir(exist_ok=True)
    (snap / "git_head.txt").write_text(git(["rev-parse", "HEAD"]) + "\n")
    (snap / "git_log.txt").write_text(git(["log", "--oneline", "-20"]) + "\n")
    (snap / "git_status.txt").write_text(git(["status", "--short", "--branch"]) + "\n")
    (snap / "date.txt").write_text(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
    try:
        (snap / "nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    except (OSError, subprocess.CalledProcessError):
        pass
    with tarfile.open(out, "w:gz") as tar:
        for rel in files:
            tar.add(_ROOT / rel, arcname=f"{name}/{rel}")
        for extra in sorted(work.rglob("*")):
            if extra.is_file():
                tar.add(extra, arcname=f"{name}/{extra.relative_to(work)}")
        progress = _ROOT / "docs" / "ops" / "00_进度.txt"
        if progress.is_file():
            tar.add(progress, arcname=f"{name}/docs/ops/00_进度.txt")
    (_ROOT / "runs" / f"handoff_manifest_{stamp}.tsv").write_text((work / "manifest.tsv").read_text())
    print(f"{out}: {len(files)} files, {len(rows)} metrics rows; {len(unaligned)} unaligned metrics files left out "
          f"(summary/unaligned.tsv); manifest runs/handoff_manifest_{stamp}.tsv", flush=True)


if __name__ == "__main__":
    main()
