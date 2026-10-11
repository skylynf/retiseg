"""Freeze, per training source, the test images that show an eye the source has seen.

Reads the vessel pairs of the leak audit (bench/scripts/leak_audit.py,
--min-vessel 0.3) and writes bench/data/splits/leak_exclusions.json. A model
trained on source S has seen S's train and val images; on every test set it is
evaluated on, the test images linked to one of those at or above the pair's
threshold are left out. A model trained on several sources leaves out the
union. Thresholds come from looking at the pairs (docs/ops/00, 2026-10-12):
0.5 in general (same photographs score 0.9-1.0, same eyes 0.5-0.9, different
eyes reach 0.40 in Retinal-Lesions and DDR); FGADR against FGADR 0.33, because
there the same eye was seen down to 0.34 and different eyes from 0.32.

    python bench/scripts/leak_exclusions.py --audit runs/audit/leak_all.json
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
OUT = _ROOT / "bench" / "data" / "splits" / "leak_exclusions.json"
SOURCES = ("IDRiD", "DDR", "TJDR_std", "TJDR_uwf", "FGADR", "Retinal-Lesions")
TEST_SETS = ("IDRiD", "DDR", "TJDR_std", "TJDR_uwf", "FGADR", "Retinal-Lesions", "DiaRetDB1")
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = {("FGADR", "FGADR"): 0.33}
SEEN_SPLITS = ("train", "val")


def threshold(a, b):
    return THRESHOLDS.get((a, b), THRESHOLDS.get((b, a), DEFAULT_THRESHOLD))


def exclusions(pairs, min_listed):
    """{source: {test set: sorted ids}} and the pairs that caused them."""
    table = {source: {test: set() for test in TEST_SETS} for source in SOURCES}
    evidence = []
    for pair in pairs:
        for seen, test in ((pair["a"], pair["b"]), (pair["b"], pair["a"])):
            if seen["dataset"] not in SOURCES or seen["split"] not in SEEN_SPLITS:
                continue
            if test["dataset"] not in TEST_SETS or test["split"] != "test":
                continue
            cut = threshold(seen["dataset"], test["dataset"])
            if cut < min_listed:
                raise ValueError(f"threshold {cut} is below the audit's listing floor {min_listed}")
            if pair["vessel_ncc"] < cut:
                continue
            table[seen["dataset"]][test["dataset"]].add(test["id"])
            evidence.append(
                {
                    "source": seen["dataset"],
                    "seen": f"{seen['split']}:{seen['id']}",
                    "test_set": test["dataset"],
                    "test_id": test["id"],
                    "vessel_ncc": pair["vessel_ncc"],
                }
            )
    out = {source: {test: sorted(ids) for test, ids in tests.items()} for source, tests in table.items()}
    return out, sorted(evidence, key=lambda e: (e["source"], e["test_set"], -e["vessel_ncc"]))


def for_sources(table, sources):
    """Union of the exclusions of a training set built from several sources (E4, E8)."""
    merged = {test: set() for test in TEST_SETS}
    for source in sources:
        for test, ids in table[source].items():
            merged[test].update(ids)
    return {test: sorted(ids) for test, ids in merged.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=str(_ROOT / "runs" / "audit" / "leak_all.json"))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    audit = json.loads(Path(args.audit).read_text())
    min_listed = float(audit["summary"]["min_vessel"])
    table, evidence = exclusions(audit["vessel_pairs"], min_listed)
    payload = {
        "rule": (
            "A model trained on a source leaves out, on every test set, the test images whose vessel signature "
            "correlates with one of that source's train or val images at or above the threshold of the pair "
            "of datasets. Several sources: the union."
        ),
        "audit": {
            "file": "runs/audit/leak_all.json",
            "command": "bench/scripts/leak_audit.py --min-vessel 0.3 (all seven prepared sources)",
            "n_images": audit["summary"]["n_images"],
            "signature": audit["summary"]["vessel"],
        },
        "thresholds": {"default": DEFAULT_THRESHOLD, **{f"{a}~{b}": t for (a, b), t in THRESHOLDS.items()}},
        "by_training_source": table,
        "counts": {source: {test: len(ids) for test, ids in tests.items() if ids} for source, tests in table.items()},
        "evidence": evidence,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["counts"], indent=1))


if __name__ == "__main__":
    main()
