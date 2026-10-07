"""Split lists and exports for the independent sources.

    python bench/scripts/prepare_datasets.py splits TJDR|FGADR|Retinal-Lesions [--force]
    python bench/scripts/prepare_datasets.py check-diaretdb1
    python bench/scripts/prepare_datasets.py export TJDR|FGADR|DiaRetDB1|Retinal-Lesions
    python bench/scripts/prepare_datasets.py report NAME

splits draws once and writes bench/data/splits/<name>.json; commit it before
any training. Rerunning with the same draw is a no-op, a different draw is an
error unless --force. export reads only the committed lists and resumes after
the last finished image. check-diaretdb1 reads ground truth only. All of it
streams one image at a time.
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.data import splits as split_lists  # noqa: E402
from bench.data.prepare import PREPARED, PREPARERS, write_report  # noqa: E402

SPLIT_SOURCES = ("TJDR", "FGADR", "Retinal-Lesions")


def make_splits(name, force=False):
    if name == "TJDR":
        from bench.data import tjdr

        written = []
        for field_name, lists in tjdr.open_tjdr(_ROOT).make_splits().items():
            written.append((field_name, lists, tjdr.SPLIT_RULE))
    elif name == "FGADR":
        from bench.data import fgadr

        written = [("FGADR", fgadr.open_fgadr(_ROOT).make_splits(), fgadr.SPLIT_RULE)]
    elif name == "Retinal-Lesions":
        from bench.data import retlesion

        written = [("Retinal-Lesions", retlesion.open_retlesion(_ROOT).make_splits(), retlesion.SPLIT_RULE)]
    else:
        raise SystemExit(f"splits takes one of {SPLIT_SOURCES}")
    for field_name, lists, rule in written:
        path = split_lists.write_frozen(field_name, lists, rule, force=force)
        sizes = {k: len(lists[k]) for k in ("train", "val", "test")}
        print(f"{field_name} {sizes} -> {path.relative_to(_ROOT)}")


def check_diaretdb1():
    from bench.data import diaretdb

    data = diaretdb.open_diaretdb1(_ROOT)
    print("ellipse sense", json.dumps(diaretdb.ellipse_sense(data)))
    counts = diaretdb.positive_image_counts(data)
    for split, expected in diaretdb.PAPER_POSITIVE_IMAGES.items():
        row = ", ".join(f"{c} {counts[split][c]}/{expected[c]}" for c in ("MA", "HE", "EX", "SE"))
        print(f"{split} (ours/paper): {row}")
    problems = diaretdb.count_problems(counts)
    for item in problems:
        print(f"  differs: {item}")
    if not problems:
        print("matches the paper except the recorded", dict(diaretdb.KNOWN_COUNT_DIFFERENCES))
    return 1 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("splits")
    p.add_argument("name", choices=SPLIT_SOURCES)
    p.add_argument("--force", action="store_true")
    sub.add_parser("check-diaretdb1")
    p = sub.add_parser("export")
    p.add_argument("name", choices=sorted(PREPARERS))
    p = sub.add_parser("report")
    p.add_argument("name", choices=sorted(n for n in PREPARED if n not in ("IDRiD", "DDR")))
    args = parser.parse_args(argv)

    if args.command == "splits":
        make_splits(args.name, args.force)
    elif args.command == "check-diaretdb1":
        return check_diaretdb1()
    elif args.command == "export":
        result = PREPARERS[args.name](_ROOT, log=lambda line: print(line, flush=True))
        reports = result if args.name == "TJDR" else {args.name: result}
        for name, report in reports.items():
            sizes = {s: v["n_images"] for s, v in report["splits"].items()}
            print(f"{name} exported {sizes} -> {PREPARED[name]}/report.json")
    elif args.command == "report":
        write_report(_ROOT / PREPARED[args.name])
        print(f"{PREPARED[args.name]}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
