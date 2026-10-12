#!/usr/bin/env python3
"""Freeze the training subsets of the factor experiments (docs/paper/16 section 3).

    bench/data/splits/subsets/DDR_train_10pct.json     38 of the 383 DDR training images (E8)
    bench/data/splits/subsets/DDR_train_50pct.json    192 of 383 (E8)
    bench/data/splits/subsets/FGADR_train_383.json    383 of the 1096 FGADR training images (E4, "as many as DDR")

Sampling: numpy PCG64 seeded with 20261012, one draw per list, uniform without
replacement over the frozen split in its splits.json order; the 10% list is a
subset of the 50% list (the first 38 of the same permutation), so the learning
curve nests. Written once, before any factor run, and committed; a second run
refuses to overwrite a file whose ids differ.

    python bench/scripts/make_subsets.py            # needs dataset/prepared/DDR and FGADR
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.common.io import load_split  # noqa: E402

OUT = REPO / "bench" / "data" / "splits" / "subsets"
SEED = 20261012
SPECS = [  # file, dataset, split, size, nested-from
    ("DDR_train_50pct.json", "DDR", "train", 192, None),
    ("DDR_train_10pct.json", "DDR", "train", 38, "DDR_train_50pct.json"),
    ("FGADR_train_383.json", "FGADR", "train", 383, None),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    permutations = {}
    for name, dataset, split, size, parent in SPECS:
        ids = load_split(REPO / "dataset" / "prepared" / dataset, split)
        key = (dataset, split)
        if key not in permutations:
            rng = np.random.Generator(np.random.PCG64(SEED))
            permutations[key] = [ids[i] for i in rng.permutation(len(ids))]
        chosen = sorted(permutations[key][:size], key=ids.index)
        doc = {
            "dataset": dataset,
            "split": split,
            "n": len(chosen),
            "of": len(ids),
            "seed": SEED,
            "rule": "PCG64 permutation of the frozen split; first n ids; nested lists share the permutation",
            "nested_in": parent,
            "ids": chosen,
        }
        path = OUT / name
        if path.exists():
            old = json.loads(path.read_text())
            if old["ids"] != chosen:
                raise SystemExit(f"{path} exists with different ids; subsets are frozen, not regenerated")
            print(f"{path.name}: unchanged ({len(chosen)} ids)")
            continue
        path.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {path.name}: {len(chosen)} of {len(ids)}")


if __name__ == "__main__":
    main()
