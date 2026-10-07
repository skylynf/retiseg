"""Seeded stratified holdouts, frozen as JSON lists under bench/data/splits/.

Each dataset draws from its own Generator(PCG64(SEED)). Strata are visited in
sorted order and the groups inside a stratum are sorted before the
permutation, so the draw depends only on the ids, the strata and the seed.
The number held out per stratum is a largest-remainder share of
round(fraction * total groups); ties go to the earlier stratum. A group is a
patient where the release has one, otherwise a single image. The lists are
generated once, committed before any training, and read from the JSON after
that. bench/tests/test_new_datasets.py regenerates them when the raw data is
present and checks that nothing moved.
"""

import json
import math
from pathlib import Path

import numpy as np

SEED = 20261003
FROZEN = Path(__file__).with_name("splits")


def generator():
    return np.random.Generator(np.random.PCG64(SEED))


def _quotas(sizes, fraction):
    total = sum(sizes.values())
    target = int(round(fraction * total))
    exact = {key: fraction * size for key, size in sizes.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    order = sorted(sizes, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[: target - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def holdout(strata, fraction, rng):
    """Split {stratum: [group ids]} into (kept, held) sorted lists of group ids."""
    strata = {key: sorted(groups) for key, groups in strata.items() if groups}
    quotas = _quotas({key: len(groups) for key, groups in strata.items()}, fraction)
    kept, held = [], []
    for key in sorted(strata):
        groups = strata[key]
        order = rng.permutation(len(groups))
        chosen = {groups[i] for i in order[: quotas[key]]}
        held.extend(g for g in groups if g in chosen)
        kept.extend(g for g in groups if g not in chosen)
    return sorted(kept), sorted(held)


def by_stratum(ids, stratum_of):
    out = {}
    for image_id in ids:
        out.setdefault(stratum_of(image_id), []).append(image_id)
    return out


def expand(groups, members):
    return sorted(image_id for group in groups for image_id in members[group])


def check_disjoint(splits):
    seen = {}
    for split, ids in splits.items():
        for image_id in ids:
            if image_id in seen:
                raise ValueError(f"{image_id} is in both {seen[image_id]} and {split}")
            seen[image_id] = split


def frozen_path(name):
    return FROZEN / f"{name}.json"


def write_frozen(name, splits, rule, force=False):
    check_disjoint({k: splits[k] for k in ("train", "val", "test")})
    path = frozen_path(name)
    if path.is_file() and not force:
        old = json.loads(path.read_text())
        if {k: old[k] for k in ("train", "val", "test")} != {k: list(splits[k]) for k in ("train", "val", "test")}:
            raise RuntimeError(f"{path} exists with different lists; pass force only before any training")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "seed": SEED, "rule": rule}
    payload.update({k: list(v) for k, v in splits.items()})
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def read_frozen(name):
    path = frozen_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run bench/scripts/prepare_datasets.py splits {name}")
    payload = json.loads(path.read_text())
    return {k: list(payload[k]) for k in ("train", "val", "test")}
