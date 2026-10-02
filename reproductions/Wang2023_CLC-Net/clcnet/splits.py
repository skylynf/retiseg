"""Dataset splits. IDRiD and DDR stay on their official folders.

e-ophtha has no official 15/6 file. Assumption A17 fixes the draw.
Once splits/eophtha_seed20230108.json exists, that file is the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clcnet.assumptions import EOPHTHA_EXPECTED, EOPHTHA_N_TEST, EOPHTHA_N_TRAIN, SEED

SPLIT_NAME = f"eophtha_seed{SEED}.json"


def eophtha_split(stems, seed: int = SEED):
    stems = sorted(stems)
    if len(stems) != EOPHTHA_EXPECTED:
        raise ValueError(
            f"e-ophtha images that contain both MA and EX must be {EOPHTHA_EXPECTED}, found {len(stems)}"
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(len(stems))
    ordered = [stems[int(index)] for index in order]
    train = ordered[:EOPHTHA_N_TRAIN]
    test = ordered[EOPHTHA_N_TRAIN : EOPHTHA_N_TRAIN + EOPHTHA_N_TEST]
    return train, test


def load_or_create_eophtha_split(stems, path: Path, seed: int = SEED):
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("seed") != seed:
            raise ValueError(f"split file seed is {payload.get('seed')}, locked seed is {seed}")
        return payload["train"], payload["test"], False
    train, test = eophtha_split(stems, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "generator": "numpy.random.Generator(PCG64)",
                "rule": "sort stems, permute, first 15 train, last 6 test",
                "train": train,
                "test": test,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return train, test, True
