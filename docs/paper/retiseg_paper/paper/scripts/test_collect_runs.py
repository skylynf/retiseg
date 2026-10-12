"""End-to-end check of collect_runs.py against a report written by the real evaluator.

    python paper/scripts/test_collect_runs.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
for _p in [PAPER, *PAPER.parents]:
    if (_p / "bench" / "configs" / "models.yaml").exists():
        sys.path.insert(0, str(_p))
        break
sys.path.insert(0, str(PAPER / "scripts"))

import collect_runs  # noqa: E402
from bench.common import io  # noqa: E402
from bench.eval.evaluate import evaluate, load_protocol  # noqa: E402

CLASSES = ["MA", "HE", "EX", "SE"]


def disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def toy(root):
    shape, rng = (120, 160), np.random.default_rng(0)
    ids = {"train": [], "val": ["v0", "v1"], "test": ["t0", "t1", "t2"]}
    truth = {}
    for image_id in ids["val"] + ids["test"]:
        fov = disk(shape, 60, 80, 58)
        (root / "fov").mkdir(parents=True, exist_ok=True)
        Image.fromarray(fov.astype(np.uint8) * 255).save(root / "fov" / f"{image_id}.png")
        truth[image_id] = {}
        for c in CLASSES:
            m = np.zeros(shape, bool)
            for _ in range(rng.integers(1, 4)):
                m |= disk(shape, rng.integers(20, 100), rng.integers(30, 130), rng.integers(1, 6))
            m &= fov
            (root / "masks" / c).mkdir(parents=True, exist_ok=True)
            Image.fromarray(m.astype(np.uint8) * 255).save(root / "masks" / c / f"{image_id}.png")
            truth[image_id][c] = m
    (root / "meta.json").write_text(json.dumps({"name": "TOY", "classes": CLASSES, "granularity": "fine"}))
    (root / "splits.json").write_text(json.dumps(ids))
    return ids, truth


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        ids, truth = toy(tmp / "TOY")
        rng = np.random.default_rng(1)
        run = tmp / "runs" / "BS_unet_ddr_seed0"
        for split, name in (("val", "prob_val"), ("test", "prob_test")):
            for image_id in ids[split]:
                for c in CLASSES:
                    p = np.clip(truth[image_id][c] * 0.8 + 0.05 + rng.normal(0, 0.05, truth[image_id][c].shape), 0, 1)
                    io.write_prob(run / name, c, image_id, p)
        proto = load_protocol()
        report = evaluate(tmp / "TOY", "test", run / "prob_test", proto, val_pred_dir=run / "prob_val", out_prefix=run / "metrics_test")
        (tmp / "runs" / "BSprobe_unet_ddr_seed0").mkdir()  # must be ignored
        new, seen = collect_runs.collect(tmp / "runs")
        assert len(seen) == 1 and seen[0][0] == "BS_unet_ddr_seed0", seen
        assert set(new["cls"]) == set(CLASSES)
        assert (new[["experiment", "model", "variant", "train_set", "seed", "test_set"]].drop_duplicates().values.tolist()
                == [["E2", "U-Net", "bstd", "DDR", 0, "DDR"]])
        got = new[(new.cls == "MA") & (new.metric == "aupr")]["value"].iloc[0]
        assert abs(got - report["per_class"]["MA"]["aupr"]) < 1e-12
        needed = {"aupr", "aupr_11pt", "aupr_ci_lo", "aupr_ci_hi", "dice_fixed", "dice_valopt", "iou_valopt", "lesion_precision",
                  "lesion_recall_small", "lesion_recall_medium", "lesion_recall_large"}
        assert needed <= set(new["metric"]), needed - set(new["metric"])
        table = pd.read_csv(PAPER / "data" / "results.csv", keep_default_na=False)
        # the test starts from a placeholder-only table so that the merge arithmetic is checked
        table = table[(table.experiment != "E2") | (table.source != "measured")]
        merged, replaced, added, stale = collect_runs.merge(table, new)
        key = (merged.experiment == "E2") & (merged.model == "U-Net") & (merged.train_set == "DDR") & (merged.seed == 0) & (merged.test_set == "DDR")
        cell = merged[key & merged.metric.isin(needed)]
        assert (cell["source"] == "measured").all() and len(cell) == len(new), (len(cell), len(new))
        assert replaced + added == len(new), (replaced, added)
        # placeholder seeds of the measured (U-Net, DDR, DDR) group are dropped; other models keep theirs
        other = merged[(merged.experiment == "E2") & (merged.model == "U-Net") & (merged.variant == "bstd") & (merged.train_set == "DDR") & (merged.test_set == "DDR")]
        assert (other["source"] == "measured").all() and set(other["seed"]) == {0}, set(other["seed"])
        assert len(merged) == len(table) - replaced - stale + len(new)
        assert new[new.metric == "lesion_recall_small"]["value"].isna().all()  # toy lesions are all large
        print(f"ok: {len(new)} rows from one report replaced {replaced} placeholder rows; nothing else changed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
