"""bench/scripts/clinical_strata.py: grade lookups and the two strata read from the evaluator's own npz (26 N5a)."""

import json

import numpy as np
import pytest
from PIL import Image

from bench.common import io
from bench.data.ddr import parse_grading_text
from bench.scripts import clinical_strata as cs


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _prepared(root, ids, grades, seed):
    """A toy prepared set with export_cases.jsonl carrying a grade per image (as prepare writes for FGADR)."""
    shape = (64, 80)
    rng = np.random.default_rng(seed)
    truth = {}
    (root / "fov").mkdir(parents=True)
    lines = []
    for image_id in ids:
        fov = _disk(shape, 32, 40, 30)
        Image.fromarray(fov.astype(np.uint8) * 255).save(root / "fov" / f"{image_id}.png")
        masks = {}
        for c in ("MA", "HE", "EX", "SE"):
            m = np.zeros(shape, bool)
            if grades[image_id] >= 2 or c == "MA":
                for _ in range(1 + grades[image_id]):
                    m |= _disk(shape, rng.integers(10, 54), rng.integers(12, 68), rng.integers(1, 4))
            masks[c] = m & fov
            (root / "masks" / c).mkdir(parents=True, exist_ok=True)
            Image.fromarray(masks[c].astype(np.uint8) * 255).save(root / "masks" / c / f"{image_id}.png")
        truth[image_id] = masks
        lines.append(json.dumps({"id": image_id, "grade": {f"g{grades[image_id]}": 1}}))
    (root / "meta.json").write_text(json.dumps({"name": "FGADR", "classes": ["MA", "HE", "EX", "SE"], "granularity": "coarse"}))
    (root / "splits.json").write_text(json.dumps({"train": [], "val": ids[:2], "test": ids}))
    (root / "export_cases.jsonl").write_text("\n".join(lines) + "\n")
    return truth


def test_parse_grading_text_keeps_grades_0_to_4_only():
    text = "007-0004-000.jpg 2\n007-0005-000.jpg 5\nnot a row\n007-0006-000.jpg 0\n"
    assert parse_grading_text(text) == {"007-0004-000": 2, "007-0006-000": 0}


def test_grade_of_reads_both_prepare_formats():
    assert cs._grade_of({"grade": {"g3": 1}}) == 3
    assert cs._grade_of({"dr": {"R4A": 1}}) == 4
    assert cs._grade_of({"dr": {"": 1}}) is None and cs._grade_of({}) is None
    assert cs._grade_of({"grade": {"g5": 1}}) is None


@pytest.fixture
def scored(tmp_path):
    """Six toy images (grades 0,1,1,2,3,4) scored by the real evaluator; the npz is what the strata read."""
    from bench.eval.evaluate import evaluate, load_protocol

    ids = [f"img{i}" for i in range(6)]
    grades = dict(zip(ids, [0, 1, 1, 2, 3, 4]))
    prepared = tmp_path / "dataset" / "prepared" / "FGADR"
    truth = _prepared(prepared, ids, grades, 5)
    rng = np.random.default_rng(1)
    pred = tmp_path / "runs" / "BS_unet_ddr_seed0" / "prob_test"
    for image_id, masks in truth.items():
        for c, m in masks.items():
            io.write_prob(pred, c, image_id, np.clip(m * 0.7 + 0.1 + rng.normal(0, 0.15, m.shape), 0, 1))
    out = tmp_path / "runs" / "E3a" / "BS_unet_ddr_seed0" / "FGADR"
    evaluate(prepared, "test", pred, load_protocol(), val_pred_dir=pred, val_split="val", out_prefix=out / "metrics")
    (out / "done.json").write_text("{}")
    return tmp_path, prepared, grades, out


def test_grades_from_prepared_and_strata_split_the_npz_by_grade(scored):
    tmp_path, prepared, grades, out = scored
    assert cs.grades_from_prepared(prepared) == grades
    result = cs.stratify(out / "metrics.npz", grades)
    assert result["n_images"] == 6 and result["n_without_grade"] == 0
    lo, hi = result["strata"]["grade_0_1"], result["strata"]["grade_2_4"]
    assert lo["n_images"] == 3 and lo["grades"] == [0, 1] and hi["n_images"] == 3 and hi["grades"] == [2, 3, 4]
    assert set(hi["per_class"]) == {"MA", "HE", "EX", "SE"} and 0 <= hi["per_class"]["MA"]["aupr"] <= 1
    assert hi["summary"]["mAUPR"] is not None
    # the mild stratum has no HE/EX/SE lesions by construction: those classes have no positive pixels and no AUPR
    assert lo["per_class"]["HE"]["n_positive_pixels"] == 0 and lo["per_class"]["HE"]["aupr"] is None
    assert lo["per_class"]["MA"]["aupr"] is not None
    # an image without a grade is counted and scored in neither stratum
    partial = cs.stratify(out / "metrics.npz", {k: v for k, v in grades.items() if k != "img5"})
    assert partial["n_without_grade"] == 1 and partial["strata"]["grade_2_4"]["n_images"] == 2


def test_targets_and_run_strata_write_one_file_per_test(scored):
    tmp_path, _prepared_dir, grades, _out = scored
    runs = tmp_path / "runs"
    # an internal test on a graded set is found through metrics_test.json's dataset name; an IDRiD one is not
    internal = runs / "BF_e3b_unet_fgadr_seed0"
    internal.mkdir()
    (internal / "metrics_test.json").write_text(json.dumps({"dataset": "FGADR"}))
    np.savez(internal / "metrics_test.npz", **dict(np.load(_out / "metrics.npz")))
    other = runs / "BS_unet_idrid_seed0"
    other.mkdir()
    (other / "metrics_test.json").write_text(json.dumps({"dataset": "IDRiD"}))
    np.savez(other / "metrics_test.npz", image_ids=np.array(["a"]))
    found = cs.targets(runs)
    assert [(r, s, k) for r, s, _p, k in found] == [("BS_unet_ddr_seed0", "FGADR", "independent"), ("BF_e3b_unet_fgadr_seed0", "FGADR", "internal")]
    n = cs.run_strata({"FGADR": grades}, runs, tmp_path / "CLIN" / "strata", log=lambda *_: None)
    assert n == 2 and (tmp_path / "CLIN" / "strata" / "BS_unet_ddr_seed0" / "FGADR" / "strata.json").is_file()
    assert cs.run_strata({"FGADR": grades}, runs, tmp_path / "CLIN" / "strata", log=lambda *_: None) == 0, "a second run is a no-op"
    rows = cs.summarise(tmp_path / "CLIN" / "strata")
    assert len(rows) == 2 and rows[0]["grade_2_4_n"] == 3 and rows[0]["grade_2_4_mAUPR"] is not None
