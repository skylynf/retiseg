"""bench/scripts/diaretdb1_experts.py on a toy DiaRetDB1 with four synthetic experts (docs/paper/14 M16a)."""

import json

import numpy as np
import pytest
from PIL import Image

from bench.data.diaretdb import EXPERTS, pack_levels
from bench.scripts import diaretdb1_experts as de


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


@pytest.fixture
def toy(tmp_path):
    """Three true lesions per class; experts 01-03 mark them High with small shifts, expert 04 marks one Low and adds a false one."""
    shape = (96, 128)
    prepared = tmp_path / "dataset" / "prepared" / "DiaRetDB1"
    rng = np.random.default_rng(3)
    ids = {"train": [], "val": ["v0", "v1"], "test": ["t0", "t1", "t2"]}
    classes = ["MA", "HE", "EX", "SE"]
    for split, split_ids in ids.items():
        for image_id in split_ids:
            fov = _disk(shape, 48, 64, 46)
            (prepared / "fov").mkdir(parents=True, exist_ok=True)
            Image.fromarray(fov.astype(np.uint8) * 255).save(prepared / "fov" / f"{image_id}.png")
            for cls in classes:
                centres = [(rng.integers(15, 80), rng.integers(20, 108), rng.integers(3, 7)) for _ in range(3)]
                levels = {}
                for k, expert in enumerate(EXPERTS):
                    level = np.zeros(shape, np.uint8)
                    for i, (cy, cx, r) in enumerate(centres):
                        if expert == "04" and i == 0:
                            level |= _disk(shape, cy, cx, r).astype(np.uint8) * 1          # Low
                        else:
                            level |= _disk(shape, cy + k - 1, cx + (k % 2), r).astype(np.uint8) * 3   # High, shifted
                    if expert == "04":
                        level |= _disk(shape, 85, 20, 4).astype(np.uint8) * 3               # a region nobody else marked
                    levels[expert] = np.where(fov, level, 0).astype(np.uint8)
                (prepared / "experts" / cls).mkdir(parents=True, exist_ok=True)
                Image.fromarray(pack_levels(levels)).save(prepared / "experts" / cls / f"{image_id}.png")
    (prepared / "meta.json").write_text(json.dumps({"name": "DiaRetDB1", "classes": classes, "granularity": "region"}))
    (prepared / "splits.json").write_text(json.dumps(ids))
    return prepared


def test_consensus_and_expert_map_follow_the_frozen_mapping(toy):
    levels = de.read_levels(toy, "MA", "t0")
    assert set(levels) == set(EXPERTS)
    values = de.confidence_values()
    assert list(values) == [0.0, 0.25, 0.75, 1.0]
    m = de.expert_map(levels, "01", values)
    assert set(np.unique(m)) <= {0.0, 0.25, 0.75, 1.0}
    # without expert 04, the three High markings agree on the shifted overlap: positive where the mean of 01-03 >= 0.75
    truth = de.others_consensus(levels, "04", values)
    assert truth.any() and truth.sum() < (levels["01"] > 0).sum()
    # the region only expert 04 drew is never in the others' consensus, and expert 04's Low mark alone is not positive either
    assert not truth[_disk(levels["01"].shape, 85, 20, 3)].any()


def test_leave_one_out_scores_every_expert_with_the_protocol_and_summarises(toy, tmp_path, monkeypatch):
    from bench.eval.evaluate import load_protocol

    proto = load_protocol()
    out = tmp_path / "runs" / "HUMAN" / "DiaRetDB1"
    reports = {e: de.score_expert(toy, out / f"expert_{e}", e, proto, log=lambda *_: None) for e in EXPERTS}
    for e in EXPERTS:
        stored = json.loads((out / f"expert_{e}" / "metrics.json").read_text())
        assert stored["human_reference"]["held_out_expert"] == e and stored["n_images"] == 3
        assert "val_opt" in stored["per_class"]["MA"]["thresholds"], "thresholds come from the official training list"
        assert (out / f"expert_{e}" / "truth" / "masks" / "MA" / "t0.png").is_file() and (out / f"expert_{e}" / "prob" / "MA" / "t0.png").is_file()
    summary = de.summarise(reports)
    assert summary["experts"] == list(EXPERTS)
    band = summary["across_experts"]["mAUPR"]
    assert band["n"] == 4 and 0 <= band["min"] <= band["median"] <= band["max"] <= 1
    assert all(0 <= summary["per_expert"][e]["mAUPR"] <= 1 for e in EXPERTS)
    assert set(summary["across_experts"]["per_class"]) == {"MA", "HE", "EX", "SE"}
    assert summary["across_experts"]["per_class"]["MA"]["lesion_recall"]["n"] == 4


def test_main_writes_the_summary_and_can_drop_the_maps(toy, tmp_path):
    out = tmp_path / "human"
    de.main(["--prepared", str(toy), "--out", str(out), "--experts", "01,02", "--drop-maps"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["experts"] == ["01", "02"] and summary["n_test_images"] == 3 and "protocol_sha256" in summary
    assert not (out / "expert_01" / "prob").exists() and (out / "expert_01" / "metrics.json").is_file()
    with pytest.raises(ValueError, match="unknown experts"):
        de.main(["--prepared", str(toy), "--out", str(out), "--experts", "05"])
