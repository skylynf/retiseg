"""bench/reference.py: RetiSeg-Ref on new images (field of view, averaging, thresholds, summary) and the release
bundle. The members' networks are stubbed; the torch path (predict_member) runs on the server."""

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from bench import reference as ref
from bench.common import io
from bench.common.io import LESION_CLASSES


def _fundus(path, size=64, value=160):
    """A bright disk on a black border, as a fundus photograph looks to bench.data.fov."""
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size * 0.4) ** 2
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[disk] = (value, value // 2, value // 4)
    Image.fromarray(image).save(path)
    return disk


def _member(root, name, model):
    run = root / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({
        "experiment": "BF", "factor": "REF", "model": model, "dataset": "dataset/prepared/DDR", "split_val": "val",
        "split_test": "test", "fov_diameter": 1440, "train": [{"dataset": f"dataset/prepared/{s}", "split": "train"} for s in ("IDRiD", "DDR", "TJDR_std")],
        "budget_reference": "DDR", "seed": 0, "paper": {"experiment": "N8"}}))
    (run / "best.pt").write_bytes(b"best")
    (run / "last.pt").write_bytes(b"last")
    (run / "checkpoint_roles.json").write_text(json.dumps({"primary": "best.pt", "sensitivity": "last.pt"}))
    (run / "recipe.json").write_text("{}")
    (run / "environment.json").write_text("{}")
    (run / "metrics_test.json").write_text(json.dumps({"summary": {"mAUPR": 0.45, "mDice_pooled_val_opt": 0.4}}))
    return run


def test_list_images_takes_directories_and_files_and_refuses_repeated_stems(tmp_path):
    (tmp_path / "a").mkdir()
    _fundus(tmp_path / "a" / "img2.png")
    _fundus(tmp_path / "a" / "img1.jpg")
    (tmp_path / "a" / "notes.txt").write_text("x")
    _fundus(tmp_path / "img3.png")
    paths = ref.list_images([tmp_path / "a", tmp_path / "img3.png"])
    assert [p.name for p in paths] == ["img1.jpg", "img2.png", "img3.png"]
    _fundus(tmp_path / "b.png")
    (tmp_path / "c").mkdir()
    _fundus(tmp_path / "c" / "b.jpg")
    try:
        ref.list_images([tmp_path / "b.png", tmp_path / "c"])
    except ValueError as error:
        assert "b" in str(error)
    else:
        raise AssertionError("repeated stems must be refused")


def test_prepare_images_writes_a_prepared_directory_with_the_project_field_of_view(tmp_path):
    disk = _fundus(tmp_path / "eye.png")
    work = tmp_path / "prepared"
    ids = ref.prepare_images([tmp_path / "eye.png"], work)
    assert ids == ["eye"]
    assert io.load_split(work, "all") == ["eye"]
    fov = io.read_fov(work, "eye")
    assert fov.shape == disk.shape and fov[32, 32] and not fov[1, 1]
    assert abs(fov.sum() - disk.sum()) < 0.05 * disk.sum()
    assert (work / "images" / "eye.png").is_file()
    config = ref.member_config(_member(tmp_path / "runs", "BF_ref_samed_fine_seed0", "SAMed"), work)
    assert config["dataset"] == str(work) and config["split_test"] == "all" and config["fov_diameter"] == 1440


def test_predict_averages_the_members_thresholds_the_maps_and_summarizes(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    a = _member(runs, "BF_ref_retfound_fine_seed0", "RETFound")
    b = _member(runs, "BF_ref_samed_fine_seed0", "SAMed")
    _fundus(tmp_path / "eye.png")
    levels = {"BF_ref_retfound_fine_seed0": 0.8, "BF_ref_samed_fine_seed0": 0.2}

    def fake_member(run_dir, work_dir, ids, out_dir, device=None):
        fov = io.read_fov(work_dir, ids[0])
        for image_id in ids:
            for cls in LESION_CLASSES:
                plane = np.where(fov, levels[Path(run_dir).name], 0.0)
                if cls == "SE":
                    plane = plane * 0.1
                io.write_prob(out_dir, cls, image_id, plane)
        return 1.0

    monkeypatch.setattr(ref, "predict_member", fake_member)
    scores = tmp_path / "scores"
    (scores / "DDR").mkdir(parents=True)
    (scores / "DDR" / "metrics.json").write_text(json.dumps({"summary": {"mAUPR": 0.5}, "per_class": {
        cls: {"thresholds": {"val_opt": {"threshold": 0.3 if cls != "SE" else 0.02}}} for cls in LESION_CLASSES}}))
    monkeypatch.setattr(ref, "_ROOT", tmp_path)
    monkeypatch.setattr(ref, "SCORES", "scores")
    out = tmp_path / "out"
    summary = ref.predict([tmp_path / "eye.png"], out, members=[a, b])
    prob = io.read_prob_quantized(out / "prob", "MA", "eye") / io.PROB_LEVELS
    fov = io.read_fov(out / "prepared", "eye")
    assert np.allclose(prob[fov], 0.5, atol=1e-3) and prob[~fov].max() == 0
    mask = np.asarray(Image.open(out / "masks" / "MA" / "eye.png")) > 0
    assert mask[fov].all() and not mask[~fov].any()
    se = np.asarray(Image.open(out / "masks" / "SE" / "eye.png")) > 0
    assert se[fov].all()   # 0.05 >= 0.02
    assert summary["thresholds"]["MA"] == 0.3 and summary["thresholds"]["SE"] == 0.02
    assert summary["images"]["eye"]["MA"]["max_probability_in_fov"] > 0.49
    assert summary["images"]["eye"]["MA"]["positive_pixels"] == int(fov.sum())
    assert json.loads((out / "summary.json").read_text())["n_images"] == 1
    assert (out / "members" / "BF_ref_samed_fine_seed0" / "HE" / "eye.png").is_file()
    assert not (out / "prepared" / "images").exists()   # the copies are removed unless --keep-work


def test_load_thresholds_falls_back_to_fixed_and_reads_a_release_file(tmp_path):
    thresholds, source = ref.load_thresholds(tmp_path / "nowhere")
    assert thresholds == {cls: 0.5 for cls in LESION_CLASSES} and "fixed" in source
    (tmp_path / "thresholds.json").write_text(json.dumps({"source": "release", "thresholds": {cls: 0.1 for cls in LESION_CLASSES}}))
    thresholds, source = ref.load_thresholds(tmp_path)
    assert thresholds["EX"] == 0.1 and source == "release"


def test_bundle_copies_the_primary_checkpoints_and_writes_thresholds_scores_and_sums(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    a = _member(runs, "BF_ref_retfound_fine_seed0", "RETFound")
    b = _member(runs, "BF_ref_samed_fine_seed0", "SAMed")
    (a / "insource_IDRiD").mkdir()
    (a / "insource_IDRiD" / "metrics.json").write_text(json.dumps({"summary": {"mAUPR": 0.7}}))
    (runs / "E3a" / "BF_ref_samed_fine_seed0" / "MAPLES-DR").mkdir(parents=True)
    (runs / "E3a" / "BF_ref_samed_fine_seed0" / "MAPLES-DR" / "metrics.json").write_text(json.dumps({"summary": {"mAUPR": 0.41}}))
    scores = runs / "ENS" / "REF" / "reference"
    for name, value in (("DDR", 0.49), ("MAPLES-DR", 0.44)):
        (scores / name).mkdir(parents=True)
        (scores / name / "metrics.json").write_text(json.dumps({"summary": {"mAUPR": value}, "per_class": {
            cls: {"thresholds": {"val_opt": {"threshold": 0.25}}} for cls in LESION_CLASSES}}))
    monkeypatch.setattr(ref, "_ROOT", tmp_path)
    out = ref.bundle(tmp_path / "release", members=[a, b], scores_root="runs/ENS/REF/reference")
    assert (out / "BF_ref_retfound_fine_seed0" / "weights.pt").read_bytes() == b"best"
    assert (out / "BF_ref_samed_fine_seed0" / "config.yaml").is_file()
    thresholds = json.loads((out / "thresholds.json").read_text())
    assert thresholds["thresholds"]["MA"] == 0.25 and "DDR validation" in thresholds["source"]
    scores_json = json.loads((out / "scores.json").read_text())
    assert scores_json["ensemble"]["DDR"]["mAUPR"] == 0.49
    assert scores_json["members"]["BF_ref_retfound_fine_seed0"]["IDRiD"]["mAUPR"] == 0.7
    assert scores_json["members"]["BF_ref_samed_fine_seed0"]["MAPLES-DR"]["mAUPR"] == 0.41
    assert scores_json["members"]["BF_ref_samed_fine_seed0"]["DDR"]["mAUPR"] == 0.45
    readme = (out / "README.md").read_text()
    assert "| DDR | 0.490 |" in readme and "0.410" in readme
    sums = (out / "sha256sums.txt").read_text().splitlines()
    assert any(line.endswith("BF_ref_retfound_fine_seed0/weights.pt") for line in sums)
    assert json.loads((out / "members.json").read_text())["members"][0]["checkpoint"] == "best.pt"
    # a bundled member serves as a member directory: predict reads weights.pt
    assert ref.primary_checkpoint(out / "BF_ref_samed_fine_seed0").name == "weights.pt"
