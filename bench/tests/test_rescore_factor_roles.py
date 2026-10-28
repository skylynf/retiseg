"""The factor cells score the validation-selected checkpoint (docs/ops/21 #97), and the re-scoring script
swaps the maps a cell predicted for both checkpoints without deleting anything."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.scripts import rescore_factor_roles as rs


class _Declared:
    def __init__(self, iterations=0, epochs=0):
        self.iterations = iterations
        self.epochs = epochs


def test_roles_for_scores_best_validation_for_bstd_and_factor_cells_and_last_step_for_b1():
    pytest.importorskip("torch")
    from bench.train import roles_for

    author_budget = _Declared(iterations=40000)
    assert roles_for({"experiment": "BS"}, author_budget) == ("best.pt", "last.pt")
    assert roles_for({"experiment": "BF"}, author_budget) == ("best.pt", "last.pt")
    assert roles_for({"experiment": "B1"}, author_budget) == ("last.pt", "best.pt")
    assert roles_for({"experiment": "B1"}, _Declared()) == ("best.pt", "last.pt")


def _maps(directory, ids, value):
    for cls in LESION_CLASSES:
        (directory / cls).mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            Image.fromarray(np.full((4, 4), value, dtype=np.uint16)).save(directory / cls / f"{image_id}.png")


def _cell(runs, name, primary="last.pt", with_sensitivity=True, e3a=True):
    run = runs / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({
        "experiment": "BF", "factor": "E3b", "model": "M2MRF", "dataset": "dataset/prepared/DDR", "split_val": "val",
        "split_test": "test", "fov_diameter": 1440, "train": [{"dataset": "dataset/prepared/FGADR", "split": "train"}],
        "budget_reference": "DDR", "seed": 0}))
    for ckpt in ("best.pt", "last.pt"):
        (run / ckpt).write_bytes(b"x")
    (run / "checkpoint_roles.json").write_text(json.dumps({
        "primary": primary, "sensitivity": "best.pt" if primary == "last.pt" else "last.pt",
        "primary_directories": ["prob_val", "prob_test"], "sensitivity_directories": ["prob_val_sensitivity", "prob_test_sensitivity"]}))
    _maps(run / "prob_val", ["v1"], 1)
    _maps(run / "prob_test", ["t1", "t2"], 1)
    if with_sensitivity:
        _maps(run / "prob_val_sensitivity", ["v1"], 2)
        _maps(run / "prob_test_sensitivity", ["t1", "t2"], 2)
    (run / "metrics_test.json").write_text(json.dumps({"summary": {"mAUPR": 0.30}}))
    (run / "metrics_test.npz").write_bytes(b"npz")
    if e3a:
        (runs / "E3a" / name / "IDRiD").mkdir(parents=True)
        (runs / "E3a" / name / "IDRiD" / "done.json").write_text("{}")
    return run


def test_affected_cells_lists_last_step_cells_and_reports_what_blocks_a_swap(tmp_path):
    runs = tmp_path / "runs"
    _cell(runs, "BF_e3b_m2mrf_fgadr_seed0")
    _cell(runs, "BF_e3b_unet_fgadr_seed0", primary="best.pt")
    _cell(runs, "BF_e8_m2mrf_fine_seed0", with_sensitivity=False)
    (runs / "BS_hrnet_ddr_seed0").mkdir()   # not a BF directory
    items = {run.name: status for run, status in rs.affected_cells(runs)}
    assert items["BF_e3b_m2mrf_fgadr_seed0"] == "ready"
    assert "BF_e3b_unet_fgadr_seed0" not in items
    assert items["BF_e8_m2mrf_fine_seed0"].startswith("sensitivity maps incomplete")


def test_apply_swaps_the_maps_keeps_the_last_step_files_and_moves_the_e3a_directory(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = _cell(runs, "BF_e3b_m2mrf_fgadr_seed0")
    moved = rs.apply(run, runs, evaluate=False)
    # best.pt maps (value 2) are now the primary directories, last.pt maps (value 1) the sensitivity ones
    assert np.asarray(Image.open(run / "prob_test" / "MA" / "t1.png")).max() == 2
    assert np.asarray(Image.open(run / "prob_val" / "MA" / "v1.png")).max() == 2
    assert np.asarray(Image.open(run / "prob_test_sensitivity" / "MA" / "t2.png")).max() == 1
    roles = json.loads((run / "checkpoint_roles.json").read_text())
    assert roles["primary"] == "best.pt" and roles["sensitivity"] == "last.pt"
    assert json.loads((run / "checkpoint_roles_laststep.json").read_text())["primary"] == "last.pt"
    assert not (run / "metrics_test.json").exists()
    assert json.loads((run / "metrics_test_laststep.json").read_text())["summary"]["mAUPR"] == 0.30
    assert (run / "metrics_test_laststep.npz").is_file()
    assert moved == runs / "E3a_laststep" / run.name and (moved / "IDRiD" / "done.json").is_file()
    assert not (runs / "E3a" / run.name).exists()
    # a second apply would find nothing to swap
    assert rs.affected_cells(runs) == []


def test_evaluate_argv_is_the_factor_launchers_command_with_the_cells_exclusions(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run = _cell(runs, "BF_e3b_m2mrf_fgadr_seed0", e3a=False)
    monkeypatch.setattr(rs, "_ROOT", tmp_path)
    import bench.scripts.launch_factor as lf

    monkeypatch.setattr(lf, "internal_exclusions", lambda cell: ["0001_1", "0002_2"])
    argv = rs.evaluate_argv(run, python="python")
    assert argv[:4] == ["python", "-m", "bench.eval.cli", "evaluate"]
    assert "--pred" in argv and argv[argv.index("--pred") + 1] == "runs/BF_e3b_m2mrf_fgadr_seed0/prob_test"
    assert argv[argv.index("--val-pred") + 1] == "runs/BF_e3b_m2mrf_fgadr_seed0/prob_val"
    assert argv[argv.index("--out") + 1] == "runs/BF_e3b_m2mrf_fgadr_seed0/metrics_test"
    assert argv[argv.index("--exclude") + 1] == "0001_1,0002_2"


def test_launch_e3a_queues_a_reference_cell_only_when_it_is_named(tmp_path, monkeypatch):
    from bench.scripts import launch_e3a

    runs = tmp_path / "runs"
    ref = _cell(runs, "BF_ref_samed_fine_seed0", primary="best.pt", e3a=False)
    cfg = yaml.safe_load((ref / "config.yaml").read_text())
    cfg.update({"factor": "REF", "model": "SAMed", "train": [{"dataset": f"dataset/prepared/{s}", "split": "train"} for s in ("IDRiD", "DDR", "TJDR_std")]})
    (ref / "config.yaml").write_text(yaml.safe_dump(cfg))
    e3b = _cell(runs, "BF_e3b_m2mrf_fgadr_seed0", primary="best.pt", e3a=False)
    monkeypatch.setattr(launch_e3a, "_ROOT", tmp_path)
    monkeypatch.setattr(launch_e3a, "_BINDING", {"SAMed": ("retiseg-sam3",), "M2MRF": ("retiseg",)})
    monkeypatch.setattr(launch_e3a.e3a, "train_sources_of", lambda config: [Path(item["dataset"]).name for item in config["train"]])
    monkeypatch.setattr(launch_e3a, "DATASET_IDS", {}, raising=False)
    import bench.scripts.nnunet_bridge as bridge

    monkeypatch.setattr(bridge, "DATASET_IDS", {})
    monkeypatch.setattr(launch_e3a.e3a, "load_frozen", lambda path=None: {"canvas_reference": 1440, "author_weights": {}})
    names = {source for _tier, source, _env, _argv, _train in launch_e3a.discover()}
    assert names == {"BF_e3b_m2mrf_fgadr_seed0"}
    names = {source for _tier, source, _env, _argv, _train in launch_e3a.discover(only={"BF_ref_samed_fine_seed0"})}
    assert names == {"BF_ref_samed_fine_seed0", "BF_e3b_m2mrf_fgadr_seed0"}
    assert e3b.is_dir()
