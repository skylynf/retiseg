"""bench/scripts/launch_e5_infer.py: the E5 inference queue (plan, filters, commands, gates)."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from bench.e5 import frozen as fz
from bench.e5 import outputs as out
from bench.scripts import launch_e5_infer as launch

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "bench/configs/e5_frozen.yaml"


def _filled(tmp_path):
    doc = yaml.safe_load(FROZEN.read_text())
    doc["zero_shot"]["grid"]["selected"] = {"points_per_side": 64, "area_cap_fraction": 0.02}
    doc["zero_shot"]["sam3"]["selected"] = {"MA": "red dot", "HE": "retinal hemorrhage", "EX": "hard exudate", "SE": "soft exudate"}
    path = tmp_path / "filled.yaml"
    path.write_text(yaml.safe_dump(doc))
    return fz.load_frozen(path)


def test_plan_before_the_selections_has_pool_runs_and_the_oracle_tier_only(tmp_path, monkeypatch):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    items = launch.discover(frozen)
    pool = [it for it in items if it.phase == launch.POOL]
    test = [it for it in items if it.phase == launch.TEST]
    # pool: 2 models x 2 densities x 2 sets, 3 SAM 3 indices x 2 sets, 4 oracle (model, kind) x 2 sets
    assert len(pool) == 8 + 6 + 8
    assert {it.env for it in pool if "sam3" in it.label} == {"retiseg-sam3"} and {it.env for it in pool if "sam3" not in it.label} == {"retiseg"}
    # test: no selection yet -> no SAM / SAM 2 / SAM 3 items; the oracle tier on 7 sets + the TJDR_uwf fov variant
    assert all(it.label.startswith("b_") for it in test) and len(test) == 4 * 8
    assert any(it.label == "b_sam2_hieral21_point:TJDR_uwf:test:fov" for it in test)
    labels = {it.label for it in pool}
    assert "a_sam_vith_p64:DDR:val:main" in labels and "a_sam2_hieral21_p128:IDRiD:val:main" in labels and "a_sam3_ph2:DDR:val:main" in labels
    item = next(it for it in pool if it.label == "a_sam_vith_p64:DDR:val:main")
    assert item.argv == ["-m", "bench.e5.amg", "--model", "SAM", "--density", "64", "--set", "DDR", "--split", "val", "--variant", "main"]
    assert len(item.dirs) == 9 and item.log_dir in item.dirs and item.log_dir.name == "DDR-val"
    # queue order: pool first, then the largest canvases (TJDR_uwf main at 4256) first
    ordered = launch.plan(frozen)
    assert [it.phase for it in ordered] == sorted(it.phase for it in ordered)
    first_test = next(it for it in ordered if it.phase == launch.TEST)
    assert first_test.set_name == "TJDR_uwf" and first_test.label.endswith(":main")
    assert len(launch.missing_prompt_files(frozen)) == 2 + 8  # pool sets and the eight test (set, variant) pairs


def test_plan_after_the_selections_adds_the_zero_shot_tests_and_the_gated_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = _filled(tmp_path)
    items = launch.discover(frozen)
    test = [it for it in items if it.phase == launch.TEST]
    amg_items = [it for it in test if it.label.startswith("a_sam_") or it.label.startswith("a_sam2_")]
    assert len(amg_items) == 2 * 8 and all(":p64" not in it.label and "_p64:" in it.label for it in amg_items)  # the selected density only
    assert all(len(it.dirs) == 3 for it in amg_items)
    sam3_items = [it for it in test if it.label.startswith("a_sam3_ph")]
    assembled = [it for it in test if it.label.startswith("a_sam3_selected")]
    assert len(sam3_items) == 3 * 8 and len(assembled) == 8
    ddr = next(it for it in assembled if it.set_name == "DDR")
    assert len(ddr.gate_dirs) == 3 and ddr.env == "retiseg" and ddr.argv[:3] == ["-m", "bench.e5.sam3_text", "--assemble"]
    gate = launch.gate_for(ddr)
    assert gate() == "wait"
    for path in ddr.gate_dirs:
        path.mkdir(parents=True)
        (path / "done.json").write_text("{}")
    assert gate() == "open" and launch.gate_for(amg_items[0]) is None
    # the registered budget rule: at the largest density the TJDR_uwf main variant is left out
    doc = yaml.safe_load((tmp_path / "filled.yaml").read_text())
    doc["zero_shot"]["grid"]["selected"]["points_per_side"] = 128
    (tmp_path / "largest.yaml").write_text(yaml.safe_dump(doc))
    largest = fz.load_frozen(tmp_path / "largest.yaml")
    labels = {it.label for it in launch.discover(largest)}
    assert "a_sam_vith_p128:TJDR_uwf:test:fov" in labels and "a_sam_vith_p128:TJDR_uwf:test:main" not in labels


def test_done_running_filters_commands_and_only(tmp_path, monkeypatch):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    item = next(it for it in launch.discover(frozen) if it.label == "b_sam_vitb_box:DDR:val:main")
    for path in item.dirs:
        path.mkdir(parents=True)
        (path / "done.json").write_text("{}")
    assert item.done() and item.label not in {it.label for it in launch.plan(frozen)}
    other = next(it for it in launch.discover(frozen) if it.label == "b_sam2_hieral21_box:DDR:val:main")
    other.log_dir.mkdir(parents=True)
    (other.log_dir / "console.log").write_text("== start e5 2026-10-12T00:00:00 gpu 0 host elsewhere pid 1\n")
    assert other.running() is not None and other.label not in {it.label for it in launch.plan(frozen)}
    only = launch.plan(frozen, only={"a_sam3_ph0"})
    assert only and all(it.label.startswith("a_sam3_ph0:") for it in only) and len(only) == 2
    assert all(it.phase == launch.POOL for it in launch.plan(frozen, pool_only=True))
    assert all(it.phase == launch.TEST for it in launch.plan(frozen, test_only=True))
    test_item = next(it for it in launch.discover(frozen) if it.label == "b_medsam_vitb_box:FGADR:test:main")
    text = launch.command(test_item, py_argv=["python"], drop_maps=("FGADR", "Retinal-Lesions"), limit=2)
    assert "-m bench.e5.prompted --model MedSAM --kind boxes --set FGADR --split test --variant main --drop-maps FGADR --limit 2" in text
    assert text.startswith("mkdir -p ") and "== start e5" in text and "unset WORLD_SIZE" in text and str(test_item.log_dir / "console.log") in text
    pool_text = launch.command(item, py_argv=["python"], drop_maps=("DDR",))
    assert "--drop-maps" not in pool_text  # a pool run never drops its maps


def test_main_prints_the_plan_without_submitting(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    launch.main([])
    text = capsys.readouterr().out
    assert "E5 inference queue:" in text and "not submitted" in text and "prompt files to write first" in text
    with pytest.raises(SystemExit, match="RETISEG_GPUS"):
        monkeypatch.delenv("RETISEG_GPUS", raising=False)
        launch.main(["--submit"])


def test_server_run_has_the_e5_infer_stage():
    subprocess.run(["bash", "-n", str(REPO / "bench/scripts/server_run.sh")], check=True)
    text = (REPO / "bench/scripts/server_run.sh").read_text()
    assert "e5-infer)" in text and "launch_e5_infer.py --submit" in text and "RetinalLesions" in text
