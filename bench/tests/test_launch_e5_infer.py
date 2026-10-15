"""bench/scripts/launch_e5_infer.py: the E5 inference queue (plan, filters, commands, gates)."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.e5 import outputs as out
from bench.scripts import launch_e5_infer as launch

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "bench/configs/e5_frozen.yaml"



def _n_targets(frozen):
    """Test (set, variant) pairs: every test set of e3a_frozen.yaml plus the TJDR_uwf fov variant
    (8 with the seven sets of 2026-10-12, 9 once MAPLES-DR was appended on 2026-10-14)."""
    return sum(len(geo.variants_for(name, frozen)) for name in geo.test_sets())

def _filled(tmp_path):
    doc = yaml.safe_load(FROZEN.read_text())
    doc["zero_shot"]["grid"]["selected"] = {model: {"points_per_side": 64, "area_cap_fraction": 0.02} for model in ("SAM", "SAM 2")}
    doc["zero_shot"]["sam3"]["selected"] = {"MA": "red dot", "HE": "retinal hemorrhage", "EX": "hard exudate", "SE": "soft exudate"}
    path = tmp_path / "filled.yaml"
    path.write_text(yaml.safe_dump(doc))
    return fz.load_frozen(path)


@pytest.fixture(autouse=True)
def _no_machine_state(tmp_path, monkeypatch):
    """The same-information references are planned only where runs/BS_m2mrf_<src>_seed0/prob_test and
    runs/NN_default_<src>_fold0/prob_test exist (the server, 2026-10-14: 40 test items instead of 36), and
    the queue order weighs each set by its image count, which the prepared data of the machine decides
    (the server, 2026-10-15: FGADR's 553 images outrank TJDR_uwf's 58 on the 4256 canvas). Every test here
    starts from a root without reference maps and with the 100-image placeholder count of a machine without
    data; test_same_information_references_appear_with_their_maps creates the maps on purpose."""
    monkeypatch.setattr(launch, "_ROOT", tmp_path)
    monkeypatch.setattr(launch, "n_images", lambda set_name, split: 100)


def test_plan_before_the_selections_has_pool_runs_and_the_oracle_tier_only(tmp_path, monkeypatch):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    items = launch.discover(frozen)
    pool = [it for it in items if it.phase == launch.POOL]
    test = [it for it in items if it.phase == launch.TEST]
    # pool: 2 models x 2 densities x 2 sets, 3 SAM 3 indices x 2 sets, 4 oracle (model, kind) x 2 sets
    assert len(pool) == 8 + 6 + 8
    assert {it.env for it in pool if "sam3" in it.label} == {"retiseg-sam3"} and {it.env for it in pool if "sam3" not in it.label} == {"retiseg"}
    # test: no selection yet -> no SAM / SAM 2 / SAM 3 items; the oracle tier on every test set + the TJDR_uwf fov variant
    assert all(it.label.startswith("b_") for it in test) and len(test) == 4 * _n_targets(frozen)
    assert any(it.label == "b_sam2_hieral21_point:TJDR_uwf:test:fov" for it in test)
    labels = {it.label for it in pool}
    assert "a_sam_vith_p64:DDR:val:main" in labels and "a_sam2_hieral21_p128:IDRiD:val:main" in labels and "a_sam3_ph2:DDR:val:main" in labels
    item = next(it for it in pool if it.label == "a_sam_vith_p64:DDR:val:main")
    assert item.argv == ["-m", "bench.e5.amg", "--model", "SAM", "--density", "64", "--set", "DDR", "--split", "val", "--variant", "main"]
    assert len(item.dirs) == 9 and item.log_dir in item.dirs and item.log_dir.name == "DDR-val"
    # queue order: pool first, then by cost = weight x images x canvas area; with equal image counts the
    # largest canvas (TJDR_uwf main at 4256) comes first, with real counts a large set such as FGADR may
    ordered = launch.plan(frozen)
    assert [it.phase for it in ordered] == sorted(it.phase for it in ordered)
    first_test = next(it for it in ordered if it.phase == launch.TEST)
    assert first_test.set_name == "TJDR_uwf" and first_test.label.endswith(":main")
    monkeypatch.setattr(launch, "n_images", lambda set_name, split: 553 if set_name == "FGADR" else 58)
    assert next(it for it in launch.plan(frozen) if it.phase == launch.TEST).set_name == "FGADR"
    assert len(launch.missing_prompt_files(frozen)) == 2 + _n_targets(frozen)  # pool sets and every test (set, variant) pair


def test_same_information_references_appear_with_their_maps(tmp_path, monkeypatch):
    """One reference item per (model, internal test set) whose probability maps are on this machine, ungated."""
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    assert not [it for it in launch.discover(frozen) if "same_info" in " ".join(it.argv)]
    for pred in ("runs/BS_m2mrf_idrid_seed0/prob_test", "runs/NN_default_ddr_fold0/prob_test"):
        (tmp_path / pred).mkdir(parents=True)
    refs = [it for it in launch.discover(frozen) if "bench.e5.same_info" in it.argv]
    assert len(refs) == 2 and all(it.phase == launch.TEST and not it.gate_dirs and it.needs_prompts for it in refs)
    assert sorted((it.set_name, it.argv[it.argv.index("--model") + 1]) for it in refs) == [("DDR", "nnU-Net"), ("IDRiD", "M2MRF")]
    assert all(it.label.endswith(":test:main") for it in refs)


def test_plan_after_the_selections_adds_the_zero_shot_tests_and_the_gated_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = _filled(tmp_path)
    items = launch.discover(frozen)
    test = [it for it in items if it.phase == launch.TEST]
    amg_items = [it for it in test if it.label.startswith("a_sam_") or it.label.startswith("a_sam2_")]
    assert len(amg_items) == 2 * _n_targets(frozen) and all(":p64" not in it.label and "_p64:" in it.label for it in amg_items)  # the selected density only
    assert all(len(it.dirs) == 3 for it in amg_items)
    sam3_items = [it for it in test if it.label.startswith("a_sam3_ph")]
    assembled = [it for it in test if it.label.startswith("a_sam3_selected")]
    assert len(sam3_items) == 3 * _n_targets(frozen) and len(assembled) == _n_targets(frozen)
    ddr = next(it for it in assembled if it.set_name == "DDR")
    # the test assembly waits for its three phrase runs and for the pool assembly of both pool sets (its thresholds)
    assert len(ddr.gate_dirs) == 3 + 2 and ddr.env == "retiseg" and ddr.argv[:3] == ["-m", "bench.e5.sam3_text", "--assemble"]
    assert sorted(p.name for p in ddr.gate_dirs[3:]) == ["DDR-val", "IDRiD-val"] and all(p.parent.name == "a_sam3_selected" for p in ddr.gate_dirs[3:])
    gate = launch.gate_for(ddr)
    assert gate() == "wait"
    for path in ddr.gate_dirs:
        path.mkdir(parents=True)
        (path / "done.json").write_text("{}")
    assert gate() == "open"
    # every test item that takes thresholds from the pool is gated on the pool directories of its own source
    assert len(amg_items[0].gate_dirs) == 2 * 9 and all(p.name.split("__")[0].endswith("-val") and p.parent.name == "a_sam_vith_p64" for p in amg_items[0].gate_dirs)
    # the registered budget rule, per model: SAM at the largest density leaves out its TJDR_uwf main variant, SAM 2 at 64 keeps it
    doc = yaml.safe_load((tmp_path / "filled.yaml").read_text())
    doc["zero_shot"]["grid"]["selected"]["SAM"]["points_per_side"] = 128
    (tmp_path / "largest.yaml").write_text(yaml.safe_dump(doc))
    largest = fz.load_frozen(tmp_path / "largest.yaml")
    labels = {it.label for it in launch.discover(largest)}
    assert "a_sam_vith_p128:TJDR_uwf:test:fov" in labels and "a_sam_vith_p128:TJDR_uwf:test:main" not in labels
    assert "a_sam2_hieral21_p64:TJDR_uwf:test:main" in labels and not any(l.startswith("a_sam2_hieral21_p128:") and ":test:" in l for l in labels)


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


# ---- review fix 2 (A5): smoke runs, and a plan that asks only for what its items need ---------------


def test_a_smoke_run_does_not_count_as_done_for_the_full_queue(tmp_path, monkeypatch):
    """17 section 6 runs --limit 2 first; the full queue must still list those items."""
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    item = next(it for it in launch.discover(frozen) if it.label == "b_sam_vitb_box:DDR:val:main")
    for path in item.dirs:
        path.mkdir(parents=True)
        (path / "done.json").write_text(json.dumps({"limit": 2}))
    assert item.done(limit=2) and not item.done()
    assert item.label in {it.label for it in launch.plan(frozen)} and item.label not in {it.label for it in launch.plan(frozen, limit=2)}


def test_a_plan_asks_only_for_the_sets_and_prompt_files_of_its_items(tmp_path, monkeypatch):
    """A --pool-only smoke on a machine holding IDRiD and DDR alone is not stopped by the five sets it does not run."""
    from bench.e5 import geometry as geo

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    monkeypatch.setattr(fz, "PROMPTS_DIR", tmp_path / "prompts")
    prepared = tmp_path / "prepared"
    for name in ("IDRiD", "DDR"):
        (prepared / name).mkdir(parents=True)
    monkeypatch.setattr(launch.geo, "data_dir", lambda name: prepared / name)
    frozen = fz.load_frozen()
    pool = launch.plan(frozen, pool_only=True)
    assert pool and launch.absent_sets(pool) == []
    full = launch.plan(frozen)
    assert set(launch.absent_sets(full)) == set(geo.test_sets()) - {"IDRiD", "DDR"}
    # prompt files: the oracle tier and the references read them, the automatic masks and SAM 3 do not
    assert all(it.split == "val" and it.variant == "main" for it in pool)
    assert sorted(launch.missing_prompt_files(frozen, pool)) == [("DDR", "val", "main"), ("IDRiD", "val", "main")]
    assert {it.needs_prompts for it in pool if it.label.startswith("b_")} == {True} and {it.needs_prompts for it in pool if it.label.startswith("a_")} == {False}
    amg_only = launch.plan(frozen, only={"a_sam_vith_p64"}, pool_only=True)
    assert amg_only and launch.missing_prompt_files(frozen, amg_only) == []
    assert len(launch.missing_prompt_files(frozen)) == 2 + _n_targets(frozen)  # every target, as before
    uwf = next(it for it in full if it.label == "b_sam2_hieral21_point:TJDR_uwf:test:fov")
    assert (uwf.set_name, uwf.split, uwf.variant) == ("TJDR_uwf", "test", "fov")


# ---- review fix 3 (A3): test items wait for their source's pool; --test-only keeps the pool assembly ---


def test_test_only_keeps_the_pool_assembly_and_gates_close_when_nothing_will_produce(tmp_path, monkeypatch):
    """The documented order is pool -> selections -> --test-only; a gate never waits for what nothing will write."""
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = _filled(tmp_path)
    items = launch.plan(frozen, test_only=True)
    pool_assembly = [it for it in items if it.phase == launch.POOL]
    assert sorted(it.label for it in pool_assembly) == ["a_sam3_selected:DDR:val:main", "a_sam3_selected:IDRiD:val:main"]
    assert all(it.prerequisite for it in pool_assembly) and items[0].phase == launch.POOL  # they run first
    assert all(it.phase == launch.POOL for it in launch.plan(frozen, pool_only=True))
    # gates by source: the oracle tier on its k directories, a SAM 3 phrase on its own index, the references on nothing
    box = next(it for it in items if it.label == "b_sam_vitb_box:DDR:test:main")
    assert len(box.gate_dirs) == 2 * 4 and all(p.name.startswith(("IDRiD-val__k", "DDR-val__k")) and p.parent.name == "b_sam_vitb_box" for p in box.gate_dirs)
    ph0 = next(it for it in items if it.label == "a_sam3_ph0:FGADR:test:main")
    assert sorted(p.name for p in ph0.gate_dirs) == ["DDR-val", "IDRiD-val"] and all(p.parent.name == "a_sam3_ph0" for p in ph0.gate_dirs)
    assert all(not it.gate_dirs for it in items if "_ref_" in it.label)
    # a gate waits while its producer is queued or running, and closes when nothing writes the directory
    assert launch.gate_for(box, producers={})() == "never"
    producers = launch.producers_of([launch.Item("p", launch.POOL, 0.0, "retiseg", [], path, [path], "DDR") for path in box.gate_dirs])
    assert launch.gate_for(box, producers)() == "wait"
    log_dir = box.gate_dirs[0]
    log_dir.mkdir(parents=True)
    (log_dir / "console.log").write_text("== start e5 x gpu 0 host h pid 1\n")
    assert launch.completed_runs(log_dir) == 0 and launch.gate_for(box, producers)() == "wait"  # running
    (log_dir / "console.log").write_text("== start e5 x gpu 0 host h pid 1\n== exit 1 x\n")
    assert launch.completed_runs(log_dir) == 1 and launch.gate_for(box, producers)() == "never"  # finished without done.json
    # a failure from before this queue started does not close the gate: that producer is queued again
    producers = launch.producers_of([launch.Item("p", launch.POOL, 0.0, "retiseg", [], path, [path], "DDR") for path in box.gate_dirs])
    assert producers[log_dir][1] == 1 and launch.gate_for(box, producers)() == "wait"
    for path in box.gate_dirs:
        path.mkdir(parents=True, exist_ok=True)
        (path / "done.json").write_text(json.dumps({"limit": 2}))
    assert launch.gate_for(box, producers, limit=2)() == "open" and launch.gate_for(box, producers)() == "wait"  # a smoke pool opens a smoke queue only
    for path in box.gate_dirs:
        (path / "done.json").write_text("{}")
    assert launch.gate_for(box, producers)() == "open"


# ---- review fix 4 (A2): one writer per directory in the SAM 3 chain -----------------------------------


def test_sam3_phrase_items_leave_assembly_and_map_removal_to_the_gated_item(tmp_path, monkeypatch):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    items = launch.plan(_filled(tmp_path), test_only=True)
    phrase = next(it for it in items if it.label == "a_sam3_ph0:FGADR:test:main")
    assembly = next(it for it in items if it.label == "a_sam3_selected:FGADR:test:main")
    assert "--no-assemble" in phrase.argv and not phrase.owns_drop and assembly.owns_drop
    assert "--drop-maps" not in launch.command(phrase, py_argv=["python"], drop_maps=("FGADR",))
    assert "--drop-maps FGADR" in launch.command(assembly, py_argv=["python"], drop_maps=("FGADR",))
    other = next(it for it in items if it.label == "b_medsam_vitb_box:FGADR:test:main")
    assert "--drop-maps FGADR" in launch.command(other, py_argv=["python"], drop_maps=("FGADR",))  # every other item drops its own
    assert all("--no-assemble" in it.argv for it in items if it.label.startswith("a_sam3_ph"))


# ---- review fix 9 (B3): the oracle items of the five non-curve sets have one directory --------------


def test_oracle_items_outside_the_curve_sets_have_the_k_all_directory_only(tmp_path, monkeypatch):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    items = launch.discover(fz.load_frozen())
    by_set = {}
    for it in items:
        if it.label.startswith("b_") and "_ref_" not in it.label:
            by_set.setdefault((it.set_name, it.split), set()).add(len(it.dirs))
    assert by_set[("IDRiD", "test")] == {4} and by_set[("DDR", "test")] == {4} and by_set[("IDRiD", "val")] == {4} and by_set[("DDR", "val")] == {4}
    assert all(by_set[(name, "test")] == {1} for name in ("TJDR_std", "TJDR_uwf", "FGADR", "Retinal-Lesions", "DiaRetDB1"))
    fgadr = next(it for it in items if it.label == "b_medsam_vitb_box:FGADR:test:main")
    assert fgadr.dirs[0].name == "FGADR__kall" and fgadr.log_dir == fgadr.dirs[0] and len(fgadr.gate_dirs) == 2 * 4  # its thresholds still come from every k of the pool


# ---- review fix 14 (open issue 32): --estimate from the measured seconds per image ---------------------


def test_estimate_scales_measured_seconds_per_image_by_canvas_area(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    items = launch.plan(frozen, pool_only=True)
    rows, total = launch.estimate(items, frozen)
    assert rows and all(row[3] is None for row in rows) and total == 0.0  # nothing measured yet
    # a smoke of a_sam_vith_p64 on the 1440 canvas took 10 s per image
    smoke = tmp_path / "runs" / "E5" / "a" / "a_sam_vith_p64" / "DDR-val"
    smoke.mkdir(parents=True)
    (smoke / "predicted.json").write_text(json.dumps({"n_images": 2, "resumed_images": 0, "predict_seconds": 20.0, "canvas_long_side": 1440, "limit": 2}))
    assert launch.measured_rate("a", "a_sam_vith_p64") == (10.0, 1440) and launch.measured_rate("a", "a_sam2_hieral21_p64") is None
    rows, total = launch.estimate(items, frozen)
    by = {row[0]: row for row in rows}
    ddr, idrid = by["a_sam_vith_p64:DDR:val:main"], by["a_sam_vith_p64:IDRiD:val:main"]
    assert ddr[2] == 1440 and abs(ddr[3] - 10.0 * ddr[1] / 3600) < 1e-9 and idrid[3] is not None and by["a_sam2_hieral21_p64:DDR:val:main"][3] is None
    assert abs(total - (ddr[3] + idrid[3])) < 1e-9
    # the 4256 canvas of TJDR_uwf scales the rate by the canvas area
    filled = _filled(tmp_path)
    test_items = launch.plan(filled, test_only=True)
    rows, _total = launch.estimate(test_items, filled)
    uwf = next(row for row in rows if row[0] == "a_sam_vith_p64:TJDR_uwf:test:main")
    assert uwf[2] == 4256 and abs(uwf[3] - 10.0 * (4256 / 1440) ** 2 * uwf[1] / 3600) < 1e-9
    assert launch.estimate_lines(test_items, filled)[-1].startswith("estimate:") and "no predicted.json" in launch.estimate_lines(test_items, filled)[-1]
    launch.main(["--estimate", "--pool-only"])
    text = capsys.readouterr().out
    assert "estimate:" in text and "not submitted" not in text
