"""bench/scripts/launch_queue.py: the one queue file (torch-free: shell and nnunet stages, ordering, gates)."""
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _queue(tmp_path, stages):
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump({"stages": stages}))
    return path


def test_gather_orders_by_priority_skips_done_and_reports_unplannable_stages(tmp_path, monkeypatch):
    from bench.scripts import launch_queue as lq
    from bench.scripts import nnunet_bridge

    from bench.scripts import launch_b1

    monkeypatch.setattr(lq, "_ROOT", tmp_path)
    monkeypatch.setattr(launch_b1, "_ROOT", tmp_path)   # in_progress reads console.log under launch_b1._ROOT (a packed NN_ run would look live)
    monkeypatch.setattr(nnunet_bridge, "RAW", tmp_path / "nnunet" / "raw")   # the nn stage is unplannable here, whatever this machine holds
    done = tmp_path / "runs" / "x" / "done.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    path = _queue(tmp_path, [
        {"name": "late", "priority": 50, "kind": "shell", "env": "retiseg",
         "items": [{"label": "b", "command": "python -m bench.e5 prompts --set IDRiD", "done": "runs/b/done.json"}]},
        {"name": "early", "priority": 5, "kind": "shell", "env": "retiseg",
         "items": [{"label": "a", "command": "bash run.sh", "run_dir": "runs/a"}, {"label": "x", "command": "true", "done": "runs/x/done.json"}]},
        {"name": "off", "priority": 1, "kind": "shell", "enabled": False, "items": [{"label": "never", "command": "true"}]},
        {"name": "nn", "priority": 2, "kind": "nnunet", "runs": [{"train": "FGADR", "plan": "default"}]},
    ])
    report = []
    items = lq.gather(path, dry=True, report=report)
    assert [label for _p, _s, label, _c, _g in items] == ["shell:a", "shell:b"]
    assert items[0][1] == "early" and "== start shell" in items[0][3] and "runs/a/console.log" in items[0][3]
    assert "-m bench.e5 prompts --set IDRiD" in items[1][3] and items[1][3].startswith("mkdir -p runs/b")
    assert len(report) == 1 and report[0].startswith("nn: FileNotFoundError")


def test_running_items_are_left_out_until_their_process_ends(tmp_path, monkeypatch):
    from bench.scripts import launch_b1, launch_queue as lq

    monkeypatch.setattr(lq, "_ROOT", tmp_path)
    monkeypatch.setattr(launch_b1, "_ROOT", tmp_path)
    log = tmp_path / "runs" / "a" / "console.log"
    log.parent.mkdir(parents=True)
    log.write_text("== start shell 2026-10-14T00:00:00 gpu 3 host elsewhere pid 1\n")   # another host: counts as running
    path = _queue(tmp_path, [{"name": "s", "priority": 1, "kind": "shell", "items": [{"label": "a", "command": "true", "run_dir": "runs/a"}]}])
    assert lq.gather(path, dry=True) == []
    log.write_text(log.read_text() + "== exit 0 2026-10-14T01:00:00\n")
    assert [i[2] for i in lq.gather(path, dry=True)] == ["shell:a"]


def by_name_of(stages, name):
    return next(s for s in stages if s["name"] == name)


def test_queue_file_in_the_repository_loads_and_names_known_kinds():
    from bench.scripts import launch_queue as lq

    stages = lq.load_queue(REPO / "bench" / "configs" / "queue.yaml")
    names = [s["name"] for s in stages]
    # third edit, 2026-10-15: seed-0 E3a first, the inference pool ahead of the E5 training, seed-1/2 E3a behind the training
    assert names[:6] == ["e3a", "factor_score", "e5_infer_pool", "e5_seed0", "fundus_seed0", "e5_infer_test"]
    assert by_name_of(stages, "factor_score")["score_only"] and by_name_of(stages, "factor_score")["models"] == ["U-Net", "M2MRF"]
    assert names.index("factor_cheap") < names.index("nnunet_e3b") < names.index("factor_e3b_heavy") < names.index("factor_e7_heavy")
    assert names.index("factor_e7_heavy") < names.index("e3a_rest") < names.index("hrdecoder_rest") < names.index("e5_rest") < names.index("fundus_rest")
    assert all(s["kind"] in lq.PLANNERS for s in stages)
    assert all(stages[i]["priority"] <= stages[i + 1]["priority"] for i in range(len(stages) - 1))
    by_name = {s["name"]: s for s in stages}
    assert by_name["e3a"]["max_concurrent"] == 2 and by_name["e3a"]["seed0_only"] and not by_name["e3a_rest"].get("seed0_only")
    assert by_name["factor_cheap"]["models"] == ["U-Net", "M2MRF"] and by_name["fundus_seed0"].get("enabled", True)
    assert by_name["factor_e3b_heavy"]["experiment"] == "E3b"
    # the albert merge (docs/ops/25): one e5_infer kind; off until the inference smoke (pool) and the selections (test)
    assert by_name["e5_infer_pool"]["phase"] == "pool" and by_name["e5_infer_test"]["phase"] == "test"
    assert by_name["e5_infer_pool"].get("enabled", True) and by_name["e5_infer_test"].get("enabled", True)   # pool on since 2026-10-17, test on since 2026-10-20 (after the selection)
    assert by_name["e5_infer_test"]["max_concurrent"] == 2
    # the HRDecoder case study at its author canvas (docs/paper/26 N1c): on since 2026-10-21, the longest job left
    assert by_name["factor_hrdecoder_2880"]["models"] == ["HRDecoder"] and by_name["factor_hrdecoder_2880"].get("enabled", True)
    # fourth edit, 2026-10-19 (docs/paper/26): SAMed and RETFound seeds are not cut any more; the other E5 seeds wait in e5_rest2
    assert by_name["e5_rest"]["only"] == ["SAMed", "RETFound"] and by_name["e5_rest"].get("enabled", True)
    assert by_name["e5_rest2"]["only"] == ["SAM2-UNet", "SAM-Adapter"] and not by_name["e5_rest2"].get("enabled", True)


def test_factor_models_filter_keeps_the_cheap_cells_apart(monkeypatch):
    from bench.scripts import launch_factor as lf

    jobs = lf.load_jobs()
    monkeypatch.setattr(lf, "in_progress", lambda run_dir: None)
    cheap = {job["name"] for job in lf.plan(models=("U-Net", "M2MRF"))}
    heavy = {job["name"] for job in lf.plan(experiment="E3b", models=("SAM2-UNet", "RETFound"))}
    assert cheap and heavy and not (cheap & heavy)
    assert all(job["model"] in ("U-Net", "M2MRF") for job in jobs if job["name"] in cheap)
    assert heavy == {job["name"] for job in jobs if job["experiment"] == "E3b" and job["model"] in ("SAM2-UNet", "RETFound")}


def test_max_concurrent_hands_the_card_to_the_next_stage(tmp_path, monkeypatch):
    """Two cards, a capped stage with three items: the second card takes the uncapped stage's item."""
    import threading

    from bench.scripts import launch_queue as lq

    monkeypatch.setattr(lq, "_ROOT", tmp_path)
    monkeypatch.setattr(lq, "python_argv", lambda env: ["python"])
    path = _queue(tmp_path, [
        {"name": "capped", "priority": 1, "kind": "shell", "max_concurrent": 1,
         "items": [{"label": f"c{i}", "command": "true", "done": f"runs/c{i}/done.json"} for i in range(3)]},
        {"name": "open", "priority": 2, "kind": "shell", "items": [{"label": "o", "command": "true", "done": "runs/o/done.json"}]},
    ])
    started = []
    release = threading.Event()

    class _Proc:
        def __init__(self, label):
            self.label = label

        def wait(self):
            release.wait(5)
            marker = tmp_path / "runs" / self.label / "done.json"   # the item finishes: its stage's marker appears
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}")
            return 0

    def _start(command, gpu):
        started.append(command)
        if len(started) == 2:
            release.set()
        import re

        return _Proc(re.search(r"runs/([a-z0-9]+)/console\.log", command).group(1))

    monkeypatch.setattr(lq, "_start", _start)
    monkeypatch.setattr(lq, "gpu_is_free", lambda gpu: True)
    lq.run([0, 1], path, poll_seconds=0.01, idle_exit_minutes=0.02)
    first_two = started[:2]
    assert any("runs/c0/console.log" in c for c in first_two) and any("runs/o/console.log" in c for c in first_two)
    assert len(started) == 4


def test_a_directory_claimed_by_a_live_launcher_is_passed_over_and_a_stale_claim_is_taken(tmp_path, monkeypatch):
    """Two launchers on one host: claim.json is created atomically before the start, so the second one skips the
    item (2026-10-15, three launchers in runs/E3a/BS_fct_idrid_seed1/FGADR). A claim of a dead pid is stale."""
    import os
    import socket
    import subprocess

    from bench.scripts import launch_queue as lq

    monkeypatch.setattr(lq, "_ROOT", tmp_path)
    monkeypatch.delenv("RETISEG_IGNORE_RUNNING", raising=False)
    assert lq.claim_dir("e3a:BS_x:FGADR", "mkdir -p runs/E3a/BS_x/FGADR && { echo; } >> runs/E3a/BS_x/FGADR/console.log 2>&1") == Path("runs/E3a/BS_x/FGADR")
    assert lq.claim_dir("nnunet:NN_default_fgadr_fold0", "bash bench/scripts/nnunet_run.sh FGADR default") == Path("runs/NN_default_fgadr_fold0")
    assert lq.claim_dir("shell:x", "true") is None
    # another live launcher on this host holds a: the claim fails; b is free
    live = tmp_path / "runs" / "a" / lq.CLAIM
    live.parent.mkdir(parents=True)
    live.write_text(json.dumps({"label": "shell:a", "host": socket.gethostname(), "pid": os.getpid()}))
    assert lq.claim("shell:a", Path("runs/a")) is False
    assert lq.claim("shell:b", Path("runs/b")) is True and json.loads((tmp_path / "runs" / "b" / lq.CLAIM).read_text())["pid"] == os.getpid()
    # a claim whose launcher died is stale and taken over; one from another host stands
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / "runs" / "c").mkdir()
    (tmp_path / "runs" / "c" / lq.CLAIM).write_text(json.dumps({"label": "shell:c", "host": socket.gethostname(), "pid": dead.pid}))
    assert lq.claim("shell:c", Path("runs/c")) is True
    (tmp_path / "runs" / "d").mkdir()
    (tmp_path / "runs" / "d" / lq.CLAIM).write_text(json.dumps({"label": "shell:d", "host": "elsewhere", "pid": 1}))
    assert lq.claim("shell:d", Path("runs/d")) is False
    monkeypatch.setenv("RETISEG_IGNORE_RUNNING", "1")
    assert lq.claim("shell:d", Path("runs/d")) is True
    monkeypatch.delenv("RETISEG_IGNORE_RUNNING")
    # through run(): the held item is passed over, the free one runs, and its claim is released afterwards
    monkeypatch.setattr(lq, "python_argv", lambda env: ["python"])
    path = _queue(tmp_path, [
        {"name": "s", "priority": 1, "kind": "shell",
         "items": [{"label": "a", "command": "true", "done": "runs/a/done.json"}, {"label": "e", "command": "true", "done": "runs/e/done.json"}]},
    ])
    started = []

    class _Proc:
        def wait(self):
            assert (tmp_path / "runs" / "e" / lq.CLAIM).is_file()   # claimed before the start
            marker = tmp_path / "runs" / "e" / "done.json"
            marker.write_text("{}")
            return 0

    def _start(command, gpu):
        started.append(command)
        return _Proc()

    monkeypatch.setattr(lq, "_start", _start)
    monkeypatch.setattr(lq, "gpu_is_free", lambda gpu: True)
    lq.run([0], path, poll_seconds=0.01, idle_exit_minutes=0.02)
    assert len(started) == 1 and "runs/e/console.log" in started[0]
    assert not (tmp_path / "runs" / "e" / lq.CLAIM).exists() and live.is_file()


def test_a_failing_item_is_not_restarted_before_its_retry_delay(tmp_path, monkeypatch):
    """2026-10-16: HRDecoder's scoring failed in seconds about 1,200 times in a night. An item that exits
    non-zero waits 10 min, then 30, 60, 180 and 360, before this launcher takes it again; the next item runs."""
    from bench.scripts import launch_queue as lq

    assert [lq.retry_delay(n) / 60 for n in (1, 2, 3, 4, 5, 9)] == [10, 30, 60, 180, 360, 360]
    monkeypatch.setattr(lq, "_ROOT", tmp_path)
    monkeypatch.setattr(lq, "python_argv", lambda env: ["python"])
    path = _queue(tmp_path, [
        {"name": "s", "priority": 1, "kind": "shell",
         "items": [{"label": "bad", "command": "false", "done": "runs/bad/done.json"}, {"label": "good", "command": "true", "done": "runs/good/done.json"}]},
    ])
    started = []

    class _Proc:
        def __init__(self, label):
            self.label = label

        def wait(self):
            if self.label == "bad":
                return 1
            marker = tmp_path / "runs" / "good" / "done.json"
            marker.write_text("{}")
            return 0

    def _start(command, gpu):
        import re

        label = re.search(r"runs/([a-z]+)/console\.log", command).group(1)
        started.append(label)
        return _Proc(label)

    monkeypatch.setattr(lq, "_start", _start)
    monkeypatch.setattr(lq, "gpu_is_free", lambda gpu: True)
    lq.run([0], path, poll_seconds=0.01, idle_exit_minutes=0.02)
    assert started == ["bad", "good"]   # bad once, then the card moved on; not bad again within the pause


def test_e3a_plan_leaves_out_test_sets_whose_prepared_directory_is_missing(tmp_path, monkeypatch):
    from bench import e3a
    from bench.scripts import launch_e3a as le

    frozen = e3a.load_frozen()
    assert "MAPLES-DR" in frozen["test_sets"] and list(frozen["test_sets"])[-1] == "MAPLES-DR"
    monkeypatch.setattr(le, "_ROOT", tmp_path)
    for name in ("IDRiD", "DDR"):
        (tmp_path / frozen["test_sets"][name]["data"]).mkdir(parents=True)
    missing = le.missing_test_sets(frozen)
    assert "MAPLES-DR" in missing and "IDRiD" not in missing
    monkeypatch.setattr(le, "discover", lambda seed0_only=False: [(1, "E1_m2mrf_idrid", "retiseg", ["-m", "bench.e3a"], ["IDRiD"])])
    monkeypatch.setattr(le, "in_progress", lambda run_dir: None)
    monkeypatch.setattr(le, "item_cost", lambda *args: 1.0)
    monkeypatch.setattr(e3a, "is_done", lambda source, test, variant: False)
    tests = {spec[3] for _tier, _cost, _label, spec in le.plan()}
    assert tests == {"DDR"}   # the one prepared set the IDRiD author weights have not seen


def test_maples_case_mapping_and_ignore_layers():
    import numpy as np

    from bench.data import maples

    fundus = np.zeros((8, 10, 3), dtype=np.uint8)
    fields = {name: np.zeros((8, 10), dtype=bool) for name in list(maples.BIOMARKERS.values()) + list(maples.UNCERTAIN.values()) + list(maples.EXTRA.values())}
    fields["Microaneurysms"][1, 1] = True
    fields["CottonWoolSpots"][2, 2] = True
    fields["RedUncertains"][3, 3] = True
    fields["RedUncertains"][1, 1] = True           # overlaps a certain MA: stays a positive, not an ignore pixel
    fields["BrightUncertains"][4, 4] = True
    fields["Drusens"][5, 5] = True
    case = maples.case_from_sample("x", fundus, lambda field: fields[field], {"dr": "R2", "me": "M0"})
    assert case["ext"] == "png" and case["bytes"][:8] == b"\x89PNG\r\n\x1a\n"
    assert case["masks"]["MA"][1, 1] and case["masks"]["SE"][2, 2] and not case["masks"]["HE"].any()
    ignore = case["binary"]["ignore"]
    assert ignore["MA"][3, 3] and ignore["HE"][3, 3] and not ignore["MA"][1, 1]
    assert ignore["EX"][4, 4] and ignore["SE"][4, 4] and not ignore["MA"][4, 4]
    assert case["binary"]["masks_ext"]["DRUSEN"][5, 5] and case["stats"] == {"dr": {"R2": 1}, "me": {"M0": 1}}
    bad = dict(fields, Hemorrhages=np.zeros((4, 4), dtype=bool))
    try:
        maples.case_from_sample("x", fundus, lambda field: bad[field])
    except ValueError as exc:
        assert "Hemorrhages" in str(exc)
    else:
        raise AssertionError("a mask of the wrong size must be refused")


def test_maples_official_lists_keep_the_duplicate_key_and_refuse_its_copy():
    """dataset_record.yaml maps the image kept in the official 138 (key) to the copy already left out (value);
    the first reader refused the key too and could not open the dataset (server, 2026-10-14)."""
    from bench.data import maples

    reader = maples.MaplesDR.__new__(maples.MaplesDR)
    train = [f"t{i}" for i in range(maples.EXPECTED["train"])]
    test = [f"s{i}" for i in range(maples.EXPECTED["test"])]
    reader.record = {"train": train, "test": test, "duplicates": {"t0": "copy_of_t0"}}
    assert reader.splits() == {"train": train, "val": [], "test": test}
    reader.record = {"train": train, "test": test, "duplicates": {"t0": "t1"}}
    try:
        reader.splits()
    except RuntimeError as exc:
        assert "duplicated" in str(exc)
    else:
        raise AssertionError("a left-out copy inside the official lists must be refused")


def test_maples_annotation_is_placed_in_the_roi_box_at_messidor_resolution(tmp_path):
    """The package's resize=False geometry: 1500 x 1500 label -> nearest resize to the ROI box -> pasted on the canvas."""
    import numpy as np
    import zipfile
    from PIL import Image

    from bench.data import maples

    label = np.zeros((1500, 1500), dtype=bool)
    label[:750, :750] = True                       # the top-left quarter
    canvas = maples.place_annotation(label, (420, 57, 1803, 1439), (1488, 2240))
    assert canvas.shape == (1488, 2240) and canvas[57:57 + 691, 420:420 + 691].all()
    assert not canvas[:57].any() and not canvas[:, :420].any() and not canvas[57 + 692:, :].any() and not canvas[:, 420 + 692:].any()
    try:
        maples.place_annotation(label, (420, 57, 2300, 1439), (1488, 2240))
    except ValueError:
        pass
    else:
        raise AssertionError("a box outside the image must be refused")
    # images are found flat, in Base directories or inside the released zips
    root = tmp_path / "messidor"
    (root / "Base12").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(root / "a.tif")
    Image.new("RGB", (4, 4)).save(root / "Base12" / "b.tif")
    with zipfile.ZipFile(root / "Base11.zip", "w") as handle:
        handle.writestr("Base11/c.tif", b"not read here")
    found = maples.index_messidor(root, ["a", "b", "c", "d"])
    assert found["a"] == root / "a.tif" and found["b"] == root / "Base12" / "b.tif"
    assert found["c"] == (root / "Base11.zip", "Base11/c.tif") and "d" not in found


def test_e5_infer_stage_plans_the_pool_through_launch_e5_infer(tmp_path, monkeypatch):
    """The unified queue delegates the inference tiers to launch_e5_infer: pool items only while the
    selections are empty, sets without a prepared directory left out, a gate on every test item."""
    from bench.e5 import geometry as geo
    from bench.scripts import launch_e5_infer as le
    from bench.scripts import launch_queue as lq

    from bench.e5 import outputs as out

    for name in ("IDRiD", "DDR"):   # the pool sets are on this machine, the other test sets are not
        (tmp_path / "prepared" / name).mkdir(parents=True)
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")   # not this machine's runs/E5
    monkeypatch.setattr(le, "_ROOT", tmp_path)   # no B-std / nnU-Net reference maps -> no same-information items
    monkeypatch.setattr(geo, "data_dir", lambda name: tmp_path / "prepared" / name)
    monkeypatch.setattr(le, "missing_prompt_files", lambda frozen, items=None: [])
    monkeypatch.setattr(le.Item, "running", lambda self: None)
    pool = lq.plan_e5_infer({"name": "p", "kind": "e5_infer", "phase": "pool"}, dry=True)
    assert pool and all(label.startswith("e5_infer:") for label, _c, _g in pool)
    assert all("console.log" in command and "== start e5" in command for _l, command, _g in pool)
    assert {label.split(":", 1)[1].split(":", 1)[0] for label, _c, _g in pool} >= {"b_sam_vitb_box", "b_sam2_hieral21_point"}
    test = lq.plan_e5_infer({"name": "t", "kind": "e5_infer", "phase": "test"}, dry=True)
    assert all(gate is not None for _l, _c, gate in test)   # the oracle tier waits for its pool thresholds
    import pytest

    with pytest.raises(ValueError, match="phase"):
        lq.plan_e5_infer({"name": "x", "kind": "e5_infer", "phase": "both"}, dry=True)


def test_a_stage_that_exits_or_has_an_unknown_kind_is_reported_not_fatal(tmp_path, monkeypatch):
    """python_argv raises SystemExit for a missing conda env (retiseg-sam3); a launcher older than the
    queue file meets an unknown kind. Both are reported and the other stages still plan (merge review)."""
    from bench.scripts import launch_queue as lq

    monkeypatch.setattr(lq, "_ROOT", tmp_path)

    def _exits(stage, dry):
        raise SystemExit("conda env retiseg-sam3 not found")

    monkeypatch.setitem(lq.PLANNERS, "exits", _exits)
    path = _queue(tmp_path, [
        {"name": "bad", "priority": 1, "kind": "exits"},
        {"name": "new", "priority": 2, "kind": "from_the_future"},
        {"name": "ok", "priority": 3, "kind": "shell", "items": [{"label": "a", "command": "true", "run_dir": "runs/a"}]},
    ])
    report = []
    items = lq.gather(path, dry=True, report=report)
    assert [label for _p, _s, label, _c, _g in items] == ["shell:a"]
    assert report[0].startswith("bad: SystemExit") and report[1].startswith("new: ValueError") and "restart" in report[1]
