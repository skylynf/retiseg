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

    monkeypatch.setattr(lq, "_ROOT", tmp_path)
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


def test_queue_file_in_the_repository_loads_and_names_known_kinds():
    from bench.scripts import launch_queue as lq

    stages = lq.load_queue(REPO / "bench" / "configs" / "queue.yaml")
    names = [s["name"] for s in stages]
    assert names[:6] == ["e3a", "e5_seed0", "fundus_seed0", "e5_amg", "e5_prompted", "e5_prompted_points"]
    assert names.index("factor_cheap") < names.index("nnunet_e3b") < names.index("factor_e3b_heavy") < names.index("factor_e7_heavy")
    assert names.index("factor_e7_heavy") < names.index("hrdecoder_rest") < names.index("e5_rest") < names.index("fundus_rest")
    assert all(s["kind"] in lq.PLANNERS for s in stages)
    assert all(stages[i]["priority"] <= stages[i + 1]["priority"] for i in range(len(stages) - 1))
    by_name = {s["name"]: s for s in stages}
    assert by_name["e3a"]["max_concurrent"] == 2 and by_name["factor_cheap"]["models"] == ["U-Net", "M2MRF"]
    assert by_name["factor_e3b_heavy"]["experiment"] == "E3b" and not by_name["e5_amg_g2"].get("enabled", True)
    assert "more_masks" not in by_name["e5_amg"]["generators"] and "MAPLES-DR" in by_name["e5_amg"]["sets"]


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
    lq.run([0, 1], path, poll_seconds=0.01, idle_exit_minutes=0.0001)
    first_two = started[:2]
    assert any("runs/c0/console.log" in c for c in first_two) and any("runs/o/console.log" in c for c in first_two)
    assert len(started) == 4


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
    fields["microaneurysms"][1, 1] = True
    fields["cottonWoolSpots"][2, 2] = True
    fields["redUncertains"][3, 3] = True
    fields["redUncertains"][1, 1] = True           # overlaps a certain MA: stays a positive, not an ignore pixel
    fields["brightUncertains"][4, 4] = True
    fields["drusens"][5, 5] = True
    case = maples.case_from_sample("x", fundus, lambda field: fields[field], {"dr": "R2", "me": "M0"})
    assert case["ext"] == "png" and case["bytes"][:8] == b"\x89PNG\r\n\x1a\n"
    assert case["masks"]["MA"][1, 1] and case["masks"]["SE"][2, 2] and not case["masks"]["HE"].any()
    ignore = case["binary"]["ignore"]
    assert ignore["MA"][3, 3] and ignore["HE"][3, 3] and not ignore["MA"][1, 1]
    assert ignore["EX"][4, 4] and ignore["SE"][4, 4] and not ignore["MA"][4, 4]
    assert case["binary"]["masks_ext"]["DRUSEN"][5, 5] and case["stats"] == {"dr": {"R2": 1}, "me": {"M0": 1}}
    bad = dict(fields, hemorrhages=np.zeros((4, 4), dtype=bool))
    try:
        maples.case_from_sample("x", fundus, lambda field: bad[field])
    except ValueError as exc:
        assert "hemorrhages" in str(exc)
    else:
        raise AssertionError("a mask of the wrong size must be refused")
