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
    assert [s["name"] for s in stages][:5] == ["e3a", "e5_seed0", "fundus_seed0", "factor", "nnunet_e3b"]
    assert all(s["kind"] in lq.PLANNERS for s in stages)
    assert all(stages[i]["priority"] <= stages[i + 1]["priority"] for i in range(len(stages) - 1))
