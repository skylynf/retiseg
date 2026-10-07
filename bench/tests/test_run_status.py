import json

from bench.scripts import run_status


def _run(root, name, rows, extra=None, log=None):
    run = root / name
    run.mkdir(parents=True)
    (run / "recipe.json").write_text(json.dumps({"iterations": 40}))
    (run / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for filename, payload in (extra or {}).items():
        (run / filename).write_text(json.dumps(payload))
    if log is not None:
        (run / "console.log").write_text(log)
    return run


def test_status_reads_state_selection_and_test_score(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_status, "RUNS", tmp_path)
    rows = [{"iteration": 20, "val_maupr": 0.3}, {"iteration": 40, "val_maupr": 0.25}]
    _run(tmp_path, "BS_a", rows, {"metrics_test.json": {"summary": {"mAUPR": 0.41234}}})
    _run(tmp_path, "BS_b", rows[:1], log="== start seed0 x\nTraceback\n== exit 1 y\n")
    _run(tmp_path, "B1_c", [{"iteration": 40, "val_loss": 0.5}], {"diagnostic.json": {}})
    table = {row["run"]: row for row in run_status.status_rows()}
    assert table["BS_a"]["state"] == "scored" and table["BS_a"]["best_val"] == 0.3
    assert table["BS_a"]["test_mAUPR"] == 0.4123
    assert table["BS_b"]["state"] == "failed(1)"
    assert table["B1_c"]["state"] == "trained" and table["B1_c"]["select_by"] == "val_loss"
    run_status.curves("BS_a")
    out = capsys.readouterr().out
    assert "best 0.3000 at 20/40 (50%)" in out and "still rising" not in out
