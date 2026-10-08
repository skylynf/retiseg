import json
import os
import socket
import subprocess
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from bench.data.b1_input import collate
from bench.models.unet import UNet
from bench.runtime import evaluation_interval
from bench.scripts.launch_b1 import estimated_hours, in_progress, longest_first, main, run_queue
from bench.train import run_training


class _Indexed(Dataset):
    """Deterministic tensors per index. Raises once ``fail_after`` items were read."""

    def __init__(self, n, fail_after=None):
        self.n = n
        self.fail_after = fail_after
        self.calls = 0

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated crash")
        generator = torch.Generator().manual_seed(index)
        image = torch.randn(3, 16, 16, generator=generator)
        target = (torch.rand(4, 16, 16, generator=generator) > 0.9).float()
        return image, target


def _loader(dataset, shuffle=True):
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(dataset, batch_size=2, shuffle=shuffle, collate_fn=collate, generator=generator)


def _model():
    torch.manual_seed(0)
    return UNet()


def _recipe(model, iterations):
    return replace(
        model.author_recipe("IDRiD"),
        iterations=iterations,
        epochs=0,
        batch_size=2,
        effective_batch_size=2,
        loss_params={},
    )


def test_eval_every_validates_at_epoch_ends_past_each_multiple(tmp_path):
    model = _model()
    run_training(
        model,
        _loader(_Indexed(4)),
        _loader(_Indexed(2), shuffle=False),
        _recipe(model, 6),
        {"seed": 0},
        tmp_path,
        micro_batch=2,
        steps_per_epoch=2,
        device="cpu",
        eval_every=4,
    )
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [4, 6]
    assert not (tmp_path / "resume.pt").exists()
    assert torch.load(tmp_path / "last.pt", weights_only=False)["step"] == 6
    assert json.loads((tmp_path / "timing.json").read_text())["train_seconds_total"] >= 0


def test_iteration_budgets_validate_every_1000_steps_and_epoch_budgets_every_epoch():
    recipe = UNet().author_recipe("IDRiD")
    assert evaluation_interval(replace(recipe, iterations=30000), {}) == 1000
    assert evaluation_interval(replace(recipe, iterations=0, epochs=484), {}) is None
    assert evaluation_interval(replace(recipe, iterations=30000), {"eval_every": 250}) == 250


def test_resume_after_a_crash_matches_an_uninterrupted_run(tmp_path):
    straight = tmp_path / "straight"
    straight.mkdir()
    model = _model()
    run_training(
        model,
        _loader(_Indexed(4)),
        _loader(_Indexed(2), shuffle=False),
        _recipe(model, 6),
        {"seed": 0},
        straight,
        micro_batch=2,
        steps_per_epoch=2,
        device="cpu",
        eval_every=2,
    )
    reference = torch.load(straight / "last.pt", weights_only=False)["model"]

    crashed = tmp_path / "crashed"
    crashed.mkdir()
    model = _model()
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_training(
            model,
            _loader(_Indexed(4, fail_after=9)),
            _loader(_Indexed(2), shuffle=False),
            _recipe(model, 6),
            {"seed": 0},
            crashed,
            micro_batch=2,
            steps_per_epoch=2,
            device="cpu",
            eval_every=2,
        )
    assert torch.load(crashed / "resume.pt", weights_only=False)["step"] == 4

    model = _model()
    run_training(
        model,
        _loader(_Indexed(4)),
        _loader(_Indexed(2), shuffle=False),
        _recipe(model, 6),
        {"seed": 0},
        crashed,
        micro_batch=2,
        steps_per_epoch=2,
        device="cpu",
        eval_every=2,
        resume=True,
    )
    resumed = torch.load(crashed / "last.pt", weights_only=False)["model"]
    for name, tensor in reference.items():
        assert torch.allclose(tensor.float(), resumed[name].float(), atol=1e-6), name
    rows = [json.loads(line) for line in (crashed / "history.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [2, 4, 6]
    assert not (crashed / "resume.pt").exists()

    with pytest.raises(RuntimeError, match="already finished"):
        run_training(
            _model(),
            _loader(_Indexed(4)),
            _loader(_Indexed(2), shuffle=False),
            _recipe(model, 6),
            {"seed": 0},
            crashed,
            micro_batch=2,
            steps_per_epoch=2,
            device="cpu",
            eval_every=2,
            resume=True,
        )


def test_resume_refuses_a_different_budget(tmp_path):
    model = _model()
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_training(
            model,
            _loader(_Indexed(4, fail_after=5)),
            _loader(_Indexed(2), shuffle=False),
            _recipe(model, 6),
            {"seed": 0},
            tmp_path,
            micro_batch=2,
            steps_per_epoch=2,
            device="cpu",
            eval_every=2,
        )
    with pytest.raises(RuntimeError, match="fresh run directory"):
        run_training(
            _model(),
            _loader(_Indexed(4)),
            _loader(_Indexed(2), shuffle=False),
            _recipe(model, 8),
            {"seed": 0},
            tmp_path,
            micro_batch=2,
            steps_per_epoch=2,
            device="cpu",
            eval_every=2,
            resume=True,
        )


def test_shared_queue_runs_longest_first_and_frees_cards_as_they_finish():
    jobs = [
        {"name": "short", "model": "U-Net", "dataset": "IDRiD"},
        {"name": "long", "model": "DeepLabv3", "dataset": "DDR"},
        {"name": "mid", "model": "M2MRF", "dataset": "DDR"},
        {"name": "custom", "model": "U-Net", "dataset": "DDR", "est_hours": 30},
    ]
    assert [job["name"] for job in longest_first(jobs)] == ["long", "custom", "mid", "short"]
    assert estimated_hours(jobs[3]) == 30

    started = []

    class _Done:
        def __init__(self, code):
            self.code = code

        def wait(self):
            return self.code

    def _start(command, gpu):
        started.append((command, gpu))
        return _Done(1 if command == "bad" else 0)

    errors = run_queue([("a", "ok"), ("b", "bad"), ("c", "ok")], [0, 1], start=_start)
    assert sorted(command for command, _gpu in started) == ["bad", "ok", "ok"]
    assert len(errors) == 1 and "bad" in errors[0]


def test_gated_items_wait_for_their_gate_and_closed_gates_drop_them():
    started = []
    opened = {"seed0": False}

    class _Done:
        def __init__(self, command):
            self.command = command

        def wait(self):
            if self.command == "seed0":
                opened["seed0"] = True
            return 0

    def _start(command, gpu):
        started.append(command)
        return _Done(command)

    def after_seed0():
        return "open" if opened["seed0"] else "wait"

    items = [
        ("later", "later", after_seed0),
        ("dropped", "dropped", lambda: "never"),
        ("seed0", "seed0"),
    ]
    errors = run_queue(items, [0], start=_start, poll_seconds=0.01)
    assert started == ["seed0", "later"]
    assert len(errors) == 1 and "dropped" in errors[0]


def test_a_job_counts_as_running_until_its_exit_line_or_its_process_is_gone(tmp_path, monkeypatch):
    log = tmp_path / "console.log"
    assert in_progress(tmp_path) is None
    log.write_text("== start seed0 2026-10-07T10:00:00+08:00 gpu 2\nstep 1\n")
    assert in_progress(tmp_path).startswith("== start seed0")
    monkeypatch.setenv("RETISEG_IGNORE_RUNNING", "1")
    assert in_progress(tmp_path) is None
    monkeypatch.delenv("RETISEG_IGNORE_RUNNING")
    log.write_text(log.read_text() + "== exit 1 2026-10-07T11:00:00+08:00\n")
    assert in_progress(tmp_path) is None
    host = socket.gethostname()
    log.write_text(log.read_text() + f"== start seed0 2026-10-07T12:00:00+08:00 gpu 2 host {host} pid {os.getpid()}\n")
    assert in_progress(tmp_path) is not None
    gone = subprocess.Popen(["true"])
    gone.wait()
    log.write_text(log.read_text() + f"== start seed0 2026-10-07T13:00:00+08:00 gpu 2 host {host} pid {gone.pid}\n")
    assert in_progress(tmp_path) is None
    log.write_text(log.read_text() + "== start seed0 2026-10-07T14:00:00+08:00 gpu 2 host elsewhere pid 1\n")
    assert in_progress(tmp_path) is not None


def test_only_keeps_the_named_models_of_a_wave(capsys):
    main(["--jobs", "bench/configs/bs_jobs.yaml", "--smoke", "--only", "M2MRF,HACDR-Net"])
    out = capsys.readouterr().out
    assert "smoke tasks: 4" in out
    with pytest.raises(SystemExit, match="unknown models"):
        main(["--jobs", "bench/configs/bs_jobs.yaml", "--smoke", "--only", "NoSuchNet"])
