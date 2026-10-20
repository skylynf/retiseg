"""bench/train_hrdecoder.py: the schedule and the yaml-against-author-config check (E1r HRDecoder, docs/paper/26 N1a)."""

from pathlib import Path

import pytest
import yaml

from bench import train_hrdecoder as th

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "bench" / "configs" / "e1r_hrdecoder_idrid_seed0.yaml"
AUTHOR_CONFIGS = REPO / "official_code" / "HRDecoder" / "configs" / "lesion"


def _mmcv_poly_with_linear_warmup(iteration, base_lr, min_lr, power, max_iters, warmup_iters, warmup_ratio):
    """mmcv.runner.hooks.lr_updater: PolyLrUpdaterHook.get_lr and LrUpdaterHook.get_warmup_lr('linear')."""
    coeff = (1 - iteration / max_iters) ** power
    regular = (base_lr - min_lr) * coeff + min_lr
    if iteration >= warmup_iters:
        return regular
    k = (1 - iteration / warmup_iters) * (1 - warmup_ratio)
    return regular * (1 - k)


def test_poly10warm_matches_mmcv_at_every_phase():
    config = yaml.safe_load(CONFIG.read_text())
    args = (config["lr"], config["min_lr"], config["power"], config["iterations"], config["warmup_iters"], config["warmup_ratio"])
    for iteration in (0, 1, 2, 749, 1499, 1500, 1501, 5000, 10000, 19999):
        assert th.poly10warm_lr(iteration, config) == pytest.approx(_mmcv_poly_with_linear_warmup(iteration, *args), rel=1e-12, abs=1e-15)
    # the warmup starts almost at zero and hands over continuously to the poly curve at warmup_iters
    assert th.poly10warm_lr(0, config) < 1e-7
    assert abs(th.poly10warm_lr(1499, config) - th.poly10warm_lr(1500, config)) < 1e-5
    assert th.poly10warm_lr(10000, config) == pytest.approx(0.005)


def test_cell_records_the_project_rules():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["experiment"] == "E1r" and config["model"] == "HRDecoder" and config["dataset"] == "IDRiD"
    assert config["checkpoint_selection"] == "last", "E1r reports the last step; the author's save_best on the test split is only measured"
    assert config["iterations"] % config["save_every"] == 0 and config["save_every"] == 2000, "one checkpoint per author evaluation interval"
    assert config["batch_size"] == 4 and config["micro_batch"] == 4, "BN sees the four images SyncBN would have synchronised"
    assert config["seed"] == 0 and config["author_seed"] == 2
    assert config["train_ids"] == "official_train"


@pytest.mark.skipif(not (AUTHOR_CONFIGS / "hrdecoder_fcn_hr48_idrid_2880x1920-slide.py").is_file(), reason="official_code/HRDecoder is not checked out")
def test_cell_agrees_with_the_author_configs():
    pytest.importorskip("torch")
    config = yaml.safe_load(CONFIG.read_text())
    mean, std = th.check_against_author(config)
    assert mean == pytest.approx([116.513, 56.437, 16.309]) and std == pytest.approx([80.206, 41.232, 13.293])
    drifted = dict(config, iterations=40000)
    with pytest.raises(RuntimeError, match="iterations"):
        th.check_against_author(drifted)
