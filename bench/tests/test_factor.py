"""bench/factor.py and bench/data/factor_input.py: the factor cells (E3b, E4, E7, E8, E9)."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from bench.common.io import LESION_CLASSES

REPO = Path(__file__).resolve().parents[2]


def _toy_prepared(root, name, n, seed, size=(48, 40), ignore=False):
    """A prepared dataset: images/<id>.png, fov, masks/<CLS>, splits.json (train = all but 2, val 1, test 1)."""
    rng = np.random.default_rng(seed)
    base = root / "dataset" / "prepared" / name
    for sub in ("images", "fov", *[f"masks/{cls}" for cls in LESION_CLASSES]):
        (base / sub).mkdir(parents=True, exist_ok=True)
    ids = [f"{name}_{i:02d}" for i in range(n)]
    for image_id in ids:
        Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(base / "images" / f"{image_id}.png")
        Image.fromarray(np.full(size, 255, np.uint8)).save(base / "fov" / f"{image_id}.png")
        for cls in LESION_CLASSES:
            Image.fromarray((rng.random(size) < 0.1).astype(np.uint8) * 255).save(base / "masks" / cls / f"{image_id}.png")
    if ignore:
        (base / "ignore" / "MA").mkdir(parents=True)
    (base / "splits.json").write_text(json.dumps({"train": ids[:-2], "val": ids[-2:-1], "test": ids[-1:]}))
    return base, ids


def test_multisource_concatenates_sources_and_applies_subsets(tmp_path, monkeypatch):
    monkeypatch.setenv("RETISEG_CANVAS_CACHE", "off")
    from bench.data.factor_input import MultiSource, PreparedSource

    ddr, ddr_ids = _toy_prepared(tmp_path, "DDR", 6, 0)
    fg, fg_ids = _toy_prepared(tmp_path, "FGADR", 5, 1)
    subset = tmp_path / "sub.json"
    subset.write_text(json.dumps({"dataset": "FGADR", "split": "train", "ids": fg_ids[:2]}))
    a = PreparedSource(ddr, "train", 1440, [0.5] * 3, [0.5] * 3)
    b = PreparedSource(fg, "train", 1440, [0.5] * 3, [0.5] * 3, subset=subset)
    assert len(a) == 4 and len(b) == 2 and b.ids == fg_ids[:2]
    multi = MultiSource([a, b])
    assert len(multi) == 6 and multi.ids[4] == f"FGADR/{fg_ids[0]}"
    image, masks = multi.canvas(5)
    ref, ref_masks = b.canvas(1)
    assert np.array_equal(image, ref) and all(np.array_equal(x, y) for x, y in zip(masks, ref_masks))
    _image, _masks, fov = multi.canvas_with_fov(0)
    assert fov.shape == image.shape[:2] and fov.all()
    with pytest.raises(IndexError):
        multi.canvas(6)
    with pytest.raises(ValueError, match="outside"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"dataset": "FGADR", "split": "train", "ids": ["nope"]}))
        PreparedSource(fg, "train", 1440, [0.5] * 3, [0.5] * 3, subset=bad)
    with pytest.raises(ValueError, match="is for"):
        PreparedSource(ddr, "train", 1440, [0.5] * 3, [0.5] * 3, subset=subset)


def test_native_diameter_keeps_the_crop_size(tmp_path, monkeypatch):
    monkeypatch.setenv("RETISEG_CANVAS_CACHE", "off")
    from bench.data.factor_input import PreparedSource

    ddr, _ids = _toy_prepared(tmp_path, "DDR", 3, 2, size=(48, 40))
    native = PreparedSource(ddr, "train", 0, [0.5] * 3, [0.5] * 3)
    small = PreparedSource(ddr, "train", 20, [0.5] * 3, [0.5] * 3)
    assert native.canvas(0)[0].shape[:2] == (48, 40)
    assert small.canvas(0)[0].shape[:2] == (20, 17)


def test_geometry_clips_the_patch_to_a_small_canvas():
    from bench import bstd, factor
    from bench.models.base import ModelCard

    frozen = dict(bstd.load_frozen(), fully_conv_patch=[960, 960])
    conv = ModelCard(name="c", family="general", track="B", repro_level="R0", source="t", pad_multiple=8)
    fixed = ModelCard(name="f", family="general", track="B", repro_level="R0", source="t", pad_multiple=8, forward_size=(224, 224))
    assert factor.geometry(conv, frozen, 1440) == ((960, 960), None)
    assert factor.geometry(conv, frozen, 512) == ((512, 512), None)
    assert factor.geometry(conv, frozen, 0) == ((960, 960), None)
    assert factor.geometry(fixed, frozen, 512) == ((224, 224), (224, 224))


def _cell(tmp_path, **over):
    body = {
        "experiment": "BF", "factor": "E4", "model": "M2MRF", "dataset": "dataset/prepared/DDR", "split_val": "val",
        "split_test": "test", "fov_diameter": 1440, "train": [{"dataset": "dataset/prepared/DDR", "split": "train"}],
        "budget_reference": "DDR", "seed": 0, "num_workers": 0,
        "paper": {"experiment": "E4", "variant": "DDR", "train_set": "DDR"},
    }
    body.update(over)
    return body


@pytest.mark.skipif(not (REPO / "dataset/prepared/DDR").is_dir(), reason="prepared DDR is not on this machine")
def test_assert_factor_cell_rejects_the_wrong_shapes():
    from bench import bstd, factor

    frozen = bstd.load_frozen()
    factor.assert_factor_cell(_cell(None), frozen)
    with pytest.raises(ValueError, match="uses"):
        factor.assert_factor_cell(_cell(None, model="HRNet"), frozen)
    with pytest.raises(ValueError, match="only E7"):
        factor.assert_factor_cell(_cell(None, fov_diameter=512), frozen)
    with pytest.raises(ValueError, match="E9"):
        factor.assert_factor_cell(_cell(None, loss_override={"loss": "dice"}), frozen)
    with pytest.raises(ValueError, match="budget_reference"):
        factor.assert_factor_cell(_cell(None, budget_reference="TJDR_std"), frozen)
    factor.assert_factor_cell(_cell(None, factor="E7", fov_diameter=0), frozen)
    # the HRDecoder case study (26 N1c): only E7 at the author's canvas, with the patch and the epoch budget keys
    study = _cell(None, factor="E7", model="HRDecoder", dataset="dataset/prepared/IDRiD", fov_diameter=2880, patch=1920,
                  budget_mode="equivalent_epochs", train=[{"dataset": "dataset/prepared/IDRiD", "split": "train"}], budget_reference="IDRiD",
                  paper={"experiment": "E7", "variant": "res:2880", "train_set": "IDRiD"})
    factor.assert_factor_cell(study, frozen)
    with pytest.raises(ValueError, match="author canvas"):
        factor.assert_factor_cell(dict(study, fov_diameter=1440), frozen)
    with pytest.raises(ValueError, match="E7 canvases"):
        factor.assert_factor_cell(_cell(None, factor="E7", fov_diameter=2880), frozen)
    with pytest.raises(ValueError, match="patch is"):
        factor.assert_factor_cell(_cell(None, patch=1920), frozen)
    with pytest.raises(ValueError, match="equivalent_epochs is"):
        factor.assert_factor_cell(_cell(None, budget_mode="equivalent_epochs"), frozen)
    with pytest.raises(ValueError, match="budget_mode is one of"):
        factor.assert_factor_cell(dict(study, budget_mode="steps"), frozen)


def test_apply_cell_geometry_rebuilds_a_fixed_window_wrapper_only_when_the_cell_asks():
    pytest.importorskip("torch")
    from bench import factor

    class Fixed:
        def __init__(self):
            self.patch = 960

        def set_patch(self, patch):
            self.patch = patch

    class Fully:
        pass

    fixed = factor.apply_cell_geometry(Fixed(), {"model": "HRDecoder", "patch": 1920})
    assert fixed.patch == 1920
    assert factor.apply_cell_geometry(Fixed(), {"model": "HRDecoder"}).patch == 960
    assert factor.apply_cell_geometry(Fully(), {"model": "U-Net"}) is not None
    with pytest.raises(ValueError, match="set_patch"):
        factor.apply_cell_geometry(Fully(), {"model": "U-Net", "patch": 1920})


def test_factor_jobs_cover_the_scope_of_16(tmp_path, monkeypatch):
    from bench.scripts import make_factor_jobs as mk

    monkeypatch.setattr(mk, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(mk, "JOBS", tmp_path / "factor_jobs.yaml")
    (tmp_path / "cells").mkdir()
    jobs = mk.jobs()
    by = {}
    for job in jobs:
        by.setdefault(job["experiment"], []).append(job)
    assert {k: len(v) for k, v in by.items()} == {"E3b": 8, "E4": 8, "E7": 5, "E8": 6, "E9": 2}
    assert len({job["name"] for job in jobs}) == 29 and all(job["run_dir"].startswith("runs/BF_") for job in jobs)
    for job in jobs:
        cell = yaml.safe_load((tmp_path / "cells" / f"{job['name']}.yaml").read_text())
        assert cell["experiment"] == "BF" and cell["seed"] == 0
        assert cell["budget_reference"] == ("IDRiD" if job["model"] == "HRDecoder" else "DDR")
        assert cell["paper"]["experiment"] == job["experiment"]
        if job["experiment"] != "E7":
            assert cell["fov_diameter"] == 1440
    # the HRDecoder case study (26 N1c) is the one cell with the author's canvas, the author's window and the epoch budget
    study = yaml.safe_load((tmp_path / "cells" / "e7_hrdecoder_idrid_2880_seed0.yaml").read_text())
    assert study["fov_diameter"] == 2880 and study["patch"] == 1920 and study["budget_mode"] == "equivalent_epochs"
    assert [job["env"] for job in jobs if job["model"] == "HRDecoder"] == ["retiseg-hacdr"]
    assert all("patch" not in yaml.safe_load((tmp_path / "cells" / f"{job['name']}.yaml").read_text()) for job in jobs if job["model"] != "HRDecoder")
    e9 = {job["name"]: yaml.safe_load((tmp_path / "cells" / f"{job['name']}.yaml").read_text()) for job in by["E9"]}
    assert e9["e9_unet_ddr_ce_seed0"]["loss_override"] == {"loss": "bce_with_logits", "loss_params": {}}
    assert e9["e9_unet_ddr_dice_seed0"]["loss_override"]["loss_params"]["batch_dice"] is True


def test_run_cell_trains_on_two_toy_sources_and_scores(tmp_path, monkeypatch):
    """End to end on toy data with a one-layer model: budget from the reference source, two training
    sources, validation selection, internal test maps. Needs torch; skipped where it is missing."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from bench import bstd, factor
    from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

    monkeypatch.setenv("RETISEG_CANVAS_CACHE", "off")
    ddr, _ = _toy_prepared(tmp_path, "DDR", 5, 0)
    fg, _ = _toy_prepared(tmp_path, "FGADR", 4, 1, ignore=True)
    monkeypatch.setattr(factor, "REPO", tmp_path)
    frozen = dict(bstd.load_frozen(), fully_conv_patch=[16, 16], n_evals=2, equivalent_epochs={"IDRiD": 2, "DDR": 2}, recipe_revisions={})
    monkeypatch.setattr(bstd, "load_frozen", lambda path=None: frozen)
    # the reference amount: DDR train (3 toy canvases of 48x40) at the frozen diameter would be resized; count 3 x 16 x 16
    monkeypatch.setattr(factor, "reference_pixels", lambda ref, fz, diameter=None: 3 * 16 * 16)

    class Tiny(nn.Module, SegmentationWrapper):
        card = ModelCard(name="M2MRF", family="fundus_multiscale", track="B", repro_level="R0", source="test", pad_multiple=8)
        classes = tuple(LESION_CLASSES)

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            return self.conv(x)

        def author_recipe(self, dataset):
            assert dataset == "DDR"
            return AuthorRecipe(loss="bce_with_logits", optimizer="adam", lr=1e-3, weight_decay=0.0, schedule="none",
                                iterations=0, batch_size=2, crop_size=(16, 16), pretrained="", loss_params={})

    config = _cell(tmp_path, train=[{"dataset": "dataset/prepared/DDR", "split": "train"}, {"dataset": "dataset/prepared/FGADR", "split": "train"}])
    config["dataset"] = "dataset/prepared/DDR"
    config_path = tmp_path / "cell.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "run"
    written = []

    def predict_fn(cfg, ckpt, pred_dir, split):
        written.append((Path(ckpt).name, Path(pred_dir).name, split))

    torch.manual_seed(0)
    model = Tiny()
    factor.run_cell(config_path, config, model, model.author_recipe("DDR"), run_dir, predict_fn)
    recipe = json.loads((run_dir / "recipe.json").read_text())
    assert recipe["n_train"] == 3 + 2
    assert [item["dataset"] for item in recipe["factor"]["train"]] == ["DDR", "FGADR"]
    assert recipe["bstd"]["samples_per_epoch"] == 4 and recipe["iterations"] == 4  # 3 patches per epoch, batch 2 -> 2 steps x 2 epochs
    assert any("ignore regions" in d for d in recipe["deviations"])
    rows = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [2, 4]
    assert (run_dir / "best.pt").is_file() and (run_dir / "environment.json").is_file()
    assert {(c, s) for c, _p, s in written} == {("best.pt", "val"), ("best.pt", "test"), ("last.pt", "val"), ("last.pt", "test")}


def test_factor_evaluation_excludes_the_same_eye_images_of_the_training_sources(tmp_path):
    """FGADR-trained cells drop FGADR's 106 same-eye test images (leak_exclusions.json); a metrics_test.json
    written without them is queued for re-evaluation; DDR-trained cells exclude nothing."""
    from bench.scripts import launch_factor as lf

    fgadr = _cell(None, factor="E3b", dataset="dataset/prepared/FGADR", train=[{"dataset": "dataset/prepared/FGADR", "split": "train"}])
    excluded = lf.internal_exclusions(fgadr)
    assert len(excluded) == 106 and excluded == sorted(excluded)
    assert lf.internal_exclusions(_cell(None)) == []
    (tmp_path / "cell.yaml").write_text(yaml.safe_dump(fgadr))
    job = {"config": str(tmp_path / "cell.yaml"), "run_dir": "runs/x", "name": "x"}
    argv = lf.evaluate_argv(job, ["python"])
    assert argv[argv.index("--exclude") + 1].split(",") == excluded
    (tmp_path / "ddr.yaml").write_text(yaml.safe_dump(_cell(None)))
    assert "--exclude" not in lf.evaluate_argv({"config": str(tmp_path / "ddr.yaml"), "run_dir": "runs/y"}, ["python"])
    run = tmp_path / "runs" / "x"
    run.mkdir(parents=True)
    (run / "metrics_test.json").write_text(json.dumps({"excluded_ids": []}))
    job["run_dir"] = str(run.relative_to(tmp_path))
    import unittest.mock as mock

    with mock.patch.object(lf, "_ROOT", tmp_path):
        assert lf.needs_reevaluation(job)
        (run / "metrics_test.json").write_text(json.dumps({"excluded_ids": excluded}))
        assert not lf.needs_reevaluation(job)

def test_the_2880_cell_budget_is_the_bstd_step_count_at_four_times_the_cost():
    """26 N1c as registered: the IDRiD equivalent epochs counted at canvas 2880 (canvas pixels x4) with 1920 patches
    (patch pixels x4) give the step count of the B-std IDRiD run; the run of 2026-10-21 counted 960 patches instead
    (83,000 steps) because the wrapper lost its patch (docs/ops/21 #82)."""
    pytest.importorskip("torch")
    from bench import bstd
    from bench.models.base import AuthorRecipe

    recipe = AuthorRecipe(loss="dice", optimizer="sgd", lr=0.01, weight_decay=5e-4, schedule="poly", iterations=20000,
                          batch_size=4, crop_size=(1920, 1920), pretrained="")
    frozen = {"equivalent_epochs": {"IDRiD": 1000}, "n_evals": 20, "fov_diameter": 1440, "recipe_revisions": {}}
    pixels_1440 = 75689280              # the IDRiD training split at the frozen diameter (runs/BS_hrdecoder_idrid_seed0/recipe.json)
    pixels_2880 = 302725440             # the same split at canvas 2880 (runs/BF_e7_hrdecoder_idrid_2880_seed0/recipe.json)
    bstd_run = bstd.budget(recipe, "IDRiD", frozen, pixels_1440, (960, 960))
    cell = bstd.budget(recipe, "IDRiD", frozen, pixels_2880, (1920, 1920))
    wrong = bstd.budget(recipe, "IDRiD", frozen, pixels_2880, (960, 960))
    assert bstd_run[2] == cell[2] == 21 and bstd_run[0].iterations == cell[0].iterations == 21000
    assert wrong[0].iterations == 83000
