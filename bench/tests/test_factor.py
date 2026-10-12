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
    base = root / "prepared" / name
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


def test_factor_jobs_cover_the_scope_of_16(tmp_path, monkeypatch):
    from bench.scripts import make_factor_jobs as mk

    monkeypatch.setattr(mk, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(mk, "JOBS", tmp_path / "factor_jobs.yaml")
    (tmp_path / "cells").mkdir()
    jobs = mk.jobs()
    by = {}
    for job in jobs:
        by.setdefault(job["experiment"], []).append(job)
    assert {k: len(v) for k, v in by.items()} == {"E3b": 8, "E4": 8, "E7": 4, "E8": 6, "E9": 2}
    assert len({job["name"] for job in jobs}) == 28 and all(job["run_dir"].startswith("runs/BF_") for job in jobs)
    for job in jobs:
        cell = yaml.safe_load((tmp_path / "cells" / f"{job['name']}.yaml").read_text())
        assert cell["experiment"] == "BF" and cell["budget_reference"] == "DDR" and cell["seed"] == 0
        assert cell["paper"]["experiment"] == job["experiment"]
        if job["experiment"] != "E7":
            assert cell["fov_diameter"] == 1440
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
    monkeypatch.setattr(factor, "reference_pixels", lambda ref, fz: 3 * 16 * 16)

    class Tiny(nn.Module, SegmentationWrapper):
        card = ModelCard(name="M2MRF", family="fundus", track="B", repro_level="R0", source="test", pad_multiple=8)
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
