import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from bench import bstd
from bench.common.io import LESION_CLASSES
from bench.data import canvas_cache
from bench.data.b1_input import load_example
from bench.data.bs_input import EpochSampler, PatchTrain, augment_patch, scaled_window
from bench.infer import canvas_logits, window_starts
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

REPO = Path(__file__).resolve().parents[2]
IDRID = REPO / "dataset/prepared/IDRiD"
FROZEN = bstd.load_frozen()


class _Tiny(nn.Module, SegmentationWrapper):
    card = ModelCard(name="tiny", family="general", track="B", repro_level="R0", source="test", pad_multiple=8)
    classes = tuple(LESION_CLASSES)

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return self.conv(x)

    def author_recipe(self, dataset):
        return AuthorRecipe(
            loss="bce_with_logits",
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            schedule="none",
            iterations=0,
            batch_size=2,
            crop_size=(64, 64),
            pretrained="",
            loss_params={"early_stop_patience": 20},
        )


class _Split:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    forward_size = None

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def canvas(self, index):
        image, masks, _ = self.items[index]
        return image, masks

    def canvas_with_fov(self, index):
        return self.items[index]


def _canvas(seed, height=90, width=70):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    masks = [rng.random((height, width)) < 0.1 for _ in LESION_CLASSES]
    fov = np.ones((height, width), dtype=bool)
    return image, masks, fov


def test_scale_one_window_is_a_plain_crop():
    image, masks, _ = _canvas(0)
    rng = np.random.default_rng(3)
    probe = np.random.default_rng(3)
    top = int(probe.integers(0, 90 - 32 + 1))
    left = int(probe.integers(0, 70 - 32 + 1))
    window, planes, filled = scaled_window(image, masks, (32, 32), 1.0, rng)
    assert np.array_equal(window, image[top : top + 32, left : left + 32])
    for got, mask in zip(planes, masks):
        assert np.array_equal(got, mask[top : top + 32, left : left + 32])
    assert filled.all()


def test_window_past_a_small_canvas_is_padded_and_marked():
    image, masks, _ = _canvas(1, 40, 30)
    window, planes, filled = scaled_window(image, masks, (64, 64), 1.0, np.random.default_rng(0))
    assert window.shape == (64, 64, 3)
    assert filled[:40, :30].all() and not filled[40:].any() and not filled[:, 30:].any()
    assert not window[40:].any() and not any(plane[40:].any() for plane in planes)


def test_patches_depend_only_on_seed_and_sample_id():
    split = _Split([_canvas(seed) for seed in range(3)])
    dataset = PatchTrain(split, (32, 32), 12, FROZEN["augment"], seed=7)
    first = dataset[25]
    again = PatchTrain(split, (32, 32), 12, FROZEN["augment"], seed=7)[25]
    other = PatchTrain(split, (32, 32), 12, FROZEN["augment"], seed=8)[25]
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    assert not torch.equal(first[0], other[0])
    assert first[0].shape == (3, 32, 32) and first[1].shape == (4, 32, 32)
    assert set(np.unique(first[1].numpy())) <= {0.0, 1.0}


def test_augment_keeps_masks_aligned_with_the_image():
    height, width = 80, 80
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[20:40, 30:50] = 255
    mask = np.zeros((height, width), dtype=bool)
    mask[20:40, 30:50] = True
    params = dict(FROZEN["augment"])
    params["photometric"] = dict(params["photometric"], p=0.0)
    for seed in range(20):
        out, planes, filled = augment_patch(image, [mask] * 4, (48, 48), np.random.default_rng(seed), params)
        bright = out.mean(axis=-1) > 127
        interior = planes[0] & filled
        assert bright[interior].mean() > 0.9 if interior.any() else True
        assert not (bright & ~planes[0] & filled).sum() > 0.1 * max(1, planes[0].sum())


def test_epoch_sampler_never_repeats_an_id_and_resumes():
    sampler = EpochSampler(6, seed=1)
    epochs = [list(iter(sampler)) for _ in range(3)]
    assert sorted(epochs[1]) == list(range(6, 12))
    assert len(set(sum(epochs, []))) == 18
    resumed = EpochSampler(6, seed=1)
    resumed.set_epoch(2)
    assert list(iter(resumed)) == epochs[2]


def test_sliding_windows_match_the_whole_canvas_for_a_pointwise_model():
    torch.manual_seed(0)
    model = _Tiny().eval()
    tensor = torch.randn(3, 50, 37)
    whole = canvas_logits(model, tensor, model.card, None)
    for window in ((16, 16), (24, 40), (64, 64)):
        slid = canvas_logits(model, tensor, model.card, window, overlap=0.5, batch=3)
        assert slid.shape == whole.shape
        assert torch.allclose(slid, whole, atol=1e-5)
    assert window_starts(50, 16, 8)[-1] == 34 and window_starts(10, 16, 8) == [0]


def test_budget_counts_pixels_not_images():
    recipe = _Tiny().author_recipe("IDRiD")
    pixels = 44 * 1440 * 1172
    resolved, deviations, per_epoch, samples, eval_every = bstd.budget(recipe, "IDRiD", FROZEN, pixels, (960, 960))
    patches = -(-pixels // (960 * 960))
    assert per_epoch == -(-patches // 2) and samples == per_epoch * 2
    assert resolved.iterations == 1000 * per_epoch and resolved.epochs == 1000
    assert eval_every == 50 * per_epoch
    assert "early_stop_patience" not in resolved.loss_params
    small = bstd.budget(recipe, "IDRiD", FROZEN, pixels, (224, 224))[2]
    assert small > 17 * per_epoch


def test_budget_probe_doubles_the_amount_and_keeps_the_validation_spacing():
    recipe = _Tiny().author_recipe("IDRiD")
    pixels = 44 * 1440 * 1172
    frozen_run = bstd.budget(recipe, "IDRiD", FROZEN, pixels, (960, 960))
    probe_run = bstd.budget(recipe, "IDRiD", FROZEN, pixels, (960, 960), scale=2)
    assert probe_run[0].iterations == 2 * frozen_run[0].iterations
    assert probe_run[4] == frozen_run[4]
    assert bstd.probe_scale({"budget_probe": 2}) == 2 and bstd.probe_scale({}) == 1
    with pytest.raises(ValueError, match="budget_probe"):
        bstd.probe_scale({"budget_probe": 3})


def test_geometry_follows_forward_size():
    assert bstd.geometry(_Tiny.card, FROZEN) == ((960, 960), None)

    class _Card:
        forward_size = (224, 224)

    assert bstd.geometry(_Card, FROZEN) == ((224, 224), (224, 224))
    assert bstd.inference_for({"experiment": "B1"}, _Card) is None
    assert bstd.inference_for({"experiment": "BS"}, _Card) == {"window": (224, 224), "overlap": 0.5}


def test_val_scorer_gives_one_for_a_perfect_model_and_ignores_outside_fov():
    image, masks, fov = _canvas(2, 32, 32)
    fov[:, :4] = False
    split = _Split([(image, masks, fov)])

    class _Oracle(_Tiny):
        def forward(self, x):
            target = torch.from_numpy(np.stack(masks).astype(np.float32))[None]
            out = (target * 20 - 10)[..., : x.shape[-2], : x.shape[-1]]
            out[..., :, :4] = 10.0
            return out

    model = _Oracle()
    scorer = bstd.ValScorer(split, model.card, model.build_loss(model.author_recipe("IDRiD")), None, 0.5)
    row = scorer(model, torch.device("cpu"))
    assert row["val_maupr"] == pytest.approx(1.0)
    assert all(row[f"val_aupr_{cls}"] == pytest.approx(1.0) for cls in LESION_CLASSES)


def test_bs_cell_refuses_per_cell_budget_keys():
    config = {"fov_diameter": 1440, "split_train": "train", "split_val": "val", "split_test": "test", "epochs": 5}
    with pytest.raises(ValueError, match="does not set epochs"):
        bstd.assert_bs_frozen(config, "IDRiD", FROZEN)


@pytest.mark.skipif(not IDRID.is_dir(), reason="prepared IDRiD is not on this machine")
def test_cached_canvas_fov_equals_the_on_the_fly_fov(tmp_path, monkeypatch):
    monkeypatch.setattr(canvas_cache, "CACHE_ROOT", tmp_path)
    canvas_cache.build(IDRID, 1440, splits=("train",), only_ids={"IDRiD_01"})
    cache = canvas_cache.CanvasCache(IDRID, 1440, ["IDRiD_01"])
    fresh = load_example(IDRID, "IDRiD_01", 1440, [0.5] * 3, [0.5] * 3, None, masks=False)
    assert np.array_equal(cache.load_fov("IDRiD_01"), fresh["canvas_fov"])
    assert fresh["canvas_fov"].shape == fresh["image"].shape[:2]


def _small_protocol(monkeypatch):
    monkeypatch.setenv("RETISEG_CANVAS_CACHE", "off")
    frozen = dict(FROZEN, fully_conv_patch=[64, 64], n_evals=2, equivalent_epochs={"IDRiD": 2, "DDR": 2})
    monkeypatch.setattr(bstd, "load_frozen", lambda path=None: frozen)
    monkeypatch.setattr(bstd, "canvas_pixels", lambda dataset_dir, ids, diameter: 4 * 64 * 64)
    return {
        "experiment": "BS",
        "model": "tiny",
        "dataset": str(IDRID),
        "split_train": "train",
        "split_val": "val",
        "split_test": "test",
        "fov_diameter": 1440,
        "seed": 0,
        "num_workers": 0,
    }


@pytest.mark.skipif(not IDRID.is_dir(), reason="prepared IDRiD is not on this machine")
@pytest.mark.parametrize("workers", [0, 2])
def test_run_cell_resumes_to_the_same_weights_after_a_crash(tmp_path, monkeypatch, workers):
    config = dict(_small_protocol(monkeypatch), num_workers=workers)
    config_path = tmp_path / "cell.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def _run(run_dir, resume=False):
        torch.manual_seed(0)
        model = _Tiny()
        bstd.run_cell(config_path, config, model, model.author_recipe("IDRiD"), run_dir, None, True, resume)
        return torch.load(run_dir / "last.pt", weights_only=False)["model"]

    straight = _run(tmp_path / "straight")
    original = PatchTrain.__getitem__

    def _crash(self, sample_id):
        if sample_id >= 4:
            raise RuntimeError("simulated crash in epoch 2")
        return original(self, sample_id)

    monkeypatch.setattr(PatchTrain, "__getitem__", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path / "crashed")
    assert (tmp_path / "crashed" / "resume.pt").is_file()
    monkeypatch.setattr(PatchTrain, "__getitem__", original)
    resumed = _run(tmp_path / "crashed", resume=True)
    for key in straight:
        assert torch.equal(straight[key], resumed[key]), key


@pytest.mark.skipif(not IDRID.is_dir(), reason="prepared IDRiD is not on this machine")
def test_run_cell_trains_and_selects_by_validation_maupr(tmp_path, monkeypatch):
    config = _small_protocol(monkeypatch)
    config_path = tmp_path / "cell.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "run"
    torch.manual_seed(0)
    bstd.run_cell(config_path, config, _Tiny(), _Tiny().author_recipe("IDRiD"), run_dir, None, diagnostic=True)
    rows = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [2, 4]
    assert all(0.0 <= row["val_maupr"] <= 1.0 for row in rows)
    recipe = json.loads((run_dir / "recipe.json").read_text())
    assert recipe["bstd"]["samples_per_epoch"] == 4 and recipe["iterations"] == 4 and recipe["n_train"] == 44
    assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file()
    assert json.loads((run_dir / "diagnostic.json").read_text())["test_split_read"] is False
    best = max(rows, key=lambda row: row["val_maupr"])
    assert torch.load(run_dir / "best.pt", weights_only=False)["step"] == best["iteration"]
