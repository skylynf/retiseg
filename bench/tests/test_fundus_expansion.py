"""The fundus-specific expansion (docs/paper/18): output margins in the runner, and the five wrappers' contracts.

Everything needs torch; the wrapper forwards also need the reproductions / submodules / timm /
efficientnet_pytorch, and skip where those are missing. Run on the server before the smoke.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_card_margin_validation():
    from bench.models.base import ModelCard

    ModelCard(name="m", family="general", track="C", repro_level="R0", source="t", forward_size=(512, 512), output_margin=128)
    with pytest.raises(ValueError, match="forward_size"):
        ModelCard(name="m", family="general", track="C", repro_level="R0", source="t", output_margin=8)
    with pytest.raises(ValueError, match="supervised centre"):
        ModelCard(name="m", family="general", track="C", repro_level="R0", source="t", forward_size=(64, 64), output_margin=32)


def test_margin_inference_tiles_the_canvas_and_the_trainer_crops_the_target():
    torch = pytest.importorskip("torch")
    from torch import nn

    from bench.infer import canvas_logits
    from bench.models.base import ModelCard
    from bench.train import supervised_target

    class Centre(nn.Module):
        """Returns the centre of its input's first channel as one logit plane, shifted by the window position."""

        card = ModelCard(name="c", family="general", track="C", repro_level="R0", source="t", forward_size=(32, 32), output_margin=8)

        def forward(self, x):
            return x[:, :1, 8:-8, 8:-8]

    model = Centre().eval()
    canvas = torch.arange(50 * 70, dtype=torch.float32).reshape(1, 50, 70).repeat(3, 1, 1)
    out = canvas_logits(model, canvas, model.card, window=(32, 32), overlap=0.5, batch=4)
    assert out.shape == (1, 50, 70)
    assert torch.allclose(out[0], canvas[0])   # every pixel reconstructed from valid window centres only
    targets = torch.zeros(2, 4, 32, 32)
    targets[:, :, 8:-8, 8:-8] = 1
    cropped = supervised_target(targets, model)
    assert cropped.shape == (2, 4, 16, 16) and bool(cropped.all())
    model.loss_takes_full_target = True
    assert supervised_target(targets, model).shape == (2, 4, 32, 32)


def test_budget_counts_the_supervised_centre():
    from bench import bstd
    from bench.models.base import AuthorRecipe

    frozen = dict(bstd.load_frozen(), equivalent_epochs={"IDRiD": 2, "DDR": 2}, n_evals=1, recipe_revisions={})
    recipe = AuthorRecipe("dice", "sgd", 0.01, 0.0, "none", 0, 4, (0, 0), "", loss_params={})
    pixels = 10 * 512 * 512
    _r, dev, per_epoch, _s, _e = bstd.budget(recipe, "IDRiD", frozen, pixels, (512, 512), model_name="x")
    _r, dev_m, per_epoch_m, _s, _e = bstd.budget(recipe, "IDRiD", frozen, pixels, (512, 512), model_name="x", margin=128)
    assert per_epoch == 3 and per_epoch_m == 10 and "supervised centre 256x256" in dev_m[0]


def _recipe_ok(recipe, dataset):
    assert recipe.lr > 0 and recipe.effective_batch_size >= 1 and recipe.optimizer in ("adam", "sgd", "adamw")
    assert recipe.notes and recipe.source_of_settings


def test_recipes_read_without_building_a_network():
    from bench.models import clcnet, hrdecoder, pmcnet, tcnet, wsrfnet

    for module, name in ((pmcnet.PMCNet, "PMCNet"), (wsrfnet.WSRFNet, "WSRFNet")):
        for dataset in ("IDRiD", "DDR"):
            _recipe_ok(module.recipe_for(dataset), dataset)
    assert wsrfnet.WSRFNet.recipe_for("IDRiD").iterations == 80000 and wsrfnet.WSRFNet.recipe_for("DDR").iterations == 100000
    if (REPO / "official_code" / "HRDecoder" / "configs").is_dir():
        for dataset in ("IDRiD", "DDR"):
            recipe = hrdecoder.HRDecoder.recipe_for(dataset)
            _recipe_ok(recipe, dataset)
            assert recipe.optimizer == "sgd" and recipe.lr == 0.01 and recipe.effective_batch_size == 4
            assert recipe.optimizer_params["crop_num"] == (2 if dataset == "IDRiD" else 4) and recipe.loss_params["hr_loss_weight"] == 0.1
    if (REPO / "reproductions" / "Wang2023_CLC-Net").is_dir():
        recipe = clcnet.CLCNet.recipe_for("IDRiD")
        _recipe_ok(recipe, "IDRiD")
        assert recipe.effective_batch_size == 16 and recipe.optimizer_params["milestone_fractions"] == pytest.approx([20 / 60, 50 / 60])
        assert clcnet.CLCNet.card.output_margin == 128 and clcnet.CLCNet.card.class_index == (2, 1, 3, 4)
    if (REPO / "reproductions" / "Zhang2023_TC-Net").is_dir():
        recipe = tcnet.TCNet.recipe_for("DDR")
        _recipe_ok(recipe, "DDR")
        assert recipe.effective_batch_size == 4 and tcnet.TCNet.card.class_index == (3, 2, 1, 4) and tcnet.TCNet.card.pad_multiple == 64


def test_pmcnet_forward_shapes_and_taps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from bench.models.pmcnet import PMCNet, TAP_CHANNELS

    model = PMCNet().eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 128, 160))
    assert out.shape == (1, 5, 128, 160)
    assert tuple(model.net._taps[i].shape[1] for i in range(4)) == TAP_CHANNELS
    assert tuple(model.net._taps[0].shape[-2:]) == (64, 80) and tuple(model.net._taps[3].shape[-2:]) == (8, 10)
    loss = model.build_loss(model.author_recipe("DDR"))
    value = loss(model(torch.rand(1, 3, 64, 64)), (torch.rand(1, 4, 64, 64) > 0.8).float())
    assert value.ndim == 0 and torch.isfinite(value)


def test_wsrfnet_forward_and_loop_loss():
    torch = pytest.importorskip("torch")
    pytest.importorskip("efficientnet_pytorch")
    if not (REPO / "official_code" / "WSRFNet" / "model.py").is_file():
        pytest.skip("official_code/WSRFNet is not checked out")
    from bench.models.wsrfnet import WSRFNet

    model = WSRFNet()
    recipe = model.author_recipe("DDR")
    assert model.card.normalization["mean"][0] == pytest.approx(81.205 / 255.0)
    model.train()
    x = torch.rand(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 4, 64, 64) and len(model.net.loops_out) == 4
    loss = model.build_loss(recipe)(logits, (torch.rand(2, 4, 64, 64) > 0.8).float())
    assert loss.ndim == 0 and torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        assert model(x).shape == (2, 4, 64, 64) and model.net.loops_out is None


def test_clcnet_wrapper_contract():
    torch = pytest.importorskip("torch")
    if not (REPO / "reproductions" / "Wang2023_CLC-Net").is_dir():
        pytest.skip("reproductions/Wang2023_CLC-Net is not in this checkout")
    from bench.models.clcnet import CLCNet

    model = CLCNet()
    recipe = model.author_recipe("IDRiD")
    model.train()
    # SKConv's BatchNorm sits on a 1x1 pooled vector, so a training batch of 1 is invalid.
    x = torch.rand(2, 3, 512, 512)
    logits = model(x)
    assert logits.shape == (2, 5, 256, 256)
    target = (torch.rand(2, 4, 512, 512) > 0.9).float()
    loss = model.build_loss(recipe)(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    opt = model.build_optimizer(model.parameters(), recipe)
    model.adjust_lr(opt, 0, 60, recipe)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)
    model.adjust_lr(opt, 20, 60, recipe)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-5)
    model.adjust_lr(opt, 55, 60, recipe)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-6)


def test_tcnet_wrapper_contract():
    torch = pytest.importorskip("torch")
    if not (REPO / "reproductions" / "Zhang2023_TC-Net").is_dir():
        pytest.skip("reproductions/Zhang2023_TC-Net is not in this checkout")
    from bench.models.tcnet import TCNet

    model = TCNet()
    recipe = model.author_recipe("DDR")
    model.train()
    logits = model(torch.rand(1, 3, 128, 128))
    assert logits.shape == (1, 5, 128, 128)
    loss_fn = model.build_loss(recipe)
    target = (torch.rand(1, 4, 128, 128) > 0.9).float()
    opt = model.build_optimizer(model.parameters(), recipe)
    model.adjust_lr(opt, 0, 100, recipe)
    early = loss_fn(logits, target)
    model.adjust_lr(opt, 99, 100, recipe)
    late = loss_fn(logits, target)
    assert torch.isfinite(early) and torch.isfinite(late) and model._progress == pytest.approx(0.99)
