"""Swin-Unet forward shape and the Synapse recipe, cited back to the author repo."""

from pathlib import Path

import pytest
import torch

from bench.models.registry import build_model
from bench.models.swin_unet import SwinUnet

REPO = Path(__file__).resolve().parents[2]
WRAPPER = Path(__file__).resolve().parents[1] / "models" / "swin_unet.py"


def _line(relative, number):
    return (REPO / relative).read_text().splitlines()[number - 1]


def test_forward_returns_five_logits_at_224():
    model = build_model("Swin-Unet").eval()
    assert model.card.name == "Swin-Unet"
    assert model.card.output_activation == "softmax"
    assert model.card.class_index == (1, 2, 3, 4)
    assert model.card.forward_size == (224, 224)
    assert model.card.normalization["mean"] == [0.0, 0.0, 0.0]
    assert model.card.normalization["std"] == [1.0, 1.0, 1.0]
    assert 224 % model.card.pad_multiple == 0
    image = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == (1, 5, 224, 224)
    assert not torch.allclose(logits.sum(dim=1), torch.ones(1, 224, 224), atol=1e-3)


def test_recipe_is_the_same_on_idrid_and_ddr_and_matches_source_lines():
    recipe = SwinUnet().author_recipe("IDRiD")
    assert recipe == SwinUnet().author_recipe("DDR")
    assert recipe.loss == "ce_dice"
    assert recipe.loss_params == {"smooth": 1e-5, "ce_weight": 0.4, "dice_weight": 0.6}
    assert recipe.optimizer == "sgd"
    assert recipe.optimizer_params == {"momentum": 0.9, "power": 0.9}
    assert recipe.lr == 0.01
    assert recipe.weight_decay == 0.0001
    assert recipe.schedule == "poly"
    assert recipe.epochs == 150
    assert recipe.iterations == 0
    assert recipe.batch_size == 24
    assert recipe.effective_batch_size == 24
    assert recipe.crop_size == (224, 224)
    assert recipe.pretrained == "swin_tiny_patch4_window7_224.pth"
    assert "1440" in recipe.notes
    assert "0.05" in recipe.notes
    model = SwinUnet()
    optimizer = model.build_optimizer([torch.nn.Parameter(torch.zeros(1))], recipe)
    total = 100
    model.adjust_lr(optimizer, 0, total, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    model.adjust_lr(optimizer, 1, total, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    model.adjust_lr(optimizer, 2, total, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * (1 - 1 / total) ** 0.9)
    assert "1234" not in WRAPPER.read_text()
    cited = {
        "train.py:24": ("official_code/Swin-Unet/train.py", 24, "default=150"),
        "train.py:26": ("official_code/Swin-Unet/train.py", 26, "default=24"),
        "train.py:27": ("official_code/Swin-Unet/train.py", 27, "default=1"),
        "train.py:30": ("official_code/Swin-Unet/train.py", 30, "default=0.01"),
        "train.py:33": ("official_code/Swin-Unet/train.py", 33, "default=224"),
        "trainer.py:26": ("official_code/Swin-Unet/trainer.py", 26, "args.batch_size * args.n_gpu"),
        "trainer.py:48": ("official_code/Swin-Unet/trainer.py", 48, "CrossEntropyLoss"),
        "trainer.py:49": ("official_code/Swin-Unet/trainer.py", 49, "DiceLoss"),
        "trainer.py:50": ("official_code/Swin-Unet/trainer.py", 50, "momentum=0.9, weight_decay=0.0001"),
        "trainer.py:54": ("official_code/Swin-Unet/trainer.py", 54, "max_epochs * len(train_loader)"),
        "trainer.py:68": ("official_code/Swin-Unet/trainer.py", 68, "softmax=True"),
        "trainer.py:69": ("official_code/Swin-Unet/trainer.py", 69, "0.4 * loss_ce + 0.6 * loss_dice"),
        "trainer.py:73": ("official_code/Swin-Unet/trainer.py", 73, "** 0.9"),
        "utils.py:24": ("official_code/Swin-Unet/utils.py", 24, "smooth = 1e-5"),
        "utils.py:27": ("official_code/Swin-Unet/utils.py", 27, "score * score"),
        "dataset_synapse.py:44": ("official_code/Swin-Unet/datasets/dataset_synapse.py", 44, "float32"),
        "swin_tiny_patch4_window7_224_lite.yaml:5": (
            "official_code/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml",
            5,
            "swin_tiny_patch4_window7_224.pth",
        ),
        "config.py:50": (
            "official_code/Swin-Unet/config.py",
            50,
            "swin_tiny_patch4_window7_224.pth",
        ),
    }
    for label, (relative, number, snippet) in cited.items():
        assert label in recipe.source_of_settings
        assert snippet in _line(relative, number)
    readme = _line("official_code/Swin-Unet/datasets/README.md", 4)
    assert "[0, 1]" in readme
    assert "datasets/README.md:4" in recipe.source_of_settings
    assert "mean" not in _line("official_code/Swin-Unet/datasets/dataset_synapse.py", 44)


def test_loss_is_weighted_ce_plus_squared_dice():
    recipe = SwinUnet().author_recipe("DDR")
    logits = torch.zeros(1, 5, 2, 2)
    logits[0, 1] = 2
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    target[0, 0, 0] = 1
    loss = SwinUnet().build_loss(recipe)
    ce = torch.nn.functional.cross_entropy(logits, target)
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, 5).permute(0, 3, 1, 2).float()
    dice = 0.0
    for index in range(5):
        score = probs[:, index]
        truth = one_hot[:, index]
        intersect = torch.sum(score * truth)
        y_sum = torch.sum(truth * truth)
        z_sum = torch.sum(score * score)
        dice = dice + (1 - (2 * intersect + 1e-5) / (z_sum + y_sum + 1e-5))
    expected = 0.4 * ce + 0.6 * (dice / 5)
    assert torch.allclose(loss(logits, target), expected)
    masks = torch.zeros(1, 4, 2, 2)
    masks[0, 0, 0, 0] = 1  # MA -> channel 1
    masks[0, 3, 0, 0] = 1  # SE on the same pixel; MA is written last
    masks[0, 2, 0, 1] = 1  # EX -> channel 3
    encoded = loss(torch.zeros(1, 5, 2, 2), masks)
    assert torch.isfinite(encoded)


def test_sgd_poly_uses_the_trainer_exponent():
    recipe = SwinUnet().author_recipe("IDRiD")
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = SwinUnet().build_optimizer([parameter], recipe)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0001)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    SwinUnet().adjust_lr(optimizer, 0, 100, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    SwinUnet().adjust_lr(optimizer, 1, 100, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    SwinUnet().adjust_lr(optimizer, 50, 100, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * ((1 - 49 / 100) ** 0.9))


def test_init_does_not_load_a_checkpoint(monkeypatch):
    def _refuse(*_args, **_kwargs):
        raise AssertionError("pretrained weights must not be loaded")

    monkeypatch.setattr(torch, "load", _refuse)
    model = build_model("Swin-Unet")
    assert model.author_recipe("IDRiD").pretrained == "swin_tiny_patch4_window7_224.pth"


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="IDRiD or DDR"):
        SwinUnet().author_recipe("Synapse")
