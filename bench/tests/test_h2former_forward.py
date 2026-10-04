import math
from pathlib import Path

import pytest
import torch
from torch import nn

from bench.models.registry import build_model
from bench.runtime import build_loss

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def model():
    # 960 with window 60 materializes a 6 GiB attention map. Keep the check on CPU
    # so a 16 GiB GPU that is already partly occupied can still run it.
    return build_model("H2Former").cpu().eval()


def test_forward_is_five_maps_at_960(model):
    image = torch.zeros(1, 3, 960, 960)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == (1, 5, 960, 960)
    assert model.card.output_activation == "softmax"
    assert model.card.class_index == (3, 2, 1, 4)
    assert model.card.forward_size == (960, 960)
    assert model.card.in_channels == 3
    assert model.card.family == "general"
    assert "not a DR-specific method" in model.author_recipe("IDRiD").notes
    assert model.net.conv1.in_channels == 3
    assert model.net.patch_embed.in_chans == 3


def test_recipe_is_shared_and_does_not_read_the_author_list_length(model):
    wrapper = model
    idrid = wrapper.author_recipe("IDRiD")
    ddr = wrapper.author_recipe("DDR")
    assert idrid == ddr
    assert idrid.loss == "ce_dice"
    assert idrid.optimizer == "adamw"
    assert idrid.lr == 1e-4
    assert idrid.weight_decay == 3e-5
    assert idrid.schedule == "poly"
    assert idrid.epochs == 92
    assert idrid.iterations == 0
    assert idrid.batch_size == 2
    assert idrid.effective_batch_size == 2
    assert idrid.crop_size == (960, 960)
    assert idrid.optimizer_params["betas"] == [0.9, 0.999]
    assert idrid.optimizer_params["eps"] == 1e-8
    assert idrid.optimizer_params["amsgrad"] is False
    assert idrid.optimizer_params["power"] == 0.9
    assert idrid.loss_params["smooth"] == 1e-5
    text = (REPO / "bench/models/h2former.py").read_text()
    assert "train_official" not in text
    assert "383" not in text
    assert "2022" not in text
    with pytest.raises(ValueError):
        wrapper.author_recipe("TJDR")


def _author_dice(probs, target, n_classes=5, smooth=1e-5):
    loss = probs.new_zeros(())
    for index in range(n_classes):
        score = probs[:, index]
        plane = (target == index).float()
        intersect = torch.sum(score * plane)
        y_sum = torch.sum(plane * plane)
        z_sum = torch.sum(score * score)
        loss = loss + (1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth))
    return loss / n_classes


def test_loss_follows_the_author_dice_and_not_the_factory(model):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 6, 6)
    labels = torch.randint(0, 5, (2, 6, 6))
    loss_fn = model.build_loss(model.author_recipe("IDRiD"))
    ce = nn.CrossEntropyLoss()
    probs = torch.softmax(logits, dim=1)
    expected = ce(logits, labels) + _author_dice(probs, labels)
    assert torch.allclose(loss_fn(logits, labels), expected)

    masks = torch.zeros(2, 4, 6, 6)
    masks[0, 0, 1, 1] = 1  # MA -> channel 3
    masks[0, 2, 2, 2] = 1  # EX -> channel 1
    masks[0, 0, 3, 3] = 1
    masks[0, 2, 3, 3] = 1  # MA is written after EX and kept
    masks[1, :, 4, 4] = 1  # every lesion; MA is last -> channel 3
    converted = loss_fn.indices(masks)
    assert converted[0, 1, 1].item() == 3
    assert converted[0, 2, 2].item() == 1
    assert converted[0, 3, 3].item() == 3
    assert converted[1, 4, 4].item() == 3
    assert converted[0, 0, 0].item() == 0
    assert torch.allclose(loss_fn(logits, masks), loss_fn(logits, converted))

    recipe = model.author_recipe("DDR")
    factory = build_loss(recipe, "softmax")
    assert not torch.allclose(loss_fn(logits, labels), factory(logits, labels))


def test_adjust_lr_keeps_the_base_rate_for_the_first_two_updates(model):
    recipe = model.author_recipe("IDRiD")
    optimizer = model.build_optimizer([torch.nn.Parameter(torch.zeros(1))], recipe)
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(1e-4)
    assert list(group["betas"]) == [0.9, 0.999]
    assert group["eps"] == pytest.approx(1e-8)
    assert group["weight_decay"] == pytest.approx(3e-5)
    total = 100
    model.adjust_lr(optimizer, 0, total, recipe)
    assert group["lr"] == pytest.approx(1e-4)
    model.adjust_lr(optimizer, 1, total, recipe)
    assert group["lr"] == pytest.approx(1e-4)
    model.adjust_lr(optimizer, 2, total, recipe)
    assert group["lr"] == pytest.approx(1e-4 * (1 - 1 / total) ** 0.9)
    assert group["lr"] != pytest.approx(1e-4 * (0.5**0.9))
    model.adjust_lr(optimizer, total, total, recipe)
    assert group["lr"] == pytest.approx(1e-4 * (1 - (total - 1) / total) ** 0.9)
    assert math.isclose(group["lr"], 1e-4 * (1 - (total - 1) / total) ** 0.9)
