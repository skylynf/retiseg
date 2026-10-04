import inspect

import pytest
import torch
from torchvision.models import ResNet101_Weights

from bench.models.deeplabv3plus import DeepLabV3Plus
from bench.models.registry import WRAPPERS, model_names
from bench.runtime import adjust_lr, build_optimizer, lesion_probabilities


def _model():
    return DeepLabV3Plus(weights_backbone=None)


def test_cpu_forward_keeps_spatial_size_and_returns_logits():
    model = _model().cpu().eval()
    image = torch.zeros(1, 3, 31, 47)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == (1, 4, 31, 47)
    assert model.card.forward_size is None
    assert model.card.class_index == ()
    assert model.card.output_activation == "sigmoid"
    assert model.card.pad_multiple == 8
    last = model.net.classifier[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        zeros = model(torch.zeros(1, 3, 32, 40))
    assert zeros.shape == (1, 4, 32, 40)
    assert torch.count_nonzero(zeros) == 0
    probabilities = lesion_probabilities(zeros, model.card)
    assert torch.allclose(probabilities, torch.full_like(probabilities, 0.5))


def test_normalization_matches_torchvision_imagenet():
    transforms = ResNet101_Weights.IMAGENET1K_V1.transforms()
    card = DeepLabV3Plus.card
    assert card.normalization["mean"] == list(transforms.mean)
    assert card.normalization["std"] == list(transforms.std)
    assert card.normalization["mean"] == [0.485, 0.456, 0.406]
    assert card.normalization["std"] == [0.229, 0.224, 0.225]


def test_recipe_is_shared_and_each_paper_item_is_cited():
    model = _model()
    idrid = model.author_recipe("IDRiD")
    ddr = model.author_recipe("DDR")
    assert idrid == ddr
    assert idrid.loss == "bce_with_logits"
    assert idrid.optimizer == "sgd"
    assert idrid.lr == 0.007
    assert idrid.weight_decay == 0.0001
    assert idrid.schedule == "poly"
    assert idrid.iterations == 30000
    assert idrid.epochs == 0
    assert idrid.batch_size == 1
    assert idrid.effective_batch_size == 16
    assert idrid.crop_size == (513, 513)
    assert idrid.pretrained == "ResNet101_Weights.IMAGENET1K_V1"
    assert idrid.optimizer_params == {"momentum": 0.9, "power": 0.9}
    source = idrid.source_of_settings
    assert "DeepLabv3+, ECCV 2018, Section 4" in source
    assert "DeepLabv3, arXiv:1706.05587, Section 4.1" in source
    assert "DeepLabv3+ Section 3" in source
    assert "train.py:116" in source
    assert "0.0001 for ResNet" in source
    assert "Section 4.3" in source
    assert "output stride is 8" in idrid.notes
    assert "not resized to 513" in idrid.notes
    assert "loader batch is 1" in idrid.notes
    assert "16, 16 and 12" in idrid.notes
    assert model.card.name == "DeepLabv3"
    optimizer = build_optimizer([torch.nn.Parameter(torch.zeros(1))], idrid)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0001)
    adjust_lr(optimizer, 0, 30000, idrid)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.007)
    adjust_lr(optimizer, 15000, 30000, idrid)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.007 * (0.5**0.9))
    with pytest.raises(ValueError, match="IDRiD or DDR"):
        model.author_recipe("FGADR")


def test_train_freezes_batch_norm():
    model = _model()
    model.train()
    batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert batch_norms
    assert all(not module.training for module in batch_norms)
    assert all(not param.requires_grad for module in batch_norms for param in module.parameters())


def test_registry_returns_the_class_by_name():
    assert "DeepLabv3" in model_names()
    assert "DeepLabv3+" not in model_names()
    assert WRAPPERS["DeepLabv3"] is DeepLabV3Plus
    default = inspect.signature(DeepLabV3Plus.__init__).parameters["weights_backbone"].default
    assert default == "IMAGENET1K_V1"
