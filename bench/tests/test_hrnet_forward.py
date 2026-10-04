import math

import pytest
import torch

from bench.models.registry import build_model

PRETRAINED = "hrnetv2_w48_imagenet_pretrained_top1_21.pth"


def test_hrnet_cpu_forward_is_four_lesion_maps():
    model = build_model("HRNet").cpu().eval()
    image = torch.zeros(1, 3, 32, 48)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == (1, 4, 32, 48)
    assert torch.isfinite(logits).all()
    assert model.card.name == "HRNet"
    assert model.card.output_activation == "sigmoid"
    assert model.card.class_index == ()
    assert model.card.forward_size is None
    assert model.classes == ("MA", "HE", "EX", "SE")
    assert model.card.normalization["mean"] == [0.485, 0.456, 0.406]
    assert model.card.normalization["std"] == [0.229, 0.224, 0.225]


def test_hrnet_recipe_is_the_cityscapes_yaml_on_both_datasets():
    model = build_model("HRNet")
    idrid = model.author_recipe("IDRiD")
    ddr = model.author_recipe("DDR")
    assert idrid == ddr
    assert idrid.loss == "bce_with_logits"
    assert idrid.optimizer == "sgd"
    assert idrid.lr == 0.01
    assert idrid.weight_decay == 0.0005
    assert idrid.schedule == "poly"
    assert idrid.iterations == 0
    assert idrid.epochs == 484
    assert idrid.batch_size == 2
    assert idrid.effective_batch_size == 12
    assert idrid.crop_size == (512, 1024)
    assert idrid.pretrained == PRETRAINED
    assert idrid.loss_params == {}
    assert idrid.optimizer_params["momentum"] == 0.9
    assert idrid.optimizer_params["nesterov"] is False
    assert "seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml" in idrid.source_of_settings
    assert "bce_with_logits" in idrid.source_of_settings
    loss = model.build_loss(idrid)
    logits = torch.zeros(1, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8)
    assert loss(logits, target).item() == pytest.approx(math.log(2), abs=1e-5)
    optimizer = model.build_optimizer(model.parameters(), idrid)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert optimizer.param_groups[0]["nesterov"] is False
    total = 100
    model.adjust_lr(optimizer, 0, total, idrid)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    model.adjust_lr(optimizer, 1, total, idrid)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    model.adjust_lr(optimizer, 2, total, idrid)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * (1 - 1 / total) ** 0.9)
    assert "drop_last=True" in idrid.notes
    with pytest.raises(ValueError):
        model.author_recipe("FGADR")
