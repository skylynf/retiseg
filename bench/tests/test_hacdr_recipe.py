"""HACDR-Net recipes stay on the two author configs, including each max_iters."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from bench.models.hacdr import (
    CONFIGS,
    HACDRNet,
    adjust_hacdr_lr,
    author_recipe,
    author_rgb_norm,
    b1_normalization,
    fgadr_class_spec,
    hacdr_param_groups,
    lesion_class_index,
    masks_to_label,
)
from bench.models.registry import WRAPPERS, build_model
from bench.scripts.train_hacdr_b1 import training_step

REPO = Path(__file__).resolve().parents[2]


def _namespace(dataset):
    path = CONFIGS[dataset]
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def test_crop_notes_differ_and_iterations_come_from_each_runner():
    idrid_cfg = _namespace("IDRiD")
    ddr_cfg = _namespace("DDR")
    idrid = author_recipe("IDRiD")
    ddr = author_recipe("DDR")
    assert idrid.iterations == int(idrid_cfg["runner"]["max_iters"])
    assert ddr.iterations == int(ddr_cfg["runner"]["max_iters"])
    assert idrid.notes != ddr.notes
    assert "(960, 1440)" in idrid.notes
    assert "(1280, 1280)" in ddr.notes
    assert "(1280, 1280)" not in idrid.notes
    assert "(960, 1440)" not in ddr.notes
    assert idrid.crop_size == (960, 1440)
    assert ddr.crop_size == (1280, 1280)


def test_recipe_fields_are_taken_from_that_config():
    for dataset in ("IDRiD", "DDR"):
        cfg = _namespace(dataset)
        recipe = author_recipe(dataset)
        assert recipe.optimizer == str(cfg["optimizer"]["type"]).lower() == "adamw"
        assert recipe.lr == float(cfg["optimizer"]["lr"])
        assert recipe.weight_decay == float(cfg["optimizer"]["weight_decay"])
        assert recipe.optimizer_params["betas"] == [float(v) for v in cfg["optimizer"]["betas"]]
        assert recipe.schedule == str(cfg["lr_config"]["policy"]).lower() == "poly"
        assert recipe.optimizer_params["power"] == float(cfg["lr_config"]["power"])
        assert recipe.optimizer_params["min_lr"] == float(cfg["lr_config"]["min_lr"])
        assert recipe.optimizer_params["warmup"] == cfg["lr_config"]["warmup"]
        assert recipe.optimizer_params["warmup_iters"] == int(cfg["lr_config"]["warmup_iters"])
        assert recipe.optimizer_params["warmup_ratio"] == float(cfg["lr_config"]["warmup_ratio"])
        assert recipe.optimizer_params["paramwise_cfg"] == cfg["optimizer"]["paramwise_cfg"]
        assert recipe.loss_params["loss_decode"] == cfg["model"]["decode_head"]["loss_decode"]
        assert recipe.batch_size == int(cfg["data"]["samples_per_gpu"])
        assert recipe.effective_batch_size == recipe.batch_size == 1
        assert recipe.pretrained == "none"
        assert recipe.epochs == 0
        assert "divided by 255" in recipe.notes
        assert "0-255" in recipe.notes
        assert f"HACDR_{dataset.lower()}.py" in recipe.source_of_settings
        assert "max_iters=40000" in recipe.source_of_settings
    idrid = author_recipe("IDRiD")
    ddr = author_recipe("DDR")
    assert idrid.loss == "ce"
    assert [item["type"] for item in idrid.loss_params["loss_decode"]] == ["CrossEntropyLoss"]
    assert idrid.loss_params["loss_decode"][0]["loss_weight"] == 1.0
    assert ddr.loss == "ce_naloss"
    assert [item["type"] for item in ddr.loss_params["loss_decode"]] == ["CrossEntropyLoss", "NALoss"]
    assert ddr.loss_params["loss_decode"][1]["loss_weight"] == 0.2
    with pytest.raises(ValueError):
        author_recipe("TJDR")


def test_channels_and_normalization_follow_fgadr_and_the_config():
    assert fgadr_class_spec() == ("background", "EX", "MA", "SE", "HE")
    assert lesion_class_index() == (2, 4, 1, 3)
    assert HACDRNet.card.class_index == (2, 4, 1, 3)
    assert HACDRNet.card.output_activation == "softmax"
    assert HACDRNet.card.forward_size is None
    assert HACDRNet.card.pad_multiple == 32
    assert HACDRNet.classes == ("MA", "HE", "EX", "SE")
    mean, std = author_rgb_norm("IDRiD")
    assert mean == [123.675, 116.28, 103.53]
    assert std == [58.395, 57.12, 57.375]
    card_mean, card_std = b1_normalization("IDRiD")
    assert HACDRNet.card.normalization["mean"] == card_mean
    assert HACDRNet.card.normalization["std"] == card_std
    assert card_mean == pytest.approx([value / 255.0 for value in mean])
    assert card_std == pytest.approx([value / 255.0 for value in std])
    assert b1_normalization("DDR") == (card_mean, card_std)


def test_registry_lists_hacdr_without_building_a_foreign_head():
    assert WRAPPERS["HACDR-Net"] is HACDRNet
    model = build_model("HACDR-Net")
    assert model.card.name == "HACDR-Net"
    assert model.author_recipe("DDR").crop_size == (1280, 1280)


def test_exclusive_labels_use_fgadr_ids_and_later_masks_replace_earlier_ones():
    target = torch.zeros(1, 4, 2, 2)
    target[0, 0, 0, 0] = 1  # MA -> 2
    target[0, 1, 0, 1] = 1  # HE -> 4
    target[0, 2, 1, 0] = 1  # EX -> 1
    target[0, 3, 1, 1] = 1  # SE -> 3
    label = masks_to_label(target, (2, 4, 1, 3))
    assert label[0, 0, 0].item() == 2
    assert label[0, 0, 1].item() == 4
    assert label[0, 1, 0].item() == 1
    assert label[0, 1, 1].item() == 3
    assert label[0, 1, 1].item() != 0
    overlap = target.clone()
    overlap[0, 0, 1, 1] = 1  # MA on the SE pixel; MA is written last and kept
    replaced = masks_to_label(overlap, (2, 4, 1, 3))
    assert replaced[0, 1, 1].item() == 2
    empty = masks_to_label(torch.zeros(1, 4, 1, 1), (2, 4, 1, 3))
    assert empty.item() == 0


def test_linear_warmup_scales_the_configured_learning_rate():
    recipe = author_recipe("IDRiD")
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=recipe.lr, weight_decay=recipe.weight_decay)
    optimizer.param_groups[0]["initial_lr"] = recipe.lr
    adjust_hacdr_lr(optimizer, 0, recipe.iterations, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(recipe.lr * recipe.optimizer_params["warmup_ratio"])
    step = recipe.optimizer_params["warmup_iters"]
    adjust_hacdr_lr(optimizer, step, recipe.iterations, recipe)
    coeff = (1.0 - step / recipe.iterations) ** recipe.optimizer_params["power"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(recipe.lr * coeff)


def test_paramwise_keys_follow_the_longest_match():
    recipe = author_recipe("DDR")

    class _Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.conv = torch.nn.Conv2d(1, 1, 1)
            self.backbone.norm1 = torch.nn.BatchNorm2d(1)
            self.backbone.pos_block = torch.nn.Linear(1, 1)
            self.decode_head = torch.nn.Module()
            self.decode_head.norm = torch.nn.BatchNorm2d(1)
            self.decode_head.conv = torch.nn.Conv2d(1, 1, 1)

    module = _Toy()
    ordered = list(module.named_parameters())
    built = hacdr_param_groups(ordered, recipe.lr, recipe.weight_decay, recipe.optimizer_params["paramwise_cfg"])
    by_name = {name: group for (name, _), group in zip(ordered, built)}
    assert by_name["backbone.conv.weight"]["lr"] == pytest.approx(recipe.lr)
    assert by_name["backbone.conv.weight"]["weight_decay"] == pytest.approx(recipe.weight_decay)
    assert by_name["backbone.norm1.weight"]["weight_decay"] == 0.0
    assert by_name["backbone.pos_block.weight"]["weight_decay"] == 0.0
    assert by_name["decode_head.conv.weight"]["lr"] == pytest.approx(recipe.lr * 15.0)
    assert by_name["decode_head.norm.weight"]["lr"] == pytest.approx(recipe.lr * 15.0)
    assert by_name["decode_head.norm.weight"]["weight_decay"] == pytest.approx(recipe.weight_decay)


def test_ddr_loss_channels_pretrained_and_probability_export(tmp_path):
    from bench.common.io import read_fov, read_prob_quantized
    from bench.predict import predict_image
    from bench.runtime import lesion_probabilities
    from bench.scripts.predict_hacdr_b1 import _check_channels

    ddr = author_recipe("DDR")
    text = CONFIGS["DDR"].read_text()
    assert ddr.loss == "ce_naloss"
    assert ddr.pretrained == "none"
    assert "load_from = None" in text
    assert "pretrained=None" in text
    assert "type='NALoss'" in text
    model = HACDRNet()
    _check_channels(model)
    assert model.card.class_index == (2, 4, 1, 3)

    class _Stub(torch.nn.Module):
        card = HACDRNet.card
        classes = HACDRNet.classes

        def forward(self, images):
            logits = torch.full((images.shape[0], 5, images.shape[-2], images.shape[-1]), -2.0)
            logits[:, 2] = 4.0
            return logits

        def eval(self):
            return self

    root = REPO / "dataset" / "prepared" / "DDR"
    image_id = json.loads((root / "splits.json").read_text())["train"][0]
    stub = _Stub()
    predict_image(stub, root, image_id, 1440, tmp_path, torch.device("cpu"))
    fov = read_fov(root, image_id)
    planes = [read_prob_quantized(tmp_path, cls, image_id).astype("float64") / 65535 for cls in ("MA", "HE", "EX", "SE")]
    assert planes[0].shape == fov.shape
    assert all(plane[~fov].max() == 0 for plane in planes)
    inside = np.stack(planes)[:, fov]
    assert inside[0].mean() > inside[1:].mean()
    logits = torch.zeros(1, 5, 2, 2)
    logits[0, 2] = 4
    prob = lesion_probabilities(logits, HACDRNet.card)
    assert float(prob[0, :, 0, 0].sum()) < 1.0

    if model.net is None:
        return
    if not torch.cuda.is_available():
        with pytest.raises(Exception):
            model.build_loss(ddr)
        return
    loss = model.build_loss(ddr).cuda()
    assert [type(module).__name__ for module in loss.loss_modules] == ["CrossEntropyLoss", "NALoss"]
    value = training_step(root, device="cuda")
    assert math.isfinite(value)


def test_one_prepared_image_step_is_finite_when_the_network_imports():
    try:
        value = training_step(REPO / "dataset" / "prepared" / "IDRiD")
    except ImportError as exc:
        text = str(exc)
        if any(token in text for token in ("mmcv", "mmseg", "No module named", "bench/compat")):
            pytest.skip(text)
        raise
    assert math.isfinite(value)
