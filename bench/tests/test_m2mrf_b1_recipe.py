"""M2MRF B1 recipe, read from the author configs without loading HRNet weights."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from PIL import Image

from bench.common.io import PROB_LEVELS, read_fov, read_prob_quantized, write_prob
from bench.data.b1_input import PreparedSplit, load_example
from bench.models.m2mrf import (
    CHANNEL_INDEX,
    M2MRF,
    author_recipe,
    b1_normalization,
    merged_model_dict,
)
from bench.models.registry import WRAPPERS, build_model
from bench.runtime import adjust_lr, build_optimizer
from bench.scripts.predict_m2mrf_b1 import write_image

REPO = Path(__file__).resolve().parents[2]


def test_idrid_and_ddr_iterations_differ_and_do_not_name_train_official():
    idrid = author_recipe("IDRiD")
    ddr = author_recipe("DDR")
    assert idrid.iterations != ddr.iterations
    assert idrid.iterations == 40000
    assert ddr.iterations == 60000
    assert "schedule_40k_idrid.py" in idrid.source_of_settings
    assert "schedule_60k_ddr.py" in ddr.source_of_settings
    for recipe in (idrid, ddr):
        blob = json.dumps(asdict(recipe))
        assert "train_official" not in blob
        assert recipe.loss == "dice"
        assert recipe.loss_params["smooth"] == pytest.approx(1e-5)
        assert recipe.optimizer == "sgd"
        assert recipe.lr == pytest.approx(0.01)
        assert recipe.weight_decay == pytest.approx(0.0005)
        assert recipe.optimizer_params["momentum"] == pytest.approx(0.9)
        assert recipe.schedule == "poly"
        assert recipe.optimizer_params["power"] == pytest.approx(0.9)
        assert recipe.optimizer_params["min_lr"] == pytest.approx(1e-4)
        assert recipe.batch_size == 4
        assert recipe.effective_batch_size == 4
        assert "BatchNorm sees 4 images" in recipe.notes
        assert recipe.pretrained == "open-mmlab://msra/hrnetv2_w48"
        assert "not resized to image_scale" in recipe.notes
        assert "not randomly cropped" in recipe.notes
        assert "PhotoMetricDistortion is not used" in recipe.notes
    assert idrid.crop_size == (960, 1440)
    assert ddr.crop_size == (1024, 1024)
    idrid_line = (REPO / "official_code/M2MRF/configs/_base_/schedules/schedule_40k_idrid.py").read_text().splitlines()
    ddr_line = (REPO / "official_code/M2MRF/configs/_base_/schedules/schedule_60k_ddr.py").read_text().splitlines()
    assert "40000" in idrid_line[6]
    assert "60000" in ddr_line[6]
    assert "schedule_40k_idrid.py:7" in idrid.source_of_settings
    assert "schedule_60k_ddr.py:7" in ddr.source_of_settings


def test_learning_rate_stays_at_the_author_value():
    recipe = M2MRF(construct=False).author_recipe("DDR")
    optimizer = build_optimizer([torch.nn.Parameter(torch.zeros(1))], recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    adjust_lr(optimizer, 0, recipe.iterations, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    adjust_lr(optimizer, recipe.iterations, recipe.iterations, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_normalization_follows_each_dataset_config():
    idrid_mean, idrid_std = b1_normalization("IDRiD")
    ddr_mean, ddr_std = b1_normalization("DDR")
    assert idrid_mean[0] == pytest.approx(116.513 / 255.0)
    assert idrid_std[2] == pytest.approx(13.293 / 255.0)
    assert ddr_mean[0] == pytest.approx(81.205 / 255.0)
    assert ddr_mean != idrid_mean
    assert ddr_std != idrid_std


def test_merged_config_is_m2mrf_c_without_loading_weights():
    cfg = merged_model_dict("IDRiD")
    assert cfg["backbone"]["type"] == "HRNet_M2MRF_C"
    assert cfg["decode_head"]["num_classes"] == 4
    assert cfg["pretrained"] == "open-mmlab://msra/hrnetv2_w48"
    assert cfg["backbone"]["extra"]["stage4"]["num_channels"] == (48, 96, 192, 384)
    assert merged_model_dict("DDR")["backbone"]["type"] == "HRNet_M2MRF_C"


def test_registry_names_m2mrf_and_construct_false_skips_the_network():
    assert WRAPPERS["M2MRF"] is M2MRF
    model = M2MRF(construct=False)
    assert model.network is None
    assert model.card.name == "M2MRF"
    assert model.card.output_activation == "sigmoid"
    assert model.card.class_index == ()
    assert model.card.forward_size is None
    assert CHANNEL_INDEX == (3, 1, 0, 2)


def test_forward_permutes_ex_he_se_ma_into_ma_he_ex_se():
    model = M2MRF(construct=False)

    class _Net(torch.nn.Module):
        def encode_decode(self, x, metas):
            value = torch.arange(1, 5, dtype=x.dtype, device=x.device).view(1, 4, 1, 1)
            return value.expand(x.shape[0], 4, x.shape[2], x.shape[3]).clone()

    model.network = _Net()
    out = model(torch.zeros(1, 3, 8, 16))
    assert out.shape == (1, 4, 8, 16)
    assert out[0, :, 0, 0].tolist() == [4, 2, 1, 3]


def test_scripts_call_b1_input_and_write_prob():
    import bench.scripts.predict_m2mrf_b1 as predict_script
    import bench.scripts.train_m2mrf_b1 as train_script

    assert train_script.PreparedSplit is PreparedSplit
    assert predict_script.load_example is load_example
    assert predict_script.write_prob is write_prob
    root = REPO / "dataset/prepared/IDRiD"
    mean, std = b1_normalization("IDRiD")
    split = PreparedSplit(root, "train", 1440, False, mean, std)
    assert len(split) == 44


def test_one_prepared_image_writes_four_full_size_maps(tmp_path):
    model = build_model("M2MRF")
    if model.network is None:
        pytest.skip(model.import_error)
    assert model.n_parameters() > 50_000_000
    assert str(model.versions["torch"])
    assert model.versions["mmcv"] == "1.2.0"
    assert "compat/mmcv" in str(model.versions["mmcv_file"])
    from bench.models.m2mrf import bind_normalization

    bind_normalization(model, "IDRiD")
    root = REPO / "dataset/prepared/IDRiD"
    image_id = PreparedSplit(
        root,
        "train",
        1440,
        False,
        model.card.normalization["mean"],
        model.card.normalization["std"],
    ).ids[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    write_image(model, root, image_id, 1440, tmp_path, device)
    original = Image.open(root / "images" / f"{image_id}.jpg")
    fov = read_fov(root, image_id)
    assert fov.shape == (original.size[1], original.size[0])
    for cls in ("MA", "HE", "EX", "SE"):
        quantized = read_prob_quantized(tmp_path, cls, image_id)
        assert quantized.shape == (original.size[1], original.size[0])
        assert int(quantized.min()) >= 0 and int(quantized.max()) <= PROB_LEVELS
        assert quantized[~fov].max() == 0
