import pytest
import torch

from bench.models.fct import FCT
from bench.models.registry import WRAPPERS, build_model


def test_registry_name_is_fct():
    assert WRAPPERS["FCT"] is FCT
    model = build_model("FCT")
    assert model.card.name == "FCT"
    assert model.card.forward_size == (384, 384)
    assert model.card.normalization["mean"] == [0.0, 0.0, 0.0]
    assert model.card.normalization["std"] == [1.0, 1.0, 1.0]
    assert model.card.output_activation == "sigmoid"
    assert model.card.class_index == ()
    assert model.card.in_channels == 3


def test_cpu_forward_matches_the_input_spatial_size():
    model = build_model("FCT").cpu().eval()
    for height, width in ((32, 32), (64, 64)):
        image = torch.randn(1, 3, height, width)
        with torch.no_grad():
            logits = model(image)
        assert logits.shape == (1, 4, height, width)
        assert torch.isfinite(logits).all()


def test_nonsquare_input_is_rejected():
    model = build_model("FCT").cpu().eval()
    with pytest.raises(ValueError, match="square"):
        model(torch.zeros(1, 3, 32, 64))


def test_idrid_and_ddr_share_the_paper_scalars_the_pytorch_file_does_not_define():
    model = build_model("FCT")
    idrid = model.author_recipe("IDRiD")
    ddr = model.author_recipe("DDR")
    assert idrid == ddr
    assert idrid.loss == "bce_with_logits"
    assert idrid.optimizer == "adam"
    assert idrid.lr == 1e-3
    assert idrid.weight_decay == 0.0
    assert idrid.schedule == "none"
    assert idrid.epochs == 300
    assert idrid.iterations == 0
    assert idrid.batch_size == 4
    assert idrid.effective_batch_size == 4
    assert idrid.crop_size == (384, 384)
    assert idrid.pretrained == "none"
    assert "warmup_run_epochs 120" in idrid.notes
    assert "are not used" in idrid.notes
    assert "384" in idrid.notes
    assert "mean 0 and std 1" in idrid.notes
    assert "ImageNet" in idrid.notes
    assert "official PyTorch port" in idrid.notes
    assert "This project's adaptation" in idrid.notes


def test_other_datasets_are_rejected():
    with pytest.raises(ValueError, match="IDRiD or DDR"):
        build_model("FCT").author_recipe("ACDC")
