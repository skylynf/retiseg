import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from bench.common.io import PROB_LEVELS, read_fov, read_prob_quantized
from bench.data.b1_input import PreparedSplit, collate
from bench.models.base import AuthorRecipe, ModelCard
from bench.models.registry import build_model
from bench.models.unet import UNet
from bench.predict import predict_image
from bench.runtime import (
    accumulation_steps,
    adjust_lr,
    build_loss,
    build_optimizer,
    compact_step_sizes,
    lesion_probabilities,
    optimizer_step_sizes,
    resolve_recipe,
    steps_per_epoch,
)
from bench.scripts.launch_b1 import main as launch_main
from bench.train import run_training

REPO = Path(__file__).resolve().parents[2]
SOURCE = (
    "Ronneberger 2015 uses cross-entropy from scratch. "
    "Adam and 1e-4 are the E0 pipeline settings, not a searched optimum. "
    "bce_dice is unweighted binary cross-entropy with logits plus sigmoid Dice, each with weight 1. "
    "The 2026-10-04 revision from unweighted BCE to bce_dice, a 200-epoch cap and patience 20 "
    "was made after the test mAUPR of runs/B1_unet_idrid_seed0 had already been read: "
    "60-epoch BCE, validation loss 0.591 to 0.118 and still falling, test mAUPR 0.174 "
    "(MA 0.002, HE 0.025, EX 0.651, SE 0.018). "
    "That test number motivated this revision and is not a blind baseline. "
    "It is not reused to choose another loss, budget or learning rate. "
    "Later changes to this recipe may use only training loss and validation loss."
)


def test_unet_cpu_forward_is_four_lesion_maps():
    model = build_model("U-Net").cpu().eval()
    image = torch.zeros(1, 3, 32, 48)
    with torch.no_grad():
        logits = model(image)
    assert logits.shape == (1, 4, 32, 48)
    assert model.card.forward_size is None
    assert model.card.class_index == ()
    assert model.card.pad_multiple == 16


def test_unet_recipe_is_the_same_bce_dice_on_idrid_and_ddr():
    model = UNet()
    idrid = model.author_recipe("IDRiD")
    ddr = model.author_recipe("DDR")
    assert idrid == ddr
    assert idrid.loss == "bce_dice"
    assert idrid.optimizer == "adam"
    assert idrid.lr == 1e-4
    assert idrid.weight_decay == 0.0
    assert idrid.schedule == "none"
    assert idrid.batch_size == 4
    assert idrid.effective_batch_size == 4
    assert idrid.pretrained == "none"
    assert idrid.epochs == 0
    assert idrid.iterations == 0
    assert idrid.loss_params["early_stop_patience"] == 20
    assert idrid.source_of_settings == SOURCE
    loss = model.build_loss(idrid)
    logits = torch.zeros(1, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8)
    bce = build_loss(replace(idrid, loss="bce_with_logits", loss_params={}), "sigmoid")
    dice = build_loss(replace(idrid, loss="dice", loss_params={"smooth": 1.0}), "sigmoid")
    assert torch.allclose(loss(logits, target), bce(logits, target) + dice(logits, target))


def test_unknown_loss_raises_and_is_not_replaced():
    recipe = replace(UNet().author_recipe("IDRiD"), loss="focal")
    with pytest.raises(ValueError, match="focal"):
        UNet().build_loss(recipe)


def test_unknown_optimizer_and_schedule_raise():
    base = UNet().author_recipe("IDRiD")
    parameter = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="rmsprop"):
        build_optimizer([parameter], replace(base, optimizer="rmsprop"))
    optimizer = build_optimizer([parameter], base)
    with pytest.raises(ValueError, match="cosine"):
        adjust_lr(optimizer, 0, 10, replace(base, schedule="cosine"))


def test_wrapper_loss_overrides_the_default():
    class _Custom(UNet):
        def build_loss(self, recipe):
            return torch.nn.L1Loss()

    recipe = replace(_Custom().author_recipe("DDR"), loss="not-a-default")
    assert isinstance(_Custom().build_loss(recipe), torch.nn.L1Loss)


def test_effective_batch_defaults_to_batch_size():
    recipe = AuthorRecipe(
        loss="bce_with_logits",
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.0,
        schedule="none",
        iterations=0,
        batch_size=4,
        crop_size=(512, 512),
        pretrained="none",
    )
    assert recipe.effective_batch_size == 4


def test_cell_epochs_fill_iterations_from_the_effective_batch():
    config = yaml.safe_load((REPO / "bench/configs/cells/b1_unet_idrid_seed0.yaml").read_text())
    assert "lr" not in config and "loss" not in config and "batch_size" not in config
    recipe = UNet().author_recipe("IDRiD")
    resolved, deviations, per_epoch = resolve_recipe(recipe, config, n_train=44)
    assert per_epoch == math.ceil(44 / 4) == 11
    assert resolved.epochs == 200
    assert resolved.iterations == 2200
    assert deviations == []
    assert steps_per_epoch(383, 4) == 96
    assert optimizer_step_sizes(44, 1, 16) == [16, 16, 12]
    assert compact_step_sizes([16, 16, 12]) == "16x2, 12"
    assert optimizer_step_sizes(44, 2, 12) == [12, 12, 12, 8]
    assert optimizer_step_sizes(383, 1, 16)[-1] == 15
    assert len(optimizer_step_sizes(383, 2, 12)) == 32
    assert optimizer_step_sizes(383, 2, 12)[-1] == 11
    assert optimizer_step_sizes(44, 24, 24) == [24, 20]
    assert accumulation_steps(2, 4) == 2
    with pytest.raises(ValueError):
        accumulation_steps(3, 4)


def test_author_iterations_win_over_the_cell_epoch_count():
    recipe = replace(UNet().author_recipe("DDR"), iterations=40000, epochs=0)
    resolved, deviations, _per_epoch = resolve_recipe(recipe, {"epochs": 60}, n_train=44)
    assert resolved.iterations == 40000
    assert any("ignored" in item for item in deviations)


def test_poly_uses_step_over_total_steps():
    recipe = replace(UNet().author_recipe("IDRiD"), schedule="poly", lr=0.01)
    optimizer = build_optimizer([torch.nn.Parameter(torch.zeros(1))], recipe)
    adjust_lr(optimizer, 0, 100, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    adjust_lr(optimizer, 50, 100, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * (0.5**0.9))
    floored = replace(recipe, optimizer_params={"power": 0.9, "min_lr": 1e-4})
    optimizer = build_optimizer([torch.nn.Parameter(torch.zeros(1))], floored)
    adjust_lr(optimizer, 100, 100, floored)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_ce_dice_is_the_sum_of_the_two_terms():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    base = replace(UNet().author_recipe("IDRiD"), loss_params={})
    ce = build_loss(replace(base, loss="ce"), "softmax")
    dice = build_loss(replace(base, loss="dice"), "softmax")
    both = build_loss(replace(base, loss="ce_dice"), "softmax")
    assert torch.allclose(both(logits, target), ce(logits, target) + dice(logits, target))


def test_softmax_planes_are_not_renormalized():
    card = ModelCard(
        name="Toy",
        family="general",
        track="B",
        repro_level="R1",
        source="test",
        output_activation="softmax",
        class_index=(1, 2, 3, 4),
    )
    logits = torch.tensor([5.0, 1.0, 1.0, 0.0, 0.0]).view(1, 5, 1, 1)
    full = torch.softmax(logits, dim=1)
    prob = lesion_probabilities(logits, card)
    assert torch.allclose(prob, full[:, (1, 2, 3, 4)])
    assert prob.sum().item() < 0.5
    assert not torch.allclose(prob, prob / prob.sum())


def test_sigmoid_card_rejects_a_class_index():
    with pytest.raises(ValueError):
        ModelCard(
            name="Toy",
            family="general",
            track="B",
            repro_level="R1",
            source="test",
            output_activation="sigmoid",
            class_index=(0, 1, 2, 3),
        )


def test_collate_pads_with_zero_after_normalization():
    image = torch.ones(3, 17, 19)
    target = torch.ones(4, 17, 19)
    images, targets = collate([(image, target)], pad_multiple=16)
    assert images.shape == (1, 3, 32, 32)
    assert targets.shape == (1, 4, 32, 32)
    assert torch.all(images[0, :, :17, :19] == 1)
    assert torch.all(images[0, :, 17:, :] == 0)
    assert torch.all(images[0, :, :, 19:] == 0)


def test_train_official_is_rejected():
    root = REPO / "dataset/prepared/IDRiD"
    with pytest.raises(ValueError, match="train_official"):
        PreparedSplit(root, "train_official", 64, False, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def test_registry_builds_unet_and_refuses_unregistered_names():
    model = build_model("U-Net")
    assert isinstance(model, UNet)
    from bench.models.registry import WRAPPERS

    assert "H2Former" in WRAPPERS
    with pytest.raises(KeyError, match="models.yaml"):
        build_model("not-a-model")


def test_jobs_file_lists_the_formal_table_and_launch_only_prints(capsys):
    document = yaml.safe_load((REPO / "bench/configs/b1_jobs.yaml").read_text())
    assert len(document["jobs"]) == 64
    assert document["jobs"][0]["env"] == "retiseg"
    launch_main([])
    out = capsys.readouterr().out
    assert "U-Net" in out
    assert "IDRiD" in out
    assert "gpu=0" in out
    assert "not submitted" in out
    assert "runs/B1_unet_idrid_seed0_runner" in out
    assert "formal tasks: 64" in out
    assert "Do not launch all 64" in out
    with pytest.raises(SystemExit, match="refusing to submit all 64"):
        launch_main(["--submit"])
    launch_main(["--smoke"])
    smoke_out = capsys.readouterr().out
    assert "smoke tasks: 16" in smoke_out
    assert "smoke_hacdr_ddr_seed0" in smoke_out
    assert "dataset=DDR" in smoke_out
    launch_main(["--seed0"])
    seed_out = capsys.readouterr().out
    assert "seed0 tasks: 16" in seed_out
    assert "--diagnostic" in seed_out
    assert "bench.eval.cli" not in seed_out
    assert "bench.eval.cli" in out
    assert "prob_test" in out
    assert "bench/predict.py" not in out
    assert "predict_m2mrf_b1.py" not in out
    assert "predict_hacdr_b1.py" not in out


class _Counted(Dataset):
    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self.calls += 1
        return torch.zeros(3, 16, 16), torch.zeros(4, 16, 16)


def test_optimizer_steps_follow_the_effective_batch(tmp_path):
    model = UNet()
    recipe = replace(model.author_recipe("IDRiD"), iterations=2, epochs=1)
    train_set = _Counted(8)
    val_set = _Counted(2)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate)
    best = run_training(
        model,
        train_loader,
        val_loader,
        recipe,
        {"seed": 0},
        tmp_path,
        micro_batch=2,
        steps_per_epoch=2,
        device="cpu",
    )
    assert train_set.calls == 8
    assert best is not None
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert rows == [
        {
            "epoch": 1,
            "iteration": 2,
            "train_loss": rows[0]["train_loss"],
            "val_loss": rows[0]["val_loss"],
            "lr": rows[0]["lr"],
        }
    ]
    assert rows[0]["lr"] == pytest.approx(1e-4)
    state = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert set(state) == {"model", "config", "step", "model_name"}
    assert state["step"] == 2
    assert state["model_name"] == "U-Net"
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert last["step"] == 2


def test_one_idrid_training_image_writes_a_full_size_probability(tmp_path):
    root = REPO / "dataset/prepared/IDRiD"
    model = UNet().cpu().eval()
    dataset = PreparedSplit(
        root,
        "train",
        1440,
        False,
        model.card.normalization["mean"],
        model.card.normalization["std"],
        model.card.forward_size,
    )
    assert len(dataset) == 44
    image_id = dataset.ids[0]
    images, targets = collate([dataset[0]], pad_multiple=model.card.pad_multiple)
    assert images.shape[0] == 1 and targets.shape[1] == 4
    assert images.shape[-2] % 16 == 0 and images.shape[-1] % 16 == 0
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (1, 4, images.shape[-2], images.shape[-1])
    predict_image(model, root, image_id, 1440, tmp_path, torch.device("cpu"))
    from PIL import Image

    original = Image.open(root / "images" / f"{image_id}.jpg")
    fov = read_fov(root, image_id)
    for cls in ("MA", "HE", "EX", "SE"):
        quantized = read_prob_quantized(tmp_path, cls, image_id)
        assert quantized.shape == (original.size[1], original.size[0])
        probability = quantized.astype("float64") / PROB_LEVELS
        assert probability.min() >= 0 and probability.max() <= 1
        assert quantized[~fov].max() == 0


def test_tail_step_averages_its_own_images():
    from bench.train import _run_epoch

    class _Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, images):
            return self.bias.view(1, 1, 1, 1).expand(images.shape[0], 4, images.shape[-2], images.shape[-1])

        def adjust_lr(self, optimizer, step, total_steps, recipe):
            return None

    class _Ones(Dataset):
        def __len__(self):
            return 12

        def __getitem__(self, index):
            return torch.ones(3, 8, 8), torch.ones(4, 8, 8)

    def _stack(batch):
        images, targets = zip(*batch)
        return torch.stack(images), torch.stack(targets)

    model = _Constant()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    recipe = replace(UNet().author_recipe("IDRiD"), iterations=1, effective_batch_size=16, batch_size=1)
    loader = DataLoader(_Ones(), batch_size=1, shuffle=False, collate_fn=_stack)
    counts = []
    _run_epoch(
        model,
        loader,
        torch.nn.MSELoss(),
        optimizer,
        recipe,
        torch.device("cpu"),
        16,
        16,
        1,
        0,
        1,
        counts,
    )
    assert counts == [12]
    assert model.bias.item() == pytest.approx(2.0)
