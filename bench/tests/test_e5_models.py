"""E5 training wrappers (design/15 sections 7-8): official builds, heads, losses, optimizers, schedules."""

import math

import pytest
import torch
import torch.nn.functional as F

from bench.models import foundation


def test_legacy_reduce_none_is_the_mean_bce_on_this_pytorch():
    """SAM2-UNet train.py:33 passes reduce='none'; PyTorch reads the legacy flag as reduction='mean'."""
    import warnings

    pred = torch.randn(2, 4, 8, 8)
    mask = (torch.rand(2, 4, 8, 8) > 0.7).float()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    assert legacy.ndim == 0
    assert torch.allclose(legacy, F.binary_cross_entropy_with_logits(pred, mask))


def test_structure_loss_is_mean_bce_plus_per_image_weighted_iou():
    from bench.models.sam2_unet import structure_loss

    pred = torch.randn(2, 4, 40, 40)
    mask = (torch.rand(2, 4, 40, 40) > 0.8).float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    p = torch.sigmoid(pred)
    inter = (p * mask * weit).sum(dim=(2, 3))
    union = ((p + mask) * weit).sum(dim=(2, 3))
    expected = F.binary_cross_entropy_with_logits(pred, mask) + (1 - (inter + 1) / (union - inter + 1)).mean()
    assert torch.allclose(structure_loss(pred, mask), expected, atol=1e-6)
    assert not torch.allclose(structure_loss(pred, mask), structure_loss(pred, mask, batch_iou=True))


def test_stepped_cosine_is_the_epoch_stepped_cosine_annealing():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20, eta_min=1e-7)
    expected = []
    for _ in range(20):
        expected.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    total = 2000
    for epoch in range(20):
        step = epoch * total // 20
        assert foundation.stepped_cosine(step, total, 1e-3, 1e-7, 20) == pytest.approx(expected[epoch], rel=1e-9)
        assert foundation.stepped_cosine(step + total // 20 - 1, total, 1e-3, 1e-7, 20) == pytest.approx(expected[epoch])


def test_sam2_unet_is_the_author_network_with_four_outputs():
    from bench.models.sam2_unet import SAM2UNet

    model = SAM2UNet()
    trainable = foundation.trainable(model)
    assert 4.3e6 < trainable < 4.5e6
    assert all(not p.requires_grad for n, p in model.net.encoder.named_parameters() if "prompt_learn" not in n)
    model.train()
    outs = model(torch.randn(1, 3, 64, 96))
    assert [tuple(o.shape) for o in outs] == [(1, 4, 64, 96)] * 3
    loss = model.build_loss(model.author_recipe("IDRiD"))(outs, (torch.rand(1, 4, 64, 96) > 0.9).float())
    loss.backward()
    assert model.net.encoder.blocks[0].prompt_learn[0].weight.grad is not None
    model.eval()
    with torch.no_grad():
        assert tuple(model(torch.randn(1, 3, 64, 64)).shape) == (1, 4, 64, 64)
    assert model.card.pad_multiple == 32 and model.card.forward_size is None


def test_sam2_unet_reference_trunk_is_copied(monkeypatch, tmp_path):
    import bench.models.sam2_unet as module

    model = module.SAM2UNet()
    reference = module._build(None)
    with torch.no_grad():
        for param in reference.encoder.parameters():
            param.add_(1.0)
    monkeypatch.setattr(module, "_build", lambda checkpoint=None, device="cpu": reference)
    monkeypatch.setattr(module, "weights", lambda name: tmp_path / name)
    before = model.net.encoder.blocks[2].prompt_learn[0].weight.clone()
    copied = model.load_pretrained()
    assert copied > 0
    assert torch.equal(model.net.encoder.blocks[2].block.attn.qkv.weight, reference.encoder.blocks[2].block.attn.qkv.weight)
    assert torch.equal(model.net.encoder.blocks[2].prompt_learn[0].weight, before)


def test_retfound_layer_decay_groups_and_schedule():
    from bench.models.vit_upernet import LAYER_DECAY, ViTUPerNet, layer_id

    assert layer_id("cls_token", 25) == 0 and layer_id("patch_embed.proj.weight", 25) == 0
    assert layer_id("blocks.0.attn.qkv.weight", 25) == 1 and layer_id("blocks.23.mlp.fc2.bias", 25) == 24
    model = ViTUPerNet()
    recipe = model.author_recipe("DDR")
    assert recipe.lr == pytest.approx(5e-3 * 24 / 256) and recipe.effective_batch_size == 24
    optimizer = model.build_optimizer(model.parameters(), recipe)
    by_param = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            by_param[id(param)] = group
    named = dict(model.named_parameters())
    assert by_param[id(named["encoder.pos_embed"])]["weight_decay"] == 0.0
    assert by_param[id(named["encoder.pos_embed"])]["lr_scale"] == pytest.approx(LAYER_DECAY ** 25)
    assert by_param[id(named["encoder.blocks.0.attn.qkv.weight"])]["lr_scale"] == pytest.approx(LAYER_DECAY ** 24)
    assert by_param[id(named["encoder.blocks.0.attn.qkv.weight"])]["weight_decay"] == pytest.approx(0.05)
    assert by_param[id(named["encoder.blocks.0.attn.qkv.bias"])]["weight_decay"] == 0.0
    assert by_param[id(named["decode_head.conv_seg.weight"])]["lr_scale"] == pytest.approx(1.0)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len([p for p in model.parameters() if p.requires_grad])
    rates = []
    for step in (0, 100, 200, 600, 999):
        model.adjust_lr(optimizer, step, 1000, recipe)
        rates.append(max(g["lr"] for g in optimizer.param_groups))
    assert rates[0] == 0.0 and rates[1] == pytest.approx(recipe.lr / 2) and rates[2] == pytest.approx(recipe.lr)
    assert rates[4] == pytest.approx(1e-6, rel=1e-2)
    # The PPM's 1x1 pooled branch has BatchNorm: a training batch needs at least 2 windows (B-std uses 24).
    model.train()
    main, aux = model(torch.randn(2, 3, 224, 224))
    assert main.shape == aux.shape == (2, 4, 224, 224)


def test_retfound_load_requires_every_block_tensor(monkeypatch, tmp_path):
    from bench.models import vit_upernet

    model = vit_upernet.ViTUPerNet()
    state = {f"backbone.{k}": v + 1.0 for k, v in model.encoder.state_dict().items()}
    state["decoder_pred.weight"] = torch.zeros(1)
    torch.save({"model": state}, tmp_path / "RETFound_mae_natureCFP.pth")
    monkeypatch.setattr(foundation, "PRETRAINED", tmp_path)
    info = model.load_pretrained()
    assert info["missing"] == 0 and info["unexpected"] == 1
    assert torch.equal(model.encoder.blocks[7].norm1.weight, state["backbone.blocks.7.norm1.weight"])
    del state["backbone.blocks.11.attn.proj.weight"]
    torch.save({"model": state}, tmp_path / "RETFound_mae_natureCFP.pth")
    with pytest.raises(RuntimeError, match="does not cover"):
        model.load_pretrained()


def test_encoder_comparison_variants_keep_their_own_normalization():
    from bench.models.registry import EXPERIMENT_ONLY

    assert EXPERIMENT_ONLY["ViT-UPerNet augreg"].card.normalization["mean"] == [0.5, 0.5, 0.5]
    assert EXPERIMENT_ONLY["ViT-UPerNet DINOv2"].ENCODER == "dinov2"
    assert EXPERIMENT_ONLY["ViT-UPerNet RETFound-DINOv2"].card.pad_multiple == 14


def test_samed_lora_keys_loss_and_schedule():
    from bench.models.samed import SAMed, _CEDice, lora_key

    assert lora_key("image_encoder.blocks.3.attn.qkv.weight") == "image_encoder.blocks.3.attn.qkv.qkv.weight"
    assert lora_key("image_encoder.blocks.3.attn.proj.weight") == "image_encoder.blocks.3.attn.proj.weight"
    logits = torch.randn(2, 5, 16, 16)
    target = (torch.rand(2, 4, 16, 16) > 0.85).float()
    loss = _CEDice()(logits, target)
    from bench.common.labels import exclusive_label

    labels = exclusive_label(target, (1, 2, 3, 4))
    probs = torch.softmax(logits, 1)
    onehot = F.one_hot(labels, 5).permute(0, 3, 1, 2).float()
    dice = [1 - (2 * (probs[:, i] * onehot[:, i]).sum() + 1e-5) / ((probs[:, i] ** 2).sum() + (onehot[:, i] ** 2).sum() + 1e-5) for i in range(5)]
    expected = 0.2 * F.cross_entropy(logits, labels) + 0.8 * torch.stack(dice).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    model = SAMed()
    assert 4.0e6 < foundation.trainable(model) < 6.5e6
    model.train()
    out = model(torch.rand(1, 3, 512, 512))
    assert tuple(out.shape) == (1, 5, 512, 512)
    recipe = model.author_recipe("IDRiD")
    optimizer = model.build_optimizer(model.parameters(), recipe)
    length, warmup = model.schedule(8000, recipe)
    assert length == pytest.approx(10000) and warmup == round(10000 * 250 / 18600)
    rates = {}
    for step in (0, 1, 2, warmup, warmup + 1):
        model.adjust_lr(optimizer, step, 8000, recipe)
        rates[step] = optimizer.param_groups[0]["lr"]
    assert rates[0] == rates[1] == pytest.approx(recipe.lr / warmup)
    assert rates[2] == pytest.approx(2 * recipe.lr / warmup) and rates[warmup] == pytest.approx(recipe.lr)
    assert rates[warmup + 1] == pytest.approx(recipe.lr)
    model.adjust_lr(optimizer, 7999, 8000, recipe)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(recipe.lr * (1 - (7998 - warmup) / 10000) ** 0.9)


def test_medsam_dice_is_monai_per_image_and_the_model_is_prompt_free():
    from bench.models.medsam import MedSAM, dice_loss

    logits = torch.randn(2, 4, 8, 8)
    target = (torch.rand(2, 4, 8, 8) > 0.8).float()
    p = torch.sigmoid(logits)
    per = 1 - (2 * (p * target).sum((2, 3)) + 1e-5) / ((p * p).sum((2, 3)) + (target * target).sum((2, 3)) + 1e-5)
    assert torch.allclose(dice_loss(logits, target), per.mean())
    model = MedSAM()
    assert all(not p.requires_grad for p in model.sam.prompt_encoder.parameters())
    optimizer = model.build_optimizer(model.parameters(), model.author_recipe("DDR"))
    held = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert held == {id(p) for p in model.sam.image_encoder.parameters()} | {id(p) for p in model.sam.mask_decoder.parameters()}
    model.eval()
    with torch.no_grad():
        assert tuple(model(torch.rand(1, 3, 1024, 1024)).shape) == (1, 4, 1024, 1024)


def test_sam_adapter_trains_the_prompt_generator_and_decoder_only():
    from bench.models.sam_adapter import SAMAdapter, iou_loss

    pred = torch.randn(2, 4, 8, 8)
    target = (torch.rand(2, 4, 8, 8) > 0.5).float()
    s = torch.sigmoid(pred)
    inter = (s * target).sum((2, 3))
    assert torch.allclose(iou_loss(pred, target), (1 - inter / ((s + target).sum((2, 3)) - inter)).mean())
    model = SAMAdapter()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("prompt_generator" in n or not n.startswith("net.image_encoder.") for n in names)
    assert 4.0e6 < foundation.trainable(model) < 4.5e6
    recipe = model.author_recipe("IDRiD")
    assert recipe.effective_batch_size == 2 and recipe.weight_decay == 0.01


def test_registry_reads_e5_recipes_without_building(monkeypatch):
    from bench.models import registry
    from bench.models.sam_adapter import SAMAdapter

    def refuse(self):
        raise AssertionError("recipe_and_card must not build the network")

    monkeypatch.setattr(SAMAdapter, "__init__", refuse)
    recipe, card = registry.recipe_and_card("SAM-Adapter", "DDR")
    assert recipe.lr == 2e-4 and card.forward_size == (1024, 1024)
    assert set(registry.E5_MODELS) <= set(registry.WRAPPERS)


def test_e5_jobs_join_the_bstd_table_only():
    from pathlib import Path

    from bench.scripts import launch_b1

    root = Path(launch_b1.__file__).resolve().parents[2]
    bs = launch_b1.load_document(root / "bench/configs/bs_jobs.yaml")
    models = launch_b1.active_models("BS")
    jobs = launch_b1.select_jobs(bs, False, models)
    assert {"SAMed", "SAM-Adapter", "SAM2-UNet", "MedSAM", "RETFound"} <= {job["model"] for job in jobs}
    for job in jobs:
        if job["model"] == "RETFound":
            launch_b1.check_job(job, False)
    assert "RETFound" not in launch_b1.active_models("B1")
    b1 = launch_b1.load_document(root / "bench/configs/b1_jobs.yaml")
    assert all(job["model"] not in {"SAMed", "RETFound"} for job in launch_b1.select_jobs(b1, False, launch_b1.active_models("B1")))


def test_matmul_precision_follows_the_wrapper_and_a_diagnostic_override():
    from bench.runtime import apply_matmul_precision

    class Plain:
        pass

    class Foundation:
        matmul_tf32 = True

    assert apply_matmul_precision(Foundation()) is True and torch.backends.cuda.matmul.allow_tf32
    assert apply_matmul_precision(Foundation(), {"matmul_tf32": False}) is False
    assert apply_matmul_precision(Plain()) is False and not torch.backends.cuda.matmul.allow_tf32
