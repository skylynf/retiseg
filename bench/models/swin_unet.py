"""Swin-Unet wrapper. The network class is the one in official_code/Swin-Unet.

Synapse is the author's training entry. Epochs, batch size, learning rate and
input size come from train.py. The loss mix, SGD momentum, weight decay and
poly exponent come from trainer.py. Classification defaults in config.py are
not read by that loop. Training loads the Swin-T checkpoint named by the
config. Construction does not, so a forward test can run without the file.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

_ROOT = Path(__file__).resolve().parents[2] / "official_code" / "Swin-Unet"
_PKG = "retiseg_swin_unet_official"
_LOADED = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _official():
    """Import SwinUnet and get_config without putting the repo on sys.path."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    package = _PKG + "_networks"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(_ROOT / "networks")]
    pkg.__package__ = package
    sys.modules[package] = pkg
    _load_module(
        package + ".swin_transformer_unet_skip_expand_decoder_sys",
        _ROOT / "networks" / "swin_transformer_unet_skip_expand_decoder_sys.py",
    )
    vision = _load_module(
        package + ".vision_transformer",
        _ROOT / "networks" / "vision_transformer.py",
    )
    config = _load_module(_PKG + "_config", _ROOT / "config.py")
    _LOADED = (vision.SwinUnet, config.get_config)
    return _LOADED


def _synapse_config():
    """Same yaml the README train command passes, plus train.py's batch size."""
    _, get_config = _official()
    args = argparse.Namespace(
        cfg=str(_ROOT / "configs" / "swin_tiny_patch4_window7_224_lite.yaml"),
        opts=None,
        batch_size=24,
        zip=False,
        cache_mode="part",
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="O1",
        tag=None,
        eval=False,
        throughput=False,
    )
    return get_config(args)


class _AuthorCEDice(nn.Module):
    """0.4 cross-entropy + 0.6 Dice, matching trainer.py:67-69 and utils.py:22-45.

    Dice uses softmax, smooth 1e-5, a squared denominator, and the mean over
    classes, including background. The shared ce_dice is an unweighted sum
    with a linear denominator, so it does not implement this objective.
    """

    def __init__(self, smooth, ce_weight, dice_weight, class_index):
        super().__init__()
        self.smooth = float(smooth)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.class_index = tuple(int(channel) for channel in class_index)

    def forward(self, logits, target):
        from bench.common.labels import exclusive_label

        if target.ndim == 4:
            target = exclusive_label(target, self.class_index)
        if target.ndim != logits.ndim - 1 or target.dtype.is_floating_point:
            raise ValueError(
                "Swin-Unet ce_dice expects integer class indices (N, H, W), as in trainer.py:67. "
                f"Got shape {tuple(target.shape)} and dtype {target.dtype}."
            )
        labels = target.long()
        ce = nn.functional.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(labels, logits.shape[1]).permute(0, 3, 1, 2).float()
        dice = probs.new_zeros(())
        for index in range(logits.shape[1]):
            score = probs[:, index]
            truth = one_hot[:, index]
            intersect = torch.sum(score * truth)
            y_sum = torch.sum(truth * truth)
            z_sum = torch.sum(score * score)
            dice = dice + (1 - (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth))
        dice = dice / logits.shape[1]
        return self.ce_weight * ce + self.dice_weight * dice


class SwinUnet(SegmentationWrapper, nn.Module):
    classes = ("MA", "HE", "EX", "SE")
    card = ModelCard(
        name="Swin-Unet",
        family="general",
        track="B",
        repro_level="R3",
        source="official_code/Swin-Unet",
        output_activation="softmax",
        in_channels=3,
        # dataset_synapse.py:44 casts the array to float32 and does not subtract
        # a mean. datasets/README.md:4 says those arrays are already in [0, 1].
        # The runner divides a JPEG by 255 first, so mean 0 and std 1 keep that range.
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        # Patch size 4 and three merges. 224 is divisible by 32, so a 224 canvas
        # is not padded. PatchEmbed still requires the spatial size to be exactly 224.
        pad_multiple=32,
        forward_size=(224, 224),
        class_index=(1, 2, 3, 4),
    )

    def __init__(self):
        super().__init__()
        official, _get_config = _official()
        # train.py:97. num_classes is 5 for background plus MA, HE, EX, SE.
        # The official class reads the spatial size from config.DATA.IMG_SIZE (224).
        # load_from is not called: the checkpoint named below is not downloaded.
        self.net = official(_synapse_config(), img_size=224, num_classes=5)

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
        return AuthorRecipe(
            loss="ce_dice",
            optimizer="sgd",
            lr=0.01,
            weight_decay=0.0001,
            schedule="poly",
            iterations=0,
            epochs=150,
            batch_size=24,
            effective_batch_size=24,
            crop_size=(224, 224),
            pretrained="swin_tiny_patch4_window7_224.pth",
            loss_params={"smooth": 1e-5, "ce_weight": 0.4, "dice_weight": 0.6},
            optimizer_params={"momentum": 0.9, "power": 0.9},
            notes=(
                "B1 first scales the field-of-view crop so its longer side is 1440, then resizes that canvas to 224. "
                "README.md:24 and train.sh:29 pass base_lr 0.05; this recipe uses the train.py:30 default 0.01. "
                "config.py:82-109 sets AdamW, learning rate 5e-4, weight decay 0.05 and 300 epochs, and trainer_synapse does not read them. "
                "Paper section 4.2 states batch size 24, SGD momentum 0.9 and weight decay 1e-4, and does not state the learning rate or the epoch count. "
                "Four-plane targets become class indices with background 0 and channels 1-4 for MA, HE, EX, SE. "
                "Overlaps follow M2MRF order EX, HE, SE, MA, so MA is kept."
            ),
            source_of_settings=(
                "loss ce_dice = 0.4*CE+0.6*Dice official_code/Swin-Unet/trainer.py:69 "
                "(CE trainer.py:48, Dice trainer.py:49, softmax trainer.py:68); "
                "dice smooth 1e-5 and squared denominator official_code/Swin-Unet/utils.py:24 and utils.py:27; "
                "optimizer sgd momentum 0.9 weight_decay 0.0001 official_code/Swin-Unet/trainer.py:50; "
                "lr 0.01 official_code/Swin-Unet/train.py:30; "
                "poly power 0.9 official_code/Swin-Unet/trainer.py:73; "
                "epochs 150 official_code/Swin-Unet/train.py:24; "
                "iterations 0 because trainer.py:54 sets max_iterations = max_epochs * len(train_loader) "
                "(train.py:22 default 30000 is not read); "
                "batch_size 24 official_code/Swin-Unet/train.py:26; "
                "effective batch 24 = batch_size * n_gpu with n_gpu 1 at train.py:27 and trainer.py:26; "
                "crop 224 official_code/Swin-Unet/train.py:33; "
                "pretrained filename swin_tiny_patch4_window7_224.pth "
                "official_code/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml:5 and config.py:50 "
                "(README.md:5 places that Swin-T file in pretrained_ckpt/); "
                "normalization is [0, 1] with no mean subtraction "
                "official_code/Swin-Unet/datasets/dataset_synapse.py:44 and datasets/README.md:4."
            ),
        )

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """Match trainer.py:73-77, which writes the learning rate after optimizer.step."""
        from bench.runtime import adjust_lr_after_step

        return adjust_lr_after_step(optimizer, step, total_steps, recipe)

    def build_loss(self, recipe):
        if str(recipe.loss).lower() != "ce_dice":
            return super().build_loss(recipe)
        return _AuthorCEDice(
            smooth=recipe.loss_params["smooth"],
            ce_weight=recipe.loss_params["ce_weight"],
            dice_weight=recipe.loss_params["dice_weight"],
            class_index=self.card.class_index,
        )

    def load_pretrained(self):
        from bench.pretrained import swin_tiny

        path = swin_tiny()
        before = {key: value.detach().clone() for key, value in self.net.state_dict().items()}
        self.net.config.defrost()
        self.net.config.MODEL.PRETRAIN_CKPT = str(path)
        self.net.config.freeze()
        self.net.load_from(self.net.config)
        copied = sum(
            1
            for key, value in self.net.state_dict().items()
            if key in before and not torch.equal(before[key], value)
        )
        if copied < 50:
            raise RuntimeError(f"{path} changed {copied} Swin-Unet tensors; refusing to train")
        return copied

    def forward(self, x):
        return self.net(x)
