"""FCT wrapper. Official PyTorch port, plus this project's adaptation.

The network class is taken from ``official_code/FCT/PyTorch/main.py``.
That file is the official PyTorch port, not a TensorFlow reimplementation and
not a network written for this benchmark. The adaptation is local: the ACDC
stem is widened from one channel to three, the four head channels are
reassigned in order to MA, HE, EX, SE, and the field-of-view canvas is resized
to 384. Loss and step count are the adapted B1 recipe recorded in
``author_recipe``, because ``PyTorch/utils.py`` is not in the repository.

``official_code/FCT/PyTorch/main.py`` cannot be imported. It changes directory,
imports a ``utils`` module that is not in the repository, and then trains.
The classes are executed from that file. Nothing in ``forward`` crops the image
or changes its color.

The official head is already four channels with a sigmoid, written for ACDC
(background, RV, MYO, LV). Those four channels are reassigned, in order, to
MA, HE, EX, SE. The wrapper returns the full-resolution head before that
sigmoid. The runner applies sigmoid once, which matches ``BCELoss`` on the
official probabilities. The ACDC stem takes one channel. The three scale-image
convolutions and the first block are widened to three channels so a fundus
image can enter. Filter counts and attention heads stay as written in the
PyTorch file. The attention reshape requires a square map, so the card
resizes the field-of-view canvas to 384, the larger size in the paper.
"""

import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

_PYTORCH_MAIN = Path(__file__).resolve().parents[2] / "official_code" / "FCT" / "PyTorch" / "main.py"


def _load_official():
    text = _PYTORCH_MAIN.read_text()
    start = text.index("class Attention")
    end = text.index("\ndevice = ")
    namespace = {"np": np, "torch": torch, "nn": nn, "F": F}
    exec(text[start:end], namespace)
    return namespace["FCT"], namespace["init_weights"]


_OfficialFCT, _init_weights = _load_official()


def _full_resolution_logits(module, x):
    """Official DS_out.forward without the sigmoid at PyTorch/main.py:271."""
    x1 = module.upsample(x)
    x1 = x1.permute(0, 2, 3, 1)
    x1 = module.layernorm(x1)
    x1 = x1.permute(0, 3, 1, 2)
    x1 = F.relu(module.conv1(x1))
    x1 = F.relu(module.conv2(x1))
    return module.conv3(x1)


class FCT(SegmentationWrapper, nn.Module):
    classes = ("MA", "HE", "EX", "SE")
    card = ModelCard(
        name="FCT",
        family="general",
        track="B",
        repro_level="R3",
        source="official_code/FCT/PyTorch/main.py",
        output_activation="sigmoid",
        in_channels=3,
        # PyTorch/main.py turns the array into a tensor and does not subtract a
        # mean. get_acdc returns resized intensities with no ImageNet stats.
        # The runner has already divided the JPEG by 255, so mean 0 and std 1
        # leave the image in [0, 1]. The ModelCard default is ImageNet and is
        # not this recipe.
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        pad_multiple=32,
        forward_size=(384, 384),
        class_index=(),
    )

    def __init__(self):
        super().__init__()
        self.net = _OfficialFCT()
        self._widen_stem_to_rgb()
        for head in (self.net.ds7, self.net.ds8, self.net.ds9):
            head.forward = types.MethodType(_full_resolution_logits, head)
        self.net.apply(_init_weights)

    def _widen_stem_to_rgb(self):
        first = self.net.block_1
        first.layernorm = nn.LayerNorm(3, eps=1e-5)
        first.conv1 = nn.Conv2d(3, first.conv1.out_channels, 3, 1, padding="same")
        for name in ("block_2", "block_3", "block_4"):
            block = getattr(self.net, name)
            out_channels = block.conv1.out_channels
            block.conv1 = nn.Conv2d(3, out_channels, 3, 1, padding="same")

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
        return AuthorRecipe(
            loss="bce_with_logits",
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            schedule="none",
            iterations=0,
            epochs=300,
            batch_size=4,
            effective_batch_size=4,
            crop_size=(384, 384),
            pretrained="none",
            notes=(
                "IDRiD and DDR share this recipe. "
                "The network is the official PyTorch port in official_code/FCT/PyTorch/main.py, not the TensorFlow entry. "
                "This project's adaptation widens the stem from 1 channel to 3, reassigns the four head channels "
                "to MA, HE, EX, SE, and sets forward_size 384. "
                "The wrapped entry is official_code/FCT/PyTorch/main.py. "
                "Its names learning_rate, epochs and batch_size have no values because utils.py is not in the repository. "
                "lr 1e-3 is the number in TensorFlow/main.py:68 and in Tragakis et al., WACV 2023, Implementation Details. "
                "epochs 300 is the paper's 50 warmup epochs plus the following 250 epochs. "
                "TensorFlow/main.py:41-42 uses warmup_run_epochs 120 and normal_run_epochs 30, and those counts are not used. "
                "batch_size 4 is the paper's batch for Synapse, Spleen and ISIC 2017. "
                "ACDC's batch of 10 and TensorFlow's batch of 2 are not used. "
                "weight_decay 0 is Adam's default: PyTorch/main.py:378 and TensorFlow/main.py:69 both omit a decay, and the paper states none. "
                "schedule none follows PyTorch/main.py:417-420, which never changes the learning rate. "
                "The paper's validation-loss plateau and the TensorFlow warmup plus ReduceLROnPlateau are not applied. "
                "loss bce_with_logits matches PyTorch/main.py:377 BCELoss on pred[2] after the sigmoid at line 271; this wrapper returns that map before the sigmoid. "
                "The paper's equal sum of cross-entropy and Dice, and TensorFlow's three-map weights 0.14, 0.29 and 0.57, are not used. "
                "forward_size 384 is the larger of the two published resizes, 224 and 384. "
                "384 is square and divisible by 32, which the attention reshape and the five stride-2 pools require. "
                "224 is not used. pretrained is none: PyTorch/main.py:358-371 draws Kaiming normal weights. "
                "Normalization is mean 0 and std 1. PyTorch/main.py:17 converts the array with torch.Tensor and never subtracts a mean; "
                "TensorFlow/utils.py get_acdc resizes intensities and does not apply ImageNet statistics."
            ),
            source_of_settings=(
                "optimizer Adam PyTorch/main.py:378 and TensorFlow/main.py:69; "
                "lr 1e-3 TensorFlow/main.py:68 and Tragakis et al., WACV 2023, Implementation Details; "
                "epochs 50 warmup then 250, same section; "
                "batch 4 for Synapse, Spleen and ISIC 2017, same section; "
                "weight_decay omitted at PyTorch/main.py:378; "
                "schedule unchanged at PyTorch/main.py:417-420; "
                "loss BCELoss on pred[2] at PyTorch/main.py:377 and :391; "
                "resize 384 Tragakis et al., WACV 2023, Implementation Details; "
                "Kaiming normal PyTorch/main.py:358-371."
            ),
        )

    def forward(self, x):
        if x.shape[-2] != x.shape[-1]:
            raise ValueError(
                f"FCT attention reshapes each map to a square (PyTorch/main.py:113); got {tuple(x.shape[-2:])}"
            )
        # Official training uses pred[2], the full-resolution map (main.py:391).
        _coarse, _mid, full = self.net(x)
        return full
