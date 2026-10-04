"""U-Net baseline trained from scratch.

This is the E0 pipeline model, not a reimplementation of a fundus paper.
Four lesion channels use independent sigmoids. The original U-Net paper
trains with cross-entropy. The 2026-10-04 change to an equally weighted
sigmoid Dice term, a 200-epoch cap and patience 20 was made after the test
mAUPR of the 60-epoch BCE smoke had already been read. That dependence is
recorded on the recipe. Later changes may use only training loss and
validation loss.
"""

import torch
from torch import nn

from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(SegmentationWrapper, nn.Module):
    classes = ("MA", "HE", "EX", "SE")
    card = ModelCard(
        name="U-Net",
        family="general",
        track="B",
        repro_level="R3",
        source="bench/models/unet.py",
        output_activation="sigmoid",
        normalization={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        pad_multiple=16,
        forward_size=None,
        class_index=(),
    )

    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.down1 = _ConvBlock(in_channels, base)
        self.down2 = _ConvBlock(base, base * 2)
        self.down3 = _ConvBlock(base * 2, base * 4)
        self.down4 = _ConvBlock(base * 4, base * 8)
        self.bottleneck = _ConvBlock(base * 8, base * 16)
        self.up4 = _ConvBlock(base * 16 + base * 8, base * 8)
        self.up3 = _ConvBlock(base * 8 + base * 4, base * 4)
        self.up2 = _ConvBlock(base * 4 + base * 2, base * 2)
        self.up1 = _ConvBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, len(self.classes), 1)
        self.pool = nn.MaxPool2d(2)
        self._base = base

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
        # Same recipe on both datasets. Epochs stay 0 so the cell config is the
        # only place the 200-epoch cap is written. The finished 60-epoch BCE
        # smoke stays in runs/B1_unet_idrid_seed0 and is not this recipe.
        return AuthorRecipe(
            loss="bce_dice",
            optimizer="adam",
            lr=1e-4,
            weight_decay=0.0,
            schedule="none",
            iterations=0,
            batch_size=4,
            effective_batch_size=4,
            crop_size=(512, 512),
            pretrained="none",
            epochs=0,
            loss_params={"smooth": 1.0, "early_stop_patience": 20},
            notes=(
                "Epochs come from the cell config and are a cap. "
                "Training stops after 20 epochs with no lower validation loss, and the scored checkpoint is that best loss. "
                "iterations is epochs times ceil(training images / effective batch size). "
                "crop_size is the E0 record and is not applied; B1 uses the field-of-view canvas at fov_diameter."
            ),
            source_of_settings=(
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
            ),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))
        u4 = self.up4(torch.cat([self._upsample(b, d4), d4], dim=1))
        u3 = self.up3(torch.cat([self._upsample(u4, d3), d3], dim=1))
        u2 = self.up2(torch.cat([self._upsample(u3, d2), d2], dim=1))
        u1 = self.up1(torch.cat([self._upsample(u2, d1), d1], dim=1))
        return self.head(u1)

    @staticmethod
    def _upsample(x, reference):
        return nn.functional.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
