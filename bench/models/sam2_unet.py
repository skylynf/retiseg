"""SAM2-UNet (official_code/SAM2-UNet, 01598e5) for four lesion channels under B-std (design/15 section 7.3).

The network is the author's SAM2UNet class: the SAM 2.0 Hiera-L trunk from the
bundled sam2 package, frozen, with a bottleneck-32 adapter before each of its
48 blocks (SAM2UNet.py:51-67, 135-145), RFB modules, a U-shaped decoder and
three outputs (main, and side outputs at strides 16 and 8; SAM2UNet.py:146-170).
Two things change. The model is built on the CPU: build_sam2 defaults to
device="cuda", and the queue launcher builds models only to read their recipe.
The three 64->1 output convolutions become 64->4, one sigmoid per lesion;
their weights are new either way. The Hiera input size is free (the window
position embedding is interpolated), so B-std trains on 960 patches and
predicts on the whole canvas, padded to a multiple of 32, instead of the
author's 352 resize (14 note 27).

Loss: the sum over the three outputs of the author's structure_loss
(train.py:31-39, 58-62). Its BCE term passes reduce='none', a legacy argument
that PyTorch reads as reduction='mean', so the weighted BCE is the plain mean
BCE; this file passes the same argument, so whatever the installed PyTorch
does with it is what runs (test_sam2_unet checks it is the mean). The
weighted IoU is per image and class with smoothing 1, which is the same
mechanism as the U-Net collapse (14 notes 21, 30); loss_params batch_iou
pools it over the batch, for a recipe revision if seed 0 collapses.
Optimizer and schedule: AdamW 1e-3, weight decay 5e-4, CosineAnnealingLR over
20 epochs to 1e-7, stepped per epoch (train.py:23-27, 48-49, 68), stretched
over the B-std steps as 20 equal segments. Batch 12. ImageNet mean and std
(dataset.py:56-64).
"""

import functools
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import IMAGENET_MEAN, IMAGENET_STD, official, stepped_cosine, set_lr, weights

CHECKPOINT = "sam2_hiera_large.pt"
SOURCE = "official_code/SAM2-UNet"


def _build(checkpoint=None, device="cpu"):
    """The repository's SAM2UNet(checkpoint), with build_sam2 bound to ``device``.

    Built inside the import context: the bundled sam2 registers the Hydra
    config module sam2_configs on import, and build_sam2 composes from it.
    """
    with official("SAM2-UNet", purge=("sam2", "sam2_configs", "SAM2UNet")):
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        import SAM2UNet as module
        from sam2.build_sam import build_sam2

        module.build_sam2 = functools.partial(build_sam2, device=device)
        return module.SAM2UNet(checkpoint)


def structure_loss(pred, mask, batch_iou=False):
    """train.py:31-39 for C channels; with batch_iou the IoU sums run over the batch too."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    dims = (0, 2, 3) if batch_iou else (2, 3)
    inter = ((pred * mask) * weit).sum(dim=dims)
    union = ((pred + mask) * weit).sum(dim=dims)
    wiou = 1 - (inter + 1) / (union - inter + 1)
    if batch_iou:
        return wbce.mean() + wiou.mean()
    return (wbce + wiou).mean()


class _DeepSupervision(nn.Module):
    def __init__(self, batch_iou=False):
        super().__init__()
        self.batch_iou = bool(batch_iou)

    def forward(self, outputs, target):
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        target = target.float()
        return sum(structure_loss(out, target, self.batch_iou) for out in outputs)


def _recipe():
    return AuthorRecipe(
        loss="structure_loss",
        optimizer="adamw",
        lr=1e-3,
        weight_decay=5e-4,
        schedule="stepped_cosine",
        iterations=0,
        epochs=20,
        batch_size=12,
        effective_batch_size=12,
        crop_size=(352, 352),
        pretrained="SAM 2.0 Hiera-L (sam2_hiera_large.pt), trunk frozen",
        loss_params={"batch_iou": False},
        optimizer_params={"eta_min": 1e-7, "segments": 20},
        notes=(
            "Sum of structure_loss over the main and two side outputs; BCE with reduce='none' (legacy: mean); "
            "weighted IoU per image and class, smoothing 1. AdamW without warmup, cosine per epoch. "
            "B-std: 960 patches, whole-canvas inference, shared augmentation instead of flips, binary nearest "
            "labels instead of bicubic-resized ones (dataset.py:27), 64->4 output convolutions; up4 stays "
            "unused in the optimizer, as in the author code."
        ),
        source_of_settings=(
            "official_code/SAM2-UNet/train.py:20-27 (epoch 20, lr 0.001, batch 12, weight decay 5e-4); "
            "train.py:31-39 structure_loss; train.py:48-49 AdamW and CosineAnnealingLR(eta_min 1e-7); "
            "train.py:58-62 three outputs; SAM2UNet.py:51-67, 128-173; dataset.py:56-64 ImageNet mean and std; "
            "README.md:34 SAM 2.0 weights"
        ),
    )


class SAM2UNet(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="SAM2-UNet",
        family="foundation",
        track="B",
        repro_level="R3",
        source=SOURCE,
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        pad_multiple=32,
        forward_size=None,
        class_index=(),
    )
    # E5 decision 2026-10-12: TF32 matrix multiplies for the foundation-model training (design/15 section 16).
    matmul_tf32 = True

    def __init__(self):
        super().__init__()
        self.net = _build(None)
        for name in ("side1", "side2", "head"):
            setattr(self.net, name, nn.Conv2d(64, len(self.classes), kernel_size=1))

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def load_pretrained(self):
        """The trunk from the author's own build with the checkpoint (strict SAM 2 load in build_sam2)."""
        reference = _build(str(weights(CHECKPOINT)))
        source = {k: v for k, v in reference.encoder.state_dict().items() if "prompt_learn" not in k}
        own = self.net.encoder.state_dict()
        missing = sorted(k for k in own if "prompt_learn" not in k and k not in source)
        if missing:
            raise RuntimeError(f"Hiera trunk tensors missing from the reference build: {missing[:5]}")
        own.update(source)
        self.net.encoder.load_state_dict(own)
        return len(source)

    def enable_activation_checkpointing(self):
        """Recompute each adapter-wrapped Hiera block in backward (cell key activation_checkpointing)."""
        from bench.models.foundation import recompute

        recompute(list(self.net.encoder.blocks))
        return "Hiera blocks recomputed in backward (torch.utils.checkpoint); arithmetic unchanged"

    def forward(self, x):
        out, out1, out2 = self.net(x)
        if self.training:
            return out, out1, out2
        return out

    def build_loss(self, recipe):
        return _DeepSupervision(batch_iou=bool(recipe.loss_params.get("batch_iou", False)))

    def build_optimizer(self, params, recipe):
        # train.py:48 passes model.parameters(); frozen tensors never get a gradient and AdamW skips them.
        optimizer = torch.optim.AdamW(list(params), lr=float(recipe.lr), weight_decay=float(recipe.weight_decay))
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        lr = stepped_cosine(
            step,
            total_steps,
            float(recipe.lr),
            float(recipe.optimizer_params["eta_min"]),
            int(recipe.optimizer_params["segments"]),
        )
        set_lr(optimizer, lr)
