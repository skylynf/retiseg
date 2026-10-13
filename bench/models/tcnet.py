"""B-std wrapper for TC-Net, the Track C re-implementation in reproductions/Zhang2023_TC-Net (docs/paper/18).

Zhang et al., Computers in Biology and Medicine 2023; no author code. The
network (ResNeSt-block CNN branch + Transformer branch with MHSSA, cross-
attention decoder, LLCS fusion) and the dynamic cyclical focal loss (DCFL,
equations 9-13) live in that directory with their locked assumptions.

Geometry: the paper resizes whole images to 512 x 512 and the network needs
sides divisible by 64, so B-std treats it like Swin-Unet and FCT: 512 patches
on the 1,440 canvas (forward_size) and sliding 512 windows with overlap 0.5
at inference; pad_multiple 64.

Head: five-class softmax, classes background, EX, HE, MA, SE -> class_index
(3, 2, 1, 4). Loss: DCFL, whose cyclical indicator depends on the epoch out
of the paper's 1,000; under B-std the runner's progress (update / total
updates) replaces epoch / 1000 (18 section 2.5): the runner hands every
update to adjust_lr, which stores the progress the loss reads. Class
frequencies beta come from the current batch (assumption A17).

Recipe (section 4.2): Adam 5e-4, weight decay 1e-5, batch 4, constant rate;
no pretraining is described, the reproduction initialises from scratch.
"""

import sys
from pathlib import Path

import torch
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.common.labels import exclusive_label
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import set_lr

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "reproductions" / "Zhang2023_TC-Net"
INPUT = 512
# tcnet.assumptions.FUNDUS_CLASSES = (background, EX, HE, MA, SE): channel of MA, HE, EX, SE
CLASS_INDEX = (3, 2, 1, 4)


def _import():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"{SOURCE} is missing: the TC-Net reproduction is not in this checkout")
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import tcnet
    from tcnet import assumptions, loss

    return tcnet, assumptions, loss


def _recipe():
    _tcnet, assumptions, _loss = _import()
    return AuthorRecipe(
        loss="ce",   # name for the runner's bookkeeping; build_loss returns DCFL
        optimizer=str(assumptions.OPTIMIZER).lower(),
        lr=float(assumptions.LR),
        weight_decay=float(assumptions.WEIGHT_DECAY),
        schedule="none",
        iterations=0,
        epochs=int(assumptions.EPOCHS),
        batch_size=int(assumptions.BATCH_SIZE),
        effective_batch_size=int(assumptions.BATCH_SIZE),
        crop_size=(INPUT, INPUT),
        pretrained="none",
        loss_params={"cyclical_factor": float(assumptions.CYCLICAL_FACTOR), "gamma_high": float(assumptions.GAMMA_HIGH),
                     "gamma_low": float(assumptions.GAMMA_LOW), "beta_log_offset": float(assumptions.BETA_LOG_OFFSET)},
        optimizer_params={"betas": list(assumptions.ADAM_BETAS)},
        notes=(
            "Track C: re-implemented from the paper (reproductions/Zhang2023_TC-Net, assumptions in tcnet/assumptions.py "
            "and DESIGN.md). B-std: 512 patches and sliding 512 windows instead of resizing the whole image to 512; the "
            "DCFL cycle runs over the B-std updates (progress = update / total updates) instead of the paper's 1,000 epochs; "
            "the shared augmentation replaces the paper's flips, colour and affine transforms."
        ),
        source_of_settings="reproductions/Zhang2023_TC-Net/tcnet/assumptions.py (paper sections 3.4 and 4.2)",
    )


class _DCFL(nn.Module):
    """tcnet.loss.DynamicCyclicalFocalLoss driven by the runner's progress instead of an epoch counter."""

    def __init__(self, wrapper, loss_module):
        super().__init__()
        self.wrapper = wrapper
        self.loss = loss_module
        self.epochs = int(self.loss.total_epochs)

    def forward(self, logits, target):
        labels = exclusive_label(target, CLASS_INDEX)
        # progress in [0, 1] -> 1-indexed epoch of the paper's cycle, inclusive of the last
        epoch = min(self.epochs, max(1, int(self.wrapper._progress * self.epochs) + 1))
        return self.loss(logits, labels, epoch=epoch)


class TCNet(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="TC-Net",
        family="relational_transformer",
        track="C",
        repro_level="R0",
        source="reproductions/Zhang2023_TC-Net",
        output_activation="softmax",
        in_channels=3,
        normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        pad_multiple=64,
        forward_size=(INPUT, INPUT),
        class_index=CLASS_INDEX,
    )

    def __init__(self):
        super().__init__()
        tcnet, assumptions, loss = _import()
        self.net = tcnet.TCNet(num_classes=int(assumptions.NUM_CLASSES))
        self._loss_module = loss.DynamicCyclicalFocalLoss()
        self._progress = 0.0

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def forward(self, x):
        return self.net(x)

    def build_loss(self, recipe):
        return _DCFL(self, self._loss_module)

    def build_optimizer(self, params, recipe):
        trainable = [p for p in params if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=float(recipe.lr), betas=tuple(recipe.optimizer_params["betas"]),
                                     weight_decay=float(recipe.weight_decay))
        for group in optimizer.param_groups:
            group["initial_lr"] = float(recipe.lr)
        return optimizer

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """Constant rate; the call also records the progress the DCFL cycle reads."""
        self._progress = min(1.0, float(step) / float(max(1, total_steps)))
        set_lr(optimizer, float(recipe.lr))
