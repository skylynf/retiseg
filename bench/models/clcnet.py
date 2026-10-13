"""B-std wrapper for CLC-Net, the Track C re-implementation in reproductions/Wang2023_CLC-Net (docs/paper/18).

Wang et al., Neurocomputing 2023; no author code. The network, losses and
locked assumptions (A1-A22) live in that directory; this file only adapts
them to the runner's contract.

Geometry: the local branch segments a 256 window, the contextual branch sees
the 512 window centred on it (resized to 256). The runner therefore feeds 512
patches (forward_size) and this network returns logits for the centre 256
(card.output_margin 128): the trainer scores the centre of the target, the
pixel budget counts the supervised centre, and inference pads the canvas by
128 and slides 512 windows whose 256 centres tile it (bench.infer).

Head: five-class softmax, classes background, HE, MA, EX, SE (A9) ->
class_index (2, 1, 3, 4). Loss (section 3.1, A10): Dice summed over the lesion
classes (batch pooled, eps 1) + weighted cross-entropy (1, 2, 2, 2, 2), on the
local output; the same on the contextual output against the 2x2 max-pooled
512 label (A6); plus binary cross-entropy of the two classification heads on
the lesion types present in each window (A8). The loss module reads the
auxiliary outputs the forward caches (``_aux``) and needs the full 512 target
(``loss_takes_full_target``), from which it derives every term.

Recipe (section 4.1, A11): Adam 3e-4, x0.1 at epochs 20 and 50 of 60, batch
16, no weight decay, no pretraining (A1). B-std stretches the two milestones
to 20/60 and 50/60 of its own step count. Pixels in [0, 1] without mean and
std (A14).
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
SOURCE = REPO / "reproductions" / "Wang2023_CLC-Net"
CONTEXT, LOCAL = 512, 256
# clcnet.assumptions.SEGMENTATION_CLASSES = (background, HE, MA, EX, SE): channel of MA, HE, EX, SE
CLASS_INDEX = (2, 1, 3, 4)
# clcnet.assumptions.CLASSIFICATION_CLASSES = (MA, EX, HE, SE): where each LESION_CLASSES plane goes
CLS_ORDER = ("MA", "EX", "HE", "SE")


def _import():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"{SOURCE} is missing: the CLC-Net reproduction is not in this checkout")
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from clcnet import assumptions, losses, model  # noqa: F401

    return assumptions, losses, model


def _recipe():
    assumptions, _losses, _model = _import()
    milestones = tuple(int(m) for m in assumptions.LR_MILESTONES)
    return AuthorRecipe(
        loss="dice",   # placeholder name for the runner's bookkeeping; build_loss returns the paper's combination
        optimizer=str(assumptions.OPTIMIZER).lower(),
        lr=float(assumptions.LR_INITIAL),
        weight_decay=float(assumptions.ADAM_WEIGHT_DECAY),
        schedule="none",   # adjust_lr below applies the two x0.1 milestones
        iterations=0,
        epochs=int(assumptions.EPOCHS),
        batch_size=int(assumptions.BATCH_SIZE),
        effective_batch_size=int(assumptions.BATCH_SIZE),
        crop_size=(LOCAL, LOCAL),
        pretrained="none (assumption A1)",
        loss_params={"smooth": float(assumptions.DICE_EPS), "class_weights": list(assumptions.SEGMENTATION_WEIGHTS)},
        optimizer_params={"betas": list(assumptions.ADAM_BETAS), "eps": float(assumptions.ADAM_EPS),
                          "milestone_fractions": [m / float(assumptions.EPOCHS) for m in milestones],
                          "gamma": float(assumptions.LR_GAMMA)},
        notes=(
            "Track C: re-implemented from the paper (reproductions/Wang2023_CLC-Net, assumptions A1-A22 in "
            "clcnet/assumptions.py and DESIGN.md). B-std: 512 patches whose 256 centre is supervised "
            "(card.output_margin 128); the two learning-rate milestones at epochs 20 and 50 of 60 become 1/3 and 5/6 "
            "of the B-std step count; the shared augmentation replaces A12; the budget counts supervised pixels."
        ),
        source_of_settings="reproductions/Wang2023_CLC-Net/clcnet/assumptions.py (paper section 4.1 and figure 2)",
    )


class _PaperLoss(nn.Module):
    """Local segmentation + contextual segmentation + two classification heads (clcnet.losses.total_loss)."""

    def __init__(self, wrapper, losses):
        super().__init__()
        self.wrapper = wrapper
        self.losses = losses

    def forward(self, logits, target):
        aux = self.wrapper._aux
        if aux is None or aux["local_logits"] is not logits:
            # stitched canvas logits (validation scorer): the local segmentation terms only, on the whole label
            labels = exclusive_label(target, CLASS_INDEX)
            return self.losses.segmentation_loss(logits, labels)["segmentation"]
        margin = self.wrapper.card.output_margin
        full = exclusive_label(target, CLASS_INDEX)           # (N, 512, 512) in the paper's class ids
        local = full[:, margin:-margin, margin:-margin]
        planes = {name: index for index, name in enumerate(LESION_CLASSES)}
        present_local = torch.stack([(target[:, planes[c], margin:-margin, margin:-margin] > 0.5).flatten(1).any(1) for c in CLS_ORDER], 1)
        present_context = torch.stack([(target[:, planes[c]] > 0.5).flatten(1).any(1) for c in CLS_ORDER], 1)
        output = {"local_logits": logits, "local_cls_logits": aux["local_cls_logits"],
                  "context_logits": aux["context_logits"], "context_cls_logits": aux["context_cls_logits"]}
        batch = {"local_mask": local, "context_mask": full, "local_cls": present_local.float(), "context_cls": present_context.float()}
        return self.losses.total_loss(output, batch)["total"]


class CLCNet(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="CLC-Net",
        family="fundus_multiscale",
        track="C",
        repro_level="R0",
        source="reproductions/Wang2023_CLC-Net",
        output_activation="softmax",
        in_channels=3,
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},   # A14: pixels / 255 only
        pad_multiple=32,
        forward_size=(CONTEXT, CONTEXT),
        class_index=CLASS_INDEX,
        output_margin=(CONTEXT - LOCAL) // 2,
    )
    loss_takes_full_target = True   # the contextual branch and the classification heads need the whole 512 label

    def __init__(self):
        super().__init__()
        assumptions, losses, model = _import()
        self._losses = losses
        self.net = model.CLCNet(assumptions.VARIANTS["full"], num_classes=len(assumptions.SEGMENTATION_CLASSES))
        self._aux = None

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def forward(self, x):
        if x.shape[-2:] != (CONTEXT, CONTEXT):
            raise ValueError(f"CLC-Net takes {CONTEXT}x{CONTEXT} windows (forward_size), got {tuple(x.shape[-2:])}")
        m = self.card.output_margin
        local = x[:, :, m:-m, m:-m]
        output = self.net(local, x)
        self._aux = output
        return output["local_logits"]

    def build_loss(self, recipe):
        return _PaperLoss(self, self._losses)

    def build_optimizer(self, params, recipe):
        trainable = [p for p in params if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=float(recipe.lr), betas=tuple(recipe.optimizer_params["betas"]),
                                     eps=float(recipe.optimizer_params["eps"]), weight_decay=float(recipe.weight_decay))
        for group in optimizer.param_groups:
            group["initial_lr"] = float(recipe.lr)
        return optimizer

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """A11 stretched to B-std: x gamma at each milestone fraction of the total steps."""
        progress = float(step) / float(max(1, total_steps))
        lr = float(recipe.lr)
        for fraction in recipe.optimizer_params["milestone_fractions"]:
            if progress >= float(fraction):
                lr *= float(recipe.optimizer_params["gamma"])
        set_lr(optimizer, lr)
