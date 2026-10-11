"""SAMed (official_code/SAMed, e351840) for four lesion classes under B-std (design/15 section 7.1).

The network is the author's: SAMed's forked SAM ViT-B built for 512 inputs and
num_classes 4, so the mask decoder has five mask tokens and returns five
masks, background first (build_sam.py:65-130; mask_decoder.py:51, 103-108 has
the slicing commented out); LoRA_Sam with rank 4 on q and v of every image
encoder block, image encoder frozen (sam_lora_image_encoder.py:42-108;
train.py:49). The prompt encoder and the mask decoder stay trainable; with no
prompt only no_mask_embed gets a gradient.

Weights come through the author's own loader: the first strict load of
sam_vit_b_01ec64.pth fails at 512, load_from then drops the class-dependent
decoder tensors (mask tokens, hypernetwork MLPs, IoU head; they stay random)
and resizes the position embedding and the global blocks' relative position
tables (build_sam.py:126-160). Those tensors are copied into this model.

Loss: 0.2 x cross-entropy + 0.8 x Dice over softmax including background,
pooled over the batch per class, squared denominators, smoothing 1e-5, mean
over the five classes (trainer.py:22-27; utils.py:60-98). The author computes
it on the 1/4-resolution logits against nearest-downsized labels; B-std
computes it on outputs['masks'], the logits the author's test uses
(utils.py:132), at input resolution, because a 4x smaller label loses
microaneurysms. AdamW 0.005, betas (0.9, 0.999), weight decay 0.1 over the
trainable tensors; linear warmup 250 steps, then (1 - (t - 250) / max_iter)^0.9,
written after each update, stopping at epoch 160 of 200 (trainer.py:50-95;
README.md:81 --warmup --AdamW). B-std stretches that schedule: max_iter is
the B-std step count / 0.8, warmup 250/18600 of it. Batch 24 (12 per GPU, two
GPUs). Inputs in [0, 1] with pixel mean 0 and std 1 (train.py:94-97).
"""

import re

import torch
import torch.nn.functional as F
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.common.labels import exclusive_label
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import official, set_lr, weights

CHECKPOINT = "sam_vit_b_01ec64.pth"
IMAGE_SIZE = 512
RANK = 4
CLASS_INDEX = (1, 2, 3, 4)
AUTHOR_MAX_ITER = 18600  # 200 epochs x 93 iterations of the author's Synapse run
AUTHOR_WARMUP = 250
STOP_FRACTION = 160 / 200
# Tensors load_from leaves random (build_sam.py:131).
RANDOM_KEYS = ("mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head")


def _build(checkpoint=None):
    """sam_model_registry["vit_b"] of SAMed's fork and LoRA_Sam, as train.py:91-98 builds them."""
    with official("SAMed", purge=("segment_anything", "sam_lora_image_encoder")):
        from sam_lora_image_encoder import LoRA_Sam
        from segment_anything import sam_model_registry

        sam, _ = sam_model_registry["vit_b"](
            image_size=IMAGE_SIZE, num_classes=len(LESION_CLASSES), checkpoint=checkpoint,
            pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
        )
        if checkpoint is not None:
            return sam
        return LoRA_Sam(sam, RANK)


def lora_key(name):
    """SAM's blocks.<i>.attn.qkv.<p> sits at blocks.<i>.attn.qkv.qkv.<p> once LoRA wraps it."""
    return re.sub(r"(blocks\.\d+\.attn\.qkv)\.(weight|bias)$", r"\1.qkv.\2", name)


class _CEDice(nn.Module):
    """trainer.py calc_loss with utils.DiceLoss: (1 - w) CE + w Dice, Dice pooled over the batch."""

    def __init__(self, dice_weight=0.8, smooth=1e-5):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        labels = exclusive_label(target, CLASS_INDEX)
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(labels, logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dim=dims)
        dice = 1 - (2 * inter + self.smooth) / ((probs * probs).sum(dim=dims) + (onehot * onehot).sum(dim=dims) + self.smooth)
        return (1 - self.dice_weight) * self.ce(logits, labels) + self.dice_weight * dice.mean()


def _recipe():
    return AuthorRecipe(
        loss="0.2_ce_0.8_dice",
        optimizer="adamw",
        lr=0.005,
        weight_decay=0.1,
        schedule="warmup_poly_stop80",
        iterations=0,
        epochs=160,
        batch_size=24,
        effective_batch_size=24,
        crop_size=(IMAGE_SIZE, IMAGE_SIZE),
        pretrained="SAM ViT-B (sam_vit_b_01ec64.pth), image encoder frozen, LoRA rank 4 on q and v",
        loss_params={"dice_weight": 0.8, "smooth": 1e-5},
        optimizer_params={
            "betas": [0.9, 0.999],
            "power": 0.9,
            "warmup_fraction_of_schedule": AUTHOR_WARMUP / AUTHOR_MAX_ITER,
            "stop_fraction": STOP_FRACTION,
        },
        notes=(
            "Five-class softmax with background (class_index 1-4); loss on the input-resolution masks instead of "
            "the 1/4 logits (deviation); schedule stretched: B-std steps are the first 80% of a warmup-poly schedule "
            "whose warmup is 250/18600 of it; author augmentation replaced by the shared one; two GPUs -> one."
        ),
        source_of_settings=(
            "official_code/SAMed/train.py:20-49 (base_lr 0.005, batch 12 x 2 GPUs, img 512, rank 4, max_epochs 200, "
            "stop_epoch 160, dice_param 0.8), :91-98; trainer.py:22-27, 50-95; utils.py:60-98; README.md:81"
        ),
    )


class SAMed(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="SAMed",
        family="foundation",
        track="B",
        repro_level="R3",
        source="official_code/SAMed",
        output_activation="softmax",
        in_channels=3,
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        pad_multiple=16,
        forward_size=(IMAGE_SIZE, IMAGE_SIZE),
        class_index=CLASS_INDEX,
    )
    matmul_tf32 = True

    def __init__(self):
        super().__init__()
        self.net = _build(None)

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def load_pretrained(self):
        reference = _build(str(weights(CHECKPOINT)))
        own = self.net.sam.state_dict()
        copied = {}
        for name, value in reference.state_dict().items():
            if any(key in name for key in RANDOM_KEYS):
                continue
            target = lora_key(name)
            if target not in own or tuple(own[target].shape) != tuple(value.shape):
                raise RuntimeError(f"{name} of the reference build has no place in the LoRA model")
            copied[target] = value
        left = [k for k in own if k not in copied and "linear_" not in k and not any(r in k for r in RANDOM_KEYS)]
        if left:
            raise RuntimeError(f"tensors not covered by the checkpoint: {left[:5]}")
        own.update(copied)
        self.net.sam.load_state_dict(own)
        return len(copied)

    def enable_activation_checkpointing(self):
        from bench.models.foundation import recompute

        recompute(list(self.net.sam.image_encoder.blocks))
        return "SAM ViT-B blocks recomputed in backward (torch.utils.checkpoint); arithmetic unchanged"

    def forward(self, x):
        return self.net(x, True, IMAGE_SIZE)["masks"]

    def build_loss(self, recipe):
        return _CEDice(recipe.loss_params.get("dice_weight", 0.8), recipe.loss_params.get("smooth", 1e-5))

    def build_optimizer(self, params, recipe):
        trainable = [p for p in params if p.requires_grad]
        betas = tuple(recipe.optimizer_params.get("betas", (0.9, 0.999)))
        optimizer = torch.optim.AdamW(trainable, lr=float(recipe.lr), betas=betas, weight_decay=float(recipe.weight_decay))
        for group in optimizer.param_groups:
            group["initial_lr"] = float(recipe.lr)
        return optimizer

    def schedule(self, total_steps, recipe):
        """(schedule length, warmup) of the stretched author schedule for ``total_steps`` B-std updates."""
        length = float(total_steps) / float(recipe.optimizer_params["stop_fraction"])
        warmup = max(1, int(round(length * float(recipe.optimizer_params["warmup_fraction_of_schedule"]))))
        return length, warmup

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """Rate for update ``step``; the author writes it after the previous update (trainer.py:83-95)."""
        base = float(recipe.lr)
        length, warmup = self.schedule(total_steps, recipe)
        if int(step) == 0:
            lr = base / warmup
        else:
            done = int(step) - 1
            if done < warmup:
                lr = base * (done + 1) / warmup
            else:
                lr = base * (1.0 - (done - warmup) / length) ** float(recipe.optimizer_params["power"])
        set_lr(optimizer, lr)
