"""MedSAM (official_code/MedSAM, d71e8a1) fully fine-tuned without prompts (design/15 section 8.1).

MedSAM's builder needs a checkpoint path (segment_anything/build_sam.py
takes Path(checkpoint) before its "is not None" test), so the same ViT-B SAM
is assembled here from MedSAM's own modeling classes with _build_sam's
arguments (build_sam.py:60-100), and medsam_vit_b.pth is loaded strictly, as
sam.load_state_dict does there. As in train_one_gpu.py's MedSAM class
(:189-235) the image encoder and the mask decoder train and the prompt
encoder is frozen and run under no_grad; the optimizer holds only the first
two (:274-279). The main table is fully automatic, so no box is given: the
prompt encoder's empty sparse prompt and no_mask_embed, as SAMed and
SAM-Adapter use; the box prompt belongs to the oracle-prompt setting (b).
All four mask tokens are kept, one sigmoid per lesion, upsampled bilinearly
from 256 to the input size (:228-233).

Loss: monai DiceLoss(sigmoid=True, squared_pred=True, reduction="mean") +
BCEWithLogits (:284-286), here written out: per image and channel, numerator
and denominator smoothing 1e-5, mean over images and channels; loss_params
batch_dice pools it over the batch, for a recipe revision if seed 0
collapses (14 note 30). AdamW 1e-4, weight decay 0.01, constant rate, batch
2 (:166-174, 277-279). The code's 1000 epochs and the paper's 150 are both
replaced by the B-std budget. Inputs divided by 255 without SAM's mean and
std (utils/pre_grey_rgb.py:68-69; utils/README.md:98 for fundus images).
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import official, weights

CHECKPOINT = "medsam_vit_b.pth"


def _build():
    """build_sam.py _build_sam for vit_b (embed 768, depth 12, heads 12, global 2, 5, 8, 11), no checkpoint."""
    with official("MedSAM", purge=("segment_anything",)):
        from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

        embed = 256
        image_size = 1024
        patch = 16
        return Sam(
            image_encoder=ImageEncoderViT(
                depth=12,
                embed_dim=768,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=patch,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                out_chans=embed,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=embed,
                image_embedding_size=(image_size // patch, image_size // patch),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(depth=2, embedding_dim=embed, mlp_dim=2048, num_heads=8),
                transformer_dim=embed,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )


def dice_loss(logits, target, smooth=1e-5, batch_dice=False):
    """monai DiceLoss(sigmoid=True, squared_pred=True), smooth_nr = smooth_dr = 1e-5."""
    pred = torch.sigmoid(logits)
    dims = (0, 2, 3) if batch_dice else (2, 3)
    inter = (pred * target).sum(dim=dims)
    denom = (pred * pred).sum(dim=dims) + (target * target).sum(dim=dims)
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


class _DiceBCE(nn.Module):
    def __init__(self, batch_dice=False):
        super().__init__()
        self.batch_dice = bool(batch_dice)

    def forward(self, logits, target):
        target = target.float()
        return dice_loss(logits, target, batch_dice=self.batch_dice) + F.binary_cross_entropy_with_logits(logits, target)


def _recipe():
    return AuthorRecipe(
        loss="dice_plus_bce",
        optimizer="adamw",
        lr=1e-4,
        weight_decay=0.01,
        schedule="none",
        iterations=0,
        epochs=1000,
        batch_size=2,
        effective_batch_size=2,
        crop_size=(1024, 1024),
        pretrained="MedSAM ViT-B (medsam_vit_b.pth)",
        loss_params={"batch_dice": False},
        optimizer_params={},
        notes=(
            "No prompt (empty sparse prompt, no_mask_embed); four mask tokens as four sigmoids; the box prompt is the "
            "oracle setting. Batch 2 of the single-GPU script, not the paper's 160; inputs / 255 in training and "
            "inference. No augmentation in the author code; B-std shared augmentation."
        ),
        source_of_settings=(
            "official_code/MedSAM/train_one_gpu.py:160-174 (epochs 1000, batch 2, weight decay 0.01, lr 1e-4), "
            ":189-235 MedSAM class, :274-286 optimizer and losses; segment_anything/build_sam.py:60-100"
        ),
    )


class MedSAM(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="MedSAM",
        family="foundation",
        track="B",
        repro_level="R3",
        source="official_code/MedSAM",
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        pad_multiple=16,
        forward_size=(1024, 1024),
        class_index=(),
        window_batch=2,
    )
    matmul_tf32 = True

    def __init__(self):
        super().__init__()
        self.sam = _build()
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def load_pretrained(self):
        state = torch.load(weights(CHECKPOINT), map_location="cpu", weights_only=False)
        self.sam.load_state_dict(state)
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        return len(state)

    def enable_activation_checkpointing(self):
        from bench.models.foundation import recompute

        recompute(list(self.sam.image_encoder.blocks))
        return "SAM ViT-B blocks recomputed in backward (torch.utils.checkpoint); arithmetic unchanged"

    def forward(self, x):
        embedding = self.sam.image_encoder(x)
        with torch.no_grad():
            sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        low_res, _ = self.sam.mask_decoder.predict_masks(
            image_embeddings=embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
        )
        return F.interpolate(low_res, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def build_loss(self, recipe):
        return _DiceBCE(batch_dice=bool(recipe.loss_params.get("batch_dice", False)))

    def build_optimizer(self, params, recipe):
        own = list(self.sam.image_encoder.parameters()) + list(self.sam.mask_decoder.parameters())
        optimizer = torch.optim.AdamW(own, lr=float(recipe.lr), weight_decay=float(recipe.weight_decay))
        for group in optimizer.param_groups:
            group["initial_lr"] = float(recipe.lr)
        return optimizer

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        for group in optimizer.param_groups:
            group["lr"] = float(recipe.lr)
