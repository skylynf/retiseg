"""ViT-L encoder with a UPerNet head: RETFound in the main table and the encoder comparison (design/15 section 8).

RETFound's repository (official_code/RETFound, ae9a9ec) fine-tunes for
classification only, so the decoder is this project's, fixed for every
encoder: the outputs of blocks 8, 12, 16 and 24 become a four-level pyramid
the way BEiT and MAE segment (two 2x2 transposed convolutions with BatchNorm
and GELU between them, one transposed convolution, identity, 2x2 max
pooling), then an mmseg-style UPerHead (512 channels, pyramid pooling 1, 2,
3, 6, FPN, dropout 0.1) and an auxiliary FCN head on the stride-16 level
(256 channels, loss weight 0.4). Four sigmoid outputs; BCE on both heads.

Kept from RETFound's fine-tuning (main_finetune.py, train.sh): input 224, drop
path 0.2, AdamW with weight decay 0.05 and no decay for 1-D tensors and
pos_embed/cls_token, layer-wise decay 0.65 over the 24 blocks with the head at
scale 1 (util/lr_decay.py), learning rate 5e-3 * 24 / 256 = 4.6875e-4 at
batch 24, warmup over the first 20% (10 of 50 epochs) from 0, then a half
cosine to 1e-6 (util/lr_sched.py). fp16 is not used (B-std numerics; 14 note
31). Encoders keep their own normalization.

Encoders (the class attribute ENCODER):
  retfound_mae    RETFound_mae_natureCFP, ViT-L/16, RETFound's models_vit.RETFound_mae (the main table)
  mae_in1k        mae_pretrain_vit_large.pth, the same ViT-L/16 (RETFound-MAE's starting point)
  retfound_dinov2 RETFound_dinov2_meh "teacher", timm vit_large_patch14_dinov2.lvd142m (models_vit.py:79-86)
  dinov2          timm vit_large_patch14_dinov2.lvd142m (LVD-142M)
  augreg          timm vit_large_patch16_224.augreg_in21k_ft_in1k (supervised ImageNet)
RETFound's loader drops keys it cannot match without a word
(main_finetune.py:223); here every tensor of the patch embedding, the
position embedding and the 24 blocks must come from the checkpoint, or the
run refuses to start, and the missing and unexpected key counts are returned.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import IMAGENET_MEAN, IMAGENET_STD, official, set_lr, weights

OUT_INDICES = (7, 11, 15, 23)
CHANNELS = 512
AUX_CHANNELS = 256
AUX_WEIGHT = 0.4
POOL_SCALES = (1, 2, 3, 6)
DROP_PATH = 0.2
LAYER_DECAY = 0.65
BATCH = 24
BLR = 5e-3
TIMM_ARCH = {
    "retfound_dinov2": "vit_large_patch14_dinov2.lvd142m",
    "dinov2": "vit_large_patch14_dinov2.lvd142m",
    "augreg": "vit_large_patch16_224.augreg_in21k_ft_in1k",
}
CHECKPOINTS = {
    "retfound_mae": ("RETFound_mae_natureCFP.pth", "model"),
    "mae_in1k": ("mae_pretrain_vit_large.pth", "model"),
    "retfound_dinov2": ("RETFound_dinov2_meh.pth", "teacher"),
}


def build_encoder(encoder):
    """The ViT-L of ``encoder`` without weights, head and final norm."""
    if encoder in ("retfound_mae", "mae_in1k"):
        with official("RETFound", purge=("models_vit", "util")):
            import models_vit

            vit = models_vit.RETFound_mae(img_size=224, num_classes=0, drop_path_rate=DROP_PATH, global_pool=True)
    elif encoder in TIMM_ARCH:
        import timm

        vit = timm.create_model(TIMM_ARCH[encoder], pretrained=False, img_size=224, num_classes=0, drop_path_rate=DROP_PATH)
    else:
        raise ValueError(f"unknown encoder {encoder!r}")
    for name in ("head", "fc_norm", "norm", "head_drop"):
        if hasattr(vit, name):
            setattr(vit, name, nn.Identity())
    return vit


def _clean_keys(state):
    """RETFound's key hygiene (main_finetune.py:208-210)."""
    state = {k.replace("backbone.", ""): v for k, v in state.items()}
    state = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in state.items()}
    return {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in state.items()}


def interpolate_pos_embed(vit, state):
    with official("RETFound", purge=("util",)):
        from util.pos_embed import interpolate_pos_embed as author

        author(vit, state)


def load_encoder(vit, encoder):
    """Copy pretrained tensors into ``vit``; every patch, position and block tensor must be covered."""
    required = [k for k in vit.state_dict() if k.startswith(("patch_embed.", "blocks.")) or k in ("pos_embed", "cls_token")]
    if encoder in ("dinov2", "augreg", "retfound_dinov2"):
        # timm's own weights (RETFound builds RETFound-DINOv2 the same way before loading the teacher,
        # models_vit.py:79-86). A file pretrained/<arch>.safetensors is used when present, else the hub cache.
        import timm

        from bench.models.foundation import PRETRAINED

        local = PRETRAINED / f"{TIMM_ARCH[encoder]}.safetensors"
        overlay = {"file": str(local)} if local.is_file() else None
        reference = timm.create_model(
            TIMM_ARCH[encoder], pretrained=True, img_size=224, num_classes=0, pretrained_cfg_overlay=overlay
        )
        own = vit.state_dict()
        own.update({k: v for k, v in reference.state_dict().items() if k in own})
        vit.load_state_dict(own)
        if encoder != "retfound_dinov2":
            return {"source": TIMM_ARCH[encoder], "loaded": len(required), "missing": 0, "unexpected": 0}
    name, key = CHECKPOINTS[encoder]
    blob = torch.load(weights(name), map_location="cpu", weights_only=False)
    state = _clean_keys(blob[key] if key in blob else blob)
    interpolate_pos_embed(vit, state)
    absent = [k for k in required if k not in state or tuple(state[k].shape) != tuple(vit.state_dict()[k].shape)]
    if absent:
        raise RuntimeError(f"{name} does not cover {len(absent)} encoder tensors, e.g. {absent[:4]}; refusing to train")
    result = vit.load_state_dict(state, strict=False)
    return {"source": name, "loaded": len(required), "missing": len(result.missing_keys), "unexpected": len(result.unexpected_keys)}


class ConvModule(nn.Sequential):
    """Conv (no bias), BatchNorm, ReLU: mmseg ConvModule with norm_cfg BN."""

    def __init__(self, inp, out, kernel, padding=0):
        super().__init__(nn.Conv2d(inp, out, kernel, padding=padding, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True))


class UPerHead(nn.Module):
    def __init__(self, in_channels, channels=CHANNELS, num_classes=4, pool_scales=POOL_SCALES):
        super().__init__()
        self.ppm = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(s), ConvModule(in_channels[-1], channels, 1)) for s in pool_scales)
        self.bottleneck = ConvModule(in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1)
        self.lateral = nn.ModuleList(ConvModule(c, channels, 1) for c in in_channels[:-1])
        self.fpn = nn.ModuleList(ConvModule(channels, channels, 3, padding=1) for _ in in_channels[:-1])
        self.fpn_bottleneck = ConvModule(len(in_channels) * channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        top = inputs[-1]
        pooled = [top] + [F.interpolate(m(top), size=top.shape[-2:], mode="bilinear", align_corners=False) for m in self.ppm]
        laterals = [conv(x) for conv, x in zip(self.lateral, inputs[:-1])] + [self.bottleneck(torch.cat(pooled, dim=1))]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        outs = [conv(x) for conv, x in zip(self.fpn, laterals[:-1])] + [laterals[-1]]
        outs = [outs[0]] + [F.interpolate(o, size=outs[0].shape[-2:], mode="bilinear", align_corners=False) for o in outs[1:]]
        return self.conv_seg(self.dropout(self.fpn_bottleneck(torch.cat(outs, dim=1))))


class FCNHead(nn.Module):
    def __init__(self, in_channels, channels=AUX_CHANNELS, num_classes=4):
        super().__init__()
        self.conv = ConvModule(in_channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        return self.conv_seg(self.dropout(self.conv(x)))


class _BCEWithAux(nn.Module):
    def forward(self, outputs, target):
        target = target.float()
        if not isinstance(outputs, (tuple, list)):
            return F.binary_cross_entropy_with_logits(outputs, target)
        main, aux = outputs
        return F.binary_cross_entropy_with_logits(main, target) + AUX_WEIGHT * F.binary_cross_entropy_with_logits(aux, target)


def layer_id(name, num_layers):
    """util/lr_decay.py get_layer_id_for_vit, on names inside the encoder."""
    if name in ("cls_token", "pos_embed"):
        return 0
    if name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    return num_layers


def _recipe(encoder):
    return AuthorRecipe(
        loss="bce_main_plus_0.4_aux",
        optimizer="adamw",
        lr=BLR * BATCH / 256,
        weight_decay=0.05,
        schedule="warmup_cosine",
        iterations=0,
        epochs=50,
        batch_size=BATCH,
        effective_batch_size=BATCH,
        crop_size=(224, 224),
        pretrained={
            "retfound_mae": "RETFound_mae_natureCFP (MEH-MIDAS + Kaggle EyePACS, MAE)",
            "mae_in1k": "MAE ViT-L/16 ImageNet-1k (RETFound-MAE's starting point)",
            "retfound_dinov2": "RETFound_dinov2_meh (part of AlzEye, DINOv2)",
            "dinov2": "DINOv2 ViT-L/14 LVD-142M",
            "augreg": "ViT-L/16 augreg ImageNet-21k fine-tuned on 1k (supervised)",
        }[encoder],
        loss_params={"aux_weight": AUX_WEIGHT},
        optimizer_params={
            "layer_decay": LAYER_DECAY,
            "warmup_fraction": 10 / 50,
            "min_lr": 1e-6,
            "drop_path": DROP_PATH,
            "betas": [0.9, 0.999],
        },
        notes=(
            "Encoder outputs of blocks 8, 12, 16, 24; BEiT/MAE feature pyramid; UPerHead 512 with PPM (1, 2, 3, 6); "
            "FCN auxiliary head on stride 16, weight 0.4; four sigmoids and BCE (RETFound fine-tunes a classifier "
            "with cross-entropy). Batch 24 from train.sh, learning rate blr 5e-3 * 24 / 256. No fp16, no RandAugment, "
            "random erasing or mixup; B-std shared augmentation instead."
        ),
        source_of_settings=(
            "official_code/RETFound/train.sh (batch 24, input 224, epochs 50); main_finetune.py:42-67 (drop path 0.2, "
            "weight decay 0.05, blr 5e-3, layer decay 0.65, min lr 1e-6, warmup 10 epochs), :321-323 (lr scaling), "
            ":344-355 (AdamW over param_groups_lrd); util/lr_decay.py; util/lr_sched.py; models_vit.py:47-51, 79-86"
        ),
    )


class ViTUPerNet(SegmentationWrapper, nn.Module):
    ENCODER = "retfound_mae"
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="RETFound",
        family="foundation",
        track="B",
        repro_level="R3",
        source="official_code/RETFound + bench/models/vit_upernet.py UPerNet head",
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        pad_multiple=16,
        forward_size=(224, 224),
        class_index=(),
    )
    matmul_tf32 = True

    def __init__(self):
        super().__init__()
        self.encoder = build_encoder(self.ENCODER)
        dim = int(self.encoder.embed_dim)
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, stride=2), nn.BatchNorm2d(dim), nn.GELU(), nn.ConvTranspose2d(dim, dim, 2, stride=2)
        )
        self.fpn2 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(2, 2)
        self.decode_head = UPerHead([dim] * 4, num_classes=len(self.classes))
        self.auxiliary_head = FCNHead(dim, num_classes=len(self.classes))

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe(cls.ENCODER)

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def load_pretrained(self):
        return load_encoder(self.encoder, self.ENCODER)

    def forward(self, x):
        if self.training and x.shape[0] < 2:
            raise ValueError("training batches need at least 2 windows: the UPerHead pooling branch has BatchNorm on 1x1 maps")
        size = x.shape[-2:]
        feats = self.encoder.forward_intermediates(
            x, indices=list(OUT_INDICES), norm=False, output_fmt="NCHW", intermediates_only=True
        )
        feats = [self.fpn1(feats[0]), self.fpn2(feats[1]), self.fpn3(feats[2]), self.fpn4(feats[3])]
        main = F.interpolate(self.decode_head(feats), size=size, mode="bilinear", align_corners=False)
        if not self.training:
            return main
        aux = F.interpolate(self.auxiliary_head(feats[2]), size=size, mode="bilinear", align_corners=False)
        return main, aux

    def build_loss(self, recipe):
        return _BCEWithAux()

    def param_groups(self, recipe):
        """util/lr_decay.py param_groups_lrd over the encoder; the pyramid and heads at scale 1."""
        num_layers = len(self.encoder.blocks) + 1
        scales = [LAYER_DECAY ** (num_layers - i) for i in range(num_layers + 1)]
        no_decay = set(self.encoder.no_weight_decay()) if hasattr(self.encoder, "no_weight_decay") else set()
        groups = {}
        for full, param in self.named_parameters():
            if not param.requires_grad:
                continue
            inner = full[len("encoder."):] if full.startswith("encoder.") else None
            lid = layer_id(inner, num_layers) if inner is not None else num_layers
            decay = not (param.ndim == 1 or (inner is not None and inner in no_decay))
            key = (lid, decay)
            if key not in groups:
                groups[key] = {
                    "params": [],
                    "lr_scale": scales[lid],
                    "weight_decay": float(recipe.weight_decay) if decay else 0.0,
                }
            groups[key]["params"].append(param)
        return [groups[key] for key in sorted(groups)]

    def build_optimizer(self, params, recipe):
        betas = tuple(recipe.optimizer_params.get("betas", (0.9, 0.999)))
        optimizer = torch.optim.AdamW(self.param_groups(recipe), lr=float(recipe.lr), betas=betas)
        for group in optimizer.param_groups:
            group["initial_lr"] = float(recipe.lr)
        return optimizer

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """util/lr_sched.py with epochs as the fraction of the run: warmup from 0, then half cosine."""
        base = float(recipe.lr)
        floor = float(recipe.optimizer_params["min_lr"])
        warm = float(recipe.optimizer_params["warmup_fraction"])
        t = float(step) / float(total_steps)
        if t < warm:
            lr = base * t / warm
        else:
            lr = floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * (t - warm) / (1.0 - warm)))
        set_lr(optimizer, lr)


def _variant(name, encoder, mean, std):
    card = ModelCard(
        name=name,
        family="foundation",
        track="B",
        repro_level="R3",
        source="bench/models/vit_upernet.py (E5 encoder comparison)",
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": list(mean), "std": list(std)},
        pad_multiple=16 if encoder in ("mae_in1k", "augreg") else 14,
        forward_size=(224, 224),
        class_index=(),
    )
    return type(f"ViTUPerNet_{encoder}", (ViTUPerNet,), {"ENCODER": encoder, "card": card})


# Encoder comparison (design/15 section 8.3): experimental conditions, not registered methods (not in models.yaml).
EncoderMAE = _variant("ViT-UPerNet MAE-IN1k", "mae_in1k", IMAGENET_MEAN, IMAGENET_STD)
EncoderRETFoundDINOv2 = _variant("ViT-UPerNet RETFound-DINOv2", "retfound_dinov2", IMAGENET_MEAN, IMAGENET_STD)
EncoderDINOv2 = _variant("ViT-UPerNet DINOv2", "dinov2", IMAGENET_MEAN, IMAGENET_STD)
EncoderAugreg = _variant("ViT-UPerNet augreg", "augreg", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
