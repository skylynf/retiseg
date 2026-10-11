"""SAM-Adapter (official_code/SAM-Adapter, 041277f), ViT-H, for four lesion channels (design/15 section 7.2).

The network is the author's SAM class built from configs/cod-sam-vit-h.yaml
(models/sam.py:93-151): the adapter image encoder with its high-pass FFT
prompt generator (models/mmseg/models/sam/image_encoder.py:116-123; the
config's encoder_mode keys are not read there), the SAM mask decoder with
four mask tokens, a random-Fourier position layer and no_mask_embed at the
top level. The checkpoint is loaded the author's way, strict=False
(train.py:171-172): the prompt encoder's two tensors do not match those
top-level names and stay at their random initial values, as in the author's
runs. Trainable, as in train.py:174-176: the prompt generator and everything
outside the image encoder.

The author's forward uses one prompt-free query batch that broadcasts over the
images and multimask_output=False, keeping mask token 0 (models/sam.py:166-186).
Here all four tokens are kept, one sigmoid per lesion, from the same
predict_masks call. Loss as in backward_G: BCE with logits + the IoU loss per
image and channel, without smoothing (models/sam.py:48-54, 248-254); on a
patch with no lesion the IoU term is constant. AdamW 2e-4 with PyTorch's
default weight decay 0.01 (utils.py:90-96; cod-sam-vit-h.yaml:93-98);
CosineAnnealingLR over 20 epochs to 1e-7, stepped per epoch (train.py:108,
190), here 20 equal segments of the B-std steps. Effective batch 2: the
script wraps the model in DistributedDataParallel and immediately takes
model.module back (train.py:162-169), so processes never average gradients
and the saved model saw 2 images per step. ImageNet mean and std
(datasets/wrappers.py:39-42, 75-78), 1024 inputs.
"""

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import IMAGENET_MEAN, IMAGENET_STD, OFFICIAL, official, set_lr, stepped_cosine, weights

CHECKPOINT = "sam_vit_h_4b8939.pth"
CONFIG = OFFICIAL / "SAM-Adapter" / "configs" / "cod-sam-vit-h.yaml"


def _model_args():
    return yaml.safe_load(CONFIG.read_text())["model"]["args"]


def _build():
    # models/mmseg/__init__.py imports mmcv and checks 1.1.4 <= version <= 1.7.0; the package chain needs a
    # few mmcv names (Registry, ConvModule, ...), which bench/compat/mmcv (1.2.0 API, used for M2MRF) provides.
    with official("SAM-Adapter", purge=("models",), extra_paths=(OFFICIAL.parent / "bench" / "compat",)):
        import models  # noqa: F401  registers "sam"
        from models.sam import SAM

        args = _model_args()
        return SAM(inp_size=args["inp_size"], encoder_mode=args["encoder_mode"], loss=args["loss"])


def freeze(net):
    """train.py:174-176: image encoder frozen except the prompt generator."""
    for name, param in net.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            param.requires_grad_(False)


def iou_loss(pred, target):
    """models/sam.py:48-54, per image and channel, no smoothing."""
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    return (1 - inter / union).mean()


class _BCEIoU(nn.Module):
    def forward(self, logits, target):
        target = target.float()
        return F.binary_cross_entropy_with_logits(logits, target) + iou_loss(logits, target)


def _recipe():
    return AuthorRecipe(
        loss="bce_plus_iou",
        optimizer="adamw",
        lr=2e-4,
        weight_decay=0.01,
        schedule="stepped_cosine",
        iterations=0,
        epochs=20,
        batch_size=2,
        effective_batch_size=2,
        crop_size=(1024, 1024),
        pretrained="SAM ViT-H (sam_vit_h_4b8939.pth), image encoder frozen except the prompt generator",
        loss_params={},
        optimizer_params={"eta_min": 1e-7, "segments": 20},
        notes=(
            "Four mask tokens as four sigmoids (author keeps token 0); prompt-encoder tensors not loaded and "
            "effective batch 2, both as the author code behaves (design/15 section 15); 1024 windows on the canvas."
        ),
        source_of_settings=(
            "official_code/SAM-Adapter/configs/cod-sam-vit-h.yaml (inp_size 1024, batch 2, adamw 2e-4, lr_min 1e-7, "
            "epoch_max 20); train.py:101-108, 162-176, 190; utils.py:90-96; models/sam.py:48-54, 93-151, 166-186, 248-254"
        ),
    )


class SAMAdapter(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="SAM-Adapter",
        family="foundation",
        track="B",
        repro_level="R3",
        source="official_code/SAM-Adapter",
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        pad_multiple=16,
        forward_size=(1024, 1024),
        class_index=(),
        window_batch=1,
    )
    matmul_tf32 = True

    def __init__(self):
        super().__init__()
        self.net = _build()
        freeze(self.net)

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def load_pretrained(self):
        state = torch.load(weights(CHECKPOINT), map_location="cpu", weights_only=False)
        result = self.net.load_state_dict(state, strict=False)
        needed = [
            k for k in self.net.state_dict()
            if (k.startswith("image_encoder.") and "prompt_generator" not in k) or k.startswith("mask_decoder.")
        ]
        absent = [k for k in needed if k in result.missing_keys]
        if absent:
            raise RuntimeError(f"{CHECKPOINT} does not cover {len(absent)} encoder or decoder tensors, e.g. {absent[:4]}")
        freeze(self.net)
        return {"missing": sorted(result.missing_keys), "unexpected": len(result.unexpected_keys)}

    def enable_activation_checkpointing(self):
        from bench.models.foundation import recompute

        recompute(list(self.net.image_encoder.blocks))
        return "SAM ViT-H blocks recomputed in backward (torch.utils.checkpoint); arithmetic unchanged"

    def forward(self, x):
        net = self.net
        size = net.image_embedding_size
        sparse = torch.empty((1, 0, net.prompt_embed_dim), device=x.device)
        dense = net.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, size, size)
        features = net.image_encoder(x)
        low_res, _ = net.mask_decoder.predict_masks(
            image_embeddings=features,
            image_pe=net.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
        )
        return net.postprocess_masks(low_res, net.inp_size, net.inp_size)

    def build_loss(self, recipe):
        return _BCEIoU()

    def build_optimizer(self, params, recipe):
        # utils.make_optimizer(model.parameters(), {lr: 2e-4}): torch's default weight decay 0.01.
        optimizer = torch.optim.AdamW(list(params), lr=float(recipe.lr), weight_decay=float(recipe.weight_decay))
        for group in optimizer.param_groups:
            group["initial_lr"] = float(recipe.lr)
        return optimizer

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        lr = stepped_cosine(
            step, total_steps, float(recipe.lr), float(recipe.optimizer_params["eta_min"]), int(recipe.optimizer_params["segments"])
        )
        set_lr(optimizer, lr)
