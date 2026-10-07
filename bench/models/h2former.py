"""B1 wrapper for the official H2Former IDRiD script.

The network is ``res34_swin_MS`` from official_code/H2Former. This file does
not crop, does not read the author's training-list length, and does not set
the seed. The runner owns the split, the field-of-view canvas and the seed.
"""

import sys
import types
from pathlib import Path

import torch
from torch import nn

from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

_MODELS = Path(__file__).resolve().parents[2] / "official_code" / "H2Former" / "models"
_NUM_CLASSES = 5  # idrid_train.py:29, background plus four lesions


def _ensure_timm_layers():
    """The official modules import DropPath, to_2tuple and trunc_normal_ from timm."""
    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    def to_2tuple(value):
        if isinstance(value, tuple):
            return value
        return (value, value)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0, scale_by_keep=True):
            super().__init__()
            self.drop_prob = float(drop_prob)
            self.scale_by_keep = bool(scale_by_keep)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep)
            if keep > 0.0 and self.scale_by_keep:
                mask = mask.div(keep)
            return x * mask

    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    layers.to_2tuple = to_2tuple
    layers.trunc_normal_ = trunc_normal_
    models = types.ModuleType("timm.models")
    models.layers = layers
    timm_mod = types.ModuleType("timm")
    timm_mod.models = models
    sys.modules.setdefault("timm", timm_mod)
    sys.modules.setdefault("timm.models", models)
    sys.modules["timm.models.layers"] = layers


def _load_res34_swin_MS():
    _ensure_timm_layers()
    models_dir = str(_MODELS)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    from H2Former import res34_swin_MS

    return res34_swin_MS


def _rgb_stem(module):
    """The released constructor hardcodes 4 input channels.

    models/H2Former.py:27 and :49 use 4 channels. The IDRiD loader returns RGB
    (datasets/dataset.py:50-69) and the paper initializes the convolutions from
    ImageNet ResNet-34, whose first convolution has 3 channels. Those two stems
    are rebuilt here with 3 input channels. official_code/ is not edited.
    """
    old = module.conv1
    module.conv1 = nn.Conv2d(
        3,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
    )
    rebuilt = []
    for proj in module.patch_embed.projs:
        rebuilt.append(
            nn.Conv2d(
                3,
                proj.out_channels,
                kernel_size=proj.kernel_size,
                stride=proj.stride,
                padding=proj.padding,
                dilation=proj.dilation,
                groups=proj.groups,
                bias=proj.bias is not None,
            )
        )
    module.patch_embed.projs = nn.ModuleList(rebuilt)
    module.patch_embed.in_chans = 3


class _AuthorDice(nn.Module):
    """utils.py:89-128. Smooth is 1e-5. Every class, including background, is averaged.

    The denominator sums squares, over the whole batch and spatial map together.
    """

    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.n_classes = int(n_classes)
        self.smooth = float(smooth)

    def _one_hot(self, target):
        planes = [(target == i).unsqueeze(1) for i in range(self.n_classes)]
        return torch.cat(planes, dim=1).float()

    def _dice(self, score, target):
        smooth = self.smooth
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        encoded = self._one_hot(target)
        if probs.shape != encoded.shape:
            raise ValueError(f"dice target {tuple(encoded.shape)} != probs {tuple(probs.shape)}")
        loss = logits.new_zeros(())
        for index in range(self.n_classes):
            loss = loss + self._dice(probs[:, index], encoded[:, index])
        return loss / self.n_classes


class _AuthorCEDice(nn.Module):
    """idrid_train.py:79 adds CrossEntropyLoss to DiceLoss, both on integer labels.

    The runner passes four overlapping masks in MA, HE, EX, SE order. Overlaps
    follow M2MRF_OVERWRITE_ORDER (EX, HE, SE, MA), so a microaneurysm is kept.
    The author script already reads a single-channel mask and does not state an
    overlap rule. Empty pixels stay background, class 0.
    """

    def __init__(self, n_classes, class_index, smooth=1e-5):
        super().__init__()
        self.n_classes = int(n_classes)
        self.class_index = tuple(int(i) for i in class_index)
        self.ce = nn.CrossEntropyLoss()
        self.dice = _AuthorDice(n_classes, smooth=smooth)

    def indices(self, target):
        from bench.common.labels import exclusive_label

        return exclusive_label(target, self.class_index)

    def forward(self, logits, target):
        labels = self.indices(target)
        return self.ce(logits, labels) + self.dice(logits, labels)


def _checkpointed_layer_forward(self, x):
    from torch.utils.checkpoint import checkpoint

    for blk in self.blocks:
        if self.training and torch.is_grad_enabled():
            x = checkpoint(blk, x, use_reentrant=False)
        else:
            x = blk(x)
    return x


def _checkpoint_swin_blocks(net):
    """Recompute each Swin block in backward instead of storing its activations.

    window_size is image_size // 16 = 60, so a stage-0 window holds 3600
    tokens and batch 1 does not fit in 40GB with stored activations.
    BasicLayer takes use_checkpoint but its forward never reads it
    (models/basic_module.py:301-325). The arithmetic is unchanged and the
    RNG state is replayed, so DropPath draws the same masks.
    """
    import types

    for layer in net.swin_layers:
        layer.use_checkpoint = True
        layer.forward = types.MethodType(_checkpointed_layer_forward, layer)


def _recipe():
    # idrid_train.py is the only fundus training script. DDR is named in
    # datasets/dataset.py:36 as a png path, with no separate optimizer settings.
    return AuthorRecipe(
        loss="ce_dice",
        optimizer="adamw",
        lr=1e-4,
        weight_decay=3e-5,
        schedule="poly",
        iterations=0,
        epochs=92,
        batch_size=2,
        crop_size=(960, 960),
        pretrained="ImageNet resnet34",
        loss_params={"smooth": 1e-5},
        optimizer_params={"betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False, "power": 0.9},
        notes=(
            "H2Former is a general medical segmentation model, not a DR-specific method. "
            "The IDRiD script is the training entry; DDR uses that same script. "
            "B1 takes the field-of-view crop, rescales its long side to 1440, then resizes to 960. "
            "The imgaug random crop in datasets/dataset.py:18 is not applied. "
            "DDR has no separate training script, so it uses this IDRiD recipe. "
            "The author script sets its own seed; the runner injects the cell seed. "
            "dataset.py:67-69 feeds 0-255 RGB with no mean or std, so the card std is 1/255. "
            "Released conv1 and patch_embed take 4 channels (models/H2Former.py:27 and :49); "
            "the wrapper rebuilds those two stems with 3 channels. "
            "idrid_train.py:91 applies poly after optimizer.step and before iter_num increments, "
            "so the first two updates stay at base_lr. "
            "Paper IV.B (He et al., TMI 2023) says 90 epochs, batch size 3 and weight decay 0.0001; "
            "this recipe follows idrid_train.py. "
            "Four-plane targets overlap in M2MRF order EX, HE, SE, MA, so MA is kept. "
            "Training loads torchvision ResNet-34 ImageNet weights after the stem is rebuilt to 3 channels, matched by name and shape, which includes conv1. "
            "idrid_train.py loads resnet34.pth the same way, but its released conv1 has 4 channels so that tensor is skipped. "
            "Swin blocks are recomputed in backward (activation checkpointing) so batch 2 at 960 fits a 40GB A100; "
            "the forward and gradients are the same computation."
        ),
        source_of_settings=(
            "official_code/H2Former/idrid_train.py:27 batch_size 2; :28 base_lr 0.0001; :29 num_classes 5; "
            ":30 image_size (960, 960); :34 max_epoch 92; :43-47 ImageNet resnet34.pth matched by shape; "
            ":51-52 CrossEntropyLoss + DiceLoss(5); :54 AdamW betas (0.9, 0.999), eps 1e-8, "
            "weight_decay 3e-5, amsgrad False; :78 softmax; :79 unweighted sum; :91 poly power 0.9. "
            "utils.py:106 smooth 1e-5; :108-110 squared sums; :124-128 mean over all 5 classes, background included. "
            "README.md:35 background 0 and lesions 1, 2, 3, 4, without names. "
            "idrid_train.py:132-135 one-hot channel 1 EX, 2 HE, 3 MA, 4 SE. "
            "utils.py:82-85 paints 1 red, 2 green, 3 blue, 4 magenta, then converts RGB to BGR for cv2. "
            "Paper Fig. 6: red, green, blue and pink are hard exudates, hemorrhages, microaneurysms and soft exudates. "
            "class_index (MA, HE, EX, SE) = (3, 2, 1, 4)."
        ),
    )


class H2Former(SegmentationWrapper, nn.Module):
    classes = ("MA", "HE", "EX", "SE")
    card = ModelCard(
        name="H2Former",
        family="general",
        track="B",
        repro_level="R3",
        source="official_code/H2Former",
        output_activation="softmax",
        in_channels=3,
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]},
        pad_multiple=16,
        forward_size=(960, 960),
        class_index=(3, 2, 1, 4),
    )

    def __init__(self):
        super().__init__()
        res34_swin_MS = _load_res34_swin_MS()
        self.net = res34_swin_MS(self.card.forward_size[0], _NUM_CLASSES)
        _rgb_stem(self.net)
        _checkpoint_swin_blocks(self.net)

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def build_loss(self, recipe):
        smooth = float(recipe.loss_params.get("smooth", 1e-5))
        return _AuthorCEDice(_NUM_CLASSES, self.card.class_index, smooth=smooth)

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """Match idrid_train.py:91-94, which writes the learning rate after the update."""
        from bench.runtime import adjust_lr_after_step

        return adjust_lr_after_step(optimizer, step, total_steps, recipe)

    def load_pretrained(self):
        from torchvision.models import ResNet34_Weights, resnet34

        source = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
        model_dict = self.net.state_dict()
        matched = {
            key: value
            for key, value in source.items()
            if key in model_dict and tuple(value.shape) == tuple(model_dict[key].shape)
        }
        if "conv1.weight" not in matched or len(matched) < 50:
            raise RuntimeError(
                f"ImageNet ResNet-34 matched {len(matched)} tensors and conv1 is "
                f"{'present' if 'conv1.weight' in matched else 'absent'}"
            )
        model_dict.update(matched)
        self.net.load_state_dict(model_dict)
        return len(matched)

    def forward(self, x):
        return self.net(x)
