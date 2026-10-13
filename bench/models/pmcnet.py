"""PMCNet rewritten in PyTorch from the author's Keras code, for B-std (docs/paper/18 section 2.2).

He et al., IEEE TMI 2022, official_code/PMCNet (TensorFlow 1.13 / Keras 2.3,
which no current GPU stack runs). models.py is translated layer by layer:

    encoder   EfficientNet-B0 (ImageNet); four taps at the *expand* activations of the first block of
              stages 2, 3, 4 and 6 (Keras layers swish_4, swish_10, swish_16, swish_34: each MBConv has an
              expand, a depthwise and an SE Swish, so those names land on the expand activations), i.e.
              96 ch at 1/2, 144 at 1/4, 240 at 1/8, 672 at 1/16, taken before the stride-2 depthwise conv
    PFF       progressive feature fusion between neighbouring scales (PFF_2_pre, PFF_3, PFF_3, PFF_2_nex):
              the coarser map is projected and bilinearly upsampled, the finer one depthwise 2x2 stride-2
              downsampled then projected, each multiplied with the current map, concatenated, DAB, 1x1 + ReLU;
              the result is added to the tap (residual)
    DAB       channel (global average pool -> dense -> sigmoid), spatial (channel mean -> sigmoid) and pixel
              (1x1 conv -> sigmoid) attention, the three gated maps summed
    decoder   bilinear x2 + 1x1 + ReLU (Upsample_conv), add skip (skip_block), two 3x3 + ReLU (conv_block),
              at 240, 144, 96 channels; a last x2 to full resolution at 32 channels, 1x1 to five classes, softmax

LeakyReLU(alpha=0) is ReLU. No normalization layers in the decoder (the BN
line is commented out in the author's conv_unit). Classes: bg, EX, HE, MA,
SE (test.py), so class_index (3, 2, 1, 4). Loss: categorical cross-entropy.
Optimizer (train.py): Adam 1e-4 with lr = 1e-4 * (1 - (epoch + 1) / 60) ** 0.9
per epoch over 60 epochs, batch 1, whole images (1024 for DDR, 960 x 1440 for
IDRiD), no augmentation. Under B-std the per-epoch decay becomes
(1 - progress) ** 0.9 over the B-std updates (same shape); 960 patches, whole
canvas at inference; effective batch 1 as the author.

ImageNet weights: timm tf_efficientnet_b0.in1k, the TensorFlow release the
Keras package converted (same padding reproduced by timm's tf_ variant);
loaded from pretrained/tf_efficientnet_b0.in1k.safetensors when present,
else through timm's hub download. Deviations are written to the recipe.
The author's released dataGenerator one-hot-encodes labels with 3 classes
against a 5-class head and does not run as published; this port follows the
paper's five classes (L2 funnel: R3 with a defect).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.common.labels import exclusive_label
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import set_lr

REPO = Path(__file__).resolve().parents[2]
PRETRAINED = REPO / "pretrained" / "tf_efficientnet_b0.in1k.safetensors"
TIMM_NAME = "tf_efficientnet_b0.in1k"
TAPS = (1, 2, 3, 5)                 # timm blocks index of the first block of Keras stages 2, 3, 4, 6
TAP_CHANNELS = (96, 144, 240, 672)
NB_FILTER = (32, 96, 144, 240, 672)   # models.py nb_filter
NUM_CLASSES = 5
# test.py: EX = channel 1, HE = 2, MA = 3, SE = 4
CLASS_INDEX = (3, 2, 1, 4)


class DAB(nn.Module):
    """DAB_block: channel + spatial + pixel attention, summed."""

    def __init__(self, channels):
        super().__init__()
        self.dense = nn.Linear(channels, channels)
        self.pixel = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        score_c = torch.sigmoid(self.dense(x.mean(dim=(2, 3))))[:, :, None, None]
        score_s = torch.sigmoid(x.mean(dim=1, keepdim=True))
        score_p = torch.sigmoid(self.pixel(x))
        return x * score_c + x * score_s + x * score_p


class _Down(nn.Module):
    """DepthwiseConv2D(2, strides 2, same) then 1x1: the finer map brought to the current scale."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 2, stride=2, groups=in_channels)
        self.project = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # Keras "same" with kernel 2, stride 2 on even sizes pads nothing; odd sizes pad one pixel at the end
        pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return self.project(self.depthwise(x))


class _Up(nn.Module):
    """1x1 then bilinear x2: the coarser map brought to the current scale."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, size):
        return F.interpolate(self.project(x), size=size, mode="bilinear", align_corners=False)


class PFF(nn.Module):
    """PFF_3 (both neighbours), PFF_2_pre (coarser neighbour only) or PFF_2_nex (finer only)."""

    def __init__(self, channels, pre_channels=None, nex_channels=None):
        super().__init__()
        self.down = _Down(pre_channels, channels) if pre_channels else None
        self.up = _Up(nex_channels, channels) if nex_channels else None
        self.dab = DAB(channels * 2)   # every variant concatenates two maps of ``channels``
        self.out = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, cur, pre=None, nex=None):
        parts = []
        if self.down is not None and self.up is not None:
            parts = [self.down(pre) * cur, self.up(nex, cur.shape[-2:]) * cur]          # PFF_3: mul_low, mul_high
        elif self.up is not None:
            parts = [cur, cur * self.up(nex, cur.shape[-2:])]                              # PFF_2_pre: cur, mul
        else:
            parts = [cur, self.down(pre) * cur]                                            # PFF_2_nex: cur, mul
        x = self.dab(torch.cat(parts, dim=1))
        return F.relu(self.out(x))


class _UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, size):
        return F.relu(self.conv(F.interpolate(x, size=size, mode="bilinear", align_corners=False)))


def _conv_block(channels):
    return nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))


class PMCNetTorch(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained_backbone=False):
        super().__init__()
        import timm

        self.backbone = timm.create_model(TIMM_NAME, pretrained=pretrained_backbone, features_only=False)
        self._taps = {}
        for position, index in enumerate(TAPS):
            block = self.backbone.blocks[index][0]
            block.bn1.register_forward_hook(self._hook(position))
        c2, c3, c4, c5 = TAP_CHANNELS
        self.pff5 = PFF(c5, pre_channels=c4)                 # PFF_2_nex(conv4_1, conv5_1)
        self.pff4 = PFF(c4, pre_channels=c3, nex_channels=c5)  # PFF_3(conv3_1, conv4_1, conv5_1)
        self.pff3 = PFF(c3, pre_channels=c2, nex_channels=c4)  # PFF_3(conv2_1, conv3_1, conv4_1)
        self.pff2 = PFF(c2, nex_channels=c3)                 # PFF_2_pre(conv2_1, conv3_1)
        self.de4, self.block4 = _UpsampleConv(c5, NB_FILTER[3]), _conv_block(NB_FILTER[3])
        self.de3, self.block3 = _UpsampleConv(NB_FILTER[3], NB_FILTER[2]), _conv_block(NB_FILTER[2])
        self.de2, self.block2 = _UpsampleConv(NB_FILTER[2], NB_FILTER[1]), _conv_block(NB_FILTER[1])
        self.de1, self.block1 = _UpsampleConv(NB_FILTER[1], NB_FILTER[0]), _conv_block(NB_FILTER[0])
        self.head = nn.Conv2d(NB_FILTER[0], num_classes, 1)

    def _hook(self, position):
        def fn(_module, _inputs, output):
            self._taps[position] = output

        return fn

    def forward(self, x):
        self._taps = {}
        self.backbone.forward_features(x)
        c2, c3, c4, c5 = (self._taps[i] for i in range(4))
        if tuple(t.shape[1] for t in (c2, c3, c4, c5)) != TAP_CHANNELS:
            raise RuntimeError(f"tap channels {[t.shape[1] for t in (c2, c3, c4, c5)]} != {TAP_CHANNELS}")
        p5 = self.pff5(c5, pre=c4) + c5
        p4 = self.pff4(c4, pre=c3, nex=c5) + c4
        p3 = self.pff3(c3, pre=c2, nex=c4) + c3
        p2 = self.pff2(c2, nex=c3) + c2
        d = self.block4(self.de4(p5, p4.shape[-2:]) + p4)
        d = self.block3(self.de3(d, p3.shape[-2:]) + p3)
        d = self.block2(self.de2(d, p2.shape[-2:]) + p2)
        d = self.block1(self.de1(d, x.shape[-2:]))
        return self.head(d)


def _recipe():
    return AuthorRecipe(
        loss="ce",
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.0,
        schedule="poly",
        iterations=0,
        epochs=60,
        batch_size=1,
        effective_batch_size=1,
        crop_size=(1024, 1024),
        pretrained=f"ImageNet EfficientNet-B0 ({TIMM_NAME})",
        loss_params={},
        optimizer_params={"power": 0.9, "min_lr": 0.0},
        notes=(
            "Rewritten in PyTorch from official_code/PMCNet/models.py (Keras); the backbone taps are the expand "
            "activations swish_4/10/16/34 (96, 144, 240, 672 channels at 1/2 to 1/16). ImageNet weights from timm "
            f"{TIMM_NAME} (the TensorFlow release the Keras package converted), not the author's .h5 file. The author's "
            "per-epoch decay 1e-4 x (1 - (epoch + 1)/60)^0.9 becomes (1 - progress)^0.9 over the B-std updates. The "
            "released dataGenerator one-hot-encodes labels with 3 classes against the 5-class head (does not run as "
            "published); this port follows the paper's five classes. Batch 1 as train.py; BatchNorm sees one image. "
            "960 patches and whole-canvas inference instead of whole 1024 / 960x1440 images; shared augmentation (the "
            "author uses none)."
        ),
        source_of_settings=(
            "official_code/PMCNet/train.py: Adam(lr=0.0001), step_decay (initial 1e-4, epochs_drop 60, power 0.9), "
            "epochs=60, batch_size=1, loss multi_class = categorical_crossentropy, image_h/w 1024 (DDR) 960x1440 (IDRiD); "
            "models.py: PMCNet(); test.py: EX, HE, MA, SE = channels 1-4"
        ),
    )


class _CE(nn.Module):
    def forward(self, logits, target):
        return F.cross_entropy(logits, exclusive_label(target, CLASS_INDEX))


class PMCNet(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="PMCNet",
        family="fundus_multiscale",
        track="B",
        repro_level="R3",
        source="official_code/PMCNet (Keras) rewritten in bench/models/pmcnet.py",
        output_activation="softmax",
        in_channels=3,
        # Keras EfficientNet preprocessing in the author's pipeline: ImageDataGenerator gives 0-255 pixels straight to
        # the network (no preprocess_input in train.py), so the only affine map is the runner's /255 -> mean 0, std 1/255
        # is not representable; the author network sees 0-255. We feed [0, 1] x 255 inside forward instead.
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        pad_multiple=32,
        forward_size=None,
        class_index=CLASS_INDEX,
    )

    def __init__(self):
        super().__init__()
        self.net = PMCNetTorch(pretrained_backbone=False)

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe()

    def author_recipe(self, dataset):
        return self.recipe_for(dataset)

    def load_pretrained(self):
        """ImageNet EfficientNet-B0 into the backbone: pretrained/<file> if present, else timm's hub."""
        import timm

        if PRETRAINED.is_file():
            from safetensors.torch import load_file

            state = load_file(str(PRETRAINED))
        else:
            state = timm.create_model(TIMM_NAME, pretrained=True).state_dict()
        missing, unexpected = self.net.backbone.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("classifier")]
        unexpected = [k for k in unexpected if not k.startswith("classifier")]
        if missing or unexpected:
            raise RuntimeError(f"EfficientNet-B0 weights do not match: missing {missing[:3]}, unexpected {unexpected[:3]}")
        return sum(1 for k in state if not k.startswith("classifier"))

    def forward(self, x):
        # train.py feeds 0-255 pixels (ImageDataGenerator without rescale); the runner normalised to [0, 1]
        return self.net(x * 255.0)

    def build_loss(self, recipe):
        return _CE()

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """train.py step_decay: lr = 1e-4 (1 - (epoch + 1)/60)^0.9 per epoch -> (1 - progress)^0.9 over the updates."""
        progress = min(1.0, float(step) / float(max(1, total_steps)))
        set_lr(optimizer, float(recipe.lr) * (1.0 - progress) ** float(recipe.optimizer_params["power"]))
