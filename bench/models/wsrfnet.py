"""B-std wrapper for WSRFNet (Li and Wu, IJCAI 2024; official_code/WSRFNet), docs/paper/18 section 2.3.

The repository holds the network (model.py), the wavelet transform and a
test script; no training script and no loss code. This wrapper imports the
author's modules and adds what the paper states for training:

    loss      Dice on every loop's prediction P_n, weighted n / N over the N = 4 loops (equation 6); the
              fused prediction of equation 7 (weighted mean of the loops' features) is what the released
              forward returns and what inference uses. The paper does not say whether P_n share the 3x3
              output conv; this wrapper shares spv1 across loops and the fused map (assumption, in the recipe)
    recipe    SGD 0.01, momentum 0.9, weight decay 5e-4, poly 0.9; 80k (IDRiD) / 100k (DDR) iterations at
              batch 4 (section 4.2; B-std replaces the step count by its budget)
    backbone  EfficientNet-B1 ImageNet (efficientnet_pytorch); loaded from pretrained/efficientnet-b1.pth
              when present, else through the package's download
    input     the author's test pipeline (dataset.py) resizes to 1440x960 / 1024 and normalises with the
              M2MRF RGB statistics; the card carries those statistics; 960 patches, whole-canvas inference

Head: four sigmoids in the order EX, HE, SE, MA (test.py CLASSES), permuted
to MA, HE, EX, SE. The author forward loads a checkpoint in __init__ and
returns probabilities through a sigmoid; the subclass below skips the load,
returns logits, and in training mode also returns the per-loop logits through
``_aux`` for the loss. Dice smoothing is not stated; 1e-5 as in M2MRF's
BinaryLoss, the evaluation code WSRFNet reuses.
"""

import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "official_code" / "WSRFNet"
PRETRAINED = REPO / "pretrained" / "efficientnet-b1.pth"
AUTHOR_LESIONS = ("EX", "HE", "SE", "MA")
CHANNEL_INDEX = tuple(AUTHOR_LESIONS.index(name) for name in LESION_CLASSES)
LOOPS = 4
SMOOTH = 1e-5


def _import():
    if not OFFICIAL.is_dir():
        raise FileNotFoundError(f"{OFFICIAL} is missing; git submodule update --init official_code/WSRFNet")
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    for name in ("model", "wavelets"):
        sys.modules.pop(name, None)
    import model as author_model

    return author_model


def _upsample_like(src, shape):
    return nn.functional.interpolate(src, size=shape, mode="bilinear", align_corners=True)


class _Net(nn.Module):
    """model.WSRFNet without the checkpoint load, with logits out and the per-loop maps kept."""

    def __init__(self, author_model):
        super().__init__()
        cls = author_model.WSRFNet
        # the author __init__ ends with load_state_dict(torch.load(cfg.modelpath)); build the modules the same way
        # without that line by calling nn.Module.__init__ and copying the constructor body's module definitions
        self.loop = LOOPS
        from efficientnet_pytorch import EfficientNet

        conv_bn_relu = author_model.conv_bn_relu
        self.bkbone = EfficientNet.from_name("efficientnet-b1", include_top=False)
        self.tran0 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran2 = nn.Sequential(nn.Conv2d(40, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran3 = nn.Sequential(nn.Conv2d(112, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.tran4 = nn.Sequential(nn.Conv2d(1280, 64, kernel_size=1), conv_bn_relu(64, 64, kernel=3, stride=1, padding=1))
        self.decoder1 = author_model.MFA()
        self.midlayer = author_model.RefineModule()
        self.spv1 = nn.Conv2d(64, 4, 3, 1, 1)
        self._author_class = cls
        self.loops_out = None

    def forward(self, image):
        shape = image.shape[-2:]
        endpoints = self.bkbone.extract_endpoints(image)
        tran = [self.tran0(endpoints["reduction_1"]), self.tran1(endpoints["reduction_2"]), self.tran2(endpoints["reduction_3"]),
                self.tran3(endpoints["reduction_4"]), self.tran4(endpoints["reduction_6"])]
        out_s = [self.decoder1(*tran)]
        for n in range(1, self.loop):
            b_n = self.midlayer(out_s[n - 1], *tran)
            out_s.append(self.decoder1(*b_n))
        total = float(sum(range(self.loop + 1)))
        fused = sum(out_s[i] * (i + 1) / total for i in range(self.loop))          # equation 7
        self.loops_out = [_upsample_like(self.spv1(s), shape) for s in out_s] if self.training else None   # P_n (shared conv)
        return _upsample_like(self.spv1(fused), shape)


class _LoopDice(nn.Module):
    """Equation 6: sum_n (n / N) Dice(P_n, G) on the per-loop logits the forward kept; the fused map in evaluation."""

    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    @staticmethod
    def _dice(logits, target):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - (2.0 * inter + SMOOTH) / (denom + SMOOTH)).mean()

    def forward(self, logits, target):
        target = target.float()
        loops = self.wrapper.net.loops_out
        if loops is None or not self.wrapper.training:
            return self._dice(logits, target)
        n = len(loops)
        return sum((i + 1) / float(n) * self._dice(p[:, CHANNEL_INDEX], target) for i, p in enumerate(loops))


NORMALIZATION = {"IDRiD": ([116.513, 56.437, 16.309], [80.206, 41.232, 13.293]),
                 "DDR": ([81.205, 50.636, 21.216], [76.252, 48.798, 21.625])}   # dataset.py Config, 0-255 RGB


def _recipe(dataset):
    iterations = {"IDRiD": 80000, "DDR": 100000}[dataset]
    mean, std = NORMALIZATION[dataset]
    return AuthorRecipe(
        loss="dice",
        optimizer="sgd",
        lr=0.01,
        weight_decay=5e-4,
        schedule="poly",
        iterations=iterations,
        batch_size=4,
        effective_batch_size=4,
        crop_size=(960, 1440) if dataset == "IDRiD" else (1024, 1024),
        pretrained="ImageNet EfficientNet-B1 (efficientnet_pytorch)",
        loss_params={"smooth": SMOOTH},
        optimizer_params={"momentum": 0.9, "power": 0.9, "min_lr": 0.0},
        notes=(
            "The repository has no training script: the loss (equation 6, Dice per loop weighted n/N) and the "
            "optimizer (section 4.2) are implemented here from the paper. Assumptions: the per-loop predictions P_n "
            "share the 3x3 output conv (spv1) with the fused map; Dice smoothing 1e-5 (M2MRF's BinaryLoss, whose "
            "evaluation code the author reuses). Author step count replaced by the B-std budget; shared augmentation "
            f"instead of the paper's crops, flips and 90-degree rotations; the author test normalisation (RGB mean {mean}, "
            f"std {std} on 0-255, dataset.py) is divided by 255 for the runner."
        ),
        source_of_settings="paper section 3.3 equation 6-7, section 4.2; official_code/WSRFNet/model.py, dataset.py, test.py",
    )


class WSRFNet(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="WSRFNet",
        family="fundus_multiscale",
        track="B",
        repro_level="R1",
        source="official_code/WSRFNet",
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": [116.513 / 255.0, 56.437 / 255.0, 16.309 / 255.0],
                       "std": [80.206 / 255.0, 41.232 / 255.0, 13.293 / 255.0]},
        pad_multiple=32,
        forward_size=None,
        class_index=(),
    )

    def __init__(self):
        super().__init__()
        self.net = _Net(_import())

    @classmethod
    def recipe_for(cls, dataset):
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
        return _recipe(dataset)

    def author_recipe(self, dataset):
        recipe = self.recipe_for(dataset)
        # the author normalises each dataset with its own RGB statistics (dataset.py Config); the runner reads
        # the card after this call, so the instance card carries the statistics of the training source
        mean, std = NORMALIZATION[dataset]
        self.card = replace(type(self).card, normalization={"mean": [v / 255.0 for v in mean], "std": [v / 255.0 for v in std]})
        return recipe

    def load_pretrained(self):
        from efficientnet_pytorch import EfficientNet

        if PRETRAINED.is_file():
            state = torch.load(str(PRETRAINED), map_location="cpu")
        else:
            state = EfficientNet.from_pretrained("efficientnet-b1").state_dict()
        state = {k: v for k, v in state.items() if not k.startswith("_fc")}
        missing, unexpected = self.net.bkbone.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("_fc")]
        if missing or unexpected:
            raise RuntimeError(f"EfficientNet-B1 weights do not match: missing {missing[:3]}, unexpected {unexpected[:3]}")
        return len(state)

    def forward(self, x):
        logits = self.net(x)
        return logits[:, CHANNEL_INDEX]

    def build_loss(self, recipe):
        return _LoopDice(self)
