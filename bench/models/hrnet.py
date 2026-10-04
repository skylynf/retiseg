"""HRNet-W48 semantic segmentation, the Cityscapes recipe in the official README.

The README's first Cityscapes result is HRNetV2-W48 with the plain segmentation
head, trained from
``experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml``.
This wrapper imports ``HighResolutionNet`` from that repository and changes the
classifier from 19 Cityscapes classes to four sigmoid channels.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from torch import nn
import yaml

from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

REPO = Path(__file__).resolve().parents[2]
_LIB = REPO / "official_code" / "HRNet-Semantic-Segmentation" / "lib"
_CONFIG_PATH = (
    REPO
    / "official_code"
    / "HRNet-Semantic-Segmentation"
    / "experiments"
    / "cityscapes"
    / "seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml"
)
_CONFIG_KEY = (
    "official_code/HRNet-Semantic-Segmentation/experiments/cityscapes/"
    "seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml"
)
# lib/datasets/cityscapes.py defaults, applied in base_dataset.input_transform
# after the image is divided by 255. Already the runner's 0-1 space.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_MODULE_NAME = "hrnet_seg_official.models.seg_hrnet"

_NOTES = (
    "The classifier is 4 sigmoid channels (MA, HE, EX, SE) instead of "
    "Cityscapes' 19 classes. The yaml selects CrossEntropy because "
    "LOSS.USE_OHEM is false; that loss needs one mutually exclusive class index "
    "per pixel. B1 targets are four overlapping float masks, so the recorded "
    "loss is bce_with_logits. IGNORE_LABEL, OHEMTHRES and OHEMKEEP are not "
    "applied, and no positive class weight is added. TRAIN.IMAGE_SIZE is the "
    "author crop and is not applied; forward does not crop. The network is "
    "fully convolutional, so forward_size is empty and B1 keeps the "
    "field-of-view canvas whose longer side is 1440. The yaml has no schedule "
    "key; the training loop for this file uses poly decay with power 0.9. "
    "TRAIN.FLIP and TRAIN.MULTI_SCALE are not applied. "
    "tools/train.py:141 sets drop_last=True and epoch length is integer "
    "division of the image count by the author batch, so an incomplete batch "
    "is not a step. This runner's epoch length is ceil(training images / "
    "effective batch), so the last step of an epoch can contain the remaining "
    "images. The polynomial is the one in function.py, which writes the "
    "learning rate after optimizer.step. SyncBatchNorm is "
    "replaced with BatchNorm2d so one process can run. The author batch is "
    "3 images on each of 4 GPUs. Three images at long side 1440 do not fit "
    "the 512x1024 memory budget, so the loader batch is 2 and gradients "
    "accumulate to 12. BatchNorm therefore sees 2 images. Training loads "
    "pretrained/hrnetv2_w48-d2186c55.pth, or the E1r copy of that file, "
    "before the first step. The constructor used by tests does not."
)


class _Node(dict):
    """Dict that also supports attribute access, as the official config does."""

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if isinstance(value, dict) and not isinstance(value, _Node):
            value = _Node(value)
            dict.__setitem__(self, key, value)
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _document():
    if not hasattr(_document, "cache"):
        _document.cache = yaml.safe_load(_CONFIG_PATH.read_text())
    return _document.cache


def _gpu_ids(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return tuple(value)


def _seg_hrnet_module():
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    # Numpy 2 removed np.int. seg_hrnet.py still calls it while building the head.
    if not hasattr(np, "int"):
        np.int = int
    package = "hrnet_seg_official"
    models = package + ".models"
    root = types.ModuleType(package)
    root.__path__ = [str(_LIB)]
    root.__package__ = package
    sys.modules[package] = root
    models_pkg = types.ModuleType(models)
    models_pkg.__path__ = [str(_LIB / "models")]
    models_pkg.__package__ = models
    sys.modules[models] = models_pkg
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _LIB / "models" / "seg_hrnet.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = models
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _network_config():
    document = _document()
    if document["MODEL"]["NAME"] != "seg_hrnet":
        raise ValueError(f"expected MODEL.NAME seg_hrnet, got {document['MODEL']['NAME']!r}")
    copied = yaml.safe_load(_CONFIG_PATH.read_text())
    copied["DATASET"]["NUM_CLASSES"] = 4
    return _Node(copied)


class HRNet(SegmentationWrapper, nn.Module):
    classes = ("MA", "HE", "EX", "SE")
    card = ModelCard(
        name="HRNet",
        family="general",
        track="B",
        repro_level="R3",
        source="official_code/HRNet-Semantic-Segmentation",
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": _MEAN, "std": _STD},
        pad_multiple=16,
        forward_size=None,
        class_index=(),
    )

    def __init__(self):
        super().__init__()
        module = _seg_hrnet_module()
        # bn_helper selects SyncBatchNorm on PyTorch >= 1. tools/test.py replaces
        # it with BatchNorm2d before building the network on PyTorch 1.x. PyTorch 2
        # does not match that version check, and SyncBatchNorm rejects CPU input.
        module.BatchNorm2d = nn.BatchNorm2d
        module.BatchNorm2d_class = nn.BatchNorm2d
        config = _network_config()
        self._align_corners = bool(config.MODEL.ALIGN_CORNERS)
        self.net = module.HighResolutionNet(config)
        # Empty path: init_weights loads a file only when that path exists.
        self.net.init_weights("")

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
        document = _document()
        train = document["TRAIN"]
        loss = document["LOSS"]
        model = document["MODEL"]
        if loss.get("USE_OHEM") is not False:
            raise ValueError("LOSS.USE_OHEM is not false; this wrapper does not invent an OHEM loss")
        gpus = _gpu_ids(document["GPUS"])
        per_gpu = int(train["BATCH_SIZE_PER_GPU"])
        width, height = (int(train["IMAGE_SIZE"][0]), int(train["IMAGE_SIZE"][1]))
        pretrained = Path(str(model["PRETRAINED"])).name
        num_classes = int(document["DATASET"]["NUM_CLASSES"])
        source = (
            f"{_CONFIG_KEY}: MODEL.NAME={model['NAME']}, MODEL.ALIGN_CORNERS={model['ALIGN_CORNERS']}, "
            f"MODEL.PRETRAINED={model['PRETRAINED']}, MODEL.EXTRA STAGE2 NUM_CHANNELS="
            f"{model['EXTRA']['STAGE2']['NUM_CHANNELS']} (W48), "
            f"DATASET.NUM_CLASSES={num_classes} (head built with 4 sigmoid channels). "
            f"LOSS.USE_OHEM={loss['USE_OHEM']} selects CrossEntropy in tools/train.py lines 194-201, "
            f"with TRAIN.IGNORE_LABEL={train['IGNORE_LABEL']}; "
            f"LOSS.OHEMTHRES={loss['OHEMTHRES']} and LOSS.OHEMKEEP={loss['OHEMKEEP']} are unused. "
            "The recipe loss is bce_with_logits because the head is 4 independent sigmoid channels "
            "and B1 labels are four float masks. "
            f"TRAIN.OPTIMIZER={train['OPTIMIZER']}, TRAIN.LR={train['LR']}, TRAIN.WD={train['WD']}, "
            f"TRAIN.MOMENTUM={train['MOMENTUM']}, TRAIN.NESTEROV={train['NESTEROV']}, "
            f"TRAIN.END_EPOCH={train['END_EPOCH']}, TRAIN.BATCH_SIZE_PER_GPU={train['BATCH_SIZE_PER_GPU']}, "
            f"GPUS={document['GPUS']} ({len(gpus)} devices, author batch {per_gpu * len(gpus)}). "
            f"TRAIN.IMAGE_SIZE={train['IMAGE_SIZE']} is width, height; tools/train.py line 120 "
            f"crops to ({height}, {width}). "
            "The yaml has no schedule key. lib/core/function.py lines 76-79 call "
            "lib/utils/utils.py adjust_learning_rate (lines 132-134), power default 0.9. "
            "mean/std are not yaml keys. lib/datasets/cityscapes.py lines 31-32 and "
            "lib/datasets/base_dataset.py input_transform (lines 44-48) divide by 255, then use "
            "mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]."
        )
        return AuthorRecipe(
            loss="bce_with_logits",
            optimizer=str(train["OPTIMIZER"]),
            lr=float(train["LR"]),
            weight_decay=float(train["WD"]),
            schedule="poly",
            iterations=0,
            epochs=int(train["END_EPOCH"]),
            batch_size=2,
            effective_batch_size=per_gpu * len(gpus),
            crop_size=(height, width),
            pretrained=pretrained,
            optimizer_params={"momentum": float(train["MOMENTUM"]), "nesterov": bool(train["NESTEROV"])},
            notes=_NOTES,
            source_of_settings=source,
        )

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """Match function.py:76-79, which writes the learning rate after optimizer.step."""
        from bench.runtime import adjust_lr_after_step

        return adjust_lr_after_step(optimizer, step, total_steps, recipe)

    def load_pretrained(self):
        from bench.pretrained import hrnet_w48, load_matching

        return load_matching(self.net, hrnet_w48())

    def forward(self, x):
        logits = self.net(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = nn.functional.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=self._align_corners,
            )
        return logits
