"""B-std wrapper for HRDecoder (Ding et al., MICCAI 2024; official_code/HRDecoder), docs/paper/18 section 2.1.

The segmentor, the HRNet-W48 backbone and the FCN head stay in the author's
mmseg fork (official_code/HRDecoder/mmseg, mmcv 1.x API: the retiseg-hacdr
environment). This file reads the author configs into an AuthorRecipe and
re-expresses the author's forward_train / encode_decode at patch level:

    lr path   the input resized to hr_scale -> backbone -> concatenated multi-level features -> head
    hr path   the low-resolution feature map cropped at hr_scale windows of the input, resized back to the
              feature size, decoded; at inference the windows tile the input (hr_slide_inference), in training
              crop_num random windows of a random scale in scale_ratio x hr_scale are decoded as well
    output    fused logits: the upsampled lr logits averaged with the hr window logits where a window lies
              (encode_decode); the training loss is Dice on the fused map + 0.1 x Dice on the random hr crops
              (hr_loss_weight; lr_loss_weight is 0)

Author geometry: IDRiD images 2880x1920, random 1920 crops, hr_scale 960,
crop_num 2, 20k iterations; DDR 2048, hr_scale 1024, crop_num 4, 40k. The
backbone therefore runs at half the crop's resolution by design. B-std keeps
that ratio: 960 patches, hr_scale 480, the author's crop_num; inference with
960 windows (overlap 0.5), never the whole 1440 canvas (which the author's
resize would squash to a third). So HRDecoder's backbone sees half the pixel
scale of the HRNet-W48 row while its decoder works at the full scale: that
is the method, and the backbone-controlled comparison of 18 section 1.2.

Head: four sigmoids, author classes bg, EX, HE, SE, MA -> channels EX, HE,
SE, MA, permuted to MA, HE, EX, SE. Loss BinaryLoss dice, smooth 1e-5 (per
image and class, squared denominators in the author code; bench.runtime's
Dice as for M2MRF). Optimizer SGD 0.01, momentum 0.9, weight decay 5e-4,
poly power 1.0 with 1,500 warmup iterations at ratio 1e-6 (poly10warm.py,
from DAFormer); samples_per_gpu 1 on 4 GPUs -> effective batch 4. SyncBN
becomes BN (one process); ImageNet HRNet-W48 from pretrained/ as the HRNet
and M2MRF rows. The mmseg import happens only when a network is built.
"""

import sys
from pathlib import Path

import torch
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper
from bench.models.foundation import set_lr

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "official_code" / "HRDecoder"
CONFIGS = {
    "IDRiD": OFFICIAL / "configs" / "lesion" / "hrdecoder_fcn_hr48_idrid_2880x1920-slide.py",
    "DDR": OFFICIAL / "configs" / "lesion" / "hrdecoder_fcn_hr48_ddr_2048.py",
}
BASE = OFFICIAL / "configs" / "_base_"
HR18, HR48 = BASE / "models" / "hrdecoder_fcn_hr18.py", BASE / "models" / "hrdecoder_fcn_hr48.py"
SGD, POLY = BASE / "schedules" / "sgd.py", BASE / "schedules" / "poly10warm.py"
DATASETS = {"IDRiD": BASE / "datasets" / "hr_idrid_2880x1920-slide.py", "DDR": BASE / "datasets" / "hr_ddr_2048.py"}
AUTHOR_LESIONS = ("EX", "HE", "SE", "MA")
CHANNEL_INDEX = tuple(AUTHOR_LESIONS.index(name) for name in LESION_CLASSES)
PATCH = 960
BACKBONE_RATIO = 0.5          # hr_scale / crop: 960/1920 (IDRiD), 1024/2048 (DDR)
HR_SCALE = (int(PATCH * BACKBONE_RATIO), int(PATCH * BACKBONE_RATIO))
GPUS = 4                      # README: tools/dist_train.sh <config> 4


def _exec_config(path):
    namespace = {}
    exec(compile(Path(path).read_text(), str(path), "exec"), namespace)
    return namespace


def _deep_merge(base, update):
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _bn(cfg):
    """SyncBN -> BN everywhere in a config dict (one process)."""
    if isinstance(cfg, dict):
        out = {}
        for key, value in cfg.items():
            if key == "norm_cfg" and isinstance(value, dict) and value.get("type") == "SyncBN":
                out[key] = {**value, "type": "BN"}
            else:
                out[key] = _bn(value)
        return out
    if isinstance(cfg, (list, tuple)):
        return type(cfg)(_bn(v) for v in cfg)
    return cfg


def author_model_dict(dataset):
    cfg = _deep_merge(_exec_config(HR18)["model"], _exec_config(HR48)["model"])
    return _deep_merge(cfg, _exec_config(CONFIGS[dataset])["model"])


def author_norm(dataset):
    norm = _exec_config(DATASETS[dataset])["img_norm_cfg"]
    return [float(v) for v in norm["mean"]], [float(v) for v in norm["std"]]


def author_recipe(dataset):
    if dataset not in CONFIGS:
        raise ValueError(f"dataset must be IDRiD or DDR, got {dataset!r}")
    model = author_model_dict(dataset)
    sgd = _exec_config(SGD)["optimizer"]
    lr_cfg = _exec_config(POLY)["lr_config"]
    lesion = _exec_config(CONFIGS[dataset])
    data = _exec_config(DATASETS[dataset])
    hr = model["hr_settings"]
    loss = model["decode_head"]["loss_decode"]
    if loss.get("type") != "BinaryLoss" or loss.get("loss_type") != "dice":
        raise RuntimeError(f"{dataset}: decode loss is not binary dice: {loss}")
    if tuple(data["classes"]) != ("bg",) + AUTHOR_LESIONS:
        raise RuntimeError(f"{dataset}: classes {data['classes']} are not bg, EX, HE, SE, MA")
    mean, std = author_norm(dataset)
    crop = tuple(int(v) for v in data["crop_size"])
    return AuthorRecipe(
        loss="dice",
        optimizer=str(sgd["type"]).lower(),
        lr=float(sgd["lr"]),
        weight_decay=float(sgd["weight_decay"]),
        schedule="poly",
        iterations=int(lesion["runner"]["max_iters"]),
        batch_size=int(lesion["data"]["samples_per_gpu"]) * GPUS,
        effective_batch_size=int(lesion["data"]["samples_per_gpu"]) * GPUS,
        crop_size=crop,
        pretrained=str(model["pretrained"]),
        loss_params={"smooth": float(loss["smooth"]), "hr_loss_weight": float(hr["hr_loss_weight"]),
                     "lr_loss_weight": float(hr.get("lr_loss_weight", 0))},
        optimizer_params={"momentum": float(sgd["momentum"]), "power": float(lr_cfg["power"]), "min_lr": float(lr_cfg["min_lr"]),
                          "warmup_iters": int(lr_cfg["warmup_iters"]), "warmup_ratio": float(lr_cfg["warmup_ratio"]),
                          "hr_scale_author": list(hr["hr_scale"]), "crop_num": int(hr["crop_num"]),
                          "scale_ratio": list(hr["scale_ratio"]), "divisible": int(hr.get("divisible", 8))},
        notes=(
            f"B-std geometry: {PATCH} patches with hr_scale {HR_SCALE} (the author's backbone/crop ratio {BACKBONE_RATIO}: "
            f"{list(hr['hr_scale'])} on {crop}), crop_num {hr['crop_num']}, scale_ratio {list(hr['scale_ratio'])}; inference "
            f"with {PATCH} windows, overlap 0.5. Loss: Dice on the fused map + {hr['hr_loss_weight']} x Dice on the random "
            "hr crops (forward_train); lr_loss_weight 0. SyncBN -> BN; 4 GPUs x 1 image -> one process accumulating to 4; "
            "the author resize/crop/flip/rotate pipeline replaced by the shared augmentation; author RGB mean "
            f"{mean} std {std} (0-255) divided by 255 for the runner; 1500-step linear warmup from ratio 1e-6 kept, "
            "then poly power 1.0 over the B-std steps."
        ),
        source_of_settings=(
            f"{CONFIGS[dataset].relative_to(REPO)} (hr_settings, runner.max_iters, samples_per_gpu); {HR18.relative_to(REPO)} "
            f"(decode_head FCNHead compress, BinaryLoss dice smooth 1e-5, use_sigmoid); {SGD.relative_to(REPO)}; "
            f"{POLY.relative_to(REPO)}; {DATASETS[dataset].relative_to(REPO)} (classes, img_norm_cfg, crop_size); README (4 GPUs)"
        ),
    )


class _Cfg(dict):
    """A dict the author code can read with attributes (backbone.type, hr_settings.hr_scale)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _as_cfg(cfg):
    if isinstance(cfg, dict):
        return _Cfg({key: _as_cfg(value) for key, value in cfg.items()})
    if isinstance(cfg, list):
        return [_as_cfg(value) for value in cfg]
    if isinstance(cfg, tuple):
        return tuple(_as_cfg(value) for value in cfg)
    return cfg


def _install_import_path():
    if not OFFICIAL.is_dir():
        raise FileNotFoundError(f"{OFFICIAL} is missing; git submodule update --init official_code/HRDecoder")
    official = str(OFFICIAL)
    loaded = sys.modules.get("mmseg")
    if loaded is not None and str(getattr(loaded, "__file__", "")).startswith(official + "/"):
        return
    while official in sys.path:
        sys.path.remove(official)
    sys.path.insert(0, official)
    for name in [m for m in sys.modules if m == "mmseg" or m.startswith("mmseg.")]:
        del sys.modules[name]
    import numpy as np

    if not hasattr(np, "float"):
        np.float = np.float64


def hr_scale_for(patch):
    """hr_scale of a fixed window: the author's backbone/crop ratio (960 on 1920 -> 480 on 960, 960 on 1920)."""
    return (int(patch * BACKBONE_RATIO), int(patch * BACKBONE_RATIO))


def build_network(dataset="IDRiD", patch=PATCH):
    """The author's HRDecoder segmentor, SyncBN -> BN, no pretrained download, hr_scale for ``patch`` (B-std: 960)."""
    _install_import_path()
    from mmseg.models import build_segmentor

    cfg = _as_cfg(_bn(author_model_dict(dataset)))
    cfg["pretrained"] = None
    hr = dict(cfg["hr_settings"])
    hr["hr_scale"] = hr_scale_for(patch)
    cfg["hr_settings"] = _as_cfg(hr)
    test_cfg = cfg.pop("test_cfg", _Cfg({"mode": "whole"}))
    train_cfg = cfg.pop("train_cfg", _Cfg({}))
    return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


class _FusedPlusCropDice(nn.Module):
    """Dice(fused logits, target) + hr_loss_weight x mean Dice(random hr crop logits, target crops)."""

    def __init__(self, wrapper, smooth, hr_weight):
        super().__init__()
        from bench.runtime import build_loss

        self.wrapper = wrapper
        self.dice = build_loss(AuthorRecipe("dice", "sgd", 0.0, 0.0, "none", 0, 1, (0, 0), "", loss_params={"smooth": smooth}), "sigmoid")
        self.hr_weight = float(hr_weight)

    def forward(self, logits, target):
        target = target.float()
        total = self.dice(logits, target)
        aux = self.wrapper._aux
        if aux is None or aux["fused"] is not logits or self.hr_weight == 0:
            return total
        crops = []
        for crop_logits, (y1, y2, x1, x2) in zip(aux["hr_logits"], aux["hr_boxes"]):
            resized = nn.functional.interpolate(crop_logits[None], size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)[0]
            crops.append(self.dice(resized[None][:, CHANNEL_INDEX], target[aux["hr_index"][len(crops)] : aux["hr_index"][len(crops)] + 1, :, y1:y2, x1:x2]))
        return total + self.hr_weight * torch.stack(crops).mean()


class HRDecoder(SegmentationWrapper, nn.Module):
    classes = tuple(LESION_CLASSES)
    card = ModelCard(
        name="HRDecoder",
        family="fundus_multiscale",
        track="B",
        repro_level="R4",
        source="official_code/HRDecoder",
        output_activation="sigmoid",
        in_channels=3,
        normalization={"mean": [116.513 / 255.0, 56.437 / 255.0, 16.309 / 255.0],
                       "std": [80.206 / 255.0, 41.232 / 255.0, 13.293 / 255.0]},
        pad_multiple=32,
        forward_size=(PATCH, PATCH),   # fixed window: the internal resize to hr_scale is defined for the patch
        class_index=(),
    )

    def __init__(self, *, construct=True):
        super().__init__()
        self.network = None
        self.import_error = None
        self._aux = None
        self._dataset = "IDRiD"
        self._patch = PATCH
        if not construct:
            return
        try:
            self.network = build_network(self._dataset, self._patch)
        except Exception as exc:  # the recipe stays readable without mmcv (launch_b1's gate)
            self.import_error = f"{type(exc).__name__}: {exc}"

    @classmethod
    def recipe_for(cls, dataset):
        return author_recipe(dataset)

    def author_recipe(self, dataset):
        from dataclasses import replace

        recipe = author_recipe(dataset)
        mean, std = author_norm(dataset)
        # the card keeps the window of set_patch: until 2026-10-22 it was rebuilt from the class card, so a cell's
        # patch (26 N1c) was lost between apply_cell_geometry and the budget, which then counted 960 patches
        # while the network's hr_scale was 960 (docs/ops/21 #82)
        self.card = replace(type(self).card, normalization={"mean": [v / 255.0 for v in mean], "std": [v / 255.0 for v in std]},
                            forward_size=(self._patch, self._patch))
        if self.network is not None and dataset != self._dataset:
            self.network = build_network(dataset, self._patch)   # crop_num differs between the two configs
        self._dataset = dataset
        return recipe

    def set_patch(self, patch):
        """E7 case study (docs/paper/26 N1c): the author's window on the author's canvas. The fixed window becomes
        ``patch`` x ``patch`` (card.forward_size: training patch and inference window) and the network's hr_scale
        follows the author's ratio (1920 -> 960, the released IDRiD geometry). No parameter depends on hr_scale."""
        from dataclasses import replace

        patch = int(patch)
        if patch <= 0 or patch % 32:
            raise ValueError(f"patch must be a positive multiple of 32, got {patch}")
        self._patch = patch
        self.card = replace(self.card, forward_size=(patch, patch))
        if self.network is not None:
            self.network.hr_scale = hr_scale_for(patch)
        return self.card.forward_size

    def load_pretrained(self):
        """ImageNet HRNet-W48 into the backbone (the file bench.pretrained names for the HRNet and M2MRF rows)."""
        from bench.pretrained import hrnet_w48

        if self.network is None:
            raise RuntimeError(self.import_error or "HRDecoder network was not imported")
        path = hrnet_w48()
        state = torch.load(str(path), map_location="cpu")
        state = state.get("state_dict", state)
        missing, unexpected = self.network.backbone.load_state_dict(state, strict=False)
        copied = len(state) - len(unexpected)
        if copied == 0:
            raise RuntimeError(f"no backbone tensor matched {path}")
        return copied

    def _lr_features(self, x):
        net = self.network
        from mmseg.ops import resize

        lr_img = resize(x, net.hr_scale, mode="bilinear")
        return net._transform_inputs(net.extract_feat(lr_img))

    def forward(self, x):
        if self.network is None:
            raise RuntimeError(self.import_error or "HRDecoder network was not imported")
        from mmseg.ops import resize

        net = self.network
        img_size = tuple(x.shape[2:])
        lr_feat = self._lr_features(x)
        lr_logit = net.decode_head([lr_feat])
        up_lr = resize(lr_logit, img_size, mode="bilinear")
        fused = up_lr.clone()
        slide = net.hr_slide_inference(img_size, lr_feat, None)
        for logit, _feat, box in zip(slide["logits"], slide["feats"], slide["boxes"]):
            fused[:, :, box[0]:box[1], box[2]:box[3]] = (logit + up_lr[:, :, box[0]:box[1], box[2]:box[3]]) / 2
        out = fused[:, CHANNEL_INDEX]
        self._aux = None
        if self.training:
            # hr_forward_train: crop_num random windows of a random scale, decoded from the lr feature map; the
            # loss module scores them against the same crops of the target (author: np.random, not the torch seed)
            from mmseg.models.segmentors.HRDecoder import get_crop_bbox

            n = x.shape[0]
            hr_scale = net.get_random_hr_scale()
            boxes = [get_crop_bbox(img_size, hr_scale, net.divisible) for _ in range(net.crop_num)]
            hr_feats = torch.cat([net._resizecrop_feats(img_size, lr_feat, box) for box in boxes])
            hr_logits = net.decode_head([hr_feats])
            self._aux = {"fused": out, "hr_logits": hr_logits, "hr_boxes": [box for box in boxes for _ in range(n)],
                         "hr_index": [index for _box in boxes for index in range(n)]}
        return out

    def build_loss(self, recipe):
        return _FusedPlusCropDice(self, float(recipe.loss_params["smooth"]), float(recipe.loss_params["hr_loss_weight"]))

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        """poly10warm: linear warmup from warmup_ratio over warmup_iters, then (1 - step/total)^power to min_lr."""
        base = float(recipe.lr)
        warmup = int(recipe.optimizer_params["warmup_iters"])
        ratio = float(recipe.optimizer_params["warmup_ratio"])
        power = float(recipe.optimizer_params["power"])
        floor = float(recipe.optimizer_params["min_lr"])
        if step < warmup:
            lr = base * (ratio + (1.0 - ratio) * step / float(warmup))
        else:
            lr = (base - floor) * (1.0 - min(1.0, step / float(max(1, total_steps)))) ** power + floor
        set_lr(optimizer, lr)
