"""B1 wrapper for HACDR-Net.

The network and the loss modules stay in official_code/HACDR-Net. This file
reads HACDR_idrid.py and HACDR_ddr.py. It does not import mmseg until a
network is built, so the recipe can be read before mmcv is installed.
official_code/ is not edited. No checkpoint is downloaded: both configs set
pretrained=None and load_from=None.

The decode head is exclusive softmax, num_classes=5, CrossEntropyLoss
use_sigmoid=False. Both configs name the dataset FGADRDataset. That class
lists CLASSES as background, EX, MA, SE, HE and sets reduce_zero_label=False,
so those names are channel ids 0, 1, 2, 3, 4. Prediction keeps that head and
writes MA, HE, EX, SE from channels 2, 4, 1, 3.
"""

import ast
import copy
import sys
from pathlib import Path

import torch
from torch import nn

from bench.common.io import LESION_CLASSES
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "official_code" / "HACDR-Net"
CONFIGS = {
    "IDRiD": OFFICIAL / "HACDRNet&DDHANet" / "HACDR_idrid.py",
    "DDR": OFFICIAL / "HACDRNet&DDHANet" / "HACDR_ddr.py",
}
FGADR_PY = OFFICIAL / "mmseg" / "datasets" / "FGADR.py"
DECODE_HEAD_PY = OFFICIAL / "mmseg" / "models" / "decode_heads" / "decode_head.py"
NALOSS_PY = OFFICIAL / "mmseg" / "models" / "losses" / "NALoss.py"
# Last CLASSES assignment in FGADRDataset. The commented list above it is not executed.
FGADR_CLASSES = ("background", "EX", "MA", "SE", "HE")
DATASETS = ("IDRiD", "DDR")


def _exec_config(path):
    path = Path(path)
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def _line(path, needle):
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if needle in line and not line.lstrip().startswith("#"):
            return number, line.strip()
    raise RuntimeError(f"{needle!r} not found in {path}")


def _cite(path, needle, label):
    number, line = _line(path, needle)
    relative = Path(path).resolve().relative_to(REPO)
    return f"{label}: {relative}:{number}: {line}"


def _pipeline_step(pipeline, step_type):
    found = [step for step in pipeline if step.get("type") == step_type]
    if len(found) != 1:
        raise RuntimeError(f"expected one {step_type} in the train pipeline, found {len(found)}")
    return found[0]


def fgadr_class_spec():
    """Active CLASSES and reduce_zero_label from mmseg/datasets/FGADR.py.

    Stops when the class list is not the five names this wrapper maps. Channel
    ids are not filled in from anywhere else.
    """
    tree = ast.parse(FGADR_PY.read_text())
    classes = None
    reduce_zero = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "FGADRDataset":
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "CLASSES":
                        classes = tuple(ast.literal_eval(stmt.value))
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.keyword) and sub.arg == "reduce_zero_label":
                        reduce_zero = ast.literal_eval(sub.value)
    if classes is None or reduce_zero is None:
        raise RuntimeError(
            "FGADRDataset has no CLASSES assignment or reduce_zero_label; "
            "lesion channel ids are not guessed"
        )
    if reduce_zero is not False:
        raise RuntimeError(
            f"FGADRDataset reduce_zero_label={reduce_zero!r}; class ids would shift and are not guessed"
        )
    if classes != FGADR_CLASSES or len(set(classes)) != len(classes):
        raise RuntimeError(
            f"FGADRDataset.CLASSES is {classes!r}, not {FGADR_CLASSES!r}; lesion channel ids are not guessed"
        )
    return classes


def lesion_class_index():
    """Channel ids of MA, HE, EX, SE inside FGADRDataset.CLASSES."""
    classes = fgadr_class_spec()
    missing = [name for name in LESION_CLASSES if name not in classes]
    if missing:
        raise RuntimeError(f"FGADRDataset.CLASSES is missing {missing}; lesion channel ids are not guessed")
    return tuple(classes.index(name) for name in LESION_CLASSES)


def author_rgb_norm(dataset):
    """RGB mean and std as written in the train Normalize step, on 0-255."""
    if dataset not in CONFIGS:
        raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
    namespace = _exec_config(CONFIGS[dataset])
    step = _pipeline_step(namespace["data"]["train"]["pipeline"], "Normalize")
    if step.get("to_rgb") is not True:
        raise RuntimeError(f"{dataset} Normalize to_rgb is not True")
    return list(step["mean"]), list(step["std"])


def b1_normalization(dataset="IDRiD"):
    """Author 0-255 RGB stats divided by 255, which is what b1_input.normalize applies.

    b1_input divides the JPEG by 255 before subtracting the mean. Dividing the
    author's stats by 255 leaves (pixel - mean) / std unchanged. Both configs
    use the same mean and std. Prepared images are already RGB, so to_rgb is
    not applied a second time.
    """
    stats = [author_rgb_norm(name) for name in DATASETS]
    if stats[0] != stats[1]:
        raise RuntimeError(f"IDRiD and DDR Normalize stats differ: {stats[0]} vs {stats[1]}")
    if dataset == "IDRiD":
        mean, std = stats[0]
    elif dataset == "DDR":
        mean, std = stats[1]
    else:
        raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
    return [float(value) / 255.0 for value in mean], [float(value) / 255.0 for value in std]


def _loss_decode(namespace, dataset):
    specs = namespace["model"]["decode_head"]["loss_decode"]
    if isinstance(specs, dict):
        specs = [specs]
    specs = copy.deepcopy(list(specs))
    if not specs:
        raise RuntimeError(f"{dataset} loss_decode is empty")
    num_classes = int(namespace["model"]["decode_head"]["num_classes"])
    if num_classes != len(FGADR_CLASSES):
        raise RuntimeError(
            f"{dataset} num_classes={num_classes} does not match FGADRDataset.CLASSES {FGADR_CLASSES}"
        )
    saw_ce = False
    for spec in specs:
        if spec.get("type") != "CrossEntropyLoss":
            continue
        saw_ce = True
        if spec.get("use_sigmoid", False):
            raise RuntimeError(f"{dataset} CrossEntropyLoss use_sigmoid is true; this is not an exclusive softmax head")
        weight = spec.get("class_weight")
        if weight is not None and len(weight) != num_classes:
            raise RuntimeError(f"{dataset} class_weight length {len(weight)} != num_classes {num_classes}")
    if not saw_ce:
        raise RuntimeError(f"{dataset} loss_decode has no CrossEntropyLoss")
    for split in ("train", "val", "test"):
        kind = namespace["data"][split]["type"]
        if kind != "FGADRDataset":
            raise RuntimeError(
                f"{dataset} {split} dataset type is {kind!r}, not FGADRDataset; "
                "lesion channel ids are not guessed"
            )
    return specs


def _architecture(namespace):
    model = copy.deepcopy(namespace["model"])
    model["decode_head"].pop("loss_decode", None)
    return model


def _author_crop(namespace):
    step = _pipeline_step(namespace["data"]["train"]["pipeline"], "RandomCrop")
    crop = tuple(int(value) for value in step["crop_size"])
    if "crop_size" in namespace:
        named = tuple(int(value) for value in namespace["crop_size"])
        if named != crop:
            raise RuntimeError(f"named crop_size {named} != RandomCrop crop_size {crop}")
    return crop


def author_recipe(dataset: str) -> AuthorRecipe:
    if dataset not in CONFIGS:
        raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
    fgadr_class_spec()
    namespaces = {name: _exec_config(CONFIGS[name]) for name in DATASETS}
    if _architecture(namespaces["IDRiD"]) != _architecture(namespaces["DDR"]):
        raise RuntimeError("IDRiD and DDR HACDR model dicts differ outside loss_decode")
    namespace = namespaces[dataset]
    if namespace["model"].get("pretrained") is not None or namespace.get("load_from") is not None:
        raise RuntimeError(f"{dataset} sets pretrained or load_from; refusing to load weights")
    if namespace["runner"].get("type") != "IterBasedRunner":
        raise RuntimeError(f"{dataset} runner is not IterBasedRunner")
    lr_config = namespace["lr_config"]
    if lr_config.get("by_epoch") is not False or str(lr_config.get("policy", "")).lower() != "poly":
        raise RuntimeError(f"{dataset} lr_config is not iter-based poly")
    if str(lr_config.get("warmup", "")).lower() != "linear":
        raise RuntimeError(f"{dataset} warmup is not linear")
    gpu_ids = list(namespace["gpu_ids"])
    if len(gpu_ids) != 1:
        raise RuntimeError(f"{dataset} gpu_ids={gpu_ids} is not one GPU; the batch is not guessed")
    specs = _loss_decode(namespace, dataset)
    optimizer = namespace["optimizer"]
    samples = int(namespace["data"]["samples_per_gpu"])
    crop = _author_crop(namespace)
    mean, std = author_rgb_norm(dataset)
    iterations = int(namespace["runner"]["max_iters"])
    path = CONFIGS[dataset]
    citations = [
        _cite(path, "num_classes=5", "num_classes"),
        _cite(path, "use_sigmoid=False", "use_sigmoid"),
        _cite(path, "loss_weight=1.0", "ce loss_weight"),
        _cite(path, "class_weight=[0.5, 0.7, 1.5, 0.7, 1.1]", "class_weight"),
        _cite(path, "type='FGADRDataset'", "dataset"),
        _cite(FGADR_PY, 'CLASSES = ["background", "EX","MA", "SE", "HE"]', "CLASSES"),
        _cite(FGADR_PY, "reduce_zero_label=False", "reduce_zero_label"),
        _cite(DECODE_HEAD_PY, "ignore_index=255", "ignore_index"),
        _cite(path, "samples_per_gpu=1", "samples_per_gpu"),
        _cite(path, "gpu_ids = [5]", "gpu_ids"),
        _cite(path, "type='RandomCrop'", "RandomCrop"),
        _cite(path, "mean=[123.675, 116.28, 103.53]", "mean"),
        _cite(path, "std=[58.395, 57.12, 57.375]", "std"),
        _cite(path, "to_rgb=True", "to_rgb"),
        _cite(path, "type='AdamW'", "AdamW"),
        _cite(path, "lr=6e-05", "lr"),
        _cite(path, "betas=(0.9, 0.999)", "betas"),
        _cite(path, "weight_decay=0.01", "weight_decay"),
        _cite(path, "pos_block=dict(decay_mult=0.0)", "pos_block"),
        _cite(path, "norm=dict(decay_mult=0.0)", "norm"),
        _cite(path, "head=dict(lr_mult=15.0)", "head lr_mult"),
        _cite(path, "policy='poly'", "poly"),
        _cite(path, "warmup='linear'", "warmup"),
        _cite(path, "warmup_iters=1500", "warmup_iters"),
        _cite(path, "warmup_ratio=1e-06", "warmup_ratio"),
        _cite(path, "power=1.0", "power"),
        _cite(path, "min_lr=0.0", "min_lr"),
        _cite(path, "max_iters=40000", "max_iters"),
        _cite(path, "pretrained=None", "pretrained"),
        _cite(path, "load_from = None", "load_from"),
    ]
    if dataset == "DDR":
        citations.extend(
            [
                _cite(path, "type='NALoss'", "NALoss"),
                _cite(path, "loss_weight=0.2", "NALoss loss_weight"),
                _cite(path, "thres=0.5", "NALoss thres"),
                _cite(path, "sigma=4", "NALoss sigma"),
                _cite(NALOSS_PY, "torch.cuda.FloatTensor", "NALoss CUDA tensor"),
            ]
        )
    loss_text = ", ".join(f"{spec['type']} loss_weight={spec['loss_weight']}" for spec in specs)
    notes = (
        "B1 deviations from the author train pipeline: "
        f"images are not resized with ratio_range and are not RandomCrop'd to crop_size {crop}. "
        "PhotoMetricDistortion is not used. "
        "The field-of-view crop keeps its aspect ratio and its longer side is 1440, "
        "then a horizontal flip with probability 0.5. "
        f"Author RandomCrop crop_size is {crop}, copied as written and not reinterpreted as height, width. "
        f"loss_decode is {loss_text}. "
        "The head stays exclusive softmax (CrossEntropyLoss use_sigmoid=False, num_classes=5). "
        "FGADRDataset.CLASSES is background, EX, MA, SE, HE with reduce_zero_label=False, "
        "so the MA, HE, EX, SE channels are (2, 4, 1, 3). "
        "Four-plane targets overlap in M2MRF order EX, HE, SE, MA, so MA is kept. "
        "Empty pixels stay background, class 0. "
        f"samples_per_gpu is {samples} and gpu_ids has one entry, so the effective batch is {samples}. "
        "The learning rate stays at the configured value. "
        "A loader batch smaller than 1 cannot be accumulated back up to that effective batch. "
        f"Author RGB mean {mean} and std {std} are 0-255. "
        "b1_input.normalize divides the image by 255 first, so the card stores those stats divided by 255 "
        "and the affine map is unchanged. "
        "Prepared images are already RGB, so to_rgb=True is not applied a second time. "
        "pretrained and load_from are None. No lesion weights are downloaded."
    )
    if dataset == "DDR":
        notes += (
            " NALoss.__init__ builds a torch.cuda.FloatTensor, so constructing the DDR loss needs a CUDA device. "
            "That line is left as the author wrote it."
        )
    loss_name = "ce" if len(specs) == 1 else "ce_naloss"
    return AuthorRecipe(
        loss=loss_name,
        optimizer=str(optimizer["type"]).lower(),
        lr=float(optimizer["lr"]),
        weight_decay=float(optimizer["weight_decay"]),
        schedule=str(lr_config["policy"]).lower(),
        iterations=iterations,
        epochs=0,
        batch_size=samples,
        effective_batch_size=samples,
        crop_size=crop,
        pretrained="none",
        loss_params={
            "loss_decode": specs,
            "ignore_index": 255,
        },
        optimizer_params={
            "betas": [float(value) for value in optimizer["betas"]],
            "power": float(lr_config["power"]),
            "min_lr": float(lr_config["min_lr"]),
            "warmup": str(lr_config["warmup"]),
            "warmup_iters": int(lr_config["warmup_iters"]),
            "warmup_ratio": float(lr_config["warmup_ratio"]),
            "paramwise_cfg": copy.deepcopy(optimizer["paramwise_cfg"]),
        },
        notes=notes,
        source_of_settings="\n".join(citations),
    )


def masks_to_label(target, class_index):
    """Integer map for the exclusive head.

    ``class_index`` is the network channel of MA, HE, EX, SE. Overlaps follow
    M2MRF order, so MA replaces EX, HE and SE. Pixels with no lesion stay 0,
    which FGADRDataset calls background.
    """
    from bench.common.labels import exclusive_label

    return exclusive_label(target, class_index)


class _AuthorLoss(nn.Module):
    """Sum of the official decode losses, each already multiplied by its loss_weight."""

    def __init__(self, modules, class_index, ignore_index=255):
        super().__init__()
        self.loss_modules = nn.ModuleList(modules)
        self.class_index = tuple(int(value) for value in class_index)
        self.ignore_index = int(ignore_index)

    def forward(self, logits, target):
        label = masks_to_label(target, self.class_index)
        if logits.shape[-2:] != label.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        total = logits.new_zeros(())
        for module in self.loss_modules:
            total = total + module(logits, label, weight=None, ignore_index=self.ignore_index)
        return total


def _import_mmseg():
    compat = str(REPO / "bench" / "compat")
    while compat in sys.path:
        sys.path.remove(compat)
    loaded = sys.modules.get("mmcv")
    if loaded is not None:
        origin = str(getattr(loaded, "__file__", "") or "").replace("\\", "/")
        if "bench/compat/mmcv" in origin:
            raise ImportError(
                "mmcv currently loaded is bench/compat/mmcv, not the mmcv HACDR-Net needs. "
                "Build HACDR-Net in a process that has not imported the M2MRF compatibility package."
            )
    official = str(OFFICIAL)
    if official not in sys.path:
        sys.path.insert(0, official)
    try:
        from mmseg.models import build_loss, build_segmentor
    except ImportError as exc:
        missing = getattr(exc, "name", None) or exc
        raise ImportError(
            f"HACDR-Net official network did not import; missing module {missing}. "
            "Vendored mmseg is 0.30.0 (official_code/HACDR-Net/mmseg/version.py) and requires "
            "installed mmcv>=1.3.13,<1.8.0 (models.yaml notes mmcv-full 1.6.2). "
            "official_code was not modified and no weights were downloaded."
        ) from exc
    return build_segmentor, build_loss


def build_network():
    """EncoderDecoder from the IDRiD config. DDR uses the same architecture.

    The two model dicts differ only in loss_decode. The DDR loss is built from
    the DDR recipe, not by putting NALoss inside this module. pretrained stays
    None, so this is a random initialization.
    """
    build_segmentor, _ = _import_mmseg()
    namespaces = {name: _exec_config(CONFIGS[name]) for name in DATASETS}
    if _architecture(namespaces["IDRiD"]) != _architecture(namespaces["DDR"]):
        raise RuntimeError("IDRiD and DDR HACDR model dicts differ outside loss_decode")
    cfg = copy.deepcopy(namespaces["IDRiD"]["model"])
    if cfg.get("pretrained") is not None:
        raise RuntimeError("refusing to load HACDR pretrained weights")
    return build_segmentor(cfg)


def build_author_loss(recipe, class_index):
    _, build_loss = _import_mmseg()
    modules = []
    for spec in recipe.loss_params["loss_decode"]:
        modules.append(build_loss(copy.deepcopy(spec)))
    ignore_index = int(recipe.loss_params.get("ignore_index", 255))
    return _AuthorLoss(modules, class_index, ignore_index=ignore_index)


def hacdr_param_groups(named_params, base_lr, base_wd, paramwise_cfg):
    """mmcv 1.6.2 DefaultOptimizerConstructor custom_keys, longest key first.

    Equal-length keys keep alphabetical order. The first matching key wins, so
    ``head`` is chosen over ``norm`` when a name contains both.
    """
    custom = {} if paramwise_cfg is None else paramwise_cfg.get("custom_keys", {})
    sorted_keys = sorted(sorted(custom), key=len, reverse=True)
    groups = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        lr = float(base_lr)
        weight_decay = float(base_wd)
        for key in sorted_keys:
            if key in name:
                lr = float(base_lr) * float(custom[key].get("lr_mult", 1.0))
                weight_decay = float(base_wd) * float(custom[key].get("decay_mult", 1.0))
                break
        groups.append({"params": [param], "lr": lr, "weight_decay": weight_decay, "initial_lr": lr})
    if not groups:
        raise RuntimeError("HACDR-Net has no trainable parameters")
    return groups


def hacdr_optimizer(module, params, recipe):
    owned = {id(param) for param in params}
    named = [(name, param) for name, param in module.named_parameters() if id(param) in owned]
    groups = hacdr_param_groups(
        named,
        recipe.lr,
        recipe.weight_decay,
        recipe.optimizer_params.get("paramwise_cfg"),
    )
    betas = tuple(float(value) for value in recipe.optimizer_params["betas"])
    return torch.optim.AdamW(groups, lr=float(recipe.lr), betas=betas, weight_decay=float(recipe.weight_decay))


def adjust_hacdr_lr(optimizer, step, total_steps, recipe):
    """Iter poly with linear warmup, as mmcv 1.6.2 applies it before the update.

    ``step`` is 0-based. For step < warmup_iters the poly learning rate is
    multiplied by warmup_ratio + (1 - warmup_ratio) * step / warmup_iters.
    min_lr is absolute, not multiplied by a param group's lr_mult. The values
    come from the author lr_config.
    """
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    power = float(recipe.optimizer_params["power"])
    min_lr = float(recipe.optimizer_params["min_lr"])
    warmup = str(recipe.optimizer_params["warmup"]).lower()
    warmup_iters = int(recipe.optimizer_params["warmup_iters"])
    warmup_ratio = float(recipe.optimizer_params["warmup_ratio"])
    if warmup != "linear":
        raise ValueError(f"HACDR warmup is linear, got {warmup!r}")
    progress = min(float(step), float(total_steps))
    coeff = (1.0 - progress / float(total_steps)) ** power
    for group in optimizer.param_groups:
        base = float(group.setdefault("initial_lr", group["lr"]))
        regular = (base - min_lr) * coeff + min_lr
        if warmup_iters > 0 and int(step) < warmup_iters:
            factor = 1.0 - (1.0 - float(step) / float(warmup_iters)) * (1.0 - warmup_ratio)
            group["lr"] = regular * factor
        else:
            group["lr"] = regular
    return float(optimizer.param_groups[0]["lr"])


_MEAN, _STD = b1_normalization("IDRiD")
_CLASS_INDEX = lesion_class_index()


class HACDRNet(SegmentationWrapper, nn.Module):
    classes = LESION_CLASSES
    card = ModelCard(
        name="HACDR-Net",
        family="fundus_multiscale",
        track="B",
        repro_level="R3",
        source="official_code/HACDR-Net",
        output_activation="softmax",
        in_channels=3,
        normalization={"mean": _MEAN, "std": _STD},
        # StemConv is two stride-2 convolutions, then three stride-2 patch embeds.
        pad_multiple=32,
        forward_size=None,
        class_index=_CLASS_INDEX,
    )

    def __init__(self):
        super().__init__()
        self.net = None
        self.import_error = None
        try:
            self.net = build_network()
        except ImportError as exc:
            self.import_error = str(exc)

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        return author_recipe(dataset)

    def build_loss(self, recipe):
        if self.net is None:
            raise ImportError(self.import_error or "HACDR-Net network was not imported")
        return build_author_loss(recipe, self.card.class_index)

    def build_optimizer(self, params, recipe):
        if self.net is None:
            raise ImportError(self.import_error or "HACDR-Net network was not imported")
        return hacdr_optimizer(self, params, recipe)

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        return adjust_hacdr_lr(optimizer, step, total_steps, recipe)

    def forward(self, x):
        if self.net is None:
            raise ImportError(self.import_error or "HACDR-Net network was not imported")
        logits = self.net.encode_decode(x, None)
        if logits.shape[1] != len(FGADR_CLASSES):
            raise RuntimeError(f"expected {len(FGADR_CLASSES)} exclusive classes, got {logits.shape[1]}")
        if logits.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                f"logits spatial {tuple(logits.shape[-2:])} != input {tuple(x.shape[-2:])}"
            )
        return logits
