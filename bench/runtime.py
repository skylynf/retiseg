"""Shared B1 training mechanics: loss, optimizer, schedule, checkpoints, run files.

One process uses one GPU. This module never constructs DistributedDataParallel.
A wrapper's own build_loss, build_optimizer or adjust_lr replaces the default
for that piece; an unknown default name is an error.
"""

import json
import math
import os
import socket
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
SCHEDULE_KEYS = ("power", "min_lr")
LOSS_NAMES = ("bce_with_logits", "ce", "dice", "ce_dice", "bce_dice")
OPTIMIZER_NAMES = ("adam", "adamw", "sgd")
SCHEDULE_NAMES = ("none", "poly")
LOSS_PARAM_KEYS = {
    "bce_with_logits": {"pos_weight", "reduction"},
    "ce": {"ignore_index", "label_smoothing", "reduction"},
    "dice": {"smooth"},
    "ce_dice": {"smooth", "ignore_index", "label_smoothing", "reduction"},
    "bce_dice": {"smooth", "pos_weight", "reduction"},
}
# Read by the runner, not by the loss module.
LOSS_META_KEYS = {"early_stop_patience"}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def run_directory(config):
    dataset = Path(config["dataset"]).name
    return Path("runs") / f"{config['experiment']}_{config['model']}_{dataset}_seed{int(config['seed'])}"


def assert_not_finished_run(path):
    """Leave the finished E0 run and the B1 smoke run untouched."""
    parts = Path(path).resolve().parts
    if "runs" not in parts:
        return
    for part in parts[parts.index("runs") + 1 :]:
        if part == "B1_unet_idrid_seed0" or part.startswith("E0_"):
            raise RuntimeError(f"refusing to write {path}: that finished run stays as it is")


def refuse_distributed():
    world = os.environ.get("WORLD_SIZE", "1")
    if world not in ("", "1"):
        raise RuntimeError(
            "this runner is one process on one GPU; DistributedDataParallel is not started"
        )


def set_seed(seed):
    seed = int(seed)
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _params(recipe, allowed):
    unknown = set(recipe.loss_params) - allowed - set(SCHEDULE_KEYS) - LOSS_META_KEYS
    if unknown:
        raise ValueError(f"unknown loss_params {sorted(unknown)} for loss {recipe.loss!r}")
    return {key: recipe.loss_params[key] for key in allowed if key in recipe.loss_params}


class _BCEWithLogits(nn.Module):
    def __init__(self, pos_weight=None, reduction="mean"):
        super().__init__()
        if pos_weight is None:
            self.pos_weight = None
        else:
            weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", weight)
        self.reduction = reduction

    def forward(self, logits, target):
        return nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight, reduction=self.reduction
        )


class _Dice(nn.Module):
    def __init__(self, smooth=1.0, activation="sigmoid"):
        super().__init__()
        self.smooth = float(smooth)
        self.activation = activation

    def forward(self, logits, target):
        if target.dtype in (torch.int32, torch.int64, torch.long):
            if target.ndim != logits.ndim - 1:
                raise ValueError(
                    f"dice class indices must have shape (N, H, W), got {tuple(target.shape)}"
                )
            target = nn.functional.one_hot(target.long(), logits.shape[1]).permute(0, 3, 1, 2).float()
        else:
            target = target.float()
        if self.activation == "sigmoid":
            probs = torch.sigmoid(logits)
        elif self.activation == "softmax":
            probs = torch.softmax(logits, dim=1)
        else:
            raise ValueError(f"dice activation must be sigmoid or softmax, got {self.activation!r}")
        if probs.shape != target.shape:
            raise ValueError(f"dice target shape {tuple(target.shape)} != logits {tuple(probs.shape)}")
        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        score = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1 - score.mean()


class _CrossEntropy(nn.Module):
    def __init__(self, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            ignore_index=int(ignore_index),
            label_smoothing=float(label_smoothing),
            reduction=reduction,
        )

    def forward(self, logits, target):
        if target.dtype.is_floating_point:
            raise ValueError(
                "ce expects integer class indices (N, H, W). "
                "Overlapping multilabel masks stay four channels; that wrapper should override build_loss."
            )
        return self.loss(logits, target.long())


class _BCEDice(nn.Module):
    """Binary cross-entropy with logits plus sigmoid Dice, each with weight 1."""

    def __init__(self, smooth=1.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.bce = _BCEWithLogits(pos_weight=pos_weight, reduction=reduction)
        self.dice = _Dice(smooth=smooth, activation="sigmoid")

    def forward(self, logits, target):
        return self.bce(logits, target) + self.dice(logits, target)


class _CEDice(nn.Module):
    """Cross-entropy plus Dice, each with weight 1."""

    def __init__(self, smooth=1.0, activation="sigmoid", ignore_index=-100, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.ce = _CrossEntropy(ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction)
        self.dice = _Dice(smooth=smooth, activation=activation)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)


def build_loss(recipe, activation):
    name = str(recipe.loss).lower()
    if name not in LOSS_NAMES:
        raise ValueError(f"unknown loss {recipe.loss!r}; expected one of {', '.join(LOSS_NAMES)}")
    params = _params(recipe, LOSS_PARAM_KEYS[name])
    if name == "bce_with_logits":
        return _BCEWithLogits(**params)
    if name == "ce":
        return _CrossEntropy(**params)
    if name == "dice":
        return _Dice(activation=activation, **params)
    if name == "bce_dice":
        return _BCEDice(**params)
    return _CEDice(activation=activation, **params)


def build_optimizer(params, recipe):
    name = str(recipe.optimizer).lower()
    if name not in OPTIMIZER_NAMES:
        raise ValueError(f"unknown optimizer {recipe.optimizer!r}; expected one of {', '.join(OPTIMIZER_NAMES)}")
    kwargs = {key: value for key, value in recipe.optimizer_params.items() if key not in SCHEDULE_KEYS}
    if "lr" in kwargs and float(kwargs["lr"]) != float(recipe.lr):
        raise ValueError(f"optimizer_params lr {kwargs['lr']} disagrees with recipe.lr {recipe.lr}")
    if "weight_decay" in kwargs and float(kwargs["weight_decay"]) != float(recipe.weight_decay):
        raise ValueError(
            f"optimizer_params weight_decay {kwargs['weight_decay']} disagrees with recipe.weight_decay {recipe.weight_decay}"
        )
    kwargs.pop("lr", None)
    kwargs.pop("weight_decay", None)
    kwargs["lr"] = float(recipe.lr)
    kwargs["weight_decay"] = float(recipe.weight_decay)
    if name == "adam":
        optimizer = torch.optim.Adam(params, **kwargs)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(params, **kwargs)
    else:
        optimizer = torch.optim.SGD(params, **kwargs)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def _schedule_value(recipe, key, default):
    in_opt = key in recipe.optimizer_params
    in_loss = key in recipe.loss_params
    if in_opt and in_loss and recipe.optimizer_params[key] != recipe.loss_params[key]:
        raise ValueError(f"{key} is in both optimizer_params and loss_params and the values differ")
    if in_opt:
        return recipe.optimizer_params[key]
    if in_loss:
        return recipe.loss_params[key]
    return default


def adjust_lr_after_step(optimizer, step, total_steps, recipe):
    """Polynomial rate for loops that write the learning rate after ``optimizer.step``.

    The runner calls this before update ``step`` (0-based). H2Former
    ``idrid_train.py``, Swin-Unet ``trainer.py`` and HRNet ``function.py``
    assign the rate after the update and only then increment the counter, so
    the first two updates both stay at the base rate. Those three loops have
    no ``min_lr``.
    """
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    power = float(_schedule_value(recipe, "power", 0.9))
    used = 0 if int(step) <= 0 else int(step) - 1
    coeff = (1.0 - min(1.0, used / float(total_steps))) ** power
    for group in optimizer.param_groups:
        base = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = base * coeff


def adjust_lr(optimizer, step, total_steps, recipe):
    """Set the learning rate for the update numbered ``step`` (0-based).

    ``none`` keeps the initial learning rate. ``poly`` follows
    (1 - step / total_steps) ** power, with power defaulting to 0.9.
    A ``min_lr`` in the recipe's optimizer_params or loss_params uses the
    same form as the author M2MRF run: (base - min_lr) * coeff + min_lr.
    Without that field the floor is 0.
    """
    name = str(recipe.schedule).lower()
    if name not in SCHEDULE_NAMES:
        raise ValueError(f"unknown schedule {recipe.schedule!r}; expected one of {', '.join(SCHEDULE_NAMES)}")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    ratio = min(1.0, float(step) / float(total_steps))
    if name == "poly":
        power = float(_schedule_value(recipe, "power", 0.9))
        min_lr = float(_schedule_value(recipe, "min_lr", 0.0))
        coeff = (1.0 - ratio) ** power
    else:
        min_lr = 0.0
        coeff = 1.0
    for group in optimizer.param_groups:
        base = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = (base - min_lr) * coeff + min_lr


def steps_per_epoch(n_train, effective_batch_size):
    if n_train < 1:
        raise ValueError("the training split is empty")
    if effective_batch_size < 1:
        raise ValueError("effective_batch_size must be positive")
    return math.ceil(int(n_train) / int(effective_batch_size))


def optimizer_step_sizes(n_train, micro_batch, effective_batch_size):
    """Images in each optimizer step of one epoch, including the tail.

    The loader does not drop the last incomplete micro-batch, and the runner
    steps on a remainder that does not fill ``effective_batch_size``. With 44
    images, micro-batch 1 and a nominal batch of 16, the steps are 16, 16, 12.
    """
    n_train = int(n_train)
    micro_batch = int(micro_batch)
    accum = accumulation_steps(micro_batch, effective_batch_size)
    if n_train < 1:
        raise ValueError("the training split is empty")
    sizes = []
    remaining = n_train
    while remaining:
        pending = 0
        held = 0
        while remaining and held < accum:
            take = min(micro_batch, remaining)
            remaining -= take
            pending += take
            held += 1
        sizes.append(pending)
    if len(sizes) != steps_per_epoch(n_train, effective_batch_size):
        raise RuntimeError(
            f"optimizer steps {len(sizes)} != ceil({n_train} / {int(effective_batch_size)})"
        )
    return sizes


def compact_step_sizes(sizes):
    parts = []
    run = 0
    previous = None
    for size in list(sizes) + [None]:
        if size == previous:
            run += 1
            continue
        if previous is not None:
            parts.append(f"{previous}x{run}" if run > 1 else str(previous))
        previous = size
        run = 1
    return ", ".join(parts)


def accumulation_steps(micro_batch, effective_batch_size):
    micro_batch = int(micro_batch)
    effective_batch_size = int(effective_batch_size)
    if micro_batch < 1:
        raise ValueError("batch_size must be positive")
    if micro_batch > effective_batch_size:
        raise ValueError(
            f"batch_size {micro_batch} is larger than effective_batch_size {effective_batch_size}"
        )
    if effective_batch_size % micro_batch != 0:
        raise ValueError(
            f"effective_batch_size {effective_batch_size} must be a multiple of batch_size {micro_batch}"
        )
    return effective_batch_size // micro_batch


def resolve_recipe(recipe, config, n_train):
    """Fill iterations from epochs when the recipe does not set a step count.

    Returns the resolved recipe, a deviation list, and the optimizer steps in
    one pass over the training images. Author iterations win over the cell
    config. Epochs are taken from the recipe when it sets them, and from the
    cell config only when the recipe leaves both epochs and iterations at 0.
    """
    deviations = []
    per_epoch = steps_per_epoch(n_train, recipe.effective_batch_size)
    config_epochs = int(config["epochs"]) if "epochs" in config and config["epochs"] is not None else 0
    if recipe.iterations > 0:
        iterations = int(recipe.iterations)
        epochs = int(recipe.epochs)
        if config_epochs > 0 and epochs == 0:
            deviations.append(
                f"cell config epochs {config_epochs} ignored; recipe.iterations is {iterations}"
            )
        elif epochs > 0 and epochs * per_epoch != iterations:
            deviations.append(
                f"recipe.iterations {iterations} kept; "
                f"epochs {epochs} * steps_per_epoch {per_epoch} = {epochs * per_epoch}"
            )
    elif recipe.epochs > 0:
        epochs = int(recipe.epochs)
        if config_epochs > 0 and config_epochs != epochs:
            deviations.append(f"cell config epochs {config_epochs} ignored; recipe.epochs is {epochs}")
        iterations = epochs * per_epoch
    elif config_epochs > 0:
        epochs = config_epochs
        iterations = epochs * per_epoch
    else:
        raise ValueError("the recipe has no iterations or epochs, and the cell config has no epochs")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    resolved = replace(recipe, epochs=epochs, iterations=iterations)
    return resolved, deviations, per_epoch


def micro_batch_and_deviations(config, recipe):
    """Loader batch from the cell config, or the recipe batch when the cell omits it."""
    deviations = []
    if "batch_size" in config and config["batch_size"] is not None:
        micro = int(config["batch_size"])
    else:
        micro = int(recipe.batch_size)
    repeats = accumulation_steps(micro, recipe.effective_batch_size)
    if repeats > 1:
        deviations.append(
            f"gradient accumulation x{repeats}: loader batch {micro}, "
            f"effective batch {recipe.effective_batch_size}; "
            "BatchNorm statistics follow the loader batch"
        )
    for key in ("lr", "weight_decay", "loss", "optimizer", "schedule"):
        if key not in config or config[key] is None:
            continue
        recipe_value = getattr(recipe, key)
        if key in ("lr", "weight_decay"):
            same = float(config[key]) == float(recipe_value)
        else:
            same = str(config[key]) == str(recipe_value)
        if not same:
            deviations.append(f"cell config {key} {config[key]!r} ignored; recipe {key} is {recipe_value!r}")
    return micro, deviations


def lesion_probabilities(logits, card):
    """Map network logits to four lesion probabilities, MA HE EX SE.

    Softmax models keep the background channel in the normalization. The four
    selected planes are not divided by their own sum.
    """
    if card.output_activation == "sigmoid":
        if logits.shape[1] != 4:
            raise ValueError(f"sigmoid models return 4 lesion channels, got {logits.shape[1]}")
        return torch.sigmoid(logits)
    if card.output_activation == "softmax":
        if max(card.class_index) >= logits.shape[1]:
            raise ValueError(
                f"class_index {card.class_index} does not fit logit channels {logits.shape[1]}"
            )
        full = torch.softmax(logits, dim=1)
        return full[:, list(card.class_index)]
    raise ValueError(f"output_activation must be sigmoid or softmax, got {card.output_activation!r}")


def save_checkpoint(path, model, config, step):
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "step": int(step),
            "model_name": model.card.name,
        },
        path,
    )


def write_recipe(path, dataset, recipe, declared, n_train, per_epoch, micro_batch, deviations):
    nominal = int(recipe.effective_batch_size)
    sizes = optimizer_step_sizes(n_train, micro_batch, nominal)
    if len(sizes) != int(per_epoch):
        raise RuntimeError(f"steps_per_epoch {per_epoch} != {len(sizes)} optimizer steps")
    deviations = list(deviations)
    if any(size != nominal for size in sizes):
        deviations.append(
            f"nominal effective batch is {nominal}; one epoch submits image counts "
            f"{compact_step_sizes(sizes)}. A short step is the mean over those images, "
            f"not over {nominal}."
        )
    payload = {
        "dataset": dataset,
        "n_train": int(n_train),
        "steps_per_epoch": int(per_epoch),
        "iterations": int(recipe.iterations),
        "epochs": int(recipe.epochs),
        "batch_size": int(micro_batch),
        "effective_batch_size": nominal,
        "nominal_effective_batch_size": nominal,
        "step_image_counts": sizes,
        "accumulation": nominal // int(micro_batch),
        "declared_iterations": int(declared.iterations),
        "declared_epochs": int(declared.epochs),
        "loss": str(recipe.loss),
        "optimizer": str(recipe.optimizer),
        "lr": float(recipe.lr),
        "weight_decay": float(recipe.weight_decay),
        "schedule": str(recipe.schedule),
        "deviations": deviations,
        "recipe": asdict(recipe),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def budget_signature(recipe_json):
    """Fields that define the executed budget. A change means seed 0 is rerun."""
    keys = (
        "iterations",
        "epochs",
        "loss",
        "optimizer",
        "lr",
        "weight_decay",
        "schedule",
        "batch_size",
        "nominal_effective_batch_size",
        "step_image_counts",
    )
    missing = [key for key in keys if key not in recipe_json]
    if missing:
        raise KeyError("recipe.json is missing " + ", ".join(missing))
    signature = {key: recipe_json[key] for key in keys}
    signature["lr"] = float(signature["lr"])
    signature["weight_decay"] = float(signature["weight_decay"])
    signature["step_image_counts"] = [int(size) for size in signature["step_image_counts"]]
    return signature


def write_environment(path, seed, parameters, seconds, best_val_loss):
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = "cpu"
    payload = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": int(seed),
        "git": git_commit(),
        "parameters": int(parameters),
        "seconds": seconds,
        "best_val_loss": best_val_loss,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def assert_b1_frozen(config, dataset_name):
    if config.get("experiment") != "B1":
        return
    frozen = json_load_yaml(REPO / "bench" / "configs" / "b1_frozen.yaml")
    if int(config["fov_diameter"]) != int(frozen["fov_diameter"]):
        raise ValueError(
            f"B1 fov_diameter is {frozen['fov_diameter']} in b1_frozen.yaml, got {config['fov_diameter']}"
        )
    if dataset_name not in frozen["datasets"]:
        raise ValueError(f"{dataset_name} is not a B1 dataset in b1_frozen.yaml")
    splits = frozen["splits"][dataset_name]
    for config_key, frozen_key in (("split_train", "train"), ("split_val", "val"), ("split_test", "test")):
        if config[config_key] != splits[frozen_key]:
            raise ValueError(
                f"B1 {dataset_name} {config_key} must be {splits[frozen_key]!r}, got {config[config_key]!r}"
            )


def json_load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())
