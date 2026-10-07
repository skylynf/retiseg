"""B-std: the shared-budget retraining table. See bench/configs/bs_frozen.yaml.

bench/train.py, bench/scripts/train_m2mrf_b1.py and train_hacdr_b1.py call
``run_cell`` when the cell says ``experiment: BS``. The model, its
normalization and author_recipe come from the caller; data, budget,
augmentation, validation and checkpoint roles come from here.
"""

import json
import math
import shutil
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from bench.common.io import LESION_CLASSES, PROB_LEVELS
from bench.data.b1_input import PreparedSplit, canvas_hw, collate, dataset_name, normalize
from bench.data.bs_input import EpochSampler, PatchTrain
from bench.eval import pixel
from bench.infer import canvas_logits
from bench.runtime import (
    REPO,
    lesion_probabilities,
    micro_batch_and_deviations,
    write_environment,
    write_recipe,
)

FROZEN_PATH = REPO / "bench" / "configs" / "bs_frozen.yaml"
# Set by the launcher's capped smoke run only.
MAX_STEPS = None


def load_frozen(path=FROZEN_PATH):
    return yaml.safe_load(Path(path).read_text())


def is_bstd(config):
    return config.get("experiment") == "BS"


def assert_bs_frozen(config, name, frozen):
    if int(config["fov_diameter"]) != int(frozen["fov_diameter"]):
        raise ValueError(f"BS fov_diameter is {frozen['fov_diameter']}, got {config['fov_diameter']}")
    if name not in frozen["datasets"]:
        raise ValueError(f"{name} is not a BS dataset")
    splits = frozen["splits"][name]
    for config_key, frozen_key in (("split_train", "train"), ("split_val", "val"), ("split_test", "test")):
        if config[config_key] != splits[frozen_key]:
            raise ValueError(f"BS {name} {config_key} must be {splits[frozen_key]!r}, got {config[config_key]!r}")
    for key in ("epochs", "eval_every", "lr", "loss", "optimizer", "schedule"):
        if config.get(key) is not None:
            raise ValueError(f"a BS cell does not set {key}; the budget and recipe are not per cell")


def geometry(card, frozen):
    """(training patch, inference window). Window None is the whole canvas."""
    if card.forward_size is not None:
        size = (int(card.forward_size[0]), int(card.forward_size[1]))
        return size, size
    patch = frozen["fully_conv_patch"]
    return (int(patch[0]), int(patch[1])), None


def inference_for(config, card, frozen=None):
    """None for B1. For BS, the dict predict_image uses: window and overlap."""
    if not is_bstd(config):
        return None
    frozen = frozen or load_frozen()
    _, window = geometry(card, frozen)
    return {"window": window, "overlap": float(frozen["window_overlap"])}


def canvas_pixels(dataset_dir, ids, diameter):
    total = 0
    for image_id in ids:
        height, width = canvas_hw(dataset_dir, image_id, diameter)
        total += int(height) * int(width)
    return total


PROBE_SCALES = (1, 2)


def probe_scale(config):
    """``budget_probe: 2`` runs twice the frozen amount with the same validation spacing."""
    scale = int(config.get("budget_probe") or 1)
    if scale not in PROBE_SCALES:
        raise ValueError(f"budget_probe must be one of {PROBE_SCALES}, got {scale}")
    return scale


def budget(declared, name, frozen, pixels, patch, scale=1):
    """Resolve the BS step count. Returns resolved recipe, deviations, steps per epoch, samples per epoch, eval_every."""
    epochs = int(frozen["equivalent_epochs"][name]) * int(scale)
    effective = int(declared.effective_batch_size)
    patch_pixels = int(patch[0]) * int(patch[1])
    patches = math.ceil(int(pixels) / patch_pixels)
    per_epoch = math.ceil(patches / effective)
    samples = per_epoch * effective
    iterations = epochs * per_epoch
    eval_epochs = max(1, epochs // (int(frozen["n_evals"]) * int(scale)))
    eval_every = eval_epochs * per_epoch
    loss_params = {key: value for key, value in declared.loss_params.items() if key != "early_stop_patience"}
    deviations = [
        f"B-std budget: {epochs} equivalent epochs of {pixels} canvas pixels; "
        f"{patches} patches of {patch[0]}x{patch[1]} per epoch, rounded up to {samples} "
        f"({per_epoch} steps of {effective}); {iterations} steps in total",
        f"author iterations {int(declared.iterations)} and epochs {int(declared.epochs)} are not used",
        f"validation mAUPR every {eval_epochs} epochs ({eval_every} steps); best.pt is the highest",
    ]
    if "early_stop_patience" in declared.loss_params:
        deviations.append("early_stop_patience dropped; B-std runs the whole budget")
    if int(scale) != 1:
        deviations.append(f"budget probe: {int(scale)} x the frozen equivalent epochs; train and validation only")
    if MAX_STEPS is not None and iterations > int(MAX_STEPS):
        deviations.append(f"smoke stops after {int(MAX_STEPS)} steps; {iterations} are not run")
        iterations = int(MAX_STEPS)
        eval_every = min(eval_every, iterations)
    resolved = replace(declared, iterations=iterations, epochs=epochs, loss_params=loss_params)
    return resolved, deviations, per_epoch, samples, eval_every


def budget_record(name, frozen, pixels, patch, window, per_epoch, samples, eval_every, resolved, scale=1):
    return {
        "equivalent_epochs": int(resolved.epochs),
        "budget_probe": int(scale),
        "canvas_pixels": int(pixels),
        "patch": [int(patch[0]), int(patch[1])],
        "window": None if window is None else [int(window[0]), int(window[1])],
        "window_overlap": float(frozen["window_overlap"]),
        "steps_per_epoch": int(per_epoch),
        "samples_per_epoch": int(samples),
        "eval_every": int(eval_every),
        "iterations": int(resolved.iterations),
        "augment": frozen["augment"],
    }


class ValScorer:
    """Validation loss and mAUPR on the canvas, inside the canvas field of view.

    Probabilities are quantized to PROB_LEVELS as write_prob does, so the
    AUPR is the exact pooled value for those levels at canvas resolution.
    The test evaluation is at original resolution and is not this number.
    """

    def __init__(self, split, card, loss_fn, window, overlap):
        self.split = split
        self.card = card
        self.loss_fn = loss_fn
        self.window = window
        self.overlap = float(overlap)

    def __call__(self, model, device):
        model.eval()
        levels = PROB_LEVELS
        pos = np.zeros((len(LESION_CLASSES), levels + 1), dtype=np.int64)
        neg = np.zeros_like(pos)
        loss_sum = 0.0
        for index in range(len(self.split)):
            image, masks, fov = self.split.canvas_with_fov(index)
            tensor = normalize(image, self.split.mean, self.split.std).to(device)
            target = torch.from_numpy(np.stack(masks).astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = canvas_logits(model, tensor, self.card, self.window, self.overlap).unsqueeze(0)
                loss_sum += float(self.loss_fn(logits, target))
                prob = lesion_probabilities(logits, self.card)[0]
                q = torch.round(prob.clamp(0.0, 1.0) * levels).long().cpu().numpy()
            for channel in range(len(LESION_CLASSES)):
                p, n = pixel.histograms(q[channel], np.asarray(masks[channel], dtype=bool), fov, levels)
                pos[channel] += p
                neg[channel] += n
        scores = [pixel.average_precision(pos[c], neg[c]) for c in range(len(LESION_CLASSES))]
        defined = [value for value in scores if not math.isnan(value)]
        row = {
            "val_loss": loss_sum / max(1, len(self.split)),
            "val_maupr": float(np.mean(defined)) if defined else float("nan"),
        }
        for cls, value in zip(LESION_CLASSES, scores):
            row[f"val_aupr_{cls}"] = None if math.isnan(value) else float(value)
        return row


def train_loader(dataset, micro, num_workers, pad_multiple, seed):
    workers = int(num_workers)
    return DataLoader(
        dataset,
        batch_size=int(micro),
        sampler=EpochSampler(len(dataset), seed),
        num_workers=workers,
        collate_fn=partial(collate, pad_multiple=int(pad_multiple)),
        persistent_workers=workers > 0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def run_cell(
    config_path,
    config,
    model,
    declared,
    run_dir,
    predict_fn,
    diagnostic=False,
    resume=False,
    before_training=None,
    extra_deviations=(),
):
    from bench.train import run_training, score_run

    config_path = Path(config_path)
    run_dir = Path(run_dir)
    name = dataset_name(config["dataset"])
    frozen = load_frozen()
    assert_bs_frozen(config, name, frozen)
    card = model.card
    patch, window = geometry(card, frozen)
    mean = card.normalization["mean"]
    std = card.normalization["std"]
    diameter = int(config["fov_diameter"])
    source = PreparedSplit(config["dataset"], config["split_train"], diameter, False, mean, std, None)
    val_split = PreparedSplit(config["dataset"], config["split_val"], diameter, False, mean, std, None)
    scale = probe_scale(config)
    if scale != 1 and not diagnostic:
        raise ValueError("a budget probe is train and validation only; run it with --diagnostic")
    pixels = canvas_pixels(config["dataset"], source.ids, diameter)
    resolved, deviations, per_epoch, samples, eval_every = budget(declared, name, frozen, pixels, patch, scale)
    micro, batch_deviations = micro_batch_and_deviations(config, resolved)
    deviations = deviations + batch_deviations + list(extra_deviations)
    train_set = PatchTrain(source, patch, samples, frozen["augment"], config["seed"])
    loader = train_loader(train_set, micro, config["num_workers"], card.pad_multiple, config["seed"])

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    write_recipe(run_dir / "recipe.json", name, resolved, declared, samples, per_epoch, micro, deviations)
    payload = json.loads((run_dir / "recipe.json").read_text())
    payload["n_train"] = len(source.ids)
    payload["bstd"] = budget_record(
        name, frozen, pixels, patch, window, per_epoch, samples, eval_every, resolved, scale
    )
    (run_dir / "recipe.json").write_text(json.dumps(payload, indent=2) + "\n")

    best = None
    started = time.perf_counter()
    try:
        if before_training is not None:
            before_training()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = ValScorer(val_split, card, model.build_loss(resolved).to(device), window, frozen["window_overlap"])
        best = run_training(
            model,
            loader,
            None,
            resolved,
            config,
            run_dir,
            micro,
            per_epoch,
            device=device,
            eval_every=eval_every,
            resume=resume,
            select_by=frozen["select_by"],
            val_scorer=scorer,
        )
    finally:
        seconds = time.perf_counter() - started
        write_environment(
            run_dir / "environment.json", config["seed"], model.n_parameters(), seconds, best, best_key="best_val_maupr"
        )
    if diagnostic:
        (run_dir / "diagnostic.json").write_text(
            json.dumps(
                {
                    "test_split_read": False,
                    "look_at": ["history.jsonl", "recipe.json", "environment.json"],
                    "note": "Train and validation only. Test metrics are not an input to a budget change.",
                },
                indent=2,
            )
            + "\n"
        )
        print("diagnostic: test split was not read", flush=True)
        print(f"trained in {seconds:.1f}s", flush=True)
        return seconds
    score_run(run_dir / "config.yaml", run_dir, declared, predict_fn)
    print(f"trained in {seconds:.1f}s", flush=True)
    return seconds
