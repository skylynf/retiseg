"""Train HACDR-Net for one B1 cell and write the same run directory as bench/train.py.

One process, one GPU. This does not call official_code slurm scripts and does
not start DistributedDataParallel. Loss, AdamW, the poly warmup and max_iters
come from the dataset config via author_recipe. The loader batch is
samples_per_gpu. When a cell config sets a smaller batch, gradients accumulate
until the configured effective batch and the learning rate stays at the
configured value. samples_per_gpu is 1, so there is no smaller batch that
still adds up to the author batch.

last.pt is the author iteration and is the scored checkpoint. best.pt is the
lowest validation loss and is written beside it. The test split is not read
until those probability maps are written.

    python bench/scripts/train_hacdr_b1.py --config CONFIG
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import yaml

from bench.common.io import LESION_CLASSES
from bench.data.b1_input import PreparedSplit, dataset_name, pad_to_multiple
from bench.models.hacdr import HACDRNet
from bench.models.registry import build_model
from bench.runtime import (
    assert_b1_frozen,
    assert_not_finished_run,
    micro_batch_and_deviations,
    refuse_distributed,
    resolve_recipe,
    run_directory,
    set_seed,
    write_environment,
    write_recipe,
)
from bench.scripts.predict_hacdr_b1 import predict
from bench.train import _loader, run_training, score_run


def _require_network(model):
    if model.net is None:
        raise ImportError(model.import_error or "HACDR-Net network was not imported")
    return model


def training_step(dataset_dir=None, device=None):
    """One optimizer step on one prepared training image. Random initialization.

    This is the import check. It does not run runner.max_iters.
    """
    root = Path(dataset_dir) if dataset_dir is not None else _ROOT / "dataset" / "prepared" / "IDRiD"
    name = dataset_name(root)
    frozen = yaml.safe_load((_ROOT / "bench" / "configs" / "b1_frozen.yaml").read_text())
    model = _require_network(HACDRNet())
    recipe = model.author_recipe(name)
    split = PreparedSplit(
        root,
        "train",
        int(frozen["fov_diameter"]),
        True,
        model.card.normalization["mean"],
        model.card.normalization["std"],
        model.card.forward_size,
    )
    image, target = split[0]
    image = pad_to_multiple(image, model.card.pad_multiple).unsqueeze(0)
    target = pad_to_multiple(target, model.card.pad_multiple).unsqueeze(0)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device).train()
    loss_fn = model.build_loss(recipe).to(device)
    optimizer = model.build_optimizer(model.parameters(), recipe)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(image.to(device)), target.to(device))
    value = float(loss.detach())
    if value != value or value in (float("inf"), float("-inf")):
        raise RuntimeError(f"HACDR-Net training loss is not finite: {value}")
    loss.backward()
    if not any(param.grad is not None for param in model.parameters()):
        raise RuntimeError("HACDR-Net training step produced no gradients")
    model.adjust_lr(optimizer, 0, recipe.iterations, recipe)
    optimizer.step()
    return value


def train(config_path, run_dir=None, diagnostic=False, score_only=False):
    refuse_distributed()
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if config.get("model") != "HACDR-Net":
        raise ValueError(f"this script trains HACDR-Net, got {config.get('model')!r}")
    if "seed" not in config:
        raise ValueError("the cell config must set seed")
    name = dataset_name(config["dataset"])
    assert_b1_frozen(config, name)
    if tuple(LESION_CLASSES) != ("MA", "HE", "EX", "SE"):
        raise RuntimeError("lesion order must stay MA, HE, EX, SE")
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    assert_not_finished_run(run_dir)
    if score_only:
        from bench.train import _score_existing

        _score_existing(config_path, config, run_dir, predict)
        return 0.0
    for split_name in (config["split_val"], config["split_test"]):
        if split_name == "train_official":
            raise ValueError("B1 does not read train_official")

    set_seed(config["seed"])
    model = _require_network(build_model(config["model"]))
    if tuple(model.classes) != tuple(LESION_CLASSES):
        raise ValueError(f"{config['model']} classes {model.classes} != {LESION_CLASSES}")
    if int(model.card.in_channels) != 3:
        raise ValueError("B1 prepared images are RGB; in_channels must be 3")
    if model.card.class_index != (2, 4, 1, 3):
        raise RuntimeError(f"HACDR-Net class_index {model.card.class_index} is not MA, HE, EX, SE = (2, 4, 1, 3)")
    declared = model.author_recipe(name)
    mean = model.card.normalization["mean"]
    std = model.card.normalization["std"]
    train_set = PreparedSplit(
        config["dataset"],
        config["split_train"],
        config["fov_diameter"],
        True,
        mean,
        std,
        model.card.forward_size,
    )
    val_set = PreparedSplit(
        config["dataset"],
        config["split_val"],
        config["fov_diameter"],
        False,
        mean,
        std,
        model.card.forward_size,
    )
    resolved, deviations, per_epoch = resolve_recipe(declared, config, len(train_set))
    micro, batch_deviations = micro_batch_and_deviations(config, resolved)
    deviations = deviations + batch_deviations
    deviations.append(
        f"author RandomCrop {tuple(declared.crop_size)} is not applied; "
        f"B1 uses the field-of-view canvas at fov_diameter {config['fov_diameter']} with horizontal flip"
    )
    train_loader = _loader(
        train_set, micro, True, config["num_workers"], model.card.pad_multiple, config["seed"]
    )
    val_loader = _loader(val_set, 1, False, config["num_workers"], model.card.pad_multiple, config["seed"])

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    write_recipe(
        run_dir / "recipe.json",
        name,
        resolved,
        declared,
        len(train_set),
        per_epoch,
        micro,
        deviations,
    )
    best = None
    started = time.perf_counter()
    try:
        best = run_training(
            model, train_loader, val_loader, resolved, config, run_dir, micro, per_epoch
        )
    finally:
        seconds = time.perf_counter() - started
        write_environment(run_dir / "environment.json", config["seed"], model.n_parameters(), seconds, best)
    saved = run_dir / "config.yaml"
    if diagnostic:
        (run_dir / "diagnostic.json").write_text(
            json.dumps(
                {
                    "test_split_read": False,
                    "look_at": ["history.jsonl", "recipe.json", "environment.json"],
                    "note": "Train and validation only. Test metrics are not an input to a recipe change.",
                },
                indent=2,
            )
            + "\n"
        )
        print("diagnostic: test split was not read", flush=True)
        print(f"trained in {seconds:.1f}s", flush=True)
        return seconds
    score_run(saved, run_dir, declared, predict)
    print(f"trained in {seconds:.1f}s", flush=True)
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)
    if args.diagnostic and args.score_only:
        raise SystemExit("--diagnostic does not score the test split; drop one of the flags")
    train(args.config, args.run_dir, diagnostic=args.diagnostic, score_only=args.score_only)


if __name__ == "__main__":
    main()
