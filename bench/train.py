"""Train one registered wrapper and write a B1 run directory.

The cell config selects the model, the dataset, the seed and, when the recipe
does not set a step count, the epoch budget. Loss, optimizer and learning rate
come from author_recipe. The checkpoint with the lowest validation loss is
best.pt. last.pt is the final step. A recipe that sets its own iterations or
epochs is scored from last.pt, because that is the author budget. U-Net leaves
both at 0 and is scored from best.pt. The other checkpoint is written beside
it. The test split is never read during training.

A cell with ``experiment: BS`` is handed to bench.bstd.run_cell: shared
pixel budget, patch augmentation, and best.pt by validation mAUPR.

    python -m bench.train --config bench/configs/cells/b1_unet_idrid_seed0.yaml
    python bench/train.py --config bench/configs/cells/b1_unet_idrid_seed0.yaml
"""

import argparse
import json
import math
import shutil
import sys
import time
from functools import partial
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from bench.common.io import LESION_CLASSES
from bench.data.b1_input import PreparedSplit, collate, dataset_name
from bench.models.registry import build_model
from bench.predict import predict
from bench.runtime import (
    accumulation_steps,
    assert_b1_frozen,
    assert_not_finished_run,
    atomic_save,
    evaluation_interval,
    micro_batch_and_deviations,
    refuse_distributed,
    resolve_recipe,
    run_directory,
    save_checkpoint,
    set_seed,
    write_environment,
    write_recipe,
)


def _seed_worker(worker_id):
    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def _loader(dataset, batch_size, shuffle, num_workers, pad_multiple, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=partial(collate, pad_multiple=int(pad_multiple)),
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _scale_grads(model, images_in_step):
    """Turn the summed micro-batch gradients into the mean over this step."""
    if images_in_step < 1:
        raise RuntimeError("an optimizer step accumulated no images")
    scale = 1.0 / float(images_in_step)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


def _run_epoch(
    model, loader, loss_fn, optimizer, recipe, device, accum, effective, max_steps, step, total_steps, step_counts=None
):
    """One pass over the loader. A short tail is its own step.

    ``effective`` is the nominal full step. The tail is divided by the images
    actually accumulated, so 44 images with accumulation 16 update on 16, 16
    and 12 rather than treating the last 12 as a batch of 16.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    held = 0
    pending = 0
    taken = 0
    loss_sum = 0.0
    seen = 0

    def _commit():
        nonlocal held, pending, taken, step
        if pending > int(effective):
            raise RuntimeError(f"a step accumulated {pending} images, above the nominal batch {effective}")
        _scale_grads(model, pending)
        if step_counts is not None:
            step_counts.append(int(pending))
        model.adjust_lr(optimizer, step, total_steps, recipe)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        held = 0
        pending = 0
        taken += 1
        step += 1

    for images, targets in loader:
        if taken >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        raw = loss_fn(model(images), targets)
        count = int(images.shape[0])
        loss_sum += float(raw.detach()) * count
        seen += count
        # ``raw`` is a mean over the micro-batch. Multiply by the count so the
        # step can divide by the images that actually landed in it.
        (raw * count).backward()
        pending += count
        held += 1
        if held == accum:
            _commit()
    if held and taken < max_steps:
        _commit()
    if seen == 0:
        raise RuntimeError("the training loader produced no images")
    if taken != max_steps:
        raise RuntimeError(f"an epoch produced {taken} optimizer steps, expected {max_steps}")
    lr = float(optimizer.param_groups[0]["lr"])
    return loss_sum / seen, step, lr


@torch.no_grad()
def _validate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss_sum += float(loss_fn(model(images), targets)) * images.shape[0]
        seen += int(images.shape[0])
    if seen == 0:
        raise RuntimeError("the validation loader produced no images")
    return loss_sum / seen


def _rng_state():
    import random

    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state):
    import random

    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _improved(value, best, select_by):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if best is None:
        return True
    return value > best if select_by == "val_maupr" else value < best


def run_training(
    model,
    train_loader,
    val_loader,
    recipe,
    config,
    run_dir,
    micro_batch,
    steps_per_epoch,
    device=None,
    eval_every=None,
    resume=False,
    select_by="val_loss",
    val_scorer=None,
    resume_every_seconds=900,
):
    """Run every optimizer step in ``recipe.iterations``. Returns the best selection value.

    Validation happens at the end of an epoch: every epoch when ``eval_every``
    is None, otherwise at the first epoch end at or past each multiple of
    ``eval_every``, and always at the final step. best.pt is written when the
    selection value improves; last.pt once, at the final or early-stopped step.
    resume.pt is written at a validation, at every one when ``eval_every`` is
    set and at most every ``resume_every_seconds`` otherwise. Saving only at
    epoch ends keeps the loader state exact on resume. ``select_by`` is ``val_loss`` (lower is better) or ``val_maupr``
    (higher is better; needs ``val_scorer(model, device)`` returning a dict
    with ``val_maupr`` and ``val_loss``). Early-stopping patience counts
    validations.

    With ``resume`` and an existing resume.pt, the model, optimizer, step,
    selection state, RNG streams and loader generator are restored and the
    loop continues. resume.pt is removed when the budget is finished.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if select_by not in ("val_loss", "val_maupr"):
        raise ValueError(f"select_by must be val_loss or val_maupr, got {select_by!r}")
    if select_by == "val_maupr" and val_scorer is None:
        raise ValueError("select_by val_maupr needs a val_scorer")
    run_dir = Path(run_dir)
    model.to(device)
    loss_fn = model.build_loss(recipe).to(device)
    optimizer = model.build_optimizer(model.parameters(), recipe)
    accum = accumulation_steps(micro_batch, recipe.effective_batch_size)
    effective = int(recipe.effective_batch_size)
    total = int(recipe.iterations)
    per_epoch = int(steps_per_epoch)
    if per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if eval_every is not None and int(eval_every) < 1:
        raise ValueError("eval_every must be positive")
    history_path = run_dir / "history.jsonl"
    resume_path = run_dir / "resume.pt"
    signature = {
        "iterations": total,
        "steps_per_epoch": per_epoch,
        "effective_batch_size": effective,
        "micro_batch": int(micro_batch),
        "eval_every": None if eval_every is None else int(eval_every),
        "select_by": select_by,
    }
    best = None
    step = 0
    epoch = 0
    stale = 0
    elapsed_before = 0.0
    if resume and not resume_path.is_file() and (run_dir / "timing.json").is_file():
        raise RuntimeError(f"{run_dir} already finished its budget; --resume does not train it again")
    if resume and resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise RuntimeError(
                f"{resume_path} was written under {state['signature']}, this run is {signature}; "
                "start a fresh run directory instead of resuming"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, stale, best = int(state["step"]), int(state["epoch"]), int(state["stale"]), state["best"]
        elapsed_before = float(state.get("elapsed", 0.0))
        _set_rng_state(state["rng"])
        if state.get("loader_generator") is not None and getattr(train_loader, "generator", None) is not None:
            train_loader.generator.set_state(state["loader_generator"])
        if hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        kept = []
        if history_path.is_file():
            for line in history_path.read_text().splitlines():
                if line.strip() and int(json.loads(line)["iteration"]) <= step:
                    kept.append(line)
        history_path.write_text("".join(line + "\n" for line in kept))
        print(f"resumed at iteration {step}, epoch {epoch}", flush=True)
    else:
        history_path.write_text("")
    patience = int(recipe.loss_params.get("early_stop_patience", 0) or 0)
    stopped_early = False
    started = time.perf_counter()
    last_resume_save = started
    next_eval = total if eval_every is None else min(total, (step // int(eval_every) + 1) * int(eval_every))
    while step < total:
        epoch += 1
        max_steps = min(int(per_epoch), total - step)
        train_loss, step, lr = _run_epoch(
            model, train_loader, loss_fn, optimizer, recipe, device, accum, effective, max_steps, step, total
        )
        if eval_every is not None and step < next_eval:
            continue
        if eval_every is not None:
            next_eval = min(total, (step // int(eval_every) + 1) * int(eval_every))
        row = {
            "epoch": epoch,
            "iteration": step,
            "train_loss": train_loss,
        }
        if val_scorer is not None:
            scores = val_scorer(model, device)
            row.update(scores)
        else:
            row["val_loss"] = _validate(model, val_loader, loss_fn, device)
        row["lr"] = lr
        with history_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        shown = row.get("val_maupr", row["val_loss"]) if select_by == "val_maupr" else row["val_loss"]
        print(
            f"epoch {epoch} iter {step} train {train_loss:.4f} {select_by} {shown:.4f} lr {lr:.6g}",
            flush=True,
        )
        value = row[select_by]
        if _improved(value, best, select_by):
            best = value
            stale = 0
            save_checkpoint(run_dir / "best.pt", model, config, step)
        else:
            stale += 1
        now = time.perf_counter()
        if step < total and (eval_every is not None or now - last_resume_save >= resume_every_seconds):
            last_resume_save = now
            atomic_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "stale": stale,
                    "best": best,
                    "elapsed": elapsed_before + time.perf_counter() - started,
                    "rng": _rng_state(),
                    "loader_generator": (
                        train_loader.generator.get_state()
                        if getattr(train_loader, "generator", None) is not None
                        else None
                    ),
                    "signature": signature,
                },
                resume_path,
            )
        if patience and stale >= patience:
            stopped_early = True
            print(
                f"early stop at epoch {epoch}: {select_by} did not improve for {patience} validations",
                flush=True,
            )
            break
    if not stopped_early and step != total:
        raise RuntimeError(f"stopped at step {step}, recipe iterations is {total}")
    save_checkpoint(run_dir / "last.pt", model, config, step)
    if not (run_dir / "best.pt").is_file():
        save_checkpoint(run_dir / "best.pt", model, config, step)
    if resume_path.is_file():
        resume_path.unlink()
    (run_dir / "timing.json").write_text(
        json.dumps({"train_seconds_total": elapsed_before + time.perf_counter() - started}, indent=2) + "\n"
    )
    if patience:
        (run_dir / "early_stop.json").write_text(
            json.dumps(
                {
                    "patience": patience,
                    "stopped_early": stopped_early,
                    "iteration": int(step),
                    "best_val_loss" if select_by == "val_loss" else "best_val_maupr": best,
                },
                indent=2,
            )
            + "\n"
        )
    return best


def checkpoint_roles(declared):
    """Author-set budgets are scored at the last step. U-Net uses the best validation loss."""
    if int(declared.iterations) > 0 or int(declared.epochs) > 0:
        return "last.pt", "best.pt"
    return "best.pt", "last.pt"


def roles_for(config, declared):
    """B-std scores the highest validation mAUPR; B1 keeps checkpoint_roles."""
    if config.get("experiment") == "BS":
        return "best.pt", "last.pt"
    return checkpoint_roles(declared)


def score_run(config_path, run_dir, declared, predict_fn):
    config = yaml.safe_load(Path(config_path).read_text())
    primary, sensitivity = roles_for(config, declared)
    run_dir = Path(run_dir)
    (run_dir / "checkpoint_roles.json").write_text(
        json.dumps(
            {
                "primary": primary,
                "sensitivity": sensitivity,
                "primary_directories": ["prob_val", "prob_test"],
                "sensitivity_directories": ["prob_val_sensitivity", "prob_test_sensitivity"],
            },
            indent=2,
        )
        + "\n"
    )
    predict_fn(config_path, run_dir / primary, run_dir / "prob_val", config["split_val"])
    predict_fn(config_path, run_dir / primary, run_dir / "prob_test", config["split_test"])
    predict_fn(config_path, run_dir / sensitivity, run_dir / "prob_val_sensitivity", config["split_val"])
    predict_fn(config_path, run_dir / sensitivity, run_dir / "prob_test_sensitivity", config["split_test"])


def _score_existing(config_path, config, run_dir, predict_fn):
    run_dir = Path(run_dir)
    missing = [name for name in ("best.pt", "last.pt", "recipe.json") if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing {missing}; the diagnostic run has not finished")
    model = build_model(config["model"])
    declared = model.author_recipe(dataset_name(config["dataset"]))
    score_run(config_path, run_dir, declared, predict_fn)
    print("scored the saved checkpoints", flush=True)


def train(config_path, run_dir=None, diagnostic=False, score_only=False, resume=False):
    refuse_distributed()
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if "seed" not in config:
        raise ValueError("the cell config must set seed")
    name = dataset_name(config["dataset"])
    assert_b1_frozen(config, name)
    if tuple(LESION_CLASSES) != ("MA", "HE", "EX", "SE"):
        raise RuntimeError("lesion order must stay MA, HE, EX, SE")
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    assert_not_finished_run(run_dir)
    if score_only:
        _score_existing(config_path, config, run_dir, predict)
        return 0.0
    for split_name in (config["split_val"], config["split_test"]):
        if split_name == "train_official":
            raise ValueError("B1 does not read train_official")

    set_seed(config["seed"])
    model = build_model(config["model"])
    if tuple(model.classes) != tuple(LESION_CLASSES):
        raise ValueError(f"{config['model']} classes {model.classes} != {LESION_CLASSES}")
    if int(model.card.in_channels) != 3:
        raise ValueError("B1 prepared images are RGB; in_channels must be 3")
    declared = model.author_recipe(name)
    if config.get("experiment") == "BS":
        from bench.bstd import run_cell

        def _pretrained():
            if hasattr(model, "load_pretrained"):
                print(f"pretrained tensors copied: {model.load_pretrained()}", flush=True)

        return run_cell(
            config_path, config, model, declared, run_dir, predict, diagnostic, resume, before_training=_pretrained
        )
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
        if hasattr(model, "load_pretrained"):
            copied = model.load_pretrained()
            print(f"pretrained tensors copied: {copied}", flush=True)
        best = run_training(
            model,
            train_loader,
            val_loader,
            resolved,
            config,
            run_dir,
            micro,
            per_epoch,
            eval_every=evaluation_interval(declared, config),
            resume=resume,
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
    parser.add_argument("--resume", action="store_true", help="continue from run-dir/resume.pt when it exists")
    args = parser.parse_args(argv)
    if args.diagnostic and args.score_only:
        raise SystemExit("--diagnostic does not score the test split; drop one of the flags")
    train(args.config, args.run_dir, diagnostic=args.diagnostic, score_only=args.score_only, resume=args.resume)


if __name__ == "__main__":
    main()
