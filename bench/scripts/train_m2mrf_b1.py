"""Train M2MRF-C on the B1 split and write the same run directory as bench/train.py.

Loss, optimizer and iteration count come from author_recipe. The data are
dataset/prepared train, val and test: IDRiD train has 44 images. The canvas
is the field-of-view crop at fov_diameter, with horizontal flip probability
0.5. The loader batch is the author's 4-GPU effective batch, so BatchNorm
sees four images. The learning rate is not scaled. One process, one GPU.

    python bench/scripts/train_m2mrf_b1.py --config CONFIG
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

import yaml

from bench.common.io import LESION_CLASSES
from bench.data.b1_input import PreparedSplit, dataset_name
from bench.models.m2mrf import M2MRF, bind_normalization, refuse_author_runs
from bench.pretrained import hrnet_w48
from bench.runtime import (
    assert_b1_frozen,
    micro_batch_and_deviations,
    refuse_distributed,
    resolve_recipe,
    run_directory,
    set_seed,
    write_environment,
    write_recipe,
)
from bench.scripts.predict_m2mrf_b1 import predict
from bench.train import _loader, run_training, score_run


def _record_versions(path, model):
    payload = json.loads(Path(path).read_text())
    payload["torch"] = model.versions.get("torch")
    payload["mmcv"] = model.versions.get("mmcv")
    payload["mmcv_file"] = model.versions.get("mmcv_file")
    if model.versions.get("mmcv_error"):
        payload["mmcv_error"] = model.versions["mmcv_error"]
    payload["import_error"] = model.import_error
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def train(config_path, run_dir=None, diagnostic=False, score_only=False):
    refuse_distributed()
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if config.get("experiment") != "B1" or config.get("model") != "M2MRF":
        raise ValueError("this script trains B1 M2MRF; the author retrain stays in bench/train_m2mrf.py")
    if "seed" not in config:
        raise ValueError("the cell config must set seed")
    name = dataset_name(config["dataset"])
    assert_b1_frozen(config, name)
    if tuple(LESION_CLASSES) != ("MA", "HE", "EX", "SE"):
        raise RuntimeError("lesion order must stay MA, HE, EX, SE")
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    refuse_author_runs(run_dir)
    if score_only:
        from bench.train import _score_existing

        _score_existing(config_path, config, run_dir, predict)
        return 0.0

    set_seed(config["seed"])
    model = M2MRF()
    bind_normalization(model, name)
    if tuple(model.classes) != tuple(LESION_CLASSES):
        raise ValueError(f"M2MRF classes {model.classes} != {LESION_CLASSES}")
    declared = model.author_recipe(name)
    mean = model.card.normalization["mean"]
    std = model.card.normalization["std"]
    # PreparedSplit is the B1 canvas. Its augment flag is the 0.5 horizontal flip.
    train_set = PreparedSplit(
        config["dataset"], config["split_train"], config["fov_diameter"], True, mean, std, model.card.forward_size
    )
    val_set = PreparedSplit(
        config["dataset"], config["split_val"], config["fov_diameter"], False, mean, std, model.card.forward_size
    )
    resolved, deviations, per_epoch = resolve_recipe(declared, config, len(train_set))
    micro, batch_deviations = micro_batch_and_deviations(config, resolved)
    deviations = deviations + batch_deviations
    deviations.append(
        "learning rate stays at the author value; it is not multiplied by the GPU count"
    )
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
    _record_versions(run_dir / "recipe.json", model)

    best = None
    started = time.perf_counter()
    try:
        if model.network is None:
            raise RuntimeError(model.import_error or "M2MRF network was not imported")
        weight_path = config.get("pretrained") or hrnet_w48()
        copied = model.load_imagenet(weight_path)
        print(f"pretrained tensors copied into backbone: {copied} from {weight_path}", flush=True)
        train_loader = _loader(
            train_set, micro, True, config["num_workers"], model.card.pad_multiple, config["seed"]
        )
        val_loader = _loader(val_set, 1, False, config["num_workers"], model.card.pad_multiple, config["seed"])
        best = run_training(
            model, train_loader, val_loader, resolved, config, run_dir, micro, per_epoch
        )
    finally:
        seconds = time.perf_counter() - started
        parameters = model.n_parameters() if model.network is not None else 0
        write_environment(run_dir / "environment.json", config["seed"], parameters, seconds, best)
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
