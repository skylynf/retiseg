"""Retrain HRDecoder with the author's IDRiD recipe (E1r; docs/paper/26 N1a, docs/ops/17 section 4 item 20).

The segmentor, backbone, head, loss and the training-time geometry are the
author's (official_code/HRDecoder, the mmseg fork of the retiseg-hacdr
environment): the network is built from the author configs exactly as
bench/predict_hrdecoder.py builds it for the released checkpoints, and each
step calls the author's ``forward_train`` (lr path at hr_scale, crop_num
random hr crops, the fused map; Dice on the fused map + 0.1 x Dice on the
crops) and sums the losses the way mmseg's ``train_step`` does. Training
images are the official 54 (IDRiD_01-54) with the author's merged label
(overwrite order EX, HE, SE, MA). The augmentation is the author's train
pipeline run through the fork's own transform classes: Resize to 2880x1920
with ratio_range 0.5-2.0, RandomCrop 1920 (cat_max_ratio 0.75), horizontal
flip 0.5, rotation of +-45 degrees on every image, Normalize, Pad. The
schedule is poly10warm (1500-step linear warmup from 1e-6, then poly power
1.0 to 0) over 20,000 steps, SGD 0.01 / 0.9 / 5e-4. The backbone starts from
the ImageNet HRNet-W48 file the M2MRF rows use (open-mmlab://msra/hrnetv2_w48).

Deviations from the author's launch, all recorded in run_params.json:
  * four GPUs x 1 image with SyncBN become one process with 4 images and BN
    (same statistics per step; micro_batch < 4 would change that);
  * seed 0 on one process instead of the config's seed 2 on four;
  * the author's runner evaluates every 2000 steps on the TEST split and keeps
    the best mIoU (evaluation.save_best in the config). E1r reports the last
    step. Every 2000-step checkpoint is kept, so ``score --all-checkpoints``
    can show what that test-split selection would have returned
    (docs/paper/14 note 46; the released checkpoint's step is in its meta).

    conda run -n retiseg-hacdr python bench/train_hrdecoder.py train \\
        --config bench/configs/e1r_hrdecoder_idrid_seed0.yaml --run-dir runs/E1r_hrdecoder_idrid_seed0_bs4
    conda run -n retiseg-hacdr python bench/train_hrdecoder.py score --run-dir runs/E1r_hrdecoder_idrid_seed0_bs4 --all-checkpoints
    conda run -n retiseg python -m bench.eval.cli evaluate --data dataset/prepared/IDRiD --split test \\
        --pred runs/E1r_hrdecoder_idrid_seed0_bs4/prob_test --out runs/E1r_hrdecoder_idrid_seed0_bs4/metrics_test

``score`` writes the author's metric (bench.eval.m2mrf_stream, 11 thresholds)
to author_idrid_test.json and the probability maps of the last checkpoint to
prob_test/ for the protocol; the protocol evaluation itself runs in the
retiseg environment (collect_runs reads metrics_test.json and author_*_test.json).
"""

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUTHOR_CLASSES = ("EX", "HE", "SE", "MA")
DEFAULT_CONFIG = "bench/configs/e1r_hrdecoder_idrid_seed0.yaml"
DEFAULT_RUN = "runs/E1r_hrdecoder_idrid_seed0_bs4"


def _torch_load(path):
    import torch

    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu")


# ----------------------------------------------------------------------------------------------------------------- recipe


def load_config(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def check_against_author(config):
    """The yaml must say what the author configs say; a drift is an error, not a silent change."""
    from bench.models.hrdecoder import author_norm, author_recipe

    recipe = author_recipe(config["dataset"])
    params = recipe.optimizer_params
    expected = {
        "lr": recipe.lr,
        "weight_decay": recipe.weight_decay,
        "momentum": params["momentum"],
        "power": params["power"],
        "min_lr": params["min_lr"],
        "warmup_iters": params["warmup_iters"],
        "warmup_ratio": params["warmup_ratio"],
        "iterations": recipe.iterations,
        "batch_size": recipe.effective_batch_size,
        "crop_size": list(recipe.crop_size),
        "hr_scale": list(params["hr_scale_author"]),
        "crop_num": params["crop_num"],
        "scale_ratio": list(params["scale_ratio"]),
        "divisible": params["divisible"],
        "smooth": recipe.loss_params["smooth"],
        "hr_loss_weight": recipe.loss_params["hr_loss_weight"],
        "lr_loss_weight": recipe.loss_params["lr_loss_weight"],
    }
    wrong = {}
    for key, value in expected.items():
        mine = config[key]
        mine = list(mine) if isinstance(mine, (list, tuple)) else mine
        if isinstance(value, float) or isinstance(mine, float):
            same = abs(float(mine) - float(value)) <= 1e-12
        else:
            same = mine == value
        if not same:
            wrong[key] = {"yaml": mine, "author": value}
    if wrong:
        raise RuntimeError(f"{DEFAULT_CONFIG} disagrees with the author configs: {json.dumps(wrong)}")
    mean, std = author_norm(config["dataset"])
    return mean, std


# ---------------------------------------------------------------------------------------------------------------- dataset


def _pipeline(config, mean, std):
    """The author's train pipeline after LoadImageFromFile / LoadAnnotations, with the fork's transform classes."""
    from mmseg.datasets.pipelines.transforms import Normalize, Pad, RandomCrop, RandomFlip, RandomRotate, Resize

    crop = tuple(int(v) for v in config["crop_size"])
    return [
        Resize(img_scale=tuple(int(v) for v in config["image_scale"]), ratio_range=tuple(float(v) for v in config["ratio_range"])),
        RandomCrop(crop_size=crop, cat_max_ratio=0.75),
        RandomFlip(prob=float(config["flip_prob"]), direction="horizontal"),
        RandomRotate(prob=float(config["rotate_prob"]), degree=tuple(float(v) for v in config["rotate_degree"]), pad_val=0, seg_pad_val=0, auto_bound=False),
        Normalize(mean=mean, std=std, to_rgb=True),
        Pad(size=crop, pad_val=0, seg_pad_val=0),
    ]


class OfficialTrain:
    """IDRiD_01-54 from dataset/prepared/IDRiD (prepare copies the release JPEG byte for byte), author labels."""

    def __init__(self, repo_root, config, mean, std):
        import mmcv

        from bench.data.idrid import _official_train as official_train
        from bench.predict_hrdecoder import author_label

        prepared = Path(repo_root) / "dataset" / "prepared" / config["dataset"]
        splits = json.loads((prepared / "splits.json").read_text())
        self.ids = sorted(splits["train"] + splits["val"])
        if config["dataset"] == "IDRiD" and self.ids != list(official_train()):
            raise RuntimeError("prepared IDRiD train+val is not the official IDRiD_01-54")
        self.images, self.labels = {}, {}
        for image_id in self.ids:
            image = mmcv.imread(str(prepared / "images" / f"{image_id}.jpg"))
            self.images[image_id] = image
            self.labels[image_id] = author_label(prepared, image_id, image.shape[:2]).astype(np.uint8)
        self.pipeline = _pipeline(config, mean, std)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        import torch

        image_id = self.ids[index]
        image = self.images[image_id]
        results = {
            "img": image,
            "gt_semantic_seg": self.labels[image_id],
            "seg_fields": ["gt_semantic_seg"],
            "ori_shape": image.shape,
            "img_shape": image.shape,
            "filename": image_id,
        }
        for transform in self.pipeline:
            results = transform(results)
        tensor = torch.from_numpy(np.ascontiguousarray(results["img"].transpose(2, 0, 1)))
        label = torch.from_numpy(np.ascontiguousarray(results["gt_semantic_seg"].astype(np.int64)))
        meta = {
            "filename": image_id,
            "ori_shape": results["ori_shape"],
            "img_shape": results["img_shape"],
            "pad_shape": results["pad_shape"],
            "scale_factor": results["scale_factor"],
            "flip": results.get("flip", False),
            "img_norm_cfg": results["img_norm_cfg"],
        }
        return tensor, label.unsqueeze(0), meta


def _next_batch(dataset, order, cursor, sampler_rng, batch_size):
    import torch

    images, labels, metas = [], [], []
    for _ in range(batch_size):
        if cursor >= len(order):
            order = torch.randperm(len(dataset), generator=sampler_rng).tolist()
            cursor = 0
        image, label, meta = dataset[order[cursor]]
        cursor += 1
        images.append(image)
        labels.append(label)
        metas.append(meta)
    return torch.stack(images), torch.stack(labels), metas, order, cursor


# ------------------------------------------------------------------------------------------------------------------ model


def build_training_network(dataset, pretrained):
    """The author's segmentor for ``dataset`` (SyncBN -> BN), backbone initialised from the local HRNet-W48 file."""
    import torch

    from bench.models.hrdecoder import _Cfg, _as_cfg, _bn, _install_import_path, author_model_dict

    _install_import_path()
    from mmseg.models import build_segmentor

    cfg = _as_cfg(_bn(author_model_dict(dataset)))
    cfg["pretrained"] = str(pretrained)          # init_cfg Pretrained on the backbone, a local file instead of open-mmlab://
    test_cfg = cfg.pop("test_cfg", _Cfg({"mode": "whole"}))
    test_cfg["compute_aupr"] = True
    train_cfg = cfg.pop("train_cfg", _Cfg({}))
    network = build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    network.init_weights()                       # backbone <- file; conv_seg Normal(0, 0.01) as the author's runner
    state = _torch_load(pretrained)
    state = state.get("state_dict", state)
    backbone = network.backbone.state_dict()
    copied = sum(1 for key, value in state.items() if key in backbone and tuple(backbone[key].shape) == tuple(value.shape) and torch.equal(backbone[key], value.to(backbone[key].dtype)))
    print(f"pretrained tensors copied into backbone: {copied}/{len(backbone)} (file has {len(state)})", flush=True)
    if copied == 0:
        raise RuntimeError(f"{pretrained} matched no backbone tensor")
    geometry = {"hr_scale": list(cfg["hr_settings"]["hr_scale"]), "crop_num": int(cfg["hr_settings"]["crop_num"]),
                "test_cfg": {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in test_cfg.items()}}
    return network, geometry


def _checkpoint_backbone(network):
    """Recompute one HRNet stage at a time (for cards below 40 GB; BN running statistics then update twice per step)."""
    import torch

    backbone = network.backbone

    def wrap_tensor(module):
        original = module.forward

        def forward(x):
            return torch.utils.checkpoint.checkpoint(original, x)

        module.forward = forward

    def wrap_list(module):
        original = module.forward

        def run(*tensors):
            return tuple(original(list(tensors)))

        def forward(xs):
            return list(torch.utils.checkpoint.checkpoint(run, *xs))

        module.forward = forward

    wrap_tensor(backbone.layer1)
    wrap_list(backbone.stage2)
    wrap_list(backbone.stage3)
    wrap_list(backbone.stage4)


def poly10warm_lr(iteration, config):
    """mmcv PolyLrUpdaterHook with warmup='linear': the warmup factor multiplies the poly value of the same iteration."""
    base, floor = float(config["lr"]), float(config["min_lr"])
    power, total = float(config["power"]), int(config["iterations"])
    warmup, ratio = int(config["warmup_iters"]), float(config["warmup_ratio"])
    regular = (base - floor) * (1.0 - iteration / float(total)) ** power + floor
    if iteration < warmup:
        k = (1.0 - iteration / float(warmup)) * (1.0 - ratio)
        return regular * (1.0 - k)
    return regular


# -------------------------------------------------------------------------------------------------------------- checkpoints


def _rng_state():
    import torch

    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _restore_rng(payload):
    import torch

    torch.set_rng_state(payload["rng"]["torch"])
    torch.cuda.set_rng_state(payload["rng"]["cuda"])
    np.random.set_state(payload["rng"]["numpy"])
    random.setstate(payload["rng"]["python"])


def _save(path, network, iteration, optimizer=None, sampler_rng=None):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": network.state_dict(), "meta": {"iter": iteration, "model": "HRDecoder", "experiment": "E1r"}, "iter": iteration}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["sampler_rng"] = sampler_rng.get_state()
        payload["rng"] = _rng_state()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------------------------ train


def train(config_path, run_dir, iters, log_every, resume, checkpoint_backbone, micro_batch):
    import torch

    config = load_config(config_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mean, std = check_against_author(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    batch_size = int(config["batch_size"])
    micro = int(micro_batch or config.get("micro_batch", batch_size))
    if batch_size % micro:
        raise ValueError(f"micro_batch {micro} must divide batch_size {batch_size}")
    total = int(config["iterations"]) if iters is None else int(iters)
    save_every = int(config.get("save_every", 2000))

    dataset = OfficialTrain(ROOT, config, mean, std)
    network, geometry = build_training_network(config["dataset"], ROOT / config["pretrained"])
    network.to(device)
    if checkpoint_backbone:
        _checkpoint_backbone(network)
    optimizer = torch.optim.SGD(network.parameters(), lr=float(config["lr"]), momentum=float(config["momentum"]), weight_decay=float(config["weight_decay"]))
    sampler_rng = torch.Generator()
    sampler_rng.manual_seed(seed)
    start = 0
    resume_path = Path(resume) if resume else run_dir / "last.pt"
    if resume_path.is_file():
        payload = _torch_load(resume_path)
        network.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload)
        sampler_rng.set_state(payload["sampler_rng"])
        start = int(payload["iter"])
        print(f"resumed from {resume_path} at iteration {start}", flush=True)

    params = {
        "config": str(config_path), "recipe": config, "run_dir": str(run_dir), "n_train_images": len(dataset), "train_ids": dataset.ids,
        "geometry": geometry, "normalization": {"mean": mean, "std": std, "to_rgb": True},
        "effective_batch": batch_size, "micro_batch": micro, "bn_images_per_step": micro, "checkpoint_backbone": bool(checkpoint_backbone),
        "lr_schedule": "poly10warm (mmcv warmup factor x poly of the same iteration)", "loss_sum": "hr.loss_seg + fuse.loss_seg (BaseSegmentor._parse_losses)",
        "deviations": [
            "4 GPUs x 1 image with SyncBN -> one process, BN over the stacked images",
            "seed 0 on one process; the author config's seed 2 drove four data-loader processes",
            "no evaluation during training; the author runner evaluates every 2000 iterations on the test split and keeps the best mIoU (save_best)",
        ],
        "checkpoint_selection": config.get("checkpoint_selection", "last"), "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "run_params.json").write_text(json.dumps(params, indent=1, default=str) + "\n")

    history = run_dir / "history.jsonl"
    order, cursor = [], 0
    network.train()
    print(f"effective batch {batch_size} (micro {micro}), {total} iterations, {len(dataset)} images", flush=True)
    tick, last_logged = time.time(), start
    for iteration in range(start, total):
        lr = poly10warm_lr(iteration, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logged = {}
        loss_value = 0.0
        for _ in range(batch_size // micro):
            image, label, meta, order, cursor = _next_batch(dataset, order, cursor, sampler_rng, micro)
            losses = network.forward_train(image.to(device, non_blocking=True), meta, label.to(device, non_blocking=True))
            loss, log_vars = network._parse_losses(losses)
            (loss * micro / batch_size).backward()
            loss_value += float(loss.detach()) * micro / batch_size
            for key, value in log_vars.items():
                logged[key] = logged.get(key, 0.0) + float(value) * micro / batch_size
        optimizer.step()
        done = iteration + 1
        if done % log_every == 0 or done == total:
            now = time.time()
            row = {"iter": done, "loss": loss_value, "lr": lr, "mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
                   "sec_per_iter": (now - tick) / max(1, done - last_logged), **{k: v for k, v in logged.items() if k != "loss"}}
            tick, last_logged = now, done
            print(f"iter {done} loss {loss_value:.4f} lr {lr:.6f} mem {row['mem_gb']:.2f}GB {row['sec_per_iter']:.2f}s/it", flush=True)
            with history.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        if done % save_every == 0 or done == total:
            _save(run_dir / f"iter_{done}.pt", network, done)
            _save(run_dir / "last.pt", network, done, optimizer, sampler_rng)
    print(f"finished {total}", flush=True)


# ------------------------------------------------------------------------------------------------------------------ score


def _author_test(network, config, mean, std, prepared, ids, device, prob_dir=None):
    """Author test pipeline on ``ids``; returns the author metric dict, writes maps when ``prob_dir`` is given."""
    from bench.common.io import image_path, write_prob
    from bench.eval.m2mrf_stream import AuthorAccumulator
    from bench.predict_hrdecoder import author_label, predict, preprocess

    accumulator = AuthorAccumulator()
    for image_id in ids:
        tensor, img_meta = preprocess(image_path(prepared, image_id), mean, std, tuple(int(v) for v in config["image_scale"]))
        prob = predict(network, tensor, img_meta, device)
        if prob_dir is not None:
            for channel, cls in enumerate(AUTHOR_CLASSES):
                write_prob(prob_dir, cls, image_id, np.clip(prob[channel], 0, 1))
        accumulator.add(prob, author_label(prepared, image_id, tuple(prob.shape[1:])))
    iou, f1, aupr = accumulator.scores()
    names = ("bg",) + AUTHOR_CLASSES
    return {
        "classes": list(names),
        "implementation": "bench.eval.m2mrf_stream.AuthorAccumulator (11 thresholds, as the fork's lesion_metric.py)",
        "per_class": {name: {"iou": float(iou[i]), "f1": float(f1[i]), "aupr": float(aupr[i])} for i, name in enumerate(names)},
        "mIoU": float(np.nanmean(np.nan_to_num(iou[-4:], nan=0))),
        "mF1": float(np.nanmean(np.nan_to_num(f1[-4:], nan=0))),
        "mAUPR": float(np.nanmean(np.nan_to_num(aupr[-4:], nan=0))),
    }


def score(config_path, run_dir, checkpoint, all_checkpoints, limit, device_name):
    import torch

    from bench.common.io import LESION_CLASSES, M2MRF_OVERWRITE_ORDER, load_split
    from bench.predict_hrdecoder import load_model

    config = load_config(config_path)
    run_dir = Path(run_dir)
    mean, std = check_against_author(config)
    prepared = ROOT / "dataset" / "prepared" / config["dataset"]
    ids = load_split(prepared, "test")
    if limit:
        ids = ids[:limit]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    total = int(config["iterations"])
    primary = Path(checkpoint) if checkpoint else run_dir / f"iter_{total}.pt"
    if not primary.is_file():
        raise FileNotFoundError(f"{primary}: train to {total} iterations first")

    network, info = load_model(primary, device, config["dataset"])
    prob_dir = run_dir / "prob_test"
    metric = _author_test(network, config, mean, std, prepared, ids, device, prob_dir)
    summary = {
        "dataset": config["dataset"].lower(), "model": "HRDecoder", "experiment": "E1r", "checkpoint": str(primary),
        "checkpoint_meta": {"iter": info.get("meta", {}).get("iter")}, "checkpoint_selection": config.get("checkpoint_selection", "last"),
        "test_preprocess": {"image_scale": list(config["image_scale"]), "keep_ratio": False, "flip": False, "mean": mean, "std": std, "to_rgb": True,
                            "hr_scale": info.get("hr_scale"), "test_cfg": info.get("test_cfg"),
                            "source": "official_code/HRDecoder/configs/_base_/datasets, configs/lesion (as bench/predict_hrdecoder.py)"},
        "n_images": len(ids), "image_ids": ids,
        "author_metric": {**metric, "overwrite_order": list(M2MRF_OVERWRITE_ORDER)},
        "prob_dir": str(prob_dir), "protocol_classes": list(LESION_CLASSES),
    }
    (run_dir / f"author_{config['dataset'].lower()}_test.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: metric[k] for k in ("mIoU", "mF1", "mAUPR")}), flush=True)

    if all_checkpoints:
        # what the author's evaluation.save_best='mIoU' on the test split would have picked (docs/paper/14 note 46)
        rows = []
        for path in sorted(run_dir.glob("iter_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
            step = int(path.stem.split("_")[1])
            if path == primary:
                row = {"iter": step, **{k: metric[k] for k in ("mIoU", "mF1", "mAUPR")}, "per_class": metric["per_class"]}
            else:
                net_k, _ = load_model(path, device, config["dataset"])
                m = _author_test(net_k, config, mean, std, prepared, ids, device)
                del net_k
                row = {"iter": step, **{k: m[k] for k in ("mIoU", "mF1", "mAUPR")}, "per_class": m["per_class"]}
            print(json.dumps({k: row[k] for k in ("iter", "mIoU", "mAUPR")}), flush=True)
            rows.append(row)
        best = max(rows, key=lambda r: r["mIoU"])
        last = max(rows, key=lambda r: r["iter"])
        (run_dir / "selection_on_test.json").write_text(json.dumps({
            "rule_reported": "last step (E1r)", "author_rule": "evaluation.save_best='mIoU' every 2000 iterations on the test split",
            "n_images": len(ids), "checkpoints": rows,
            "last": {k: last[k] for k in ("iter", "mIoU", "mAUPR")}, "best_mIoU_on_test": {k: best[k] for k in ("iter", "mIoU", "mAUPR")},
            "gain_of_test_selection": {"mIoU": best["mIoU"] - last["mIoU"], "mAUPR": best["mAUPR"] - last["mAUPR"]},
        }, indent=1) + "\n")
    print("protocol evaluation (retiseg environment):", flush=True)
    print(f"  python -m bench.eval.cli evaluate --data {prepared.relative_to(ROOT)} --split test --pred {prob_dir} --out {run_dir / 'metrics_test'}", flush=True)


# ------------------------------------------------------------------------------------------------------------------- cli


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="author recipe, one process")
    t.add_argument("--config", default=DEFAULT_CONFIG)
    t.add_argument("--run-dir", default=DEFAULT_RUN)
    t.add_argument("--iters", type=int, default=None, help="stop early (smoke); the schedule still spans the config's iterations")
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--resume", default="", help="checkpoint to resume from (default: <run-dir>/last.pt when present)")
    t.add_argument("--micro-batch", type=int, default=None, help="images per forward; gradients accumulate to batch_size")
    t.add_argument("--checkpoint-backbone", action="store_true", help="gradient checkpointing of the HRNet stages (small cards)")
    s = sub.add_parser("score", help="author metric and protocol maps of a checkpoint on the test split")
    s.add_argument("--config", default=DEFAULT_CONFIG)
    s.add_argument("--run-dir", default=DEFAULT_RUN)
    s.add_argument("--checkpoint", default="", help="default: iter_<iterations>.pt (the last step)")
    s.add_argument("--all-checkpoints", action="store_true", help="also score every iter_*.pt -> selection_on_test.json")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.config, args.run_dir, args.iters, args.log_every, args.resume, args.checkpoint_backbone, args.micro_batch)
    else:
        score(args.config, args.run_dir, args.checkpoint, args.all_checkpoints, args.limit, args.device)


if __name__ == "__main__":
    main()
