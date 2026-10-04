"""Retrain M2MRF-C with the author's IDRiD recipe.

The network, loss, augmentation and schedule come from
official_code/M2MRF/configs/m2mrf/fcn_hr48-M2MRF-C_40k_idrid_bdice.py.
Training images are the official 54 (IDRiD_01-54). Labels use the author's
overwrite order EX, HE, SE, MA. The backbone starts from the ImageNet
HRNet-W48 weights named in configs/_base_/models/fcn_hr48.py.
SyncBN is BatchNorm2d, as in the inference stand-in.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "compat"))
sys.path.insert(0, str(ROOT / "official_code" / "M2MRF"))
np.float = np.float64

import mmcv
import torch
import yaml

from bench.data.idrid import open_idrid
from bench.predict_m2mrf import Cfg, author_label, model_cfg
from mmseg.datasets.pipelines.transforms import (
    Normalize,
    Pad,
    PhotoMetricDistortion,
    RandomCrop,
    RandomFlip,
    Resize,
)
from mmseg.models import build_segmentor

MEAN = [116.513, 56.437, 16.309]
STD = [80.206, 41.232, 13.293]
IMAGE_SCALE = (1440, 960)
CROP_SIZE = (960, 1440)


def _pipeline():
    return [
        Resize(img_scale=IMAGE_SCALE, ratio_range=(0.5, 2.0)),
        RandomCrop(crop_size=CROP_SIZE, cat_max_ratio=0.75),
        RandomFlip(flip_ratio=0),
        PhotoMetricDistortion(),
        Normalize(mean=MEAN, std=STD, to_rgb=True),
        Pad(size=CROP_SIZE, pad_val=0, seg_pad_val=0),
    ]


def _poly_lr(iteration, max_iters, base_lr=0.01, min_lr=1e-4, power=0.9):
    coeff = (1 - iteration / max_iters) ** power
    return (base_lr - min_lr) * coeff + min_lr


class OfficialTrain(torch.utils.data.Dataset):
    def __init__(self, repo_root):
        data = open_idrid(repo_root)
        self.ids = list(data.splits()["train_official"])
        prepared = Path(repo_root) / "dataset" / "prepared" / "IDRiD"
        self.images = {}
        self.labels = {}
        for image_id in self.ids:
            image = mmcv.imread(data.image_path(image_id))
            label = author_label(prepared, image_id, image.shape[:2]).astype(np.uint8)
            self.images[image_id] = image
            self.labels[image_id] = label
        self.pipeline = _pipeline()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
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


def _build(pretrained, device):
    model = build_segmentor(model_cfg(), train_cfg=Cfg(), test_cfg=Cfg(mode="whole", compute_aupr=True))
    before = {name: tensor.detach().clone() for name, tensor in model.backbone.named_parameters()}
    model.init_weights(pretrained=str(pretrained))
    copied = 0
    for name, tensor in model.backbone.named_parameters():
        if not torch.equal(before[name], tensor):
            copied += 1
    print(f"pretrained tensors copied into backbone: {copied}/{len(before)}", flush=True)
    if copied == 0:
        raise RuntimeError("HRNet pretrained file matched no backbone parameters")
    model.to(device)
    return model


def _save(path, model, optimizer, iteration, sampler_rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": {"iter": iteration},
            "iter": iteration,
            "sampler_rng": sampler_rng.get_state(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        path,
    )


def _restore_rng(payload):
    torch.set_rng_state(payload["rng"]["torch"])
    torch.cuda.set_rng_state(payload["rng"]["cuda"])
    np.random.set_state(payload["rng"]["numpy"])
    random.setstate(payload["rng"]["python"])


def _checkpoint_backbone(model):
    """Recompute one HRNet stage at a time so four images fit in 16GB."""
    backbone = model.backbone

    def wrap_tensor(module):
        original = module.forward

        def forward(x):
            return torch.utils.checkpoint.checkpoint(original, x, use_reentrant=False)

        module.forward = forward

    def wrap_list(module):
        original = module.forward

        def run(*tensors):
            return tuple(original(list(tensors)))

        def forward(xs):
            outputs = torch.utils.checkpoint.checkpoint(run, *xs, use_reentrant=False)
            return list(outputs)

        module.forward = forward

    wrap_tensor(backbone.layer1)
    wrap_list(backbone.stage2)
    wrap_list(backbone.stage3)
    wrap_list(backbone.stage4)


def _next_batch(dataset, order, cursor, sampler_rng, batch_size):
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


def train(run_dir, pretrained, iterations, schedule_iters, save_every, log_every, seed, resume, batch_size):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    dataset = OfficialTrain(ROOT)
    model = _build(pretrained, device)
    _checkpoint_backbone(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    sampler_rng = torch.Generator()
    sampler_rng.manual_seed(seed)
    start = 0
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload)
        sampler_rng.set_state(payload["sampler_rng"])
        start = int(payload["iter"])
    history = run_dir / "history.jsonl"
    order = []
    cursor = 0
    model.train()
    print(f"effective batch {batch_size}", flush=True)
    for iteration in range(start, iterations):
        image, label, meta, order, cursor = _next_batch(dataset, order, cursor, sampler_rng, batch_size)
        lr = _poly_lr(iteration, schedule_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses = model.forward_train(image.to(device), meta, label.to(device))
        loss = losses["decode.loss_seg"]
        loss.backward()
        optimizer.step()
        done = iteration + 1
        if done % log_every == 0 or done == iterations:
            row = {
                "iter": done,
                "loss": float(loss.detach()),
                "lr": lr,
                "mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
            print(
                f"iter {done} loss {row['loss']:.4f} lr {lr:.6f} mem {row['mem_gb']:.2f}GB",
                flush=True,
            )
            with history.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        if done % save_every == 0 or done == iterations:
            _save(run_dir / "last.pt", model, optimizer, done, sampler_rng)
            _save(run_dir / f"iter_{done}.pt", model, optimizer, done, sampler_rng)
    print(f"finished {iterations}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bench/configs/e1r_m2mrf_idrid_seed0.yaml")
    parser.add_argument("--run-dir", default="runs/E1r_m2mrf_idrid_seed0_bs4")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    schedule_iters = int(config["iterations"])
    train(
        args.run_dir,
        ROOT / config["pretrained"],
        int(schedule_iters if args.iters is None else args.iters),
        schedule_iters,
        args.save_every,
        args.log_every,
        int(config["seed"]),
        args.resume or None,
        int(config["batch_size"]),
    )


if __name__ == "__main__":
    main()
