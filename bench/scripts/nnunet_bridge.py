"""nnU-Net v2 on this project's data, splits and evaluation (E2, nnU-Net column).

nnU-Net keeps its own planning, preprocessing, augmentation, schedule and
checkpoint (checkpoint_final.pth). Shared with B-std: the prepared images,
the splits and bench.eval. Labels are exclusive: later classes in
M2MRF_OVERWRITE_ORDER overwrite earlier ones where masks overlap, so the
softmax channels are read back as MA, HE, EX, SE probabilities. Images are
the prepared JPEG decoded to RGB and written as PNG at original resolution.

    python bench/scripts/nnunet_bridge.py convert --data dataset/prepared/IDRiD --dataset-id 601
    python bench/scripts/nnunet_bridge.py splits  --data dataset/prepared/IDRiD --dataset-id 601
    python bench/scripts/nnunet_bridge.py export  --data dataset/prepared/IDRiD --npz-dir DIR --pred-dir DIR --split test
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from bench.common.io import (
    LESION_CLASSES,
    M2MRF_OVERWRITE_ORDER,
    image_path,
    load_meta,
    load_split,
    read_fov,
    read_mask,
    write_prob,
)

RAW = _ROOT / "dataset" / "nnunet" / "raw"
PREPROCESSED = _ROOT / "dataset" / "nnunet" / "preprocessed"
LABELS = {"background": 0, **{cls: index for index, cls in enumerate(M2MRF_OVERWRITE_ORDER, start=1)}}
TRAIN_SPLITS = ("train", "val")
PREDICT_SPLITS = ("val", "test")


def dataset_folder(data, dataset_id):
    return f"Dataset{int(dataset_id):03d}_{load_meta(data)['name']}"


def exclusive_label(data, image_id, shape):
    label = np.zeros(shape, dtype=np.uint8)
    for cls in M2MRF_OVERWRITE_ORDER:
        mask = read_mask(data, cls, image_id, shape)
        if mask.shape != shape:
            raise ValueError(f"{image_id} {cls} mask {mask.shape} != image {shape}")
        label[mask] = LABELS[cls]
    return label


def _convert_one(job):
    data, image_id, image_out, label_out = job
    rgb = np.asarray(Image.open(image_path(data, image_id)).convert("RGB"))
    Image.fromarray(rgb).save(image_out)
    if label_out is not None:
        Image.fromarray(exclusive_label(data, image_id, rgb.shape[:2])).save(label_out)
    return image_id


def convert(data, dataset_id, workers):
    data = Path(data)
    root = RAW / dataset_folder(data, dataset_id)
    jobs = []
    train_ids = []
    for split in TRAIN_SPLITS:
        train_ids += load_split(data, split)
    for sub in ("imagesTr", "labelsTr"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for image_id in train_ids:
        jobs.append((data, image_id, root / "imagesTr" / f"{image_id}_0000.png", root / "labelsTr" / f"{image_id}.png"))
    for split in PREDICT_SPLITS:
        (root / f"images_{split}").mkdir(parents=True, exist_ok=True)
        for image_id in load_split(data, split):
            jobs.append((data, image_id, root / f"images_{split}" / f"{image_id}_0000.png", None))
    with Pool(workers) as pool:
        for count, _ in enumerate(pool.imap_unordered(_convert_one, jobs), start=1):
            if count % 100 == 0:
                print(f"{count}/{len(jobs)}", flush=True)
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": LABELS,
        "numTraining": len(train_ids),
        "file_ending": ".png",
        "overwrite_order": list(M2MRF_OVERWRITE_ORDER),
        "source": str(data.relative_to(_ROOT)) if data.is_absolute() else str(data),
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    print(f"{root}: {len(train_ids)} training cases, {len(jobs) - len(train_ids)} prediction images")


def write_splits(data, dataset_id):
    """One fold: this project's train and val. Written after planning, before training."""
    data = Path(data)
    folder = PREPROCESSED / dataset_folder(data, dataset_id)
    if not (folder / "nnUNetPlans.json").is_file():
        raise FileNotFoundError(f"{folder} has no nnUNetPlans.json; run nnUNetv2_plan_and_preprocess first")
    splits = [{"train": load_split(data, "train"), "val": load_split(data, "val")}]
    path = folder / "splits_final.json"
    if path.is_file() and json.loads(path.read_text()) != splits:
        raise RuntimeError(f"{path} exists with different splits")
    path.write_text(json.dumps(splits, indent=2) + "\n")
    print(f"{path}: train {len(splits[0]['train'])}, val {len(splits[0]['val'])}")


def export(data, npz_dir, pred_dir, split):
    """Softmax channels to write_prob maps at original size, 0 outside the field of view."""
    data = Path(data)
    npz_dir = Path(npz_dir)
    for image_id in load_split(data, split):
        probs = np.load(npz_dir / f"{image_id}.npz")["probabilities"]
        if probs.ndim == 4:
            if probs.shape[1] != 1:
                raise ValueError(f"{image_id}: expected one 2-D slice, got {probs.shape}")
            probs = probs[:, 0]
        if probs.shape[0] != len(LABELS):
            raise ValueError(f"{image_id}: {probs.shape[0]} channels, labels are {LABELS}")
        fov = read_fov(data, image_id)
        if probs.shape[1:] != fov.shape:
            raise ValueError(f"{image_id}: probabilities {probs.shape[1:]} != image {fov.shape}")
        for cls in LESION_CLASSES:
            plane = np.clip(probs[LABELS[cls]].astype(np.float64), 0.0, 1.0)
            plane[~fov] = 0.0
            write_prob(pred_dir, cls, image_id, plane)
        print(image_id, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--data", required=True)
    c.add_argument("--dataset-id", type=int, required=True)
    c.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("splits")
    s.add_argument("--data", required=True)
    s.add_argument("--dataset-id", type=int, required=True)
    e = sub.add_parser("export")
    e.add_argument("--data", required=True)
    e.add_argument("--npz-dir", required=True)
    e.add_argument("--pred-dir", required=True)
    e.add_argument("--split", required=True)
    args = parser.parse_args(argv)
    if args.command == "convert":
        convert(args.data, args.dataset_id, args.workers)
    elif args.command == "splits":
        write_splits(args.data, args.dataset_id)
    else:
        export(args.data, args.npz_dir, args.pred_dir, args.split)


if __name__ == "__main__":
    main()
