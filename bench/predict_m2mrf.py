"""Run the author's M2MRF-C checkpoint on the official IDRiD test set.

The network class is the one in official_code/M2MRF. This file only loads that
class, applies the test pipeline written in configs/_base_/datasets/idrid.py,
and scores the maps with mmseg/core/evaluation/my_metrics.py. A second pass
writes the same probabilities in this project's class order so bench.eval can
score them under the frozen protocol.

Author test settings, from that config:
  image_scale (1440, 960), keep_ratio resize, no flip, no pad
  mean [116.513, 56.437, 16.309], std [80.206, 41.232, 13.293], BGR to RGB
  sigmoid, four channels in the order EX, HE, SE, MA
  labels merged by tools/prepare_labels.py: later classes overwrite earlier ones
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "compat"))
sys.path.insert(0, str(ROOT / "official_code" / "M2MRF"))
np.float = np.float64

import mmcv
import torch

from bench.common.io import LESION_CLASSES, M2MRF_OVERWRITE_ORDER, load_split, read_mask, write_prob
from bench.data.idrid import open_idrid
from bench.eval.m2mrf_stream import AuthorAccumulator
from mmseg.models import build_segmentor

class Cfg(dict):
    """dict with attribute access, which is what the author's test_cfg expects."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


AUTHOR_CLASSES = ("EX", "HE", "SE", "MA")
NORM_CFG = dict(type="SyncBN", requires_grad=True)
# Each entry is one author's test config. IDRiD and DDR do not share a scale or a normalization.
DATASETS = {
    "idrid": {
        "mean": [116.513, 56.437, 16.309],
        "std": [80.206, 41.232, 13.293],
        "image_scale": (1440, 960),
        "checkpoint": ROOT / "runs" / "E1_m2mrf" / "fcn_hr48-M2MRF-C_40k_idrid.pth",
        "prepared": ROOT / "dataset" / "prepared" / "IDRiD",
        "prob_dir": ROOT / "runs" / "E1_m2mrf" / "prob_test",
        "summary": ROOT / "runs" / "E1_m2mrf" / "author_idrid_test.json",
        "readme": {"mIOU_percent": 50.17, "mAUPR_percent": 67.55},
        "paper_three_run_mean": {"mAUPR_percent": 67.24, "mF_percent": 65.71, "mIoU_percent": 49.94},
    },
    "ddr": {
        "mean": [81.205, 50.636, 21.216],
        "std": [76.252, 48.798, 21.625],
        "image_scale": (1024, 1024),
        "checkpoint": ROOT / "runs" / "E1_m2mrf" / "fcn_hr48-M2MRF-C_60k_ddr.pth",
        "prepared": ROOT / "dataset" / "prepared" / "DDR",
        "prob_dir": ROOT / "runs" / "E1_m2mrf" / "ddr_prob_test",
        "summary": ROOT / "runs" / "E1_m2mrf" / "author_ddr_test.json",
        "readme": {"mIOU_percent": 30.39, "mAUPR_percent": 49.20},
        "paper_three_run_mean": {"mAUPR_percent": 48.94, "mF_percent": 45.40, "mIoU_percent": 30.09},
    },
}


def model_cfg():
    """Merged fcn_hr48-M2MRF-C_40k_idrid_bdice.py. pretrained is cleared."""
    return dict(
        type="EncoderDecoder",
        pretrained=None,
        use_sigmoid=True,
        backbone=dict(
            type="HRNet_M2MRF_C",
            norm_cfg=NORM_CFG,
            norm_eval=False,
            m2mrf_patch_size=(8, 8),
            m2mrf_encode_channels_rate=4,
            m2mrf_fc_channels_rate=64,
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block="BOTTLENECK",
                    num_blocks=(4,),
                    num_channels=(64,),
                ),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block="BASIC",
                    num_blocks=(4, 4),
                    num_channels=(48, 96),
                ),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block="BASIC",
                    num_blocks=(4, 4, 4),
                    num_channels=(48, 96, 192),
                ),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block="BASIC",
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(48, 96, 192, 384),
                ),
            ),
        ),
        decode_head=dict(
            type="FCNHead",
            in_channels=[48, 96, 192, 384],
            in_index=(0, 1, 2, 3),
            channels=720,
            input_transform="resize_concat",
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=4,
            norm_cfg=NORM_CFG,
            align_corners=False,
            loss_decode=dict(type="BinaryLoss", loss_type="dice", loss_weight=1.0, smooth=1e-5),
        ),
    )


def load_model(checkpoint_path, device):
    model = build_segmentor(
        model_cfg(), train_cfg=Cfg(), test_cfg=Cfg(mode="whole", compute_aupr=True)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    real_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match HRNet_M2MRF_C: "
            f"{len(real_missing)} missing, {len(unexpected)} unexpected. "
            f"missing[:8]={real_missing[:8]} unexpected[:8]={unexpected[:8]}"
        )
    model.to(device)
    model.eval()
    return model, checkpoint.get("meta", {})


def preprocess(path, mean, std, image_scale):
    """Author test pipeline: keep-ratio resize, no flip, BGR file read, then RGB normalization."""
    image = mmcv.imread(path)
    ori_shape = image.shape
    resized, _ = mmcv.imrescale(image, image_scale, return_scale=True)
    normalized = mmcv.imnormalize(resized, mean, std, to_rgb=True)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    meta = dict(
        filename=str(path),
        ori_shape=ori_shape,
        img_shape=resized.shape,
        pad_shape=resized.shape,
        scale_factor=np.array(
            [resized.shape[1] / ori_shape[1], resized.shape[0] / ori_shape[0]] * 2,
            dtype=np.float32,
        ),
        flip=False,
        img_norm_cfg=dict(mean=np.array(mean, dtype=np.float32), std=np.array(std, dtype=np.float32), to_rgb=True),
    )
    return tensor, [meta]


def author_label(dataset_dir, image_id, shape):
    """Same pixels the protocol reads. Later classes overwrite, as in prepare_labels.py."""
    label = np.zeros(shape, dtype=np.int32)
    for index, cls in enumerate(AUTHOR_CLASSES, start=1):
        mask = read_mask(dataset_dir, cls, image_id, shape)
        if mask.shape != shape:
            raise ValueError(f"{image_id} {cls} mask {mask.shape} != image {shape}")
        label[mask] = index
    return label


def _jsonable(obj):
    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _image_path(spec, image_id):
    if spec is DATASETS["idrid"]:
        return open_idrid(ROOT).image_path(image_id)
    path = spec["prepared"] / "images" / f"{image_id}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="idrid")
    parser.add_argument("--limit", type=int, default=0, help="if positive, only the first N test images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="", help="override the author's released weight")
    parser.add_argument("--prob-dir", default="", help="where to write 16-bit probability maps")
    parser.add_argument("--summary", default="", help="where to write the author-metric json")
    args = parser.parse_args()
    spec = dict(DATASETS[args.dataset])
    if args.checkpoint:
        spec["checkpoint"] = Path(args.checkpoint)
    if args.prob_dir:
        spec["prob_dir"] = Path(args.prob_dir)
    if args.summary:
        spec["summary"] = Path(args.summary)
    if not Path(spec["checkpoint"]).is_file():
        raise FileNotFoundError(spec["checkpoint"])
    Path(spec["summary"]).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ids = load_split(spec["prepared"], "test")
    if args.limit:
        ids = ids[: args.limit]
    model, meta = load_model(spec["checkpoint"], device)
    accumulator = AuthorAccumulator()
    for image_id in ids:
        tensor, img_meta = preprocess(_image_path(spec, image_id), spec["mean"], spec["std"], spec["image_scale"])
        with torch.no_grad():
            prediction = model.simple_test(tensor.to(device), img_meta, rescale=True)
        prob = prediction[0][0]
        if prob.shape[0] != 4:
            raise RuntimeError(f"expected 4 channels, got {prob.shape}")
        height, width = img_meta[0]["ori_shape"][:2]
        if prob.shape[1:] != (height, width):
            raise RuntimeError(f"{image_id} probability {prob.shape} != original {(height, width)}")
        for channel, cls in enumerate(AUTHOR_CLASSES):
            write_prob(spec["prob_dir"], cls, image_id, np.clip(prob[channel], 0, 1))
        label = author_label(spec["prepared"], image_id, (height, width))
        accumulator.add(prob, label)
        print(image_id, "prob", [round(float(prob[c].mean()), 4) for c in range(4)], flush=True)

    iou, f1, aupr = accumulator.scores()
    names = ("bg",) + AUTHOR_CLASSES
    per_class = {}
    for index, name in enumerate(names):
        per_class[name] = {"iou": float(iou[index]), "f1": float(f1[index]), "aupr": float(aupr[index])}
    summary = {
        "dataset": args.dataset,
        "checkpoint": str(spec["checkpoint"]),
        "test_preprocess": {
            "image_scale": list(spec["image_scale"]),
            "keep_ratio": True,
            "flip": False,
            "mean": spec["mean"],
            "std": spec["std"],
            "to_rgb": True,
            "source": "official_code/M2MRF/configs/_base_/datasets",
        },
        "checkpoint_meta": {
            "iter": meta.get("iter") if isinstance(meta, dict) else None,
            "mmseg_version": meta.get("mmseg_version") if isinstance(meta, dict) else None,
            "time": meta.get("time") if isinstance(meta, dict) else None,
        },
        "n_images": len(ids),
        "image_ids": ids,
        "readme": spec["readme"],
        "paper_three_run_mean": spec["paper_three_run_mean"],
        "readme_note": "README says the paper reports the mean of three runs and the code reports the best run.",
        "author_metric": {
            "classes": list(names),
            "overwrite_order": list(M2MRF_OVERWRITE_ORDER),
            "per_class": per_class,
            "mIoU": float(np.nanmean(np.nan_to_num(iou[-4:], nan=0))),
            "mF1": float(np.nanmean(np.nan_to_num(f1[-4:], nan=0))),
            "mAUPR": float(np.nanmean(np.nan_to_num(aupr[-4:], nan=0))),
        },
        "prob_dir": str(spec["prob_dir"]),
        "protocol_classes": list(LESION_CLASSES),
    }
    spec["summary"].write_text(json.dumps(_jsonable(summary), indent=1) + "\n")
    print(json.dumps(_jsonable(summary["author_metric"]), indent=1))


if __name__ == "__main__":
    main()
