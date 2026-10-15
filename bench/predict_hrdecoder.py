"""Run the released HRDecoder checkpoints through the author's own test pipeline (E1).

The segmentor is the one in official_code/HRDecoder (an mmseg fork derived
from M2MRF's; retiseg-hacdr environment). This file builds it from the
author's configs (bench/models/hrdecoder.py reads them), loads a released
checkpoint, applies the test pipeline of the dataset config and scores the
maps with the author's 11-point AUPR (bench.eval.m2mrf_stream, the same code
the fork's lesion_metric.py carries). bench.e3a.run_author reuses load_model
and preprocess for the independent tests.

Author test settings, from configs/_base_/datasets and configs/lesion:
  IDRiD  resize to 2880 x 1920 (keep_ratio False), slide crop 1920 stride 960,
         mean [116.513, 56.437, 16.309] std [80.206, 41.232, 13.293]
  DDR    resize to 2048 x 2048 (keep_ratio False), whole image,
         mean [81.205, 50.636, 21.216] std [76.252, 48.798, 21.625]
  no flip, BGR file read then RGB normalization, sigmoid, channels EX, HE, SE, MA;
  hr_scale 960 (IDRiD) / 1024 (DDR), as trained.

    python -m bench.predict_hrdecoder --dataset idrid [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUTHOR_CLASSES = ("EX", "HE", "SE", "MA")
README = {  # README table of CVIU-CSU/HRDecoder (released model), percent
    "idrid": {"mIoU_percent": 52.20, "mAUPR_percent": 71.34},
    "ddr": {"mIoU_percent": 32.24, "mAUPR_percent": 49.24},
}
DATASETS = {
    "idrid": {
        "config_name": "IDRiD",
        "mean": [116.513, 56.437, 16.309],
        "std": [80.206, 41.232, 13.293],
        "image_scale": (2880, 1920),   # (width, height), mmseg convention
        "checkpoint": ROOT / "pretrained" / "hrdecoder_idrid.pth",
        "prepared": ROOT / "dataset" / "prepared" / "IDRiD",
        "prob_dir": ROOT / "runs" / "E1_hrdecoder" / "prob_test",
        "summary": ROOT / "runs" / "E1_hrdecoder" / "author_idrid_test.json",
        "readme": README["idrid"],
    },
    "ddr": {
        "config_name": "DDR",
        "mean": [81.205, 50.636, 21.216],
        "std": [76.252, 48.798, 21.625],
        "image_scale": (2048, 2048),
        "checkpoint": ROOT / "pretrained" / "hrdecoder_ddr.pth",
        "prepared": ROOT / "dataset" / "prepared" / "DDR",
        "prob_dir": ROOT / "runs" / "E1_hrdecoder" / "ddr_prob_test",
        "summary": ROOT / "runs" / "E1_hrdecoder" / "author_ddr_test.json",
        "readme": README["ddr"],
    },
}


def build_author_network(dataset):
    """The author's segmentor exactly as configured for ``dataset`` (hr_scale, test_cfg), SyncBN -> BN."""
    from bench.models.hrdecoder import _Cfg, _as_cfg, _bn, _install_import_path, author_model_dict

    _install_import_path()
    from mmseg.models import build_segmentor

    cfg = _as_cfg(_bn(author_model_dict(dataset)))
    cfg["pretrained"] = None
    test_cfg = cfg.pop("test_cfg", _Cfg({"mode": "whole"}))
    test_cfg["compute_aupr"] = True
    train_cfg = cfg.pop("train_cfg", _Cfg({}))
    network = build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    return network, {"hr_scale": list(cfg["hr_settings"]["hr_scale"]), "test_cfg": {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in test_cfg.items()}}


def load_model(checkpoint_path, device, dataset="IDRiD"):
    import torch

    network, geometry = build_author_network(dataset)
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in torch.load.__code__.co_varnames:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(str(checkpoint_path), **load_kwargs)
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = network.load_state_dict(state, strict=False)
    real_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the HRDecoder config for {dataset}: {len(real_missing)} missing, "
            f"{len(unexpected)} unexpected. missing[:8]={real_missing[:8]} unexpected[:8]={unexpected[:8]}"
        )
    network.to(device)
    network.eval()
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    return network, {"meta": meta, **geometry}


def preprocess(path, mean, std, image_scale):
    """Author test pipeline: resize to image_scale (keep_ratio False), no flip, BGR read, RGB normalization."""
    import mmcv
    import torch

    image = mmcv.imread(str(path))
    ori_shape = image.shape
    resized = mmcv.imresize(image, tuple(int(v) for v in image_scale))   # (w, h)
    normalized = mmcv.imnormalize(resized, np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32), to_rgb=True)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
    meta = dict(
        filename=str(path),
        ori_shape=ori_shape,
        img_shape=resized.shape,
        pad_shape=resized.shape,
        scale_factor=np.array([resized.shape[1] / ori_shape[1], resized.shape[0] / ori_shape[0]] * 2, dtype=np.float32),
        flip=False,
        img_norm_cfg=dict(mean=np.array(mean, dtype=np.float32), std=np.array(std, dtype=np.float32), to_rgb=True),
    )
    return tensor, [meta]


def predict(network, tensor, img_meta, device):
    """Four sigmoid maps (EX, HE, SE, MA) at the original image size."""
    import torch

    with torch.no_grad():
        out = network.simple_test(tensor.to(device), img_meta, rescale=True)
    prob = out[0][0] if isinstance(out[0], tuple) else out[0]
    prob = np.asarray(prob)
    if prob.shape[0] != 4:
        raise RuntimeError(f"expected 4 channels, got {prob.shape}")
    height, width = img_meta[0]["ori_shape"][:2]
    if prob.shape[1:] != (height, width):
        raise RuntimeError(f"probability {prob.shape} != original {(height, width)}")
    return prob


def author_label(dataset_dir, image_id, shape):
    """The author's merged label: later classes overwrite (prepare_labels.py of the M2MRF/HRDecoder forks)."""
    from bench.common.io import read_mask

    label = np.zeros(shape, dtype=np.int32)
    for index, cls in enumerate(AUTHOR_CLASSES, start=1):
        mask = read_mask(dataset_dir, cls, image_id, shape)
        if mask.shape != shape:
            raise ValueError(f"{image_id} {cls} mask {mask.shape} != image {shape}")
        label[mask] = index
    return label


def _jsonable(obj):
    import math

    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main(argv=None):
    import torch

    from bench.common.io import LESION_CLASSES, M2MRF_OVERWRITE_ORDER, image_path, load_split, write_prob
    from bench.eval.m2mrf_stream import AuthorAccumulator

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="idrid")
    parser.add_argument("--limit", type=int, default=0, help="if positive, only the first N test images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="", help="override the released weight path")
    parser.add_argument("--prob-dir", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)
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
    network, info = load_model(spec["checkpoint"], device, spec["config_name"])
    accumulator = AuthorAccumulator()
    for image_id in ids:
        tensor, img_meta = preprocess(image_path(spec["prepared"], image_id), spec["mean"], spec["std"], spec["image_scale"])
        prob = predict(network, tensor, img_meta, device)
        for channel, cls in enumerate(AUTHOR_CLASSES):
            write_prob(spec["prob_dir"], cls, image_id, np.clip(prob[channel], 0, 1))
        label = author_label(spec["prepared"], image_id, tuple(prob.shape[1:]))
        accumulator.add(prob, label)
        print(image_id, "prob", [round(float(prob[c].mean()), 4) for c in range(4)], flush=True)

    iou, f1, aupr = accumulator.scores()
    names = ("bg",) + AUTHOR_CLASSES
    per_class = {name: {"iou": float(iou[i]), "f1": float(f1[i]), "aupr": float(aupr[i])} for i, name in enumerate(names)}
    meta = info.get("meta") or {}
    summary = {
        "dataset": args.dataset,
        "model": "HRDecoder",
        "checkpoint": str(spec["checkpoint"]),
        "test_preprocess": {
            "image_scale": list(spec["image_scale"]),
            "keep_ratio": False,
            "flip": False,
            "mean": spec["mean"],
            "std": spec["std"],
            "to_rgb": True,
            "hr_scale": info.get("hr_scale"),
            "test_cfg": info.get("test_cfg"),
            "source": "official_code/HRDecoder/configs/_base_/datasets, configs/lesion",
        },
        "checkpoint_meta": {
            "iter": meta.get("iter") if isinstance(meta, dict) else None,
            "mmseg_version": meta.get("mmseg_version") if isinstance(meta, dict) else None,
            "time": meta.get("time") if isinstance(meta, dict) else None,
        },
        "n_images": len(ids),
        "image_ids": ids,
        "readme": spec["readme"],
        "readme_note": "README table of the released model; the paper's table is extracted in L1 when its PDF is on file.",
        "author_metric": {
            "classes": list(names),
            "overwrite_order": list(M2MRF_OVERWRITE_ORDER),
            "implementation": "bench.eval.m2mrf_stream.AuthorAccumulator (11 thresholds, as the fork's lesion_metric.py)",
            "per_class": per_class,
            "mIoU": float(np.nanmean(np.nan_to_num(iou[-4:], nan=0))),
            "mF1": float(np.nanmean(np.nan_to_num(f1[-4:], nan=0))),
            "mAUPR": float(np.nanmean(np.nan_to_num(aupr[-4:], nan=0))),
        },
        "prob_dir": str(spec["prob_dir"]),
        "protocol_classes": list(LESION_CLASSES),
    }
    Path(spec["summary"]).write_text(json.dumps(_jsonable(summary), indent=1) + "\n")
    print(json.dumps(_jsonable(summary["author_metric"]), indent=1))


if __name__ == "__main__":
    main()
