"""Evaluate one prediction directory against one dataset split under the frozen protocol."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from bench.common import io
from bench.eval import lesion, pixel, stats

DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.yaml")


def load_protocol(path=DEFAULT_PROTOCOL, **overrides):
    text = Path(path).read_text()
    proto = yaml.safe_load(text)
    proto["_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    for k, v in overrides.items():
        if v is not None:
            proto[k] = v
    if proto["prob_levels"] != io.PROB_LEVELS:
        raise ValueError("protocol prob_levels must match bench.common.io.PROB_LEVELS")
    return proto


def _classes(dataset_dir, proto):
    available = io.load_meta(dataset_dir)["classes"]
    return [c for c in proto["classes"] if c in available]


def _load_image(dataset_dir, pred_dir, image_id, classes, proto):
    fov = io.read_fov(dataset_dir, image_id)
    masks = {c: io.read_mask(dataset_dir, c, image_id, shape=fov.shape) for c in classes}
    if proto["label_policy"] == "m2mrf_overwrite":
        masks = io.apply_m2mrf_overwrite(masks)
    elif proto["label_policy"] != "multilabel":
        raise ValueError(f"unknown label_policy {proto['label_policy']}")
    valid = fov if proto["use_fov"] else np.ones_like(fov)
    probs = {}
    for c in classes:
        q = io.read_prob_quantized(pred_dir, c, image_id)
        if q.shape != fov.shape:
            raise ValueError(
                f"{image_id}/{c}: prediction {q.shape} != ground truth {fov.shape}; "
                "predictions must be resized back to the original resolution"
            )
        probs[c] = q
    return fov, valid, masks, probs


def pooled_histograms(dataset_dir, split, pred_dir, proto):
    classes = _classes(dataset_dir, proto)
    L = proto["prob_levels"]
    pooled = {c: [np.zeros(L + 1, np.int64), np.zeros(L + 1, np.int64)] for c in classes}
    for image_id in io.load_split(dataset_dir, split):
        _, valid, masks, probs = _load_image(dataset_dir, pred_dir, image_id, classes, proto)
        for c in classes:
            pos, neg = pixel.histograms(probs[c], masks[c], valid, L)
            pooled[c][0] += pos
            pooled[c][1] += neg
    return pooled


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def evaluate(dataset_dir, split, pred_dir, proto, val_pred_dir=None, val_split="val", out_prefix=None):
    classes = _classes(dataset_dir, proto)
    L = proto["prob_levels"]
    B = proto["image_hist_bins"]

    thresholds = {"fixed": {c: proto["fixed_threshold"] for c in classes}}
    if val_pred_dir is not None:
        val_hist = pooled_histograms(dataset_dir, val_split, val_pred_dir, proto)
        thresholds["val_opt"] = {c: pixel.best_dice_threshold(*val_hist[c], L) for c in classes}

    ids = io.load_split(dataset_dir, split)
    pooled = {c: [np.zeros(L + 1, np.int64), np.zeros(L + 1, np.int64)] for c in classes}
    img_pos = {c: np.zeros((len(ids), B), np.int64) for c in classes}
    img_neg = {c: np.zeros((len(ids), B), np.int64) for c in classes}
    img_dice = {(c, t): np.full(len(ids), np.nan) for c in classes for t in thresholds}
    lesion_acc = {
        (c, t): lesion.LesionAccumulator(proto["lesion_size_edges_rel_diameter"], proto["lesion_size_names"])
        for c in classes
        for t in thresholds
    }

    for i, image_id in enumerate(ids):
        fov, valid, masks, probs = _load_image(dataset_dir, pred_dir, image_id, classes, proto)
        fov_area = int(fov.sum())
        for c in classes:
            pos, neg = pixel.histograms(probs[c], masks[c], valid, L)
            pooled[c][0] += pos
            pooled[c][1] += neg
            img_pos[c][i] = pixel.coarsen(pos, B)
            img_neg[c][i] = pixel.coarsen(neg, B)
            for tname, tdict in thresholds.items():
                q_min = pixel.level_for_threshold(tdict[c], L)
                tp, fp, fn = pixel.confusion_at(pos, neg, q_min)
                if tp + fn > 0:
                    img_dice[(c, tname)][i] = 2 * tp / (2 * tp + fp + fn)
                if proto["lesion_metrics"]:
                    pred_bin = (probs[c] >= q_min) & valid
                    m = lesion.match_lesions(pred_bin, masks[c] & valid, proto["connectivity"])
                    lesion_acc[(c, tname)].add(m, fov_area)

    boot = proto["bootstrap"]
    per_class = {}
    for c in classes:
        pos, neg = pooled[c]
        entry = {
            "aupr": pixel.average_precision(pos, neg),
            "aupr_ci": list(bootstrap_or_nan(img_pos[c], img_neg[c], boot)),
            "aupr_m2mrf_impl": pixel.m2mrf_aupr(pos, neg, L, proto["m2mrf_n_thresholds"]),
            "n_positive_pixels": int(pos.sum()),
            "n_images_with_lesion": int(((img_pos[c].sum(1)) > 0).sum()),
            "thresholds": {},
        }
        for tname, tdict in thresholds.items():
            q_min = pixel.level_for_threshold(tdict[c], L)
            pooled_scores = pixel.overlap_scores(*pixel.confusion_at(pos, neg, q_min))
            d = img_dice[(c, tname)]
            entry["thresholds"][tname] = {
                "threshold": tdict[c],
                "pooled": pooled_scores,
                "per_image_dice_mean": _nanmean(d),
                "per_image_dice_median": float(np.nanmedian(d)) if np.any(~np.isnan(d)) else float("nan"),
                "lesion": lesion_acc[(c, tname)].result() if proto["lesion_metrics"] else None,
            }
        rec, prec = pixel.pr_curve(pos, neg)
        entry["pr_curve"] = {"recall": rec, "precision": prec}
        per_class[c] = entry

    summary = {
        "mAUPR": _nanmean([per_class[c]["aupr"] for c in classes]),
        "mAUPR_m2mrf_impl": _nanmean([per_class[c]["aupr_m2mrf_impl"] for c in classes]),
    }
    for tname in thresholds:
        summary[f"mDice_pooled_{tname}"] = _nanmean(
            [per_class[c]["thresholds"][tname]["pooled"]["dice"] for c in classes]
        )

    result = _clean(
        {
            "dataset": io.load_meta(dataset_dir).get("name", str(dataset_dir)),
            "split": split,
            "pred_dir": str(pred_dir),
            "val_pred_dir": str(val_pred_dir) if val_pred_dir else None,
            "n_images": len(ids),
            "classes": classes,
            "protocol": {k: v for k, v in proto.items() if k != "_sha256"},
            "protocol_sha256": proto["_sha256"],
            "summary": summary,
            "per_class": per_class,
        }
    )

    if out_prefix is not None:
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        out_prefix.with_suffix(".json").write_text(json.dumps(result, indent=1))
        arrays = {"image_ids": np.array(ids)}
        for c in classes:
            arrays[f"{c}_pos"] = img_pos[c]
            arrays[f"{c}_neg"] = img_neg[c]
            for tname in thresholds:
                arrays[f"{c}_dice_{tname}"] = img_dice[(c, tname)]
        np.savez_compressed(out_prefix.with_suffix(".npz"), **arrays)
    return result


def bootstrap_or_nan(pos, neg, boot):
    if pos.sum() == 0:
        return float("nan"), float("nan")
    return stats.bootstrap_aupr_ci(pos, neg, n_boot=boot["n"], seed=boot["seed"], ci=boot["ci"])
