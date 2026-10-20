"""Refined IDRiD (Chankhachon et al., Data 2026, doi 10.3390/data11020030; Zenodo 18676805): annotation-revision
sensitivity, docs/paper/26 N4 and docs/paper/14 note 49.

The release re-annotates the 81 IDRiD images (ophthalmologist-corrected MA, HE, EX and cotton-wool spots, plus
NV, VH, IRMA and anatomy) as ONE grayscale mask per image at 1024 x 1024 after a crop to the retinal region and a
resize that does not keep the aspect ratio (paper section 2; nearest neighbour for the masks). Table 2 of the paper
gives one value per class: background 0, VH 4, retina 8, fovea 16, vessel 24, optic disc 32, EX 63, IRMA 96, HE 127,
NV 166, CWS 191, MA 255. One class per pixel, so refined lesions never overlap. Three CPU commands:

    check    find, for every official test image, the crop of the original that reproduces the released 1024 x 1024
             image (the bounding box of this project's FOV mask with a margin of 0-32 pixels, resized to 1024 x 1024)
             and report the normalized cross-correlation per image; accept the geometry only if every image matches
    prepare  write dataset/prepared/RefinedIDRiD (refined labels at 1024: masks MA / HE / EX / SE, fov = the crop of
             this project's FOV; split test = the 27 official test images) and dataset/prepared/RefinedIDRiD_original
             (the official IDRiD masks taken through the same crop and resize), plus label-difference statistics
    score    for every IDRiD-trained run with original-resolution prob_test maps: crop + resize the maps to 1024
             (bilinear), score against both label sets with the protocol -> runs/REFINED/<run>/{refined,original}/
             metrics.json; val_opt thresholds come from the run's prob_val on the official validation split, so the
             only thing that changes between the two scores is the test labels

    python bench/scripts/refined_idrid.py check   --root dataset/incoming/RefinedIDRiD
    python bench/scripts/refined_idrid.py prepare --root dataset/incoming/RefinedIDRiD
    python bench/scripts/refined_idrid.py score

Nothing here reads the refined labels before the geometry is accepted; ``score`` refuses to run without the
geometry file that ``check`` writes.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.common import io  # noqa: E402
from bench.data.fov import crop_box  # noqa: E402

SIZE = 1024
VALUES = {"background": 0, "VH": 4, "retina": 8, "fovea": 16, "vessel": 24, "OD": 32, "EX": 63, "IRMA": 96, "HE": 127, "NV": 166, "CWS": 191, "MA": 255}
CLASS_VALUE = {"MA": 255, "HE": 127, "EX": 63, "SE": 191}          # this project's four classes; CWS is SE
MARGINS = (0, 1, 2, 4, 8, 16, 32)
PREPARED_IDRID = REPO / "dataset" / "prepared" / "IDRiD"
OUT_REFINED = REPO / "dataset" / "prepared" / "RefinedIDRiD"
OUT_ORIGINAL = REPO / "dataset" / "prepared" / "RefinedIDRiD_original"
GEOMETRY = REPO / "dataset" / "prepared" / "RefinedIDRiD_geometry.json"
REFINED_RUNS = REPO / "runs" / "REFINED"
ID = re.compile(r"IDRiD_(\d{2})")


# ------------------------------------------------------------------------------------------------------------- release


def find_release(root, split="test", ids=None):
    """{image id: {"image": path, "label": path}} for the official ids of ``split`` (or ``ids``) found under ``root``.

    The archive layout is not documented, so files are matched by name: an image is a jpg/png/tif whose name
    carries IDRiD_NN and no label marker; a label is a png whose name carries IDRiD_NN and 'label' or 'vessel'
    (the training labels of the release are named *_vessel.png) or that lies in a directory named like Labels.
    """
    ids = list(ids) if ids is not None else [f"IDRiD_{i:02d}" for i in (range(55, 82) if split == "test" else range(1, 55))]
    found = {image_id: {} for image_id in ids}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        m = ID.search(path.name)
        if not m:
            continue
        image_id = f"IDRiD_{int(m.group(1)):02d}"
        if image_id not in found:
            continue
        lower = path.name.lower()
        in_label_dir = any(part.lower().startswith("label") for part in path.parts[:-1])
        is_label = "label" in lower or "vessel" in lower or in_label_dir
        key = "label" if is_label else "image"
        if key in found[image_id]:
            continue   # first match wins; the check reports what was used
        found[image_id][key] = path
    missing = [i for i in ids if "image" not in found[i] or "label" not in found[i]]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(ids)} {split} ids have no image or no label under {root}: {missing[:5]}")
    return found


def read_label(path):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (SIZE, SIZE):
        raise ValueError(f"{path}: refined mask is {arr.shape}, expected {(SIZE, SIZE)}")
    unknown = sorted(set(np.unique(arr).tolist()) - set(VALUES.values()))
    if unknown:
        raise ValueError(f"{path}: values {unknown} are not in Table 2 of the paper")
    return arr


# ------------------------------------------------------------------------------------------------------------- geometry


def resize_gray(arr, size=SIZE, nearest=False):
    img = Image.fromarray(arr)
    return np.asarray(img.resize((size, size), Image.NEAREST if nearest else Image.BILINEAR))


def crop_with_margin(fov, margin):
    y0, y1, x0, x1 = crop_box(fov)
    h, w = fov.shape
    return max(0, y0 - margin), min(h, y1 + margin), max(0, x0 - margin), min(w, x1 + margin)


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def match_geometry(original_rgb, fov, released_rgb, margins=MARGINS):
    """The margin whose FOV-box crop, resized to 1024 x 1024, best reproduces the released image (grayscale NCC)."""
    released = np.asarray(Image.fromarray(np.asarray(released_rgb)).convert("L"), np.float64)
    if released.shape != (SIZE, SIZE):
        raise ValueError(f"released image is {released.shape}, expected {(SIZE, SIZE)}")
    gray = np.asarray(Image.fromarray(np.asarray(original_rgb)).convert("L"))
    best = None
    for margin in margins:
        y0, y1, x0, x1 = crop_with_margin(fov, margin)
        candidate = resize_gray(gray[y0:y1, x0:x1])
        score = ncc(candidate, released)
        if best is None or score > best["ncc"]:
            best = {"margin": margin, "box": [y0, y1, x0, x1], "ncc": score}
    return best


def check(root, prepared=PREPARED_IDRID, out=GEOMETRY, accept=0.97, log=print):
    ids = io.load_split(prepared, "test")
    release = find_release(root, "test", ids)
    result = {"source": str(root), "size": SIZE, "accept_ncc": accept, "images": {}}
    worst = 1.0
    for image_id in ids:
        original = np.asarray(Image.open(io.image_path(prepared, image_id)).convert("RGB"))
        fov = io.read_fov(prepared, image_id)
        released = np.asarray(Image.open(release[image_id]["image"]).convert("RGB"))
        best = match_geometry(original, fov, released)
        best["released_image"] = str(release[image_id]["image"])
        best["released_label"] = str(release[image_id]["label"])
        result["images"][image_id] = best
        worst = min(worst, best["ncc"])
        log(f"{image_id}: margin {best['margin']} box {best['box']} ncc {best['ncc']:.4f}")
    result["worst_ncc"] = worst
    result["accepted"] = bool(worst >= accept)
    margins = sorted({v["margin"] for v in result["images"].values()})
    result["margins_used"] = margins
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    log(f"worst ncc {worst:.4f} -> {'accepted' if result['accepted'] else 'NOT accepted'}; margins {margins}; written {out}")
    return result


# -------------------------------------------------------------------------------------------------------------- prepare


def _write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, bool).astype(np.uint8) * 255).save(path)


def _components(mask):
    from scipy import ndimage

    return int(ndimage.label(mask, structure=np.ones((3, 3)))[1])


def prepare(root, geometry=GEOMETRY, prepared=PREPARED_IDRID, out_refined=OUT_REFINED, out_original=OUT_ORIGINAL, log=print):
    geo = json.loads(Path(geometry).read_text())
    if not geo.get("accepted"):
        raise RuntimeError(f"{geometry}: the geometry was not accepted (worst ncc {geo.get('worst_ncc')}); fix the check first")
    ids = io.load_split(prepared, "test")
    release = find_release(root, "test", ids)
    stats = {}
    for out in (out_refined, out_original):
        Path(out).mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        y0, y1, x0, x1 = geo["images"][image_id]["box"]
        fov = io.read_fov(prepared, image_id)
        fov_1024 = resize_gray(fov[y0:y1, x0:x1].astype(np.uint8) * 255, nearest=True) > 0
        label = read_label(release[image_id]["label"])
        if str(release[image_id]["label"]) != geo["images"][image_id]["released_label"]:
            raise RuntimeError(f"{image_id}: the label file differs from the one recorded by check")
        _write_mask(Path(out_refined) / "fov" / f"{image_id}.png", fov_1024)
        _write_mask(Path(out_original) / "fov" / f"{image_id}.png", fov_1024)
        # the release image itself is the image of both directories (same pixels, same crop)
        for out in (out_refined, out_original):
            (Path(out) / "images").mkdir(parents=True, exist_ok=True)
            Image.open(release[image_id]["image"]).convert("RGB").save(Path(out) / "images" / f"{image_id}.png")
        entry = {"retina_vs_fov_iou": None, "classes": {}}
        retina = label != 0
        entry["retina_vs_fov_iou"] = float((retina & fov_1024).sum() / max(1, (retina | fov_1024).sum()))
        for cls, value in CLASS_VALUE.items():
            refined = label == value
            original = io.read_mask(prepared, cls, image_id, shape=fov.shape)
            original_1024 = resize_gray(original[y0:y1, x0:x1].astype(np.uint8) * 255, nearest=True) > 0
            _write_mask(Path(out_refined) / "masks" / cls / f"{image_id}.png", refined)
            _write_mask(Path(out_original) / "masks" / cls / f"{image_id}.png", original_1024)
            inter = int((refined & original_1024).sum())
            union = int((refined | original_1024).sum())
            entry["classes"][cls] = {
                "refined_pixels": int(refined.sum()), "original_pixels": int(original_1024.sum()),
                "refined_components": _components(refined), "original_components": _components(original_1024),
                "iou": float(inter / union) if union else None,
                "refined_only_pixels": int((refined & ~original_1024).sum()), "original_only_pixels": int((original_1024 & ~refined).sum()),
            }
        stats[image_id] = entry
        log(f"{image_id}: " + ", ".join(f"{c} {e['original_pixels']}->{e['refined_pixels']} px, {e['original_components']}->{e['refined_components']} cc, IoU {e['iou'] if e['iou'] is None else round(e['iou'], 3)}" for c, e in entry["classes"].items()))
    for out, name, note in ((out_refined, "RefinedIDRiD", "Refined IDRiD release labels (Table 2 values MA 255, HE 127, EX 63, CWS 191 -> SE)"),
                            (out_original, "RefinedIDRiD_original", "the official IDRiD masks through the same crop and 1024 x 1024 nearest-neighbour resize")):
        (Path(out) / "meta.json").write_text(json.dumps({
            "name": name, "classes": ["MA", "HE", "EX", "SE"], "granularity": "fine", "source": "Zenodo 18676805 (Chankhachon et al., Data 2026)",
            "geometry": str(geometry), "size": SIZE, "labels": note, "fov": "this project's IDRiD FOV mask cropped and resized like the release",
            "license": "CC BY 4.0"}, indent=1) + "\n")
        (Path(out) / "splits.json").write_text(json.dumps({"train": [], "val": [], "test": list(ids)}, indent=1) + "\n")
    totals = {cls: {k: sum(stats[i]["classes"][cls][k] for i in ids) for k in ("refined_pixels", "original_pixels", "refined_components", "original_components", "refined_only_pixels", "original_only_pixels")} for cls in CLASS_VALUE}
    for cls in totals:
        inter = totals[cls]["refined_pixels"] - totals[cls]["refined_only_pixels"]
        union = totals[cls]["refined_pixels"] + totals[cls]["original_only_pixels"]
        totals[cls]["pooled_iou"] = float(inter / union) if union else None
    report = {"n_images": len(ids), "per_image": stats, "totals": totals, "retina_vs_fov_iou_mean": float(np.mean([stats[i]["retina_vs_fov_iou"] for i in ids]))}
    (Path(out_refined) / "label_difference.json").write_text(json.dumps(report, indent=1) + "\n")
    log(json.dumps(totals, indent=1))
    return report


# ---------------------------------------------------------------------------------------------------------------- score


def candidate_runs(runs_root=REPO / "runs"):
    """IDRiD-trained run directories with original-resolution prob_test maps (main-table seed 0, nnU-Net, author weights)."""
    out = []
    for run in sorted(p for p in Path(runs_root).iterdir() if p.is_dir()):
        name = run.name
        if not (run / "prob_test").is_dir():
            continue
        if re.match(r"^(BS|B1)_[a-z0-9_]+_idrid_seed0$", name) or re.match(r"^NN_(default|resencm)_idrid_fold0$", name) or re.match(r"^E1r_[a-z0-9]+_idrid_seed0_bs4", name):
            out.append(run)
        elif re.match(r"^E1_[a-z0-9]+$", name) and (run / "author_idrid_test.json").is_file():
            out.append(run)
    return out


def score(runs_root=REPO / "runs", geometry=GEOMETRY, prepared=PREPARED_IDRID, out_root=REFINED_RUNS, force=False, log=print,
          refined_dir=OUT_REFINED, original_dir=OUT_ORIGINAL):
    from bench.eval.evaluate import evaluate, load_protocol

    geo = json.loads(Path(geometry).read_text())
    if not geo.get("accepted"):
        raise RuntimeError(f"{geometry}: the geometry was not accepted; nothing is scored")
    proto = load_protocol()
    ids = io.load_split(prepared, "test")
    results = {}
    for run in candidate_runs(runs_root):
        out = Path(out_root) / run.name
        if (out / "refined" / "metrics.json").is_file() and (out / "original" / "metrics.json").is_file() and not force:
            continue
        pred = out / "prob_test"
        for image_id in ids:
            y0, y1, x0, x1 = geo["images"][image_id]["box"]
            for cls in CLASS_VALUE:
                q = io.read_prob_quantized(run / "prob_test", cls, image_id).astype(np.float32) / io.PROB_LEVELS
                crop = q[y0:y1, x0:x1]
                resized = np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.BILINEAR), np.float64)
                io.write_prob(pred, cls, image_id, np.clip(resized, 0, 1))
        val_pred = run / "prob_val" if (run / "prob_val").is_dir() else None
        entry = {}
        for name, data in (("refined", refined_dir), ("original", original_dir)):
            res = evaluate(data, "test", pred, proto, val_pred_dir=val_pred, val_split="val", out_prefix=out / name / "metrics", val_dataset_dir=prepared if val_pred else None)
            entry[name] = res["summary"]["mAUPR"]
        (out / "done.json").write_text(json.dumps({"run": run.name, "geometry": str(geometry), "val_thresholds": "run prob_val on the official IDRiD validation split" if val_pred else "fixed only",
                                                   "mAUPR_refined": entry["refined"], "mAUPR_original_1024": entry["original"]}, indent=1) + "\n")
        results[run.name] = entry
        log(f"{run.name}: refined {entry['refined']:.4f}  original@1024 {entry['original']:.4f}  (delta {entry['refined'] - entry['original']:+.4f})")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--root", required=True, help="directory holding the extracted release (Test.tar contents)")
    c.add_argument("--out", default=str(GEOMETRY))
    c.add_argument("--accept", type=float, default=0.97, help="smallest per-image NCC that accepts the geometry")
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--geometry", default=str(GEOMETRY))
    s = sub.add_parser("score")
    s.add_argument("--runs", default=str(REPO / "runs"))
    s.add_argument("--geometry", default=str(GEOMETRY))
    s.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "check":
        check(args.root, out=args.out, accept=args.accept)
    elif args.command == "prepare":
        prepare(args.root, geometry=args.geometry)
    else:
        score(Path(args.runs), geometry=args.geometry, force=args.force)


if __name__ == "__main__":
    main()
