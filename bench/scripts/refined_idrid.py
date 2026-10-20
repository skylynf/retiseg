"""Refined IDRiD (Chankhachon et al., Data 2026, doi 10.3390/data11020030; Zenodo 18676805): annotation-revision
sensitivity, docs/paper/26 N4 and docs/paper/14 note 49.

The release re-annotates the 81 IDRiD images (ophthalmologist-corrected MA, HE, EX and cotton-wool spots, plus
NV, VH, IRMA and anatomy) as ONE grayscale mask per image at 1024 x 1024 after a crop to the retinal region and a
resize that does not keep the aspect ratio (paper section 2; nearest neighbour for the masks). Table 2 of the paper
gives one value per class: background 0, VH 4, retina 8, fovea 16, vessel 24, optic disc 32, EX 63, IRMA 96, HE 127,
NV 166, CWS 191, MA 255. One class per pixel, so refined lesions never overlap. Three CPU commands:

    check    find, for every official test image, the crop of the original that reproduces the released 1024 x 1024
             image: a search over the four crop edges starting from this project's FOV box, under two resize
             hypotheses (stretch to 1024 x 1024, or pad to a square first), scored by normalized cross-correlation
             at full resolution; the geometry is accepted only if every image reaches the threshold
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


def square_pad(arr, fill=0):
    """Centre ``arr`` on a square canvas of its longer side (the 'pad' hypothesis: pad, then resize uniformly)."""
    h, w = arr.shape[:2]
    side = max(h, w)
    out = np.full((side, side) + tuple(arr.shape[2:]), fill, dtype=arr.dtype)
    y, x = (side - h) // 2, (side - w) // 2
    out[y:y + h, x:x + w] = arr
    return out


def transform(arr, box, hypothesis="stretch", size=SIZE, nearest=False):
    """Crop ``box`` (y0, y1, x0, x1) and map it to size x size the way the release does under ``hypothesis``:
    'stretch' resizes the crop without keeping the aspect ratio; 'pad' centres it on a square first."""
    y0, y1, x0, x1 = box
    crop = np.ascontiguousarray(arr[y0:y1, x0:x1])
    if hypothesis == "pad":
        crop = square_pad(crop)
    elif hypothesis != "stretch":
        raise ValueError(f"unknown hypothesis {hypothesis!r}")
    return resize_gray(crop, size, nearest)


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


HYPOTHESES = ("stretch", "pad")
SEARCH_SIZE = 256


def _search_box(gray_small, released_small, box0, hypothesis, scale, shape):
    """Coordinate descent on the four box edges (full-resolution units) that maximizes the NCC between the
    transformed crop of the downscaled original and the downscaled release; steps 64, 16, 4, 1 pixels."""
    H, W = shape

    def score(box):
        y0, y1, x0, x1 = [int(round(v / scale)) for v in box]
        if y1 - y0 < 4 or x1 - x0 < 4:
            return -1.0
        return ncc(transform(gray_small, (y0, y1, x0, x1), hypothesis, size=SEARCH_SIZE), released_small)

    limits = ((0, H - 1), (1, H), (0, W - 1), (1, W))
    box = list(box0)
    best = score(box)
    for step in (64, 16, 4, 1):
        moved = True
        rounds = 0
        while moved and rounds < 6:
            moved = False
            rounds += 1
            for i in range(4):
                lo, hi = limits[i]
                candidates = []
                for k in range(-8, 9):
                    v = box[i] + k * step
                    if lo <= v <= hi and v != box[i]:
                        trial = list(box)
                        trial[i] = v
                        if (i in (0, 1) and trial[1] - trial[0] < 8) or (i in (2, 3) and trial[3] - trial[2] < 8):
                            continue
                        candidates.append((score(trial), v))
                if candidates:
                    sc, v = max(candidates)
                    if sc > best + 1e-9:
                        best, box[i], moved = sc, v, True
    return [int(v) for v in box], best


def _refine_full(gray, released, box, hypothesis, reach=4, rounds=3):
    """Final coordinate descent at full resolution, one pixel at a time within +-reach per edge."""
    H, W = gray.shape
    limits = ((0, H - 1), (1, H), (0, W - 1), (1, W))
    box = list(box)
    best = ncc(transform(gray, box, hypothesis), released)
    for _ in range(rounds):
        moved = False
        for i in range(4):
            lo, hi = limits[i]
            for v in range(box[i] - reach, box[i] + reach + 1):
                if v == box[i] or not lo <= v <= hi:
                    continue
                trial = list(box)
                trial[i] = v
                if trial[1] - trial[0] < 8 or trial[3] - trial[2] < 8:
                    continue
                sc = ncc(transform(gray, trial, hypothesis), released)
                if sc > best + 1e-9:
                    best, box, moved = sc, trial, True
        if not moved:
            break
    return [int(v) for v in box], best


def match_geometry(original_rgb, fov, released_rgb):
    """The crop box (and the resize hypothesis) that reproduces the released 1024 x 1024 image from the original.

    Starts from this project's FOV box, searches the four edges by coordinate descent at a reduced scale under
    both hypotheses (stretch: anisotropic resize of the crop; pad: square padding, then uniform resize), and
    reports the full-resolution NCC of the winner next to that of the plain FOV-box stretch.
    """
    released = np.asarray(Image.fromarray(np.asarray(released_rgb)).convert("L"))
    if released.shape != (SIZE, SIZE):
        raise ValueError(f"released image is {released.shape}, expected {(SIZE, SIZE)}")
    gray = np.asarray(Image.fromarray(np.asarray(original_rgb)).convert("L"))
    H, W = gray.shape
    scale = max(1, int(round(max(H, W) / 1024)))
    gray_small = np.asarray(Image.fromarray(gray).resize((max(1, W // scale), max(1, H // scale)), Image.BILINEAR)) if scale > 1 else gray
    released_small = resize_gray(released, SEARCH_SIZE)
    fov_box = list(crop_box(fov))
    results = {}
    for hypothesis in HYPOTHESES:
        box, search_ncc = _search_box(gray_small, released_small, fov_box, hypothesis, scale, (H, W))
        box, full = _refine_full(gray, released, box, hypothesis)
        results[hypothesis] = {"box": box, "ncc": full, "ncc_search": search_ncc}
    # the paper documents a plain resize; 'pad' wins only when it is clearly better (a square crop makes the two identical)
    best_h = "stretch" if results["stretch"]["ncc"] >= results["pad"]["ncc"] - 0.005 else "pad"
    return {
        "hypothesis": best_h, "box": results[best_h]["box"], "ncc": results[best_h]["ncc"], "ncc_search": results[best_h]["ncc_search"],
        "edge_ncc": edge_ncc(transform(gray, results[best_h]["box"], best_h), released),
        "fov_box": fov_box, "fov_box_stretch_ncc": ncc(transform(gray, fov_box, "stretch"), released),
        "alternatives": {h: {"box": r["box"], "ncc": r["ncc"]} for h, r in results.items() if h != best_h},
    }


def edge_ncc(a, b):
    """NCC of Sobel gradient magnitudes: alignment evidence that survives a global or local contrast change
    (a release that was contrast-enhanced keeps its edges where they were)."""
    from scipy import ndimage

    def edges(x):
        x = np.asarray(x, np.float64)
        return np.hypot(ndimage.sobel(x, axis=0), ndimage.sobel(x, axis=1))

    return ncc(edges(a), edges(b))


def label_extent(path):
    """Bounding box of the non-background pixels of a released mask: tells whether the crop was the retina's box
    (extent spans the frame) or something padded (bands of background)."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return None
    return [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]


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
        best["label_extent"] = label_extent(release[image_id]["label"])
        result["images"][image_id] = best
        worst = min(worst, best["ncc"])
        log(f"{image_id}: {best['hypothesis']} box {best['box']} ncc {best['ncc']:.4f} edge ncc {best['edge_ncc']:.4f} "
            f"(fov box {best['fov_box']} stretch ncc {best['fov_box_stretch_ncc']:.4f}; label extent {best['label_extent']})")
    result["worst_ncc"] = worst
    result["accepted"] = bool(worst >= accept)
    result["hypotheses_used"] = sorted({v["hypothesis"] for v in result["images"].values()})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    log(f"worst ncc {worst:.4f} -> {'accepted' if result['accepted'] else 'NOT accepted'}; hypotheses {result['hypotheses_used']}; written {out}")
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
        entry = geo["images"][image_id]
        box, hyp = entry["box"], entry.get("hypothesis", "stretch")
        fov = io.read_fov(prepared, image_id)
        fov_1024 = transform(fov.astype(np.uint8) * 255, box, hyp, nearest=True) > 0
        label = read_label(release[image_id]["label"])
        if str(release[image_id]["label"]) != geo["images"][image_id]["released_label"]:
            raise RuntimeError(f"{image_id}: the label file differs from the one recorded by check")
        _write_mask(Path(out_refined) / "fov" / f"{image_id}.png", fov_1024)
        _write_mask(Path(out_original) / "fov" / f"{image_id}.png", fov_1024)
        # the release image itself is the image of both directories (same pixels, same crop)
        for out in (out_refined, out_original):
            (Path(out) / "images").mkdir(parents=True, exist_ok=True)
            Image.open(release[image_id]["image"]).convert("RGB").save(Path(out) / "images" / f"{image_id}.png")
        entry = {"hypothesis": hyp, "box": box, "retina_vs_fov_iou": None, "classes": {}}
        retina = label != 0
        entry["retina_vs_fov_iou"] = float((retina & fov_1024).sum() / max(1, (retina | fov_1024).sum()))
        for cls, value in CLASS_VALUE.items():
            refined = label == value
            original = io.read_mask(prepared, cls, image_id, shape=fov.shape)
            original_1024 = transform(original.astype(np.uint8) * 255, box, hyp, nearest=True) > 0
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
            entry = geo["images"][image_id]
            box, hyp = entry["box"], entry.get("hypothesis", "stretch")
            for cls in CLASS_VALUE:
                q = io.read_prob_quantized(run / "prob_test", cls, image_id).astype(np.float32) / io.PROB_LEVELS
                resized = np.asarray(transform(q, box, hyp), np.float64)
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
