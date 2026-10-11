"""Near-duplicate audit between datasets and between the splits of one dataset.

Each image is cropped to its field-of-view box. Two signatures are compared:
a 64-bit DCT perceptual hash of the 32x32 grayscale crop (Hamming distance),
and the global-average-pooled ImageNet ResNet-34 feature of the 224x224 crop
(cosine similarity). Every pair from different splits or different datasets
that passes either threshold is listed for review; nothing is removed here.

Both signatures separate fundus images poorly (the median cosine to the
nearest image of another group is about 0.95), so a third, exhaustive check
compares the vessels (added 2026-10-12): the green channel of the
field-of-view box at 192x192, black top-hat 5x5 (thin dark structures:
vessels and red lesions; smooth illumination and the bright field-of-view rim
drop out), the inner field of view (20 pixels eroded), normalized cross-
correlation, the larger of the image and its horizontal mirror. Every pair at
or above --min-vessel is listed, within a group as well ("vessel_pairs");
the same eye photographed twice scores about 0.5-1.0, different eyes stay
below about 0.4. --review writes side-by-side sheets of the highest pairs.

    python bench/scripts/leak_audit.py --data dataset/prepared/IDRiD dataset/prepared/DDR --out runs/audit/leak_idrid_ddr
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
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.fft import dctn

from bench.common.io import image_path, load_meta, read_fov

HASH_SIDE = 32
HASH_KEEP = 8
FEATURE_SIDE = 224
VESSEL_SIDE = 192
VESSEL_TOPHAT = 5
VESSEL_ERODE = 20
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _box(fov):
    rows = np.flatnonzero(fov.any(axis=1))
    cols = np.flatnonzero(fov.any(axis=0))
    if len(rows) == 0:
        return 0, 0, fov.shape[1], fov.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _crop(data, image_id, with_fov=False):
    image = Image.open(image_path(data, image_id)).convert("RGB")
    fov = read_fov(data, image_id)
    box = _box(fov)
    crop = image.crop(box)
    if not with_fov:
        return crop
    return crop, fov[box[1] : box[3], box[0] : box[2]]


def vessel_signature(crop, fov_crop):
    """Unit vectors of the image and its horizontal mirror; see the module docstring."""
    side = VESSEL_SIDE
    green = np.asarray(crop.resize((side, side), Image.BILINEAR), dtype=np.float32)[..., 1]
    mask = np.asarray(Image.fromarray(fov_crop.astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 0
    if mask.any():
        green = np.where(mask, green, green[mask].mean())
    tophat = ndimage.grey_closing(green, size=(VESSEL_TOPHAT, VESSEL_TOPHAT)) - green
    inner = ndimage.binary_erosion(mask, iterations=VESSEL_ERODE)
    out = []
    for values, region in ((tophat, inner), (tophat[:, ::-1], inner[:, ::-1])):
        vector = np.zeros_like(values)
        if region.any():
            vector[region] = values[region] - values[region].mean()
        vector = vector.ravel()
        out.append((vector / (np.linalg.norm(vector) + 1e-6)).astype(np.float32))
    return out


def phash(crop):
    gray = np.asarray(crop.convert("L").resize((HASH_SIDE, HASH_SIDE), Image.BILINEAR), dtype=np.float64)
    low = dctn(gray, norm="ortho")[:HASH_KEEP, :HASH_KEEP].ravel()
    return low[1:] > np.median(low[1:])


def _signature(job):
    data, image_id = job
    crop, fov_crop = _crop(data, image_id, with_fov=True)
    small = np.asarray(crop.resize((FEATURE_SIDE, FEATURE_SIDE), Image.BILINEAR), dtype=np.float32) / 255.0
    plain, mirrored = vessel_signature(crop, fov_crop)
    return image_id, phash(crop), ((small - MEAN) / STD).transpose(2, 0, 1), plain, mirrored


def collect(datasets, workers):
    entries = []
    for data in datasets:
        name = load_meta(data)["name"]
        splits = json.loads((Path(data) / "splits.json").read_text())
        for split, ids in splits.items():
            entries += [(name, split, Path(data), image_id) for image_id in ids]
    with Pool(workers) as pool:
        signatures = pool.map(_signature, [(data, image_id) for _, _, data, image_id in entries], chunksize=8)
    return entries, signatures


def features(tensors, batch=64):
    from torchvision.models import ResNet34_Weights, resnet34

    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch):
            chunk = torch.from_numpy(np.stack(tensors[start : start + batch]))
            out.append(torch.nn.functional.normalize(model(chunk), dim=1).numpy())
    return np.concatenate(out)


def vessel_pairs(entries, signatures, min_vessel):
    """Every pair, within or between groups, whose vessel correlation is at least ``min_vessel``."""
    plain = np.stack([sig[3] for sig in signatures])
    mirrored = np.stack([sig[4] for sig in signatures])
    score = np.maximum(plain @ plain.T, plain @ mirrored.T)
    np.fill_diagonal(score, -1.0)
    found = []
    for i, j in np.argwhere(np.triu(score >= min_vessel, 1)):
        same = entries[i][:2] == entries[j][:2]
        found.append(
            {
                "a": {"dataset": entries[i][0], "split": entries[i][1], "id": entries[i][3]},
                "b": {"dataset": entries[j][0], "split": entries[j][1], "id": entries[j][3]},
                "vessel_ncc": round(float(score[i, j]), 4),
                "same_group": bool(same),
            }
        )
    found.sort(key=lambda pair: -pair["vessel_ncc"])
    upper = score[np.triu_indices(len(entries), 1)]
    quantiles = {q: round(float(np.quantile(upper, q)), 4) for q in (0.5, 0.99, 0.9999)}
    return found, quantiles


def review_sheets(datasets, pairs, directory, per_group=6, side=320):
    """Side-by-side field-of-view crops of the highest vessel pairs of each group pair."""
    roots = {load_meta(data)["name"]: Path(data) for data in datasets}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_group = {}
    for pair in pairs:
        key = " ".join(sorted([f"{pair['a']['dataset']}-{pair['a']['split']}", f"{pair['b']['dataset']}-{pair['b']['split']}"]))
        by_group.setdefault(key, []).append(pair)
    for key, members in by_group.items():
        members = members[:per_group]
        sheet = Image.new("RGB", (2 * side + 10, len(members) * (side + 20)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, pair in enumerate(members):
            top = row * (side + 20)
            for column, end in enumerate(("a", "b")):
                crop = _crop(roots[pair[end]["dataset"]], pair[end]["id"]).resize((side, side), Image.BILINEAR)
                sheet.paste(crop, (column * (side + 10), top + 18))
            draw.text(
                (4, top + 3),
                f"{pair['a']['dataset']}:{pair['a']['split']}:{pair['a']['id']} | "
                f"{pair['b']['dataset']}:{pair['b']['split']}:{pair['b']['id']}  vessel {pair['vessel_ncc']:.2f}",
                fill="black",
            )
        sheet.save(directory / (key.replace(" ", "__") + ".jpg"), quality=88)


def audit(datasets, out, max_hamming, min_cosine, workers, min_vessel=0.4, review=None):
    entries, signatures = collect(datasets, workers)
    hashes = np.stack([sig[1] for sig in signatures]).astype(np.uint8)
    feats = features([sig[2] for sig in signatures])
    hamming = (hashes[:, None, :] != hashes[None, :, :]).sum(axis=2)
    cosine = feats @ feats.T
    groups = [(name, split) for name, split, _, _ in entries]
    pairs = []
    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            if groups[i] == groups[j]:
                continue
            if hamming[i, j] <= max_hamming or cosine[i, j] >= min_cosine:
                pairs.append(
                    {
                        "a": {"dataset": entries[i][0], "split": entries[i][1], "id": entries[i][3]},
                        "b": {"dataset": entries[j][0], "split": entries[j][1], "id": entries[j][3]},
                        "hamming": int(hamming[i, j]),
                        "cosine": round(float(cosine[i, j]), 4),
                    }
                )
    pairs.sort(key=lambda p: (p["hamming"], -p["cosine"]))
    nearest = {}
    for i in range(n):
        other = [j for j in range(n) if groups[j] != groups[i]]
        j = max(other, key=lambda k: cosine[i, k])
        nearest[f"{entries[i][0]}/{entries[i][1]}/{entries[i][3]}"] = round(float(cosine[i, j]), 4)
    summary = {
        "datasets": [str(d) for d in datasets],
        "n_images": n,
        "hash": f"DCT pHash, {HASH_SIDE}x{HASH_SIDE} grayscale of the field-of-view box, {HASH_KEEP}x{HASH_KEEP} low band without DC",
        "feature": "ImageNet ResNet-34, global average pool, L2-normalized, 224x224 field-of-view box",
        "max_hamming": max_hamming,
        "min_cosine": min_cosine,
        "n_pairs": len(pairs),
        "pairs_by_group": {},
        "cosine_to_nearest_other_group_quantiles": {
            q: round(float(np.quantile(list(nearest.values()), q)), 4) for q in (0.5, 0.9, 0.99, 1.0)
        },
    }
    for pair in pairs:
        key = " / ".join(sorted([f"{pair['a']['dataset']}:{pair['a']['split']}", f"{pair['b']['dataset']}:{pair['b']['split']}"]))
        summary["pairs_by_group"][key] = summary["pairs_by_group"].get(key, 0) + 1
    vessels, quantiles = vessel_pairs(entries, signatures, min_vessel)
    summary["vessel"] = (
        f"green black top-hat {VESSEL_TOPHAT}x{VESSEL_TOPHAT} at {VESSEL_SIDE}x{VESSEL_SIDE} of the field-of-view box, "
        f"inner field of view ({VESSEL_ERODE} px eroded), normalized cross-correlation, max with the mirror"
    )
    summary["min_vessel"] = min_vessel
    summary["vessel_quantiles_all_pairs"] = quantiles
    summary["vessel_pairs_by_group"] = {}
    for pair in vessels:
        key = " / ".join(sorted([f"{pair['a']['dataset']}:{pair['a']['split']}", f"{pair['b']['dataset']}:{pair['b']['split']}"]))
        summary["vessel_pairs_by_group"][key] = summary["vessel_pairs_by_group"].get(key, 0) + 1
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "pairs": pairs, "vessel_pairs": vessels}, indent=1) + "\n"
    )
    if review:
        review_sheets(datasets, vessels, review)
    print(json.dumps(summary, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="output prefix; writes .json")
    parser.add_argument("--max-hamming", type=int, default=8)
    parser.add_argument("--min-cosine", type=float, default=0.97)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-vessel", type=float, default=0.4)
    parser.add_argument("--review", help="directory for side-by-side sheets of the vessel pairs")
    args = parser.parse_args(argv)
    torch.set_num_threads(16)
    audit(args.data, args.out, args.max_hamming, args.min_cosine, args.workers, args.min_vessel, args.review)


if __name__ == "__main__":
    main()
