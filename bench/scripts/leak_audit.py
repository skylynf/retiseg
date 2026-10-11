"""Near-duplicate audit between datasets and between the splits of one dataset.

Each image is cropped to its field-of-view box. Two signatures are compared:
a 64-bit DCT perceptual hash of the 32x32 grayscale crop (Hamming distance),
and the global-average-pooled ImageNet ResNet-34 feature of the 224x224 crop
(cosine similarity). Every pair from different splits or different datasets
that passes either threshold is listed for review; nothing is removed here.

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
from PIL import Image
from scipy.fft import dctn

from bench.common.io import image_path, load_meta, read_fov

HASH_SIDE = 32
HASH_KEEP = 8
FEATURE_SIDE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop(data, image_id):
    image = Image.open(image_path(data, image_id)).convert("RGB")
    fov = read_fov(data, image_id)
    rows = np.flatnonzero(fov.any(axis=1))
    cols = np.flatnonzero(fov.any(axis=0))
    if len(rows) == 0:
        return image
    return image.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def phash(crop):
    gray = np.asarray(crop.convert("L").resize((HASH_SIDE, HASH_SIDE), Image.BILINEAR), dtype=np.float64)
    low = dctn(gray, norm="ortho")[:HASH_KEEP, :HASH_KEEP].ravel()
    return low[1:] > np.median(low[1:])


def _signature(job):
    data, image_id = job
    crop = _crop(data, image_id)
    small = np.asarray(crop.resize((FEATURE_SIDE, FEATURE_SIDE), Image.BILINEAR), dtype=np.float32) / 255.0
    return image_id, phash(crop), ((small - MEAN) / STD).transpose(2, 0, 1)


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


def audit(datasets, out, max_hamming, min_cosine, workers):
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
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="output prefix; writes .json")
    parser.add_argument("--max-hamming", type=int, default=8)
    parser.add_argument("--min-cosine", type=float, default=0.97)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    torch.set_num_threads(16)
    audit(args.data, args.out, args.max_hamming, args.min_cosine, args.workers)


if __name__ == "__main__":
    main()
