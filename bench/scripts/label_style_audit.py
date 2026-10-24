"""Annotation style of every source: lesions per image and lesion size, from the prepared masks (figure 6a; R6).

For each source of bench/configs/e3a_frozen.yaml (the eight test sets), every image listed in its
splits.json (all splits, each id once) is read: the field-of-view box gives the FOV long side, and
each class mask is split into 8-connected components. Per source and class the script records the
number of components, the mean per image (over all images, and over the images that carry the class),
the number of positive images, and percentiles of the component's equivalent diameter relative to the
FOV long side (2 sqrt(area / pi) / FOV long side). Ignore regions (Retinal-Lesions grey 127) are not
lesions and are left out; DiaRetDB1's masks are the >= 0.75 expert consensus the protocol evaluates.

Until this audit ran, figure 6a and the opening of R6 used placeholder counts for every source but
IDRiD (docs/ops/21 item 90). CPU only; FGADR (1,842 images) takes the longest, a few minutes.

    python bench/scripts/label_style_audit.py [--out runs/audit/label_style.json] [--sets IDRiD,DDR]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench import e3a  # noqa: E402
from bench.common.io import LESION_CLASSES, read_fov, read_ignore, read_mask  # noqa: E402
from bench.data.fov import crop_box  # noqa: E402

DEFAULT_OUT = _ROOT / "runs" / "audit" / "label_style.json"
PERCENTILES = (10, 25, 50, 75, 90)
STRUCTURE = np.ones((3, 3), dtype=bool)   # 8-connectivity, as the protocol's lesion level (bench/eval/lesion.py)


def image_ids(data_dir):
    """Every id of splits.json once, in split order (train, val, test, then any other split)."""
    splits = json.loads((Path(data_dir) / "splits.json").read_text())
    order = [s for s in ("train", "val", "test") if s in splits] + sorted(s for s in splits if s not in ("train", "val", "test"))
    seen, ids = set(), []
    for split in order:
        for image_id in splits[split]:
            if image_id not in seen:
                seen.add(image_id)
                ids.append(image_id)
    return ids, {s: len(splits[s]) for s in order}


def components(mask):
    """(count, areas in pixels) of the 8-connected components of a binary mask."""
    if not mask.any():
        return 0, np.zeros(0, dtype=np.int64)
    labels, count = ndimage.label(mask, structure=STRUCTURE)
    areas = np.bincount(labels.ravel())[1:]
    return int(count), areas.astype(np.int64)


def audit_source(data_dir, classes=LESION_CLASSES, log=print):
    ids, split_sizes = image_ids(data_dir)
    counts = {cls: [] for cls in classes}
    diameters = {cls: [] for cls in classes}
    areas_px = {cls: [] for cls in classes}
    started = time.perf_counter()
    for index, image_id in enumerate(ids):
        fov = read_fov(data_dir, image_id)
        y0, y1, x0, x1 = crop_box(fov)
        long_side = float(max(y1 - y0, x1 - x0))
        for cls in classes:
            mask = read_mask(data_dir, cls, image_id, shape=fov.shape)
            ignore = read_ignore(data_dir, cls, image_id, fov.shape)
            if ignore.any():
                mask = mask & ~ignore
            count, areas = components(mask)
            counts[cls].append(count)
            if count:
                diameters[cls].extend((2.0 * np.sqrt(areas / np.pi) / long_side).tolist())
                areas_px[cls].extend(areas.tolist())
        if index % 200 == 0:
            log(f"{Path(data_dir).name}: {index + 1}/{len(ids)} images", flush=True)
    per_class = {}
    for cls in classes:
        n = np.asarray(counts[cls], dtype=np.int64)
        d = np.asarray(diameters[cls], dtype=np.float64)
        positive = int((n > 0).sum())
        per_class[cls] = {
            "n_components": int(n.sum()),
            "images_positive": positive,
            "per_image_mean": float(n.mean()) if len(n) else None,
            "per_image_mean_positive": float(n[n > 0].mean()) if positive else None,
            "rel_diameter": {f"p{p}": float(np.percentile(d, p)) for p in PERCENTILES} if len(d) else None,
            "area_px_median": float(np.median(areas_px[cls])) if areas_px[cls] else None,
        }
    return {"data": str(Path(data_dir).relative_to(_ROOT)) if str(data_dir).startswith(str(_ROOT)) else str(data_dir),
            "n_images": len(ids), "splits": split_sizes, "seconds": round(time.perf_counter() - started, 1), "per_class": per_class}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sets", help="comma-separated test-set names of e3a_frozen.yaml (default all eight)")
    args = parser.parse_args(argv)
    frozen = e3a.load_frozen()
    names = [s.strip() for s in args.sets.split(",")] if args.sets else list(frozen["test_sets"])
    out = Path(args.out)
    result = json.loads(out.read_text()) if out.is_file() else {}
    result.setdefault("rule", "8-connected components of the prepared masks (ignore regions removed), every id of splits.json once; "
                               "relative diameter = 2 sqrt(area / pi) / FOV box long side; docs/ops/21 item 90")
    result.setdefault("sources", {})
    for name in names:
        data_dir = e3a.test_data_dir(name, frozen)
        result["sources"][name] = audit_source(data_dir)
        row = result["sources"][name]
        summary = {cls: (round(v["per_image_mean"], 2), round(v["rel_diameter"]["p50"], 4) if v["rel_diameter"] else None) for cls, v in row["per_class"].items()}
        print(f"{name}: {row['n_images']} images; per-image mean and median relative diameter {summary}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
