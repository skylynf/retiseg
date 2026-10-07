"""Write the original-resolution layout consumed by bench.common.io.

Images stay at the release resolution. Masks are one binary PNG per class;
an all-negative class is omitted, which the reader treats as a negative mask.
The FOV mask is computed here and stored beside the image. Lesion-size
statistics use only the training ids passed in, never the test split.

The independent sources (TJDR_std, TJDR_uwf, FGADR, DiaRetDB1,
Retinal-Lesions) go through export_cases. Their images keep the release
bytes and extension. Besides masks/ they may have masks_ext/<CLS> (classes
outside the four), ignore/<CLS> (pixels excluded from that class in
evaluation and loss) and dataset-specific layers. Each finished image appends
one line to export_cases.jsonl, written after its FOV, so an interrupted
export resumes at the first image without a line. report.json sums those
lines per split. Lists come from bench/data/splits/*.json, never from a
fresh draw.
"""

import json
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.data.fov import field_of_view
from bench.data.lesion_size import relative_diameters

PREPARED = {
    "IDRiD": "dataset/prepared/IDRiD",
    "DDR": "dataset/prepared/DDR",
    "TJDR_std": "dataset/prepared/TJDR_std",
    "TJDR_uwf": "dataset/prepared/TJDR_uwf",
    "FGADR": "dataset/prepared/FGADR",
    "DiaRetDB1": "dataset/prepared/DiaRetDB1",
    "Retinal-Lesions": "dataset/prepared/RetinalLesions",
}
CASE_LOG = "export_cases.jsonl"


def _write_binary(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def write_case(dest, image_id, image_bytes, image_ext, masks, fov):
    dest = Path(dest)
    image_path = dest / "images" / f"{image_id}.{image_ext}"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_bytes, (bytes, bytearray)):
        image_path.write_bytes(image_bytes)
    else:
        shutil.copyfile(image_bytes, image_path)
    for cls, mask in masks.items():
        if np.any(mask):
            _write_binary(dest / "masks" / cls / f"{image_id}.png", mask)
    _write_binary(dest / "fov" / f"{image_id}.png", fov)


def write_index(dest, name, splits, granularity="fine", **extra):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "classes": list(LESION_CLASSES), "granularity": granularity}
    meta.update(extra)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (dest / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")


def measure_exported(dest, train_ids):
    dest = Path(dest)
    diameters = {cls: [] for cls in LESION_CLASSES}
    for image_id in train_ids:
        fov = np.asarray(Image.open(dest / "fov" / f"{image_id}.png")) > 0
        for cls in LESION_CLASSES:
            path = dest / "masks" / cls / f"{image_id}.png"
            if path.is_file():
                mask = np.asarray(Image.open(path)) > 0
            else:
                mask = np.zeros(fov.shape, dtype=bool)
            diameters[cls].extend(relative_diameters(mask, fov).tolist())
    return diameters


def export_split(dest, image_ids, load_case, train_ids, diameters):
    dest = Path(dest)
    train_ids = set(train_ids)
    for index, image_id in enumerate(image_ids, start=1):
        fov_path = dest / "fov" / f"{image_id}.png"
        if not fov_path.is_file():
            image_bytes, image_ext, image, masks = load_case(image_id)
            fov = field_of_view(image)
            write_case(dest, image_id, image_bytes, image_ext, masks, fov)
        else:
            fov = np.asarray(Image.open(fov_path)) > 0
            masks = {}
            for cls in LESION_CLASSES:
                path = dest / "masks" / cls / f"{image_id}.png"
                if path.is_file():
                    masks[cls] = np.asarray(Image.open(path)) > 0
                else:
                    masks[cls] = np.zeros(fov.shape, dtype=bool)
        if image_id in train_ids:
            for cls in LESION_CLASSES:
                diameters[cls].extend(relative_diameters(masks[cls], fov).tolist())
        if index % 20 == 0 or index == len(image_ids):
            print(f"{dest.name} {index}/{len(image_ids)}", flush=True)


def refresh_fovs(repo_root):
    """Rewrite stored FOV masks from the prepared images. Masks and JPEGs stay."""
    repo_root = Path(repo_root)
    for name in ("IDRiD", "DDR"):
        dest = repo_root / PREPARED[name]
        images = sorted((dest / "images").glob("*.jpg"))
        for index, path in enumerate(images, start=1):
            image = np.asarray(Image.open(path).convert("RGB"))
            _write_binary(dest / "fov" / f"{path.stem}.png", field_of_view(image))
            if index % 40 == 0 or index == len(images):
                print(f"{name} fov {index}/{len(images)}", flush=True)


def prepare_idrid(repo_root):
    from bench.data.idrid import open_idrid

    data = open_idrid(repo_root)
    splits = data.splits()
    dest = Path(repo_root) / PREPARED["IDRiD"]
    write_index(dest, "IDRiD", {k: list(splits[k]) for k in ("train", "val", "test")})

    def load_case(image_id):
        path = data.image_path(image_id)
        image = data.read_image(image_id)
        masks = {cls: data.read_mask(image_id, cls) for cls in LESION_CLASSES}
        return path, "jpg", image, masks

    diameters = {cls: [] for cls in LESION_CLASSES}
    ids = list(splits["train"]) + list(splits["val"]) + list(splits["test"])
    export_split(dest, ids, load_case, splits["train"], diameters)
    return dest, diameters


def prepare_ddr(repo_root):
    from bench.data.ddr import open_ddr

    data = open_ddr(repo_root)
    groups = {
        "train": data.segmentation_ids("train"),
        "val": data.segmentation_ids("valid"),
        "test": data.segmentation_ids("test"),
    }
    dest = Path(repo_root) / PREPARED["DDR"]
    write_index(dest, "DDR", {k: list(v) for k, v in groups.items()})
    source_split = {}
    for split, official in (("train", "train"), ("val", "valid"), ("test", "test")):
        for image_id in groups[split]:
            source_split[image_id] = official

    def load_case(image_id):
        official = source_split[image_id]
        image = data.read_image(official, image_id)
        masks = {cls: data.read_mask(official, image_id, cls) for cls in LESION_CLASSES}
        return data.read_image_bytes(official, image_id), "jpg", image, masks

    diameters = {cls: [] for cls in LESION_CLASSES}
    ids = list(groups["train"]) + list(groups["val"]) + list(groups["test"])
    export_split(dest, ids, load_case, groups["train"], diameters)
    return dest, diameters


def _write_raw(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _done(dest):
    path = Path(dest) / CASE_LOG
    if not path.is_file():
        return {}
    done = {}
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            done[entry["id"]] = entry
    return done


def _case_stats(image_id, case, fov):
    shape = fov.shape
    stats = {
        "id": image_id,
        "size": list(shape),
        "fov_fraction": float(fov.mean()),
        "positive_pixels": {},
        "outside_fov_pixels": {},
        "layers": {},
    }
    for cls in LESION_CLASSES:
        mask = case["masks"][cls]
        stats["positive_pixels"][cls] = int(np.count_nonzero(mask))
        stats["outside_fov_pixels"][cls] = int(np.count_nonzero(mask & ~fov))
    for layer, by_class in case.get("binary", {}).items():
        stats["layers"][layer] = {cls: int(np.count_nonzero(m)) for cls, m in by_class.items()}
    stats.update(case.get("stats", {}))
    return stats


def export_cases(dest, image_ids, load_case, log=print):
    """Write every id in order; resume after the last logged id."""
    dest = Path(dest)
    done = _done(dest)
    with open(dest / CASE_LOG, "a") as handle:
        for index, image_id in enumerate(image_ids, start=1):
            if image_id in done and (dest / "fov" / f"{image_id}.png").is_file():
                continue
            case = load_case(image_id)
            image = case["image"]
            shape = image.shape[:2]
            layers = [case["masks"]] + list(case.get("binary", {}).values()) + list(case.get("raw", {}).values())
            for by_class in layers:
                for cls, array in by_class.items():
                    if array.shape[:2] != shape:
                        raise ValueError(f"{image_id} {cls}: mask {array.shape[:2]} != image {shape}")
            if set(case["masks"]) != set(LESION_CLASSES):
                raise ValueError(f"{image_id} must give all four classes, got {sorted(case['masks'])}")
            fov = field_of_view(image)
            image_path = dest / "images" / f"{image_id}.{case['ext']}"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(case["bytes"])
            for cls, mask in case["masks"].items():
                if np.any(mask):
                    _write_binary(dest / "masks" / cls / f"{image_id}.png", mask)
            for layer, by_class in case.get("binary", {}).items():
                for cls, mask in by_class.items():
                    if np.any(mask):
                        _write_binary(dest / layer / cls / f"{image_id}.png", mask)
            for layer, by_class in case.get("raw", {}).items():
                for cls, array in by_class.items():
                    if np.any(array):
                        _write_raw(dest / layer / cls / f"{image_id}.png", array)
            _write_binary(dest / "fov" / f"{image_id}.png", fov)
            handle.write(json.dumps(_case_stats(image_id, case, fov)) + "\n")
            handle.flush()
            if index % 20 == 0 or index == len(image_ids):
                log(f"{dest.name} {index}/{len(image_ids)}")


def _add(total, value):
    for key, item in value.items():
        if isinstance(item, dict):
            _add(total.setdefault(key, {}), item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            total[key] = total.get(key, 0) + item


def write_report(dest, extra=None):
    """Sum export_cases.jsonl per split into report.json. Reads no model output."""
    dest = Path(dest)
    splits = json.loads((dest / "splits.json").read_text())
    done = _done(dest)
    report = {"dataset": json.loads((dest / "meta.json").read_text())["name"], "splits": {}}
    for split, ids in splits.items():
        missing = [i for i in ids if i not in done]
        if missing:
            raise RuntimeError(f"{dest.name} {split}: {len(missing)} ids not exported, e.g. {missing[0]}")
        entries = [done[i] for i in ids]
        fov = np.array([e["fov_fraction"] for e in entries]) if entries else np.zeros(0)
        summary = {
            "n_images": len(entries),
            "sizes": sorted({tuple(e["size"]) for e in entries}),
            "fov_fraction": {
                "min": float(fov.min()) if fov.size else None,
                "median": float(np.median(fov)) if fov.size else None,
                "max": float(fov.max()) if fov.size else None,
            },
            "images_with": {cls: sum(e["positive_pixels"][cls] > 0 for e in entries) for cls in LESION_CLASSES},
            "sums": {},
        }
        layer_images = {}
        for e in entries:
            for layer, by_class in e["layers"].items():
                for cls, count in by_class.items():
                    slot = layer_images.setdefault(layer, {})
                    slot[cls] = slot.get(cls, 0) + int(count > 0)
            _add(summary["sums"], {k: v for k, v in e.items() if k not in ("id", "size", "fov_fraction")})
        summary["layer_images_with"] = layer_images
        summary["sizes"] = [list(s) for s in summary["sizes"]]
        report["splits"][split] = summary
    path = dest / "report.json"
    if extra is None and path.is_file():
        old = json.loads(path.read_text())
        extra = {k: v for k, v in old.items() if k not in report}
    if extra:
        report.update(extra)
    path.write_text(json.dumps(report, indent=1) + "\n")
    return report


def _start(repo_root, name, splits, granularity, **meta):
    dest = Path(repo_root) / PREPARED[name]
    write_index(dest, name, splits, granularity, **meta)
    return dest


def prepare_tjdr(repo_root, log=print):
    from bench.data import tjdr as source

    data = source.open_tjdr(repo_root)
    out = {}
    for name, field in source.FIELDS.items():
        splits = source.frozen_splits(name)
        dest = _start(
            repo_root,
            name,
            splits,
            "fine",
            source="TJDR",
            field=field,
            field_degrees=source.FIELD_DEGREES[field],
            split_rule=source.SPLIT_RULE,
            exclusive_labels=True,
        )

        def load_case(image_id):
            raw = data.read_image_bytes(image_id)
            with Image.open(BytesIO(raw)) as handle:
                image = np.asarray(handle.convert("RGB"))
            return {"bytes": raw, "ext": "png", "image": image, "masks": data.read_masks(image_id)}

        export_cases(dest, splits["train"] + splits["val"] + splits["test"], load_case, log)
        out[name] = write_report(dest)
    return out


def prepare_fgadr(repo_root, log=print):
    from bench.data import fgadr as source
    from bench.data.splits import read_frozen

    data = source.open_fgadr(repo_root)
    splits = read_frozen("FGADR")
    dest = _start(
        repo_root,
        "FGADR",
        splits,
        "mixed",
        source="FGADR",
        extra_classes=list(source.EXTRA_FOLDERS),
        positive_at=source.POSITIVE_AT,
        split_rule=source.SPLIT_RULE,
        license="non-commercial research only; no redistribution of data or derived data",
    )

    def load_case(image_id):
        raw = data.read_image_bytes(image_id)
        image = data.read_image(image_id)
        masks, extra, dropped = data.read_masks(image_id)
        return {
            "bytes": raw,
            "ext": "png",
            "image": image,
            "masks": masks,
            "binary": {"masks_ext": extra},
            "stats": {"dropped_below_128": dropped, "grade": {f"g{data.grade(image_id)}": 1}},
        }

    export_cases(dest, splits["train"] + splits["val"] + splits["test"], load_case, log)
    return write_report(dest)


def prepare_diaretdb1(repo_root, log=print):
    """Consensus masks for evaluation only. val is the official training list of 28."""
    from bench.data import diaretdb as source

    data = source.open_diaretdb1(repo_root)
    counts = source.positive_image_counts(data)
    problems = source.count_problems(counts)
    if problems:
        raise RuntimeError("DiaRetDB1 consensus counts moved: " + "; ".join(problems))
    official = data.splits()
    splits = {"train": [], "val": list(official["train"]), "test": list(official["test"])}
    dest = _start(
        repo_root,
        "DiaRetDB1",
        splits,
        "region",
        source="DIARETDB1 v2.1",
        confidence=source.CONFIDENCE,
        consensus_at=source.CONSENSUS_AT,
        layers={
            "masks": "four-expert mean confidence >= 0.75",
            "masks_union": "any expert, any confidence",
            "experts": "per-class uint8, two bits per expert (01 lowest), level 0 none 1 Low 2 Medium 3 High",
        },
        split_rule="official lists; val is the official training list and is used only for thresholds",
    )

    def load_case(image_id):
        image = data.read_image(image_id)
        levels = data.expert_levels(image_id, size=image.shape[:2])
        return {
            "bytes": data.image_path(image_id).read_bytes(),
            "ext": "png",
            "image": image,
            "masks": {cls: source.positive(levels[cls]) for cls in LESION_CLASSES},
            "binary": {"masks_union": {cls: source.union(levels[cls]) for cls in LESION_CLASSES}},
            "raw": {"experts": {cls: source.pack_levels(levels[cls]) for cls in LESION_CLASSES}},
        }

    export_cases(dest, splits["val"] + splits["test"], load_case, log)
    extra = {
        "paper_positive_images": source.PAPER_POSITIVE_IMAGES,
        "positive_images_official_split": counts,
        "known_count_differences": {f"{s} {c}": n for (s, c), n in source.KNOWN_COUNT_DIFFERENCES.items()},
        "ellipse_sense": source.ellipse_sense(data),
    }
    return write_report(dest, extra)


def prepare_retlesion(repo_root, log=print):
    from bench.data import retlesion as source
    from bench.data.splits import read_frozen

    data = source.open_retlesion(repo_root)
    splits = read_frozen("Retinal-Lesions")
    grades = data.grades()
    dest = _start(
        repo_root,
        "Retinal-Lesions",
        splits,
        "coarse",
        source="Retinal-Lesions v20191227",
        extra_classes=list(source.EXTRA_FILES),
        ignore_value=source.IGNORE_VALUE,
        split_rule=source.SPLIT_RULE,
    )

    def load_case(image_id):
        raw = data.read_image_bytes(image_id)
        image = data.read_image(image_id)
        masks, extra, ignore = data.read_masks(image_id, image.shape[:2])
        return {
            "bytes": raw,
            "ext": "jpg",
            "image": image,
            "masks": masks,
            "binary": {"masks_ext": extra, "ignore": ignore},
            "stats": {"grade": {f"g{grades[image_id]}": 1}},
        }

    export_cases(dest, splits["train"] + splits["val"] + splits["test"], load_case, log)
    return write_report(dest)


PREPARERS = {
    "TJDR": prepare_tjdr,
    "FGADR": prepare_fgadr,
    "DiaRetDB1": prepare_diaretdb1,
    "Retinal-Lesions": prepare_retlesion,
}
