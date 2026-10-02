"""Find IDRiD, DDR and e-ophtha images.

A manifest.csv in the dataset root overrides discovery. Columns:
id,split,image,HE,MA,EX,SE
split is train, val or test. An empty mask cell means that lesion is absent.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LESIONS = ("HE", "MA", "EX", "SE")


@dataclass
class ImageRecord:
    id: str
    split: str
    image_path: Path
    mask_paths: dict = field(default_factory=dict)


def load_dataset(root: Path, dataset: str, split_file: Path | None = None) -> list[ImageRecord]:
    root = Path(root)
    manifest = root / "manifest.csv"
    if manifest.exists():
        return _read_manifest(manifest)
    dataset = dataset.lower()
    if dataset == "idrid":
        return _discover_idrid(root)
    if dataset == "ddr":
        return _discover_ddr(root)
    if dataset in ("eophtha", "e-ophtha"):
        return _discover_eophtha(root, split_file)
    raise ValueError(f"unknown dataset {dataset}")


def _read_manifest(path: Path) -> list[ImageRecord]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "split", "image", "HE", "MA", "EX", "SE"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"manifest needs columns {sorted(required)}")
        for row in reader:
            masks = {}
            for lesion in LESIONS:
                value = (row.get(lesion) or "").strip()
                masks[lesion] = Path(value) if value else None
            split = row["split"].strip().lower()
            if split in ("validation", "valid"):
                split = "val"
            if split in ("testing",):
                split = "test"
            records.append(
                ImageRecord(
                    id=row["id"].strip(),
                    split=split,
                    image_path=Path(row["image"].strip()),
                    mask_paths=masks,
                )
            )
    return records


def _discover_idrid(root: Path) -> list[ImageRecord]:
    images = []
    masks = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        lesion = _suffix_lesion(stem)
        parent = str(path.parent).lower()
        if lesion is not None:
            image_id = stem[: -(len(lesion) + 1)]
            masks.setdefault(image_id, {})[lesion] = path
            continue
        if "idrid" not in stem.lower():
            continue
        split = _split_from_path(parent)
        if split is None:
            continue
        images.append((stem, split, path))
    records = []
    for image_id, split, path in images:
        mask_paths = {lesion: masks.get(image_id, {}).get(lesion) for lesion in LESIONS}
        records.append(ImageRecord(image_id, split, path, mask_paths))
    if not records:
        raise FileNotFoundError(f"no IDRiD images under {root}")
    return records


def _discover_ddr(root: Path) -> list[ImageRecord]:
    records = []
    for split_dir in root.rglob("*"):
        if not split_dir.is_dir():
            continue
        split = _ddr_split_name(split_dir.name)
        if split is None:
            continue
        image_dir = _first_image_dir(split_dir)
        if image_dir is None:
            continue
        label_root = _label_root(split_dir)
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            masks = {}
            for lesion in LESIONS:
                masks[lesion] = _find_label(label_root, image_path.stem, lesion)
            records.append(ImageRecord(image_path.stem, split, image_path, masks))
    if not records:
        raise FileNotFoundError(f"no DDR lesion-segmentation images under {root}")
    return records


def _discover_eophtha(root: Path, split_file: Path | None) -> list[ImageRecord]:
    from clcnet.splits import load_or_create_eophtha_split

    cases = {"MA": {}, "EX": {}}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = [part.lower() for part in path.parts]
        if any("healthy" in part for part in parts):
            continue
        lesion = None
        for candidate in ("MA", "EX"):
            if any(part == candidate.lower() or f"e_ophtha_{candidate.lower()}" in part or f"eophtha_{candidate.lower()}" in part for part in parts):
                lesion = candidate
                break
        if lesion is None:
            if any("ex" == part or "ma" == part for part in parts):
                lesion = "EX" if "ex" in parts else "MA"
        if lesion is None:
            continue
        case_id = path.parent.name
        cases[lesion].setdefault(case_id, []).append(path)
    both = sorted(set(cases["MA"]) & set(cases["EX"]))
    if split_file is None:
        split_file = Path(__file__).resolve().parents[1] / "splits" / "eophtha_seed20230108.json"
    train, test, _ = load_or_create_eophtha_split(both, split_file)
    split_of = {stem: "train" for stem in train}
    split_of.update({stem: "test" for stem in test})
    records = []
    for case_id in both:
        image_path, ma_paths = _eophtha_image_and_annotations(cases["MA"][case_id])
        ex_image, ex_paths = _eophtha_image_and_annotations(cases["EX"][case_id])
        # The photograph is stored in both subsets. Keep the MA-side file as the image.
        del ex_image
        records.append(
            ImageRecord(
                id=case_id,
                split=split_of[case_id],
                image_path=image_path,
                mask_paths={"MA": ma_paths, "EX": ex_paths, "HE": None, "SE": None},
            )
        )
    return records


def _eophtha_image_and_annotations(paths: list[Path]):
    paths = sorted(paths)
    folder = paths[0].parent.name
    image = None
    for path in paths:
        if path.stem == folder:
            image = path
            break
    if image is None:
        image = max(paths, key=lambda path: path.stat().st_size)
    annotations = [path for path in paths if path != image]
    return image, annotations


def _suffix_lesion(stem: str):
    upper = stem.upper()
    for lesion in LESIONS:
        if upper.endswith("_" + lesion):
            return lesion
    return None


def _split_from_path(path_text: str):
    if "test" in path_text:
        return "test"
    if "train" in path_text:
        return "train"
    if "valid" in path_text:
        return "val"
    return None


def _ddr_split_name(name: str):
    lowered = name.lower()
    if lowered in ("train", "training"):
        return "train"
    if lowered in ("valid", "validation", "val"):
        return "val"
    if lowered in ("test", "testing"):
        return "test"
    return None


def _first_image_dir(split_dir: Path):
    for candidate in (split_dir / "image", split_dir / "images", split_dir / "Image", split_dir / "Images"):
        if candidate.is_dir() and any(path.suffix.lower() in IMAGE_SUFFIXES for path in candidate.iterdir()):
            return candidate
    return None


def _label_root(split_dir: Path):
    for candidate in (split_dir / "label", split_dir / "labels", split_dir / "segmentation", split_dir):
        if candidate.is_dir():
            return candidate
    return split_dir


def _find_label(label_root: Path, stem: str, lesion: str):
    if label_root is None:
        return None
    names = {
        "HE": ("HE", "he", "Haemorrhages", "Hemorrhages", "hemorrhages", "haemorrhages"),
        "MA": ("MA", "ma", "Microaneurysms", "microaneurysms"),
        "EX": ("EX", "ex", "HardExudates", "hard exudates", "Hard_Exudates"),
        "SE": ("SE", "se", "SoftExudates", "soft exudates", "Soft_Exudates"),
    }[lesion]
    for directory in label_root.rglob("*"):
        if not directory.is_dir() or directory.name not in names:
            continue
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None
