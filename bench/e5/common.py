"""Shared pieces of the E5 inference tiers (design/15 sections 4-6, bench/configs/e5_frozen.yaml).

Canvas: every test image is the B-std canvas (field-of-view crop, long side 1,440);
TJDR_uwf is scaled per degree as in E3a (``UWF_LONG_SIDE``). Windows of the
model's native size (SAM/SAM 2/MedSAM 1,024, SAM 3 1,008) are cut with overlap
0.5 as bench.infer does; per-window outputs are merged on the canvas by
pixelwise maximum; the canvas map is placed back on the original image and
written as a 16-bit map with bench.common.io.write_prob, the same file format
every other model writes, so bench.eval scores it unchanged.

Output tree (runs/E5):
    prompts/<set>_<split>.json                         bench.e5.prompts
    a/<model>/<generator>/<set>_<split>/masks/<id>.npz  bench.e5.amg (raw automatic masks)
    a/<model>/<generator>/<set>_<split>/<rule>/         bench.e5.rules (maps + paper.json + metrics_test.json)
    b/<model>/<kind>/k<k>/<set>_<split>/               bench.e5.prompted
    b_control/<run>/<kind>/k<k>/<set>_<split>/         bench.e5.control (supervised maps inside the same boxes)
    sam3/phrase<i>/<set>_<split>/, sam3/selected/...   bench.e5.sam3_text
Every scored directory holds paper.json (the results.csv key) next to metrics_test.json;
paper/scripts/collect_runs.py reads them (collect_e5).

Nothing here imports torch at module level, so the mask store, the prompt
generation, the scoring rules and the tests run on a machine without it.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from bench.common.io import LESION_CLASSES, load_split, write_prob

REPO = Path(__file__).resolve().parents[2]
FROZEN_PATH = REPO / "bench" / "configs" / "e5_frozen.yaml"
E3A_PATH = REPO / "bench" / "configs" / "e3a_frozen.yaml"
PROTOCOL_PATH = REPO / "bench" / "eval" / "protocol.yaml"
OUT_ROOT = REPO / "runs" / "E5"
CANVAS = 1440
WINDOW = {"sam": 1024, "sam2": 1024, "medsam": 1024, "sam3": 1008}
OVERLAP = 0.5
# Test sets of e3a_frozen.yaml; TJDR_uwf is the per-degree canvas. The zero-shot models have no
# training source, so the ultra-widefield scale follows the DDR reference of E3a (1440 x 133 / 45),
# the larger of the two training sources' per-degree scales.
UWF_LONG_SIDE = 4256
THRESHOLD_SOURCE = ("DDR", "val")   # design/15 section 4: Dice thresholds of the inference tiers come from DDR val
PAPER_SET = {"IDRiD": "IDRiD", "DDR": "DDR", "TJDR_std": "TJDR-std", "TJDR_uwf": "TJDR-uwf", "FGADR": "FGADR",
             "Retinal-Lesions": "Retinal-Lesions", "DiaRetDB1": "DiaRetDB1"}
PAPER_MODEL = {"sam": "SAM", "sam2": "SAM 2", "medsam": "MedSAM", "sam3": "SAM 3"}


def load_frozen(path=FROZEN_PATH):
    return yaml.safe_load(Path(path).read_text())


def load_e3a(path=E3A_PATH):
    return yaml.safe_load(Path(path).read_text())


def test_sets():
    return list(load_e3a()["test_sets"])


def data_dir(test_set):
    return REPO / load_e3a()["test_sets"][test_set]["data"]


def long_side_for(test_set):
    return UWF_LONG_SIDE if test_set in (load_e3a().get("per_degree_sets") or ()) else CANVAS


def split_ids(test_set, split):
    """Ids of a split without the frozen E3a exclusions (none of the zero-shot models has a training source)."""
    ids = load_split(data_dir(test_set), split)
    exclude = set((load_e3a().get("exclude") or {}).get(test_set) or ())
    return [image_id for image_id in ids if image_id not in exclude]


def load_canvas(test_set, image_id, masks=False):
    """The canvas dict of bench.data.b1_input.crop_resize (image uint8 RGB, masks, crop, canvas_hw, original_hw,
    canvas_fov); no tensor, so no torch."""
    from bench.common.io import read_fov, read_mask
    from bench.data.b1_input import crop_resize, read_image

    directory = data_dir(test_set)
    image = read_image(directory, image_id)
    fov = read_fov(directory, image_id)
    label = [read_mask(directory, cls, image_id, shape=image.shape[:2]) for cls in LESION_CLASSES] if masks else None
    canvas = crop_resize(image, fov, long_side_for(test_set), None, label)
    canvas["image_id"] = image_id
    return canvas


def window_starts(length, window, stride):
    """bench.infer.window_starts, copied so that this module stays torch-free."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def windows(height, width, size, overlap=OVERLAP):
    """(top, left, h, w) of overlapping windows covering the canvas; a canvas smaller than the window is one window."""
    size = int(size)
    stride = max(1, int(round(size * (1.0 - overlap))))
    return [(top, left, size, size) for top in window_starts(height, size, stride) for left in window_starts(width, size, stride)]


def window_crop(image, top, left, h, w):
    """Window pixels; the canvas is zero-padded at the bottom and right when a window runs past it."""
    out = np.zeros((h, w) + image.shape[2:], dtype=image.dtype)
    sub = image[top : top + h, left : left + w]
    out[: sub.shape[0], : sub.shape[1]] = sub
    return out


def window_of(box, wins):
    """Index of the window whose edges are farthest from the box centre (design/15 section 6.3)."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    best, best_margin = 0, -float("inf")
    for index, (top, left, h, w) in enumerate(wins):
        margin = min(cx - left, left + w - cx, cy - top, top + h - cy)
        if margin > best_margin:
            best, best_margin = index, margin
    return best


def paste_max(canvas_map, window_map, top, left):
    """Pixelwise maximum of a window map into the canvas map (in place), clipped at the canvas edge."""
    h = min(window_map.shape[0], canvas_map.shape[0] - top)
    w = min(window_map.shape[1], canvas_map.shape[1] - left)
    if h <= 0 or w <= 0:
        return
    region = canvas_map[top : top + h, left : left + w]
    np.maximum(region, window_map[:h, :w], out=region)


def place_on_original(canvas_prob, canvas):
    """Canvas-resolution map (H_c, W_c) in [0, 1] -> original-resolution map inside the field-of-view crop."""
    canvas_h, canvas_w = canvas["canvas_hw"]
    prob = np.ascontiguousarray(canvas_prob[:canvas_h, :canvas_w], dtype=np.float32)
    y0, y1, x0, x1 = canvas["crop"]
    crop_h, crop_w = y1 - y0, x1 - x0
    if prob.shape != (crop_h, crop_w):
        prob = np.asarray(Image.fromarray(prob).resize((crop_w, crop_h), Image.Resampling.BILINEAR), dtype=np.float32)
    placed = np.zeros(canvas["original_hw"], dtype=np.float32)
    placed[y0:y1, x0:x1] = np.clip(prob, 0.0, 1.0)
    return placed


def write_maps(pred_dir, image_id, canvas, maps):
    """``maps``: {cls: canvas map} for the four classes, or {"any": map} written to every class (class-agnostic)."""
    if "any" in maps:
        placed = place_on_original(maps["any"], canvas)
        for cls in LESION_CLASSES:
            write_prob(pred_dir, cls, image_id, placed)
        return
    for cls in LESION_CLASSES:
        write_prob(pred_dir, cls, image_id, place_on_original(maps[cls], canvas))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def done_marker(directory):
    return Path(directory) / "done.json"


def is_done(directory):
    return done_marker(directory).is_file()


def components(mask, connectivity=8):
    """Labelled connected components of a binary mask (scipy), with the frozen 8-connectivity."""
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else None
    labels, n = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return labels, int(n)


def set_dirname(test_set, split):
    return f"{test_set}_{split}"


def parse_set_dirname(name):
    """"<set>_<split>" -> (set, split); the set names of e3a_frozen contain no underscore except TJDR_*."""
    stem, _, split = name.rpartition("_")
    if stem not in PAPER_SET:
        raise ValueError(f"{name!r} is not <test set>_<split>")
    return stem, split


def paper_record(model, variant, test_set, variant_by_class=None):
    """The results.csv key of one scored directory (paper/data/results.csv, scripts/collect_runs.collect_e5)."""
    record = {"experiment": "E5", "model": PAPER_MODEL.get(model, model), "variant": variant, "train_set": "none",
              "seed": -1, "test_set": PAPER_SET[test_set]}
    if variant_by_class:
        record["variant_by_class"] = dict(variant_by_class)
    return record


def environment():
    """Host, time and library versions, for recipe-style records (torch and the official repositories if present)."""
    import datetime
    import platform
    import socket

    out = {"host": socket.gethostname(), "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "python": platform.python_version(), "numpy": np.__version__}
    try:
        import torch

        out["torch"] = getattr(torch, "__version__", None)
        out["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (ImportError, AttributeError):
        pass
    return out
