"""Oracle prompts of the E5 b tier (design/15 section 6; e5_frozen.yaml oracle): one file per set, split and variant.

The prompts come from the ground truth on the canvas the models see
(bench.e5.geometry.long_side_for), so the b tier is an upper bound and is
never ranked with trained models. The rules are the registered ones:

    components  8-connected components of each class mask on the canvas
    order       components sorted by their first pixel in row-major order (smallest
                row, then smallest column), then one permutation drawn from
                PCG64(20261003) spawned per (image, class); the first k prompts of a
                smaller k are therefore a prefix of a larger k
    box         the tight bounding box [x0, y0, x1, y1] in canvas pixels, x1 and y1
                exclusive, no jitter
    point       the component pixel farthest from its boundary (Euclidean distance
                transform inside the component; ties go to the first pixel in
                row-major order), stored [x, y], one positive point

The file records the rules, the k levels, every image's canvas size, the sha256
of every mask file (None where the class has no file, i.e. a negative image)
and the number of components per image and class, the statistic of 15 section
6.5 ("prompts per image and class"). Every model of the tier reads the same file.

    python -m bench.e5.prompts --set IDRiD --split test [--variant fov] [--out runs/E5_prompts]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy import ndimage

from bench.common.io import LESION_CLASSES, load_split
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo

SEED = 20261003
CONNECTIVITY = 8
_STRUCTURE = np.ones((3, 3), dtype=bool)
_RULE_KEYS = ("components", "order", "box", "point")


def components(mask):
    """The 8-connected components of a boolean mask, sorted by their first pixel in row-major order.

    Each is {"box": [x0, y0, x1, y1], "point": [x, y], "area": pixels, "first": [row, col]}.
    """
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=_STRUCTURE)
    if count == 0:
        return []
    out = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        rows_slice, cols_slice = slices
        piece = labels[slices] == index
        rows, cols = np.nonzero(piece)  # row-major order
        first = (int(rows[0]) + rows_slice.start, int(cols[0]) + cols_slice.start)
        # pixels outside the component (other components included) count as boundary
        distance = ndimage.distance_transform_edt(np.pad(piece, 1))[1:-1, 1:-1]
        flat = int(np.argmax(distance))  # the first maximum in row-major order
        py, px = divmod(flat, piece.shape[1])
        out.append(
            {
                "box": [int(cols_slice.start), int(rows_slice.start), int(cols_slice.stop), int(rows_slice.stop)],
                "point": [int(px + cols_slice.start), int(py + rows_slice.start)],
                "area": int(piece.sum()),
                "first": [first[0], first[1]],
            }
        )
    out.sort(key=lambda item: (item["first"][0], item["first"][1]))
    return out


def spawn_rng(image_id, cls, seed=SEED):
    """The (image, class) stream: PCG64 seeded from SeedSequence(seed, spawn_key from sha256(image_id/cls))."""
    digest = hashlib.sha256(f"{image_id}/{cls}".encode("utf-8")).digest()
    key = tuple(int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=key)))


def ordered(comps, image_id, cls, seed=SEED):
    """The prompt order: one permutation of the sorted components; ``rank`` is the position."""
    order = spawn_rng(image_id, cls, seed).permutation(len(comps))
    return [dict(comps[int(i)], rank=rank) for rank, i in enumerate(order)]


def take(prompts, k):
    """The first k prompts ("all": every one); the prefix property makes k nested."""
    return list(prompts) if k == "all" else list(prompts)[: int(k)]


def mask_sha256(data_dir, cls, image_id):
    path = Path(data_dir) / "masks" / cls / f"{image_id}.png"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def verify_masks(doc, data_dir, image_id):
    """Raise when a ground-truth mask file of the image is not the one the prompt file was built from (review fix 12)."""
    recorded = doc["images"][image_id]["mask_sha256"]
    for cls in LESION_CLASSES:
        actual = mask_sha256(data_dir, cls, image_id)
        if actual != recorded.get(cls):
            raise ValueError(f"{image_id} {cls}: the mask file differs from the one the prompt file recorded "
                             f"(sha256 {recorded.get(cls)} -> {actual}); regenerate the prompt file with bench.e5.prompts")


def prompts_for_image(cv, image_id, seed=SEED):
    """{cls: {"n": components, "prompts": [...]}} from the canvas masks, LESION_CLASSES order."""
    if cv.masks is None:
        raise ValueError("the canvas was built without masks")
    out = {}
    for cls, mask in zip(LESION_CLASSES, cv.masks):
        comps = ordered(components(mask), image_id, cls, seed)
        out[cls] = {"n": len(comps), "prompts": comps}
    return out


def build(set_name, split, variant=geo.MAIN, frozen=None, out_dir=None, data_dir=None, long_side=None):
    """Write the prompt file of (set, split, variant) and return its path.

    ``data_dir`` and ``long_side`` override the registered ones (tests); by
    default the set's prepared directory and the G1 canvas long side.
    """
    frozen = frozen or fz.load_frozen()
    if long_side is None:
        long_side = geo.long_side_for(set_name, variant, frozen)
    elif variant != geo.MAIN:
        geo.long_side_for(set_name, variant, frozen)  # the variant must still be a registered one
    data_dir = Path(data_dir) if data_dir is not None else geo.data_dir(set_name)
    ids = load_split(data_dir, split)
    images = {}
    totals = {cls: 0 for cls in LESION_CLASSES}
    for image_id in ids:
        cv = geo.canvas(data_dir, image_id, int(long_side), masks=True)
        classes = prompts_for_image(cv, image_id)
        for cls in LESION_CLASSES:
            totals[cls] += classes[cls]["n"]
        images[image_id] = {
            "canvas_hw": list(cv.hw),
            "mask_sha256": {cls: mask_sha256(data_dir, cls, image_id) for cls in LESION_CLASSES},
            "classes": classes,
        }
    doc = {
        "set": set_name,
        "split": split,
        "variant": variant,
        "long_side": int(long_side),
        "seed": SEED,
        "connectivity": CONNECTIVITY,
        "rule": {key: frozen["oracle"][key] for key in _RULE_KEYS},
        "k": list(frozen["oracle"]["k"]),
        "n_images": len(ids),
        "components_per_image": {cls: (totals[cls] / len(ids) if ids else 0.0) for cls in LESION_CLASSES},
        "images": images,
    }
    path = fz.prompt_file(set_name, split, out_dir, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench.e5.prompts", description="Write the oracle prompt file of one test set and split.")
    parser.add_argument("--set", required=True, help="a test set of bench/configs/e3a_frozen.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default=geo.MAIN, help="main, or a comparison variant such as fov for TJDR_uwf")
    parser.add_argument("--out", help="directory of the prompt files (default runs/E5_prompts)")
    args = parser.parse_args(argv)
    path = build(args.set, args.split, args.variant, out_dir=args.out)
    doc = load(path)
    counts = ", ".join(f"{cls} {doc['components_per_image'][cls]:.1f}" for cls in LESION_CLASSES)
    print(f"{path}: {doc['n_images']} images, canvas long side {doc['long_side']}, components per image {counts}")


if __name__ == "__main__":
    main()
