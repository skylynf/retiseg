"""Oracle prompts from the test labels (tier b, e5_frozen.yaml ``oracle``).

For every image and class: 8-connected components of the ground truth on the
canvas; per component a tight box and one positive point (the pixel farthest
from the component boundary, inside the component). Components are sorted by
(row, col) of their top-left pixel and permuted once with a PCG64 stream spawned
per (test set, image, class) from seed 20261003, so every k is a prefix of the
same order and every model reads the same file.

    python -m bench.e5 prompts --set IDRiD [--split test]
    -> runs/E5/prompts/<set>_<split>.json  {"sha256": ..., "images": {id: {cls: [{"box": [x0,y0,x1,y1], "point": [x,y], "area": n}, ...]}}}
"""

import numpy as np
from scipy import ndimage

from bench.common.io import LESION_CLASSES
from bench.e5 import common

SEED = 20261003


def component_prompts(mask):
    labels, n = common.components(mask, 8)
    out = []
    for index in range(1, n + 1):
        part = labels == index
        ys, xs = np.nonzero(part)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        # farthest-from-boundary pixel: distance transform inside the component
        sub = part[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        dist = ndimage.distance_transform_edt(np.pad(sub, 1))[1:-1, 1:-1]
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        point = [int(xs.min() + ix), int(ys.min() + iy)]
        out.append({"box": box, "point": point, "area": int(part.sum()), "anchor": [int(ys.min()), int(xs.min())]})
    out.sort(key=lambda item: (item["anchor"][0], item["anchor"][1]))
    return out


def build(test_set, split="test"):
    ids = common.split_ids(test_set, split)
    images = {}
    for image_id in ids:
        canvas = common.load_canvas(test_set, image_id, masks=True)
        per_class = {}
        for cls, mask in zip(LESION_CLASSES, canvas["masks"]):
            items = component_prompts(mask)
            rng = np.random.Generator(np.random.PCG64([SEED, _hash(test_set), _hash(image_id), _hash(cls)]))
            order = rng.permutation(len(items)) if items else np.zeros(0, dtype=int)
            per_class[cls] = [dict(items[i], order=int(rank)) for rank, i in enumerate(order)]
            per_class[cls].sort(key=lambda item: item["order"])
        images[image_id] = per_class
    payload = {"test_set": test_set, "split": split, "canvas_long_side": common.long_side_for(test_set), "seed": SEED,
               "connectivity": 8, "n_images": len(ids), "images": images}
    path = common.OUT_ROOT / "prompts" / f"{test_set}_{split}.json"
    common.write_json(path, payload)
    digest = common.sha256_of(path)
    payload["sha256"] = digest
    common.write_json(path, payload)
    counts = {cls: int(np.mean([len(v[cls]) for v in images.values()])) if images else 0 for cls in LESION_CLASSES}
    print(f"{path}: {len(ids)} images, mean components per image {counts}, sha256 {digest[:12]}", flush=True)
    return path


def _hash(text):
    return int.from_bytes(__import__("hashlib").sha256(str(text).encode()).digest()[:4], "little")


def load(test_set, split="test"):
    import json

    path = common.OUT_ROOT / "prompts" / f"{test_set}_{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run python -m bench.e5 prompts --set {test_set} --split {split} first")
    return json.loads(path.read_text())


def select(items, k):
    """The first k prompts of the frozen order; "all" keeps every component."""
    if k == "all":
        return list(items)
    return list(items[: int(k)])
