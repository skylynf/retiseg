"""Training sources for the factor experiments (docs/paper/16 section 3: E3b, E4, E7, E8, E9).

A ``PreparedSource`` is one split of any prepared dataset (not only IDRiD and
DDR), optionally restricted to a frozen subset list. A ``MultiSource`` joins
several of them into the one source that ``bench.data.bs_input.PatchTrain``
samples from: canvases are read through the per-dataset canvas cache when one
is built for that diameter, otherwise on the fly. Both expose the interface
PatchTrain and bstd.ValScorer use: ``ids``, ``mean``, ``std``,
``forward_size`` (always None), ``canvas(index)`` and ``canvas_with_fov(index)``.

Ignore regions (Retinal-Lesions ``ignore/<CLS>/<id>.png``, the "uncertain"
label) are left out of the evaluation by bench.eval; in training they are
background, which is recorded as a deviation of every cell that trains on such
a source (14 note 38).
"""

import json
from pathlib import Path

import numpy as np

from bench.common.io import load_split
from bench.data.b1_input import load_example

PREPARED_NAMES = ("IDRiD", "DDR", "TJDR_std", "TJDR_uwf", "FGADR", "RetinalLesions", "DiaRetDB1")


def prepared_name(dataset_dir):
    path = Path(dataset_dir)
    if path.parent.name != "prepared" or path.name not in PREPARED_NAMES:
        raise ValueError(f"expected dataset/prepared/<one of {PREPARED_NAMES}>, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path.name


def load_subset(path):
    """A frozen subset list: {"dataset": ..., "split": ..., "ids": [...], ...} (bench/scripts/make_subsets.py)."""
    doc = json.loads(Path(path).read_text())
    ids = list(doc["ids"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} lists an id twice")
    return doc, ids


class PreparedSource:
    def __init__(self, dataset_dir, split, diameter, mean, std, subset=None):
        self.root = Path(dataset_dir)
        self.name = prepared_name(self.root)
        self.split = str(split)
        self.diameter = int(diameter)
        self.mean = mean
        self.std = std
        self.forward_size = None
        ids = load_split(self.root, self.split)
        self.subset = None
        if subset is not None:
            doc, chosen = load_subset(subset)
            if doc.get("dataset") != self.name or doc.get("split") != self.split:
                raise ValueError(f"{subset} is for {doc.get('dataset')}/{doc.get('split')}, not {self.name}/{self.split}")
            missing = sorted(set(chosen) - set(ids))
            if missing:
                raise ValueError(f"{subset} lists ids outside {self.name}/{self.split}: {missing[:5]}")
            ids = [image_id for image_id in ids if image_id in set(chosen)]
            self.subset = str(subset)
        self.ids = list(ids)
        from bench.data.canvas_cache import open_cache

        self.cache = open_cache(self.root, self.diameter, self.ids)

    def __len__(self):
        return len(self.ids)

    def canvas(self, index):
        image_id = self.ids[index]
        if self.cache is not None:
            return self.cache.load(image_id)
        canvas = load_example(self.root, image_id, self.diameter, self.mean, self.std, None, masks=True)
        return canvas["image"], canvas["masks"]

    def canvas_with_fov(self, index):
        image_id = self.ids[index]
        if self.cache is not None:
            image, masks = self.cache.load(image_id)
            return image, masks, self.cache.load_fov(image_id)
        canvas = load_example(self.root, image_id, self.diameter, self.mean, self.std, None, masks=True)
        return canvas["image"], canvas["masks"], canvas["canvas_fov"]


class MultiSource:
    """Several PreparedSources in order; index i reads source k at local index i - offset_k."""

    def __init__(self, sources):
        if not sources:
            raise ValueError("MultiSource needs at least one source")
        self.sources = list(sources)
        first = self.sources[0]
        for source in self.sources[1:]:
            if source.diameter != first.diameter or list(source.mean) != list(first.mean) or list(source.std) != list(first.std):
                raise ValueError("every source of a MultiSource must share diameter and normalization")
        self.diameter = first.diameter
        self.mean = first.mean
        self.std = first.std
        self.forward_size = None
        self.ids = [f"{source.name}/{image_id}" for source in self.sources for image_id in source.ids]
        self.offsets = np.cumsum([0] + [len(source) for source in self.sources])

    def __len__(self):
        return len(self.ids)

    def _locate(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        k = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.sources[k], index - int(self.offsets[k])

    def canvas(self, index):
        source, local = self._locate(index)
        return source.canvas(local)

    def canvas_with_fov(self, index):
        source, local = self._locate(index)
        return source.canvas_with_fov(local)

    def describe(self):
        return [
            {"dataset": source.name, "split": source.split, "n": len(source), "subset": source.subset}
            for source in self.sources
        ]
