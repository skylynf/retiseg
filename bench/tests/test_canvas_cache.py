from pathlib import Path

import numpy as np
import pytest

from bench.common.io import LESION_CLASSES
from bench.data import canvas_cache
from bench.data.b1_input import apply_forward_size, load_example

REPO = Path(__file__).resolve().parents[2]
IDRID = REPO / "dataset/prepared/IDRiD"


@pytest.mark.skipif(not IDRID.is_dir(), reason="prepared IDRiD is not on this machine")
@pytest.mark.parametrize("forward_size", [None, (224, 224)])
def test_cached_canvas_equals_the_on_the_fly_canvas(tmp_path, monkeypatch, forward_size):
    monkeypatch.setattr(canvas_cache, "CACHE_ROOT", tmp_path)
    ids = ["IDRiD_01", "IDRiD_03"]
    assert canvas_cache.build(IDRID, 1440, splits=("train",), only_ids=set(ids)) == 2
    assert canvas_cache.build(IDRID, 1440, splits=("train",), only_ids=set(ids)) == 0
    cache = canvas_cache.CanvasCache(IDRID, 1440, ids)
    for image_id in ids:
        fresh = load_example(IDRID, image_id, 1440, [0.5] * 3, [0.5] * 3, forward_size, masks=True)
        image, masks = apply_forward_size(*cache.load(image_id), forward_size)
        assert np.array_equal(image, fresh["image"])
        assert len(masks) == len(LESION_CLASSES)
        for got, expected in zip(masks, fresh["masks"]):
            assert np.array_equal(got, expected)


@pytest.mark.skipif(not IDRID.is_dir(), reason="prepared IDRiD is not on this machine")
def test_changed_sources_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(canvas_cache, "CACHE_ROOT", tmp_path)
    canvas_cache.build(IDRID, 1440, splits=("train",), only_ids={"IDRiD_01"})
    original = canvas_cache._sources

    def _changed(dataset_dir, image_id):
        sources = original(dataset_dir, image_id)
        sources["images/IDRiD_01.jpg"] = [0, 0]
        return sources

    monkeypatch.setattr(canvas_cache, "_sources", _changed)
    with pytest.raises(ValueError, match="sources changed"):
        canvas_cache.CanvasCache(IDRID, 1440, ["IDRiD_01"])
    assert canvas_cache.open_cache(IDRID, 999, ["IDRiD_01"]) is None
    monkeypatch.setenv("RETISEG_CANVAS_CACHE", "off")
    assert canvas_cache.open_cache(IDRID, 1440, ["IDRiD_01"]) is None
