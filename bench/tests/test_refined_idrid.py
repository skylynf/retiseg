"""bench/scripts/refined_idrid.py on a toy IDRiD whose 'release' is made by the documented crop and resize (26 N4)."""

import json
import re

import numpy as np
import pytest
from PIL import Image

from bench.common import io
from bench.scripts import refined_idrid as ri


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


@pytest.fixture
def toy(tmp_path):
    """Two toy test images (IDRiD_55, IDRiD_56): the release = FOV box with margin 2, resized to 1024 x 1024."""
    shape = (240, 320)
    rng = np.random.default_rng(7)
    prepared = tmp_path / "dataset" / "prepared" / "IDRiD"
    root = tmp_path / "incoming"
    (root / "Test" / "Images").mkdir(parents=True)
    (root / "Test" / "Labels").mkdir(parents=True)
    ids = ["IDRiD_55", "IDRiD_56"]
    truth = {}
    for k, image_id in enumerate(ids):
        fov = _disk(shape, 120, 160 + 10 * k, 100)
        rgb = (rng.random((*shape, 3)) * 255).astype(np.uint8)
        rgb[~fov] = 0
        (prepared / "images").mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(prepared / "images" / f"{image_id}.jpg", quality=95)
        (prepared / "fov").mkdir(exist_ok=True)
        Image.fromarray(fov.astype(np.uint8) * 255).save(prepared / "fov" / f"{image_id}.png")
        masks = {}
        for cls in ("MA", "HE", "EX", "SE"):
            m = np.zeros(shape, bool)
            for _ in range(3):
                m |= _disk(shape, rng.integers(40, 200), rng.integers(80, 240), rng.integers(3, 8))
            masks[cls] = m & fov
            (prepared / "masks" / cls).mkdir(parents=True, exist_ok=True)
            Image.fromarray(masks[cls].astype(np.uint8) * 255).save(prepared / "masks" / cls / f"{image_id}.png")
        truth[image_id] = masks
        # the release: crop the FOV box with a margin of 2 and resize to 1024 without keeping the aspect ratio
        y0, y1, x0, x1 = ri.crop_with_margin(fov, 2)
        released = np.asarray(Image.fromarray(rgb[y0:y1, x0:x1]).resize((1024, 1024), Image.BILINEAR))
        Image.fromarray(released).save(root / "Test" / "Images" / f"{image_id}_f0{k + 1:02d}.jpg", quality=95)
        label = np.zeros((1024, 1024), np.uint8)
        fov_1024 = np.asarray(Image.fromarray(fov[y0:y1, x0:x1].astype(np.uint8) * 255).resize((1024, 1024), Image.NEAREST)) > 0
        label[fov_1024] = ri.VALUES["retina"]
        # refined labels: the original masks resized the same way, the MA mask shifted by 6 pixels (a "correction")
        for cls, value in ri.CLASS_VALUE.items():
            m = np.asarray(Image.fromarray(masks[cls][y0:y1, x0:x1].astype(np.uint8) * 255).resize((1024, 1024), Image.NEAREST)) > 0
            if cls == "MA":
                m = np.roll(m, 6, axis=1)
            label[m & fov_1024] = value
        Image.fromarray(label).save(root / "Test" / "Labels" / f"Label_{image_id}.png")
    (prepared / "meta.json").write_text(json.dumps({"name": "IDRiD", "classes": ["MA", "HE", "EX", "SE"], "granularity": "fine"}))
    (prepared / "splits.json").write_text(json.dumps({"train": [], "val": ids[:1], "test": ids}))
    return tmp_path, prepared, root, ids, truth


def test_find_release_matches_images_and_labels_by_id(toy):
    _tmp, _prepared, root, ids, _truth = toy
    found = ri.find_release(root, "test", ids)
    assert set(found) == set(ids)
    assert found["IDRiD_55"]["image"].name == "IDRiD_55_f001.jpg" and found["IDRiD_55"]["label"].name == "Label_IDRiD_55.png"
    with pytest.raises(FileNotFoundError, match="no image or no label"):
        ri.find_release(root, "test", ids + ["IDRiD_57"])


def test_check_recovers_the_margin_and_prepare_writes_both_label_sets(toy):
    tmp_path, prepared, root, ids, truth = toy
    geometry = tmp_path / "geometry.json"
    result = ri.check(root, prepared=prepared, out=geometry, log=lambda *_: None)
    assert result["accepted"] and result["margins_used"] == [2], result
    assert all(v["ncc"] > 0.97 for v in result["images"].values())
    out_r, out_o = tmp_path / "prep" / "RefinedIDRiD", tmp_path / "prep" / "RefinedIDRiD_original"
    report = ri.prepare(root, geometry=geometry, prepared=prepared, out_refined=out_r, out_original=out_o, log=lambda *_: None)
    assert report["n_images"] == 2 and (out_r / "masks" / "MA" / "IDRiD_55.png").is_file() and (out_o / "fov" / "IDRiD_56.png").is_file()
    assert json.loads((out_r / "splits.json").read_text())["test"] == ids
    t = report["totals"]
    # HE/EX/SE are the same labels through the same geometry (one class per pixel in the release, so overlaps cost a
    # little); the shifted MA differs clearly
    assert min(t[c]["pooled_iou"] for c in ("HE", "EX", "SE")) > 0.98
    assert t["MA"]["pooled_iou"] < 0.9 and abs(t["MA"]["refined_components"] - t["MA"]["original_components"]) <= 1   # a shifted disk may split at the FOV edge
    assert report["retina_vs_fov_iou_mean"] > 0.99
    # a rejected geometry blocks prepare
    bad = json.loads(geometry.read_text())
    bad["accepted"] = False
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="not accepted"):
        ri.prepare(root, geometry=tmp_path / "bad.json", prepared=prepared, out_refined=out_r, out_original=out_o, log=lambda *_: None)


def test_score_moves_the_maps_through_the_geometry_and_scores_both_label_sets(toy):
    tmp_path, prepared, root, ids, truth = toy
    geometry = tmp_path / "geometry.json"
    ri.check(root, prepared=prepared, out=geometry, log=lambda *_: None)
    out_r, out_o = tmp_path / "prep" / "RefinedIDRiD", tmp_path / "prep" / "RefinedIDRiD_original"
    ri.prepare(root, geometry=geometry, prepared=prepared, out_refined=out_r, out_original=out_o, log=lambda *_: None)
    rng = np.random.default_rng(3)
    runs = tmp_path / "runs"
    run = runs / "BS_unet_idrid_seed0"
    for split, name in (("val", "prob_val"), ("test", "prob_test")):
        for image_id in (ids[:1] if split == "val" else ids):
            for cls, m in truth[image_id].items():
                io.write_prob(run / name, cls, image_id, np.clip(m * 0.8 + 0.1 + rng.normal(0, 0.1, m.shape), 0, 1))
    (runs / "BS_unet_ddr_seed0" / "prob_test").mkdir(parents=True)     # DDR-trained: not a candidate
    assert [r.name for r in ri.candidate_runs(runs)] == ["BS_unet_idrid_seed0"]
    results = ri.score(runs, geometry=geometry, prepared=prepared, out_root=tmp_path / "REFINED", log=lambda *_: None, refined_dir=out_r, original_dir=out_o)
    assert set(results) == {"BS_unet_idrid_seed0"}
    done = json.loads((tmp_path / "REFINED" / "BS_unet_idrid_seed0" / "done.json").read_text())
    assert 0 < done["mAUPR_refined"] < done["mAUPR_original_1024"] <= 1, "the shifted MA labels must cost the same maps some AUPR"
    rep = json.loads((tmp_path / "REFINED" / "BS_unet_idrid_seed0" / "original" / "metrics.json").read_text())
    assert rep["n_images"] == 2 and "val_opt" in rep["per_class"]["MA"]["thresholds"]
    q = io.read_prob_quantized(tmp_path / "REFINED" / "BS_unet_idrid_seed0" / "prob_test", "HE", "IDRiD_55")
    assert q.shape == (1024, 1024)
    assert ri.score(runs, geometry=geometry, prepared=prepared, out_root=tmp_path / "REFINED", log=lambda *_: None, refined_dir=out_r, original_dir=out_o) == {}
