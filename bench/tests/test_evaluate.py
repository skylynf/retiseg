import json

import numpy as np
import pytest
from PIL import Image

from bench.common import io
from bench.eval.evaluate import evaluate, load_protocol


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _save_bin(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8) * 255).save(path)


@pytest.fixture
def toy_dataset(tmp_path):
    root = tmp_path / "TOY"
    shape = (120, 160)
    rng = np.random.default_rng(0)
    ids = {"train": [], "val": ["v0", "v1"], "test": ["t0", "t1", "t2"]}
    truth = {}
    for split in ("val", "test"):
        for image_id in ids[split]:
            fov = _disk(shape, 60, 80, 58)
            _save_bin(root / "fov" / f"{image_id}.png", fov)
            masks = {}
            for c in ("MA", "HE", "EX"):
                m = np.zeros(shape, bool)
                for _ in range(rng.integers(1, 4)):
                    m |= _disk(shape, rng.integers(20, 100), rng.integers(30, 130), rng.integers(1, 6))
                m &= fov
                masks[c] = m
                _save_bin(root / "masks" / c / f"{image_id}.png", m)
            truth[image_id] = masks
    (root / "meta.json").write_text(json.dumps({"name": "TOY", "classes": ["MA", "HE", "EX"], "granularity": "fine"}))
    (root / "splits.json").write_text(json.dumps(ids))
    return root, truth


def _write_preds(pred_dir, truth, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    for image_id, masks in truth.items():
        for c, m in masks.items():
            p = m.astype(float) * 0.9 + 0.05
            if noise:
                p = np.clip(p + rng.normal(0, noise, p.shape), 0, 1)
            io.write_prob(pred_dir, c, image_id, p)


def test_prob_png_roundtrip_is_exact(tmp_path):
    p = np.linspace(0, 1, 12).reshape(3, 4)
    io.write_prob(tmp_path, "MA", "x", p)
    q = io.read_prob_quantized(tmp_path, "MA", "x")
    assert q.dtype == np.int64 and np.array_equal(q, np.rint(p * io.PROB_LEVELS))


def test_m2mrf_overwrite_order():
    a = np.array([True, True, False])
    b = np.array([True, False, True])
    out = io.apply_m2mrf_overwrite({"EX": a, "MA": b})
    assert out["EX"].tolist() == [False, True, False]
    assert out["MA"].tolist() == b.tolist()


def test_perfect_predictions_score_one(toy_dataset, tmp_path):
    root, truth = toy_dataset
    _write_preds(tmp_path / "pred", truth)
    proto = load_protocol()
    proto["bootstrap"]["n"] = 100
    res = evaluate(root, "test", tmp_path / "pred", proto, val_pred_dir=tmp_path / "pred", out_prefix=tmp_path / "m" / "toy")
    assert res["classes"] == ["MA", "HE", "EX"]
    assert res["summary"]["mAUPR"] == pytest.approx(1.0)
    for c in res["classes"]:
        fixed = res["per_class"][c]["thresholds"]["fixed"]
        assert fixed["pooled"]["dice"] == pytest.approx(1.0)
        assert fixed["lesion"]["recall"] == pytest.approx(1.0)
        assert "val_opt" in res["per_class"][c]["thresholds"]
    saved = np.load(tmp_path / "m" / "toy.npz")
    assert saved["MA_pos"].shape == (3, proto["image_hist_bins"])
    assert json.loads((tmp_path / "m" / "toy.json").read_text())["protocol_sha256"] == proto["_sha256"]


def test_noisy_predictions_and_resolution_check(toy_dataset, tmp_path):
    root, truth = toy_dataset
    _write_preds(tmp_path / "pred", truth, noise=0.3)
    proto = load_protocol(lesion_metrics=False)
    proto["bootstrap"]["n"] = 100
    res = evaluate(root, "test", tmp_path / "pred", proto)
    assert 0 < res["summary"]["mAUPR"] < 1
    lo, hi = res["per_class"]["MA"]["aupr_ci"]
    assert lo <= res["per_class"]["MA"]["aupr"] <= hi

    io.write_prob(tmp_path / "pred", "MA", "t0", np.zeros((60, 80)))
    with pytest.raises(ValueError, match="original resolution"):
        evaluate(root, "test", tmp_path / "pred", proto)


def test_any_lesion_and_image_level_outputs(toy_dataset, tmp_path):
    root, truth = toy_dataset
    _write_preds(tmp_path / "pred", truth)
    proto = load_protocol()
    proto["bootstrap"]["n"] = 50
    res = evaluate(root, "test", tmp_path / "pred", proto, out_prefix=tmp_path / "m" / "toy")
    assert res["summary"]["AUPR_any_lesion"] == pytest.approx(1.0)
    assert res["any_lesion"]["n_positive_pixels"] > 0
    for c in res["classes"]:
        level = res["per_class"][c]["image_level"]["max"]
        assert level["n_images"] == 3 and level["n_positive"] == 3
    saved = np.load(tmp_path / "m" / "toy.npz")
    assert saved["ANY_pos"].shape == (3, proto["image_hist_bins"])
    assert saved["MA_img_label"].tolist() == [True, True, True]


def test_any_lesion_skips_ignored_pixels_unless_another_class_marks_them(tmp_path):
    from bench.eval.evaluate import any_lesion

    base = np.ones((2, 3), bool)
    masks = {"MA": np.array([[1, 0, 0], [0, 0, 0]], bool), "HE": np.array([[0, 1, 0], [0, 0, 0]], bool)}
    ignore_ma = np.array([[0, 1, 1], [0, 0, 0]], bool)
    valid = {"MA": base & ~ignore_ma, "HE": base.copy(), "_base": base}
    probs = {"MA": np.full((2, 3), 3), "HE": np.full((2, 3), 7)}
    truth, region, score = any_lesion(valid, masks, probs, ["MA", "HE"])
    assert truth.tolist() == [[True, True, False], [False, False, False]]
    # (0, 1) is ignored for MA but positive for HE, so it stays; (0, 2) is only ignored.
    assert region.tolist() == [[True, True, False], [True, True, True]]
    assert (score == 7).all()


def test_thresholds_from_another_dataset_and_excluded_images(toy_dataset, tmp_path):
    root, truth = toy_dataset
    _write_preds(tmp_path / "pred", truth, noise=0.2)
    other = tmp_path / "OTHER"
    for name in ("meta.json",):
        (other / name).parent.mkdir(parents=True, exist_ok=True)
        (other / name).write_text((root / name).read_text())
    (other / "splits.json").write_text(json.dumps({"train": [], "val": ["v0"], "test": []}))
    for sub in ("fov", "masks/MA", "masks/HE", "masks/EX"):
        (other / sub).mkdir(parents=True, exist_ok=True)
        src = root / sub / "v0.png"
        (other / sub / "v0.png").write_bytes(src.read_bytes())
    proto = load_protocol(lesion_metrics=False)
    proto["bootstrap"]["n"] = 50
    own = evaluate(root, "test", tmp_path / "pred", proto, val_pred_dir=tmp_path / "pred", val_split="val")
    moved = evaluate(
        root, "test", tmp_path / "pred", proto, val_pred_dir=tmp_path / "pred", val_split="val",
        val_dataset_dir=other, exclude=["t1"],
    )
    assert moved["n_images"] == 2 and moved["excluded_ids"] == ["t1"]
    assert moved["val_dataset_dir"] == str(other)
    # One validation image instead of two changes the chosen thresholds.
    th_own = [own["per_class"][c]["thresholds"]["val_opt"]["threshold"] for c in own["classes"]]
    th_moved = [moved["per_class"][c]["thresholds"]["val_opt"]["threshold"] for c in moved["classes"]]
    assert th_own != th_moved
    with pytest.raises(ValueError, match="not in test"):
        evaluate(root, "test", tmp_path / "pred", proto, exclude=["nope"])
