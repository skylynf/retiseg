"""bench/e5: the inference tiers' torch-free parts (windows, mask store, scoring rules, prompts, prompted decoding, control)."""
import json
from pathlib import Path

import numpy as np
import pytest

from bench.common.io import LESION_CLASSES, read_prob_quantized
from bench.e5 import common, masks, prompts, rules

REPO = Path(__file__).resolve().parents[2]


def _toy_canvas(h=40, w=48, original=(80, 96), crop=(0, 80, 0, 96)):
    rng = np.random.default_rng(0)
    gt = []
    for i in range(4):
        m = np.zeros((h, w), dtype=bool)
        m[5 + 8 * i : 9 + 8 * i, 6 : 10] = True          # one component per class
        if i == 2:
            m[20:23, 30:34] = True                         # EX has a second one
        gt.append(m)
    fov = np.ones((h, w), dtype=bool)
    return {"image": rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "masks": gt, "crop": crop, "canvas_hw": (h, w),
            "original_hw": original, "canvas_fov": fov, "image_id": "t0"}


def test_windows_cover_the_canvas_with_the_frozen_overlap():
    wins = common.windows(1440, 1440, 1024)
    assert wins[0] == (0, 0, 1024, 1024) and (416, 416, 1024, 1024) in wins and len(wins) == 4
    assert common.windows(900, 700, 1024) == [(0, 0, 1024, 1024)]
    big = common.windows(4256, 4256, 1024, 0.5)
    tops = sorted({w[0] for w in big})
    assert tops[0] == 0 and tops[-1] == 4256 - 1024 and all(b - a <= 512 for a, b in zip(tops, tops[1:]))


def test_window_of_prefers_the_window_with_the_largest_margin_and_crops_pad():
    wins = common.windows(1440, 1440, 1024)
    assert common.window_of([900, 900, 940, 940], wins) == 3     # centre 920: margin 504 in (416, 416), 104 in (0, 0)
    assert common.window_of([700, 700, 740, 740], wins) == 0     # a tie (304 either way) keeps the first window
    assert common.window_of([10, 10, 30, 30], wins) == 0
    image = np.arange(30 * 20 * 3, dtype=np.uint8).reshape(30, 20, 3)
    crop = common.window_crop(image, 20, 10, 16, 16)
    assert crop.shape == (16, 16, 3) and np.array_equal(crop[:10, :10], image[20:30, 10:20]) and not crop[10:, :].any()
    canvas = np.zeros((30, 20), np.float32)
    common.paste_max(canvas, np.full((16, 16), 0.5, np.float32), 20, 10)
    common.paste_max(canvas, np.full((16, 16), 0.2, np.float32), 20, 10)
    assert canvas[20:, 10:].min() == 0.5 and canvas[:20].max() == 0


def test_place_on_original_scales_into_the_crop():
    canvas = _toy_canvas(h=40, w=48, original=(100, 120), crop=(10, 90, 12, 108))
    prob = np.zeros((40, 48), np.float32)
    prob[10:20, 10:20] = 1.0
    placed = common.place_on_original(prob, canvas)
    assert placed.shape == (100, 120) and placed[:10].max() == 0 and placed[90:].max() == 0
    assert placed[10 + 25, 12 + 30] > 0.9 and placed[10 + 5, 12 + 5] == 0


def test_mask_store_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(5):
        crop = rng.random((7, 9)) > 0.5
        rows.append((crop, 3 * i, 2 * i, 0.9 - 0.1 * i, 0.8, i % 2))
    rows.append((np.zeros((3, 3), bool), 0, 0, 1.0, 1.0, 0))   # empty masks are dropped
    store = masks.pack(rows, (40, 48))
    masks.save(tmp_path / "m.npz", store)
    back = masks.load(tmp_path / "m.npz")
    assert masks.count(back) == 5
    for index, (crop, y0, x0, iou, _stab, win) in enumerate(rows[:5]):
        full = masks.full_mask(back, index)
        ref = np.zeros((40, 48), bool)
        ref[y0 : y0 + 7, x0 : x0 + 9] = crop
        assert np.array_equal(full, ref) and back["predicted_iou"][index] == pytest.approx(iou, abs=1e-6)
        assert back["area"][index] == crop.sum() and back["window"][index] == win


def test_rules_any_oracle_and_cap():
    canvas = _toy_canvas()
    gt = canvas["masks"]
    a = np.zeros((40, 48), bool); a[5:9, 6:10] = True            # = MA component
    b = np.zeros((40, 48), bool); b[20:24, 30:35] = True          # overlaps the second EX component
    c = np.zeros((40, 48), bool); c[:, :] = True                  # whole canvas, overlaps everything, mostly background
    d = np.zeros((40, 48), bool); d[35:38, 40:44] = True          # overlaps nothing
    store = masks.pack([(a, 0, 0, 0.9, 0.5, 0), (b, 0, 0, 0.7, 0.6, 0), (c, 0, 0, 0.3, 0.9, 0), (d, 0, 0, 0.8, 0.95, 0)], (40, 48))
    any_map = rules.any_map(store)
    assert any_map[6, 7] == pytest.approx(0.9) and any_map[21, 31] == pytest.approx(0.7) and any_map[0, 0] == pytest.approx(0.3)
    assert any_map[36, 41] == pytest.approx(0.8)
    stab = rules.any_map(store, "stability_score")
    assert stab[6, 7] == pytest.approx(0.9) and stab[36, 41] == pytest.approx(0.95)
    capped = rules.any_map(store, cap_px=100)
    assert capped[0, 0] == 0 and capped[6, 7] == pytest.approx(0.9)
    oracle = rules.oracle_maps(store, gt)
    assert oracle["MA"][6, 7] == pytest.approx(0.9) and oracle["EX"][6, 7] == pytest.approx(0.3)   # a -> MA; c -> EX (most overlap)
    assert oracle["HE"][6, 7] == 0 and oracle["EX"][21, 31] == pytest.approx(0.7) and oracle["MA"][21, 31] == 0
    assert all(oracle[cls][36, 41] == pytest.approx(0.8) for cls in LESION_CLASSES)                 # d -> all four
    assigned = rules.assign_classes(store, gt, np.arange(4))
    assert list(assigned) == [0, 2, 2, -1]   # c overlaps EX most (two components)
    assert rules.rule_name() == "any" and rules.rule_name("oracle_class", "stability_score", 0.005) == "oracle_class+stability+cap_0.5pct"
    assert rules.variant_name("any", "more_masks", "G2", "predicted_iou", "train") == "zero_shot:auto_any_lesion:more_masks:G2:cap_train"


def test_score_rule_writes_maps_and_paper_key(tmp_path, monkeypatch):
    from bench.e5 import amg

    canvas = _toy_canvas()
    monkeypatch.setattr(common, "OUT_ROOT", tmp_path / "E5")
    monkeypatch.setattr(common, "split_ids", lambda test_set, split: ["t0"])
    monkeypatch.setattr(common, "load_canvas", lambda test_set, image_id, masks=False: canvas)
    source = amg.out_dir("sam", "default", "IDRiD", "test")
    a = np.zeros((40, 48), bool); a[5:9, 6:10] = True
    masks.save(source / "masks" / "t0.npz", masks.pack([(a, 0, 0, 0.9, 0.5, 0)], (40, 48)))
    common.write_json(common.done_marker(source), {"ok": True})
    out = rules.run("sam", "default", "IDRiD", "test")
    assert out == source / "any"
    paper = json.loads((out / "paper.json").read_text())
    assert paper == {"experiment": "E5", "model": "SAM", "variant": "zero_shot:auto_any_lesion", "train_set": "none", "seed": -1, "test_set": "IDRiD"}
    for cls in LESION_CLASSES:
        q = read_prob_quantized(out, cls, "t0")
        assert q.shape == (80, 96) and abs(int(q.max()) - round(0.9 * 65535)) <= 1 and q[0, 0] == 0
    oracle = rules.run("sam", "default", "IDRiD", "test", output="oracle_class", cap=0.02)
    assert oracle.name == "oracle_class+cap_2pct"
    assert read_prob_quantized(oracle, "MA", "t0").max() > 0 and read_prob_quantized(oracle, "HE", "t0").max() == 0
    assert json.loads((oracle / "paper.json").read_text())["variant"] == "zero_shot:auto_oracle_class:cap_2pct"


def test_component_prompts_box_point_and_frozen_order(tmp_path, monkeypatch):
    canvas = _toy_canvas()
    items = prompts.component_prompts(canvas["masks"][2])
    assert len(items) == 2 and items[0]["box"] == [30, 20, 34, 23] and items[1]["box"] == [6, 21, 10, 25]   # (row, col) order
    for item, mask in zip(items, (canvas["masks"][2],) * 2):
        x, y = item["point"]
        assert mask[y, x]
    assert [item["area"] for item in items] == [12, 16]
    monkeypatch.setattr(common, "OUT_ROOT", tmp_path / "E5")
    monkeypatch.setattr(common, "split_ids", lambda test_set, split: ["t0"])
    monkeypatch.setattr(common, "load_canvas", lambda test_set, image_id, masks=False: canvas)
    path = prompts.build("DDR", "test")
    table = json.loads(path.read_text())
    assert table["sha256"] and table["seed"] == 20261003 and set(table["images"]["t0"]) == set(LESION_CLASSES)
    ex = table["images"]["t0"]["EX"]
    assert [item["order"] for item in ex] == [0, 1] and {tuple(item["box"]) for item in ex} == {(6, 21, 10, 25), (30, 20, 34, 23)}
    again = json.loads(prompts.build("DDR", "test").read_text())
    assert again["images"] == table["images"]     # the permutation is a function of (seed, set, image, class)
    assert prompts.select(ex, 1) == ex[:1] and prompts.select(ex, "all") == ex and prompts.select(ex, 20) == ex


class _BoxAdapter:
    """Fills the prompted box (or a 3x3 patch around the point) with probability 1 inside the window."""

    window = 32

    def __init__(self):
        self.encoded = 0

    def set_image(self, rgb):
        self.encoded += 1
        self.shape = rgb.shape[:2]

    def predict_box(self, box):
        prob = np.zeros(self.shape, np.float32)
        x0, y0, x1, y1 = (int(v) for v in box)
        prob[max(y0, 0) : y1, max(x0, 0) : x1] = 1.0
        return prob, 0.9

    def predict_point(self, xy):
        prob = np.zeros(self.shape, np.float32)
        x, y = (int(v) for v in xy)
        prob[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = 1.0
        return prob, 0.8


def test_prompted_decoding_snapshots_every_k_and_scores_in_box():
    from bench.e5 import prompted

    canvas = _toy_canvas()
    image = canvas["image"]
    wins = common.windows(40, 48, 32)
    assert len(wins) == 4
    items = prompts.component_prompts(canvas["masks"][2])
    adapter = _BoxAdapter()
    per_k, scores = prompted.decode_class(adapter, image, wins, items, "box", "G1", (1, 5, "all"), gt=canvas["masks"][2])
    assert per_k[1].sum() == 12 and per_k[5].sum() == 16 + 12 and np.array_equal(per_k[5], per_k["all"])
    assert [s["dice"] for s in scores] == [1.0, 1.0] and all(s["iou"] == 1.0 for s in scores)
    assert adapter.encoded <= 2
    per_k, _ = prompted.decode_class(adapter, image, wins, items, "point", "G1", (1,), gt=None)
    assert 0 < per_k[1].sum() <= 9
    empty, scores = prompted.decode_class(adapter, image, wins, [], "box", "G1", (1, "all"))
    assert empty[1].sum() == 0 and scores == []
    assert prompted.parse_k("1,5,all") == (1, 5, "all")


def test_control_keeps_the_map_inside_the_oracle_boxes(tmp_path, monkeypatch):
    from bench.common.io import write_prob
    from bench.e5 import control

    canvas = _toy_canvas()
    monkeypatch.setattr(common, "OUT_ROOT", tmp_path / "E5")
    monkeypatch.setattr(common, "split_ids", lambda test_set, split: ["t0"])
    monkeypatch.setattr(common, "load_canvas", lambda test_set, image_id, masks=False: canvas)
    prompts.build("IDRiD", "test")
    pred = tmp_path / "prob_test"
    for cls in LESION_CLASSES:
        write_prob(pred, cls, "t0", np.full((80, 96), 0.6))
    dirs = control.run(pred, "HRNet-W48", "DDR", 0, "IDRiD", "test", ks=(1, "all"), name="hrnet")
    q = read_prob_quantized(dirs["all"], "EX", "t0")
    assert abs(int(q.max()) - round(0.6 * 65535)) <= 1 and (16 + 12) * 4 <= (q > 0).sum() <= 2 * (16 + 12) * 4   # boxes x4, bilinear rim
    q1 = read_prob_quantized(dirs[1], "EX", "t0")
    assert 0 < (q1 > 0).sum() < (q > 0).sum()
    paper = json.loads((dirs[1] / "paper.json").read_text())
    assert paper["model"] == "HRNet-W48" and paper["train_set"] == "DDR" and paper["seed"] == 0 and paper["variant"] == "oracle_control:box:1"


def test_val_dir_twin_and_set_parsing():
    from bench.e5 import score

    d = Path("runs/E5/a/sam/default/TJDR_uwf_test/any")
    assert score.val_dir_of(d) == Path("runs/E5/a/sam/default/DDR_val/any") and score.set_of(d) == ("TJDR_uwf", "test")
    assert common.parse_set_dirname("Retinal-Lesions_test") == ("Retinal-Lesions", "test")
    with pytest.raises(ValueError):
        common.parse_set_dirname("masks")


def test_amg_rows_clip_at_the_canvas_edge():
    from bench.e5 import amg

    seg = np.ones((16, 16), bool)
    rows = amg.masks_from_generator([{"segmentation": seg, "predicted_iou": 0.5, "stability_score": 0.4}], 30, 40, (40, 48), 3)
    crop, top, left, iou, stab, win = rows[0]
    assert crop.shape == (10, 8) and (top, left, win) == (30, 40, 3) and iou == 0.5
    frozen = common.load_frozen()
    params, build = amg.generator_params(frozen, "sam2", "more_masks")
    assert build == {"apply_postprocessing": False} and "apply_postprocessing" not in params and params["use_m2m"] is True
    params, build = amg.generator_params(frozen, "sam", "default")
    assert build == {} and params["points_per_side"] == 32
