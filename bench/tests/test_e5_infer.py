"""E5 inference tiers (design/15 sections 4-6 under the docs/paper/16 scope): the registered protocol."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "bench/configs/e5_frozen.yaml"


def test_frozen_scope_matches_16():
    """docs/paper/16 section 3, rows E5a and E5b, as fixed on 2026-10-12 (step 0, decisions D1-D9)."""
    frozen = yaml.safe_load(FROZEN.read_text())
    # D3-D6: SAM ViT-B boxes only, SAM 2.1 points and boxes, MedSAM boxes; k on four levels
    oracle = frozen["oracle"]
    assert oracle["models"] == {"SAM": ["boxes"], "SAM 2": ["points", "boxes"], "MedSAM": ["boxes"]}
    assert oracle["checkpoints"]["SAM"] == "sam_vit_b_01ec64.pth"
    assert oracle["k"] == [1, 5, 20, "all"]
    assert oracle["curve_sets"] == ["IDRiD", "DDR"]
    assert oracle["same_information_reference"] == ["M2MRF", "nnU-Net"]
    weights = frozen["weights"]
    assert "SAM b" in weights["sam_vit_b_01ec64.pth"]["used_by"]
    assert "SAM b" not in weights["sam_vit_h_4b8939.pth"]["used_by"]
    assert "SAM a" in weights["sam_vit_h_4b8939.pth"]["used_by"]
    # D1: a registered grid chosen on the pool, not the authors' two parameter sets
    grid = frozen["zero_shot"]["grid"]
    assert grid["points_per_side"] == [64, 128]
    assert grid["area_cap_fraction"] == [0.005, 0.02, 1.0]
    assert grid["select_on"] == "validation_pool" and grid["select_by"] == "AUPR_any_lesion"
    assert grid["selected"] == {}  # filled from the pool before any test run
    for model in ("sam", "sam2"):
        assert "generator" not in frozen["zero_shot"][model]
        assert "points_per_side" not in frozen["zero_shot"][model]["generator_defaults"]
    assert frozen["zero_shot"]["sam"]["checkpoint"] == "sam_vit_h_4b8939.pth"
    # D2: the validation pool
    pool = frozen["validation_pool"]
    assert pool["IDRiD"]["split"] == pool["DDR"]["split"] == "val"
    assert frozen["zero_shot"]["sam3"]["select_on"].startswith("validation_pool")
    assert frozen["zero_shot"]["sam3"]["selected"] == {}
    # D7: the TJDR_uwf canvas of the inference tiers
    uwf = frozen["geometry"]["G1"]["TJDR_uwf"]
    assert uwf["long_side"] == round(1440 * 133 / 45) == 4256
    assert uwf["comparison_variant"] == {"fov": 1440}
    # D8: outputs
    assert frozen["zero_shot"]["outputs"] == {"main": "any_lesion", "upper_bound": "oracle_class", "sensitivity": "area_cap"}
    # SAM 3 phrases stay registered, three per class
    phrases = frozen["zero_shot"]["sam3"]["phrases"]
    assert set(phrases) == {"MA", "HE", "EX", "SE"} and all(len(v) == 3 for v in phrases.values())


def test_placeholder_keys_follow_the_16_scope():
    """The paper's placeholder generator writes only keys the measured runs will replace."""
    text = (REPO / "docs/paper/retiseg_paper/paper/scripts/build_placeholder.py").read_text()
    assert 'for k in (1, 5, 20, "all")' in text
    assert '("SAM", "point")' not in text


# ---- step 1: bench/e5/frozen.py ---------------------------------------------------------------


def _toy_frozen(tmp_path, **over):
    """A copy of the real protocol with top-level overrides applied by a function."""
    doc = yaml.safe_load(FROZEN.read_text())
    for key, change in over.items():
        change(doc) if callable(change) else doc.__setitem__(key, change)
    path = tmp_path / "e5.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_frozen_loads_the_registered_grid_pool_phrases_and_k():
    from bench.e5 import frozen as fz

    frozen = fz.load_frozen()
    assert fz.grid_pairs(frozen) == [(64, 0.005), (64, 0.02), (64, 1.0), (128, 0.005), (128, 0.02), (128, 1.0)]
    assert [name for name, _dir, _split in fz.pool_sets(frozen)] == ["IDRiD", "DDR"]
    assert all(split == "val" and directory.name == name for name, directory, split in fz.pool_sets(frozen))
    assert len(fz.phrase_ids(frozen)) == 12 and fz.phrase_ids(frozen)[("MA", 0)] == "microaneurysm"
    assert fz.k_levels(frozen) == [1, 5, 20, "all"]
    assert fz.oracle_models(frozen) == {"SAM": ["boxes"], "SAM 2": ["points", "boxes"], "MedSAM": ["boxes"]}
    assert fz.prompt_file("IDRiD", "test").name == "IDRiD_test.json"


def test_validate_rejects_a_protocol_with_a_missing_key(tmp_path):
    import pytest

    from bench.e5 import frozen as fz

    def drop_k(doc):
        del doc["oracle"]["k"]

    with pytest.raises(ValueError, match="oracle lacks"):
        fz.load_frozen(_toy_frozen(tmp_path, oracle=drop_k))

    def two_phrases(doc):
        doc["zero_shot"]["sam3"]["phrases"]["SE"] = doc["zero_shot"]["sam3"]["phrases"]["SE"][:2]

    with pytest.raises(ValueError, match="three phrases"):
        fz.load_frozen(_toy_frozen(tmp_path, zero_shot=two_phrases))


def test_assert_ready_gates_test_splits_on_the_pool_selections(tmp_path):
    import pytest

    from bench.e5 import frozen as fz

    frozen = fz.load_frozen()
    # the pool needs nothing filled; a test split needs both selections
    fz.assert_ready(frozen, "a", "val")
    with pytest.raises(ValueError, match="grid.selected is empty"):
        fz.assert_ready(frozen, "a", "test")
    with pytest.raises(ValueError, match="grid.selected is empty"):
        fz.assert_ready(frozen, "a", "test", model="SAM 2")
    with pytest.raises(ValueError, match="sam3.selected lacks"):
        fz.assert_ready(frozen, "a", "test", model="SAM 3")
    with pytest.raises(ValueError, match="zero-shot models"):
        fz.assert_ready(frozen, "a", "test", model="MedSAM")

    def fill(density, cap, phrases):
        def apply(doc):
            doc["zero_shot"]["grid"]["selected"] = {"points_per_side": density, "area_cap_fraction": cap}
            doc["zero_shot"]["sam3"]["selected"] = phrases
        return apply

    good = {"MA": "red dot", "HE": "retinal hemorrhage", "EX": "hard exudate", "SE": "soft exudate"}
    filled = fz.load_frozen(_toy_frozen(tmp_path, zero_shot=fill(128, 0.02, good)))
    fz.assert_ready(filled, "a", "test")
    assert fz.selected_pair(filled) == (128, 0.02) and fz.selected_phrases(filled) == good
    off_grid = fz.load_frozen(_toy_frozen(tmp_path, zero_shot=fill(96, 0.02, good)))
    with pytest.raises(ValueError, match="not in the registered grid"):
        fz.assert_ready(off_grid, "a", "test", model="SAM")
    unregistered = fz.load_frozen(_toy_frozen(tmp_path, zero_shot=fill(64, 1.0, dict(good, MA="tiny red thing"))))
    with pytest.raises(ValueError, match="not a registered MA phrase"):
        fz.assert_ready(unregistered, "a", "test", model="SAM 3")
    # tier b: the prompt file of the set and split, pool included
    with pytest.raises(ValueError, match="needs set_name"):
        fz.assert_ready(frozen, "b", "test")
    with pytest.raises(FileNotFoundError, match="bench.e5.prompts --set DDR --split val"):
        fz.assert_ready(frozen, "b", "val", set_name="DDR", prompts_dir=tmp_path)
    fz.prompt_file("DDR", "val", tmp_path).write_text("{}")
    fz.assert_ready(frozen, "b", "val", set_name="DDR", prompts_dir=tmp_path)
    with pytest.raises(ValueError, match="tier must be"):
        fz.assert_ready(frozen, "c", "test")


# ---- step 2: bench/e5/geometry.py -------------------------------------------------------------


def _toy_set(root, image_id="img_01", hw=(120, 90), radius=40):
    """One prepared image: a disc field of view, two lesion masks, PNG files, splits.json."""
    import json

    import numpy as np
    from PIL import Image

    from bench.common.io import LESION_CLASSES

    rng = np.random.default_rng(3)
    height, width = hw
    yy, xx = np.mgrid[:height, :width]
    fov = (yy - height // 2) ** 2 + (xx - width // 2) ** 2 <= radius**2
    for sub in ("images", "fov", *[f"masks/{cls}" for cls in LESION_CLASSES]):
        (root / sub).mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(root / "images" / f"{image_id}.png")
    Image.fromarray(fov.astype(np.uint8) * 255).save(root / "fov" / f"{image_id}.png")
    for cls in ("MA", "EX"):
        mask = np.zeros(hw, bool)
        mask[height // 2 - 3 : height // 2 + 3, width // 2 - 3 : width // 2 + 3] = True
        Image.fromarray(mask.astype(np.uint8) * 255).save(root / "masks" / cls / f"{image_id}.png")
    (root / "splits.json").write_text(json.dumps({"train": [], "val": [], "test": [image_id]}))
    return fov


def test_windows_follow_bench_infer_and_pad_small_canvases():
    import numpy as np

    from bench.e5 import geometry as geo
    from bench.infer import window_starts

    expected = [(t, l) for t in window_starts(100, 64, 32) for l in window_starts(80, 64, 32)]
    assert geo.windows((100, 80), 64, 0.5) == expected and len(expected) == 6
    assert geo.windows((50, 40), 64, 0.5) == [(0, 0)]
    image = np.arange(100 * 80 * 3, dtype=np.uint8).reshape(100, 80, 3)
    piece, valid = geo.crop_window(image, 36, 16, 64)
    assert piece.shape == (64, 64, 3) and valid == (64, 64) and np.array_equal(piece, image[36:100, 16:80])
    small = np.ones((50, 40, 3), np.uint8)
    piece, valid = geo.crop_window(small, 0, 0, 64)
    assert piece.shape == (64, 64, 3) and valid == (50, 40)
    assert piece[:50, :40].all() and not piece[50:, :].any() and not piece[:, 40:].any()
    import pytest

    with pytest.raises(ValueError, match="outside"):
        geo.crop_window(small, 50, 0, 64)
    with pytest.raises(ValueError, match="overlap"):
        geo.windows((100, 80), 64, 1.0)


def test_long_side_and_windows_follow_e3a_and_the_d7_entry():
    import pytest

    from bench.e5 import frozen as fz
    from bench.e5 import geometry as geo

    frozen = fz.load_frozen()
    assert geo.test_sets()[:2] == ["IDRiD", "DDR"] and len(geo.test_sets()) == 7
    assert geo.long_side_for("IDRiD", "main", frozen) == 1440
    assert geo.long_side_for("TJDR_uwf", "main", frozen) == 4256
    assert geo.long_side_for("TJDR_uwf", "fov", frozen) == 1440
    assert geo.variants_for("TJDR_uwf", frozen) == ["main", "fov"] and geo.variants_for("FGADR", frozen) == ["main"]
    with pytest.raises(ValueError, match="no variant"):
        geo.long_side_for("IDRiD", "fov", frozen)
    with pytest.raises(ValueError, match="unknown test set"):
        geo.long_side_for("Kaggle", "main", frozen)
    assert geo.window_size(frozen, "SAM") == 1024 and geo.window_size(frozen, "SAM 3") == 1008
    with pytest.raises(ValueError, match="no G1 window"):
        geo.window_size(frozen, "SAMed")
    assert geo.overlap(frozen) == 0.5
    assert geo.plan(frozen, "G2", "SAM", (300, 200)) == (None, [(0, 0)])
    window, boxes = geo.plan(frozen, "G1", "SAM 2", (4256, 4256))
    # window_starts(4256, 1024, 512): 0, 512, ..., 3072 then 3232 flush with the edge -> 8 x 8 windows
    assert window == 1024 and len(boxes) == 64 and boxes[-1] == (3232, 3232)
    assert len(geo.plan(frozen, "G1", "SAM", (1440, 956))[1]) == 2
    with pytest.raises(ValueError, match="geometry"):
        geo.plan(frozen, "G3", "SAM", (10, 10))


def test_canvas_and_paste_round_trip(tmp_path):
    import numpy as np
    import pytest

    from bench.data.fov import crop_box
    from bench.e5 import geometry as geo

    fov = _toy_set(tmp_path)
    y0, y1, x0, x1 = crop_box(fov)
    cv = geo.canvas(tmp_path, "img_01", 60, masks=True)
    assert cv.image.dtype == np.uint8 and max(cv.hw) == 60 and cv.fov.shape == cv.hw
    assert cv.crop == (y0, y1, x0, x1) and cv.original_hw == (120, 90) and cv.long_side == 60
    assert len(cv.masks) == 4 and cv.masks[0].any() and cv.masks[2].any() and not cv.masks[1].any()
    assert cv.masks[0].shape == cv.hw and cv.masks[0].dtype == bool
    native = geo.canvas(tmp_path, "img_01", geo.NATIVE)
    assert native.hw == (y1 - y0, x1 - x0) and native.masks is None
    # a constant map comes back as 1 inside the field of view and 0 outside, at the original size
    ones = geo.paste(np.ones(cv.hw, np.float32), cv)
    assert ones.shape == (120, 90) and ones.dtype == np.float32
    assert ones[fov].min() == pytest.approx(1.0) and not ones[~fov].any()
    # a hot pixel lands where the canvas pixel came from
    hot = np.zeros(cv.hw, np.float32)
    cy, cx = cv.hw[0] // 3, cv.hw[1] // 2
    hot[cy, cx] = 1.0
    placed = geo.paste(hot, cv)
    py, px = np.unravel_index(np.argmax(placed), placed.shape)
    scale_y, scale_x = (y1 - y0) / cv.hw[0], (x1 - x0) / cv.hw[1]
    assert abs(py - (y0 + cy * scale_y)) <= 2 and abs(px - (x0 + cx * scale_x)) <= 2
    with pytest.raises(ValueError, match="canvas size"):
        geo.paste(np.zeros((5, 5), np.float32), cv)


# ---- step 3: bench/e5/prompts.py --------------------------------------------------------------


def test_components_are_8_connected_sorted_with_tight_boxes_and_inner_points():
    import numpy as np

    from bench.e5 import prompts

    mask = np.zeros((20, 30), bool)
    mask[2:5, 3:8] = True  # a 3 x 5 block
    mask[10, 10] = True
    mask[11, 11] = True  # a diagonal pair: one component under 8-connectivity
    mask[1, 20] = True  # a single pixel, first in row-major order
    comps = prompts.components(mask)
    assert [c["first"] for c in comps] == [[1, 20], [2, 3], [10, 10]]
    assert comps[0] == {"box": [20, 1, 21, 2], "point": [20, 1], "area": 1, "first": [1, 20]}
    assert comps[1]["box"] == [3, 2, 8, 5] and comps[1]["area"] == 15
    assert comps[1]["point"] == [4, 3]  # the middle row; three pixels tie at distance 2, the first wins
    assert comps[2]["box"] == [10, 10, 12, 12] and comps[2]["area"] == 2
    for c in comps:
        x, y = c["point"]
        assert mask[y, x] and c["box"][0] <= x < c["box"][2] and c["box"][1] <= y < c["box"][3]
    assert prompts.components(np.zeros((5, 5), bool)) == []


def test_prompt_order_is_a_seeded_permutation_and_k_is_a_prefix():
    from bench.e5 import prompts

    comps = [{"box": [0, i, 1, i + 1], "point": [0, i], "area": 1, "first": [i, 0]} for i in range(7)]
    once = prompts.ordered(comps, "img_01", "MA")
    again = prompts.ordered(comps, "img_01", "MA")
    assert once == again
    assert sorted(c["first"][0] for c in once) == list(range(7)) and [c["rank"] for c in once] == list(range(7))
    assert prompts.take(once, 1) == once[:1] and prompts.take(once, 5) == once[:5]
    assert prompts.take(once, 20) == once and prompts.take(once, "all") == once
    other_class = prompts.ordered(comps, "img_01", "HE")
    other_image = prompts.ordered(comps, "img_02", "MA")
    assert other_class != once or other_image != once  # separate streams per (image, class)
    assert prompts.ordered([], "img_01", "MA") == []


def test_build_writes_a_deterministic_prompt_file_with_hashes_and_counts(tmp_path):
    import numpy as np
    import pytest
    from PIL import Image

    from bench.e5 import frozen as fz
    from bench.e5 import geometry as geo
    from bench.e5 import prompts

    root = tmp_path / "prepared" / "IDRiD"
    _toy_set(root)
    ma_path = root / "masks" / "MA" / "img_01.png"
    ma = np.asarray(Image.open(ma_path)) > 0
    ma[30:35, 30:35] = True  # a second MA component inside the field of view
    Image.fromarray(ma.astype(np.uint8) * 255).save(ma_path)
    frozen = fz.load_frozen()
    out = tmp_path / "prompts"
    first = prompts.build("IDRiD", "test", frozen=frozen, out_dir=out, data_dir=root, long_side=60)
    text = first.read_text()
    assert prompts.build("IDRiD", "test", frozen=frozen, out_dir=out, data_dir=root, long_side=60).read_text() == text
    doc = prompts.load(first)
    assert first.name == "IDRiD_test.json" and doc["long_side"] == 60 and doc["k"] == [1, 5, 20, "all"]
    assert doc["seed"] == 20261003 and doc["connectivity"] == 8 and doc["rule"]["box"].startswith("tight")
    image = doc["images"]["img_01"]
    cv = geo.canvas(root, "img_01", 60, masks=True)
    assert image["canvas_hw"] == list(cv.hw)
    assert image["classes"]["MA"]["n"] == 2 and image["classes"]["EX"]["n"] == 1 and image["classes"]["HE"]["n"] == 0
    assert image["mask_sha256"]["HE"] is None and len(image["mask_sha256"]["MA"]) == 64
    assert doc["components_per_image"] == {"MA": 2.0, "HE": 0.0, "EX": 1.0, "SE": 0.0}
    for prompt in image["classes"]["MA"]["prompts"]:
        x0, y0, x1, y1 = prompt["box"]
        assert 0 <= x0 < x1 <= cv.hw[1] and 0 <= y0 < y1 <= cv.hw[0]
        px, py = prompt["point"]
        assert x0 <= px < x1 and y0 <= py < y1 and cv.masks[0][py, px]
    assert [p["rank"] for p in image["classes"]["MA"]["prompts"]] == [0, 1]
    # a changed label changes the hash and the count, so a stale file is visible (a block, so the
    # nearest-neighbour resize to the canvas keeps it)
    ma[78:82, 58:62] = True
    Image.fromarray(ma.astype(np.uint8) * 255).save(ma_path)
    doc2 = prompts.load(prompts.build("IDRiD", "test", frozen=frozen, out_dir=out, data_dir=root, long_side=60))
    assert doc2["images"]["img_01"]["mask_sha256"]["MA"] != image["mask_sha256"]["MA"]
    assert doc2["images"]["img_01"]["classes"]["MA"]["n"] == 3
    # variants: the file name carries the comparison variant and assert_ready looks for it
    assert fz.prompt_file("TJDR_uwf", "test", out, "fov").name == "TJDR_uwf_test__fov.json"
    fz.assert_ready(frozen, "b", "test", set_name="IDRiD", prompts_dir=out)
    with pytest.raises(FileNotFoundError, match="--variant fov"):
        fz.assert_ready(frozen, "b", "test", set_name="TJDR_uwf", prompts_dir=out, variant="fov")
    with pytest.raises(ValueError, match="no variant"):
        prompts.build("IDRiD", "test", variant="fov", frozen=frozen, out_dir=out, data_dir=root, long_side=60)


# ---- step 4: bench/e5/score_rules.py ----------------------------------------------------------


def _mask(rows, cols, top=0, left=0, iou=0.9, stability=0.8, shape=(8, 8)):
    import numpy as np

    from bench.e5.score_rules import Mask

    seg = np.zeros(shape, bool)
    seg[rows, cols] = True
    return Mask(segmentation=seg, top=top, left=left, predicted_iou=iou, stability_score=stability)


def test_pixel_score_is_the_max_over_covering_masks_across_windows():
    import numpy as np
    import pytest

    from bench.e5 import score_rules as sr

    a = _mask(slice(0, 4), slice(0, 4), iou=0.6, stability=0.95)
    b = _mask(slice(2, 6), slice(2, 6), iou=0.9, stability=0.5)
    score = sr.masks_to_score([a, b], (10, 10))
    assert score.dtype == np.float32 and score.shape == (10, 10)
    assert score[0, 0] == pytest.approx(0.6) and score[3, 3] == pytest.approx(0.9) and score[5, 5] == pytest.approx(0.9)
    assert score[9, 9] == 0.0 and score[0, 5] == 0.0
    by_stability = sr.masks_to_score([a, b], (10, 10), key="stability_score")
    assert by_stability[3, 3] == pytest.approx(0.95) and by_stability[5, 5] == pytest.approx(0.5)
    # a mask from the second window lands at its offset and is clipped at the canvas edge
    c = _mask(slice(0, 8), slice(0, 8), top=6, left=6, iou=1.2)
    score = sr.masks_to_score([c], (10, 10))
    assert score[6, 6] == pytest.approx(1.0) and score[9, 9] == pytest.approx(1.0) and score[5, 6] == 0.0
    assert c.canvas_area((10, 10)) == 16 and c.canvas_area((20, 20)) == 64
    with pytest.raises(ValueError, match="score key"):
        sr.masks_to_score([a], (10, 10), key="area")
    with pytest.raises(ValueError, match="negative"):
        sr.combine_max((10, 10), [(-1, 0, np.ones((2, 2)))])
    four = sr.as_four(score)
    assert list(four) == ["MA", "HE", "EX", "SE"] and all(four[cls] is score for cls in four)
    record = {"segmentation": np.ones((3, 3), bool), "predicted_iou": 0.7, "stability_score": 0.6, "area": 9, "bbox": [0, 0, 3, 3]}
    mask = sr.Mask.from_record(record, top=2, left=3)
    assert mask.top == 2 and mask.left == 3 and mask.score() == 0.7 and mask.canvas_area((10, 10)) == 9


def test_oracle_class_credits_the_most_overlapped_class_and_spreads_false_positives():
    import numpy as np
    import pytest

    from bench.e5 import score_rules as sr

    canvas_hw = (10, 10)
    gt = [np.zeros(canvas_hw, bool) for _ in range(4)]
    gt[0][0:3, 0:3] = True  # MA
    gt[1][0:3, 2:6] = True  # HE, overlapping MA at column 2
    mostly_he = _mask(slice(0, 3), slice(1, 6), iou=0.8)  # 3 MA pixels (cols 1-2 ... ) vs HE 12
    assert sr.assign_classes(mostly_he, gt, canvas_hw) == [1]
    tie = _mask(slice(0, 3), slice(2, 3), iou=0.5)  # column 2 is both MA and HE: 3 pixels each -> MA first
    assert sr.assign_classes(tie, gt, canvas_hw) == [0]
    nowhere = _mask(slice(7, 9), slice(7, 9), iou=0.4, shape=(10, 10))  # the default 8 x 8 window would cut it
    assert sr.assign_classes(nowhere, gt, canvas_hw) == [0, 1, 2, 3]
    maps = sr.oracle_class([mostly_he, tie, nowhere], gt, canvas_hw)
    assert set(maps) == {"MA", "HE", "EX", "SE"}
    assert maps["HE"][1, 4] == pytest.approx(0.8) and maps["MA"][1, 4] == 0.0
    assert maps["MA"][1, 2] == pytest.approx(0.5) and maps["HE"][1, 2] == pytest.approx(0.8)
    assert all(maps[cls][8, 8] == pytest.approx(0.4) for cls in maps)
    assert maps["EX"][1, 4] == 0.0 and maps["SE"][1, 2] == 0.0
    with pytest.raises(ValueError, match="ground-truth masks"):
        sr.oracle_class([tie], gt[:2], canvas_hw)


def test_area_cap_drops_masks_above_the_fraction_of_the_field_of_view():
    import pytest

    from bench.e5 import score_rules as sr

    canvas_hw = (10, 10)
    small = _mask(slice(0, 3), slice(0, 5))  # 15 pixels
    large = _mask(slice(0, 5), slice(0, 5))  # 25 pixels
    padded = _mask(slice(0, 8), slice(0, 8), top=6, left=6)  # 64 in the window, 16 on the canvas
    assert sr.cap_pixels(0.02, 1000) == 20 and sr.cap_pixels(0.005, 1000) == 5 and sr.cap_pixels(1.0, 1000) is None
    kept = sr.area_cap([small, large, padded], 0.02, 1000, canvas_hw)
    assert kept == [small, padded]
    assert sr.area_cap([small, large, padded], 1.0, 1000, canvas_hw) == [small, large, padded]
    assert sr.area_cap([small, large, padded], 0.005, 1000, canvas_hw) == []
    with pytest.raises(ValueError, match="fraction"):
        sr.cap_pixels(0.0, 1000)


# ---- step 6: bench/e5/outputs.py --------------------------------------------------------------


def _toy_e5_source(root, name, ids_by_split, shape=(24, 30)):
    """A prepared set like test_e3a's: box field of view, MA and EX lesions, HE and SE negative."""
    import json

    import numpy as np
    from PIL import Image

    from bench.common.io import LESION_CLASSES

    rng = np.random.default_rng(len(name))
    for ids in ids_by_split.values():
        for image_id in ids:
            (root / "images").mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, shape + (3,), dtype=np.uint8)).save(root / "images" / f"{image_id}.png")
            fov = np.zeros(shape, bool)
            fov[2:-2, 3:-3] = True
            (root / "fov").mkdir(parents=True, exist_ok=True)
            Image.fromarray(fov.astype(np.uint8) * 255).save(root / "fov" / f"{image_id}.png")
            for cls in ("MA", "EX"):
                mask = np.zeros(shape, bool)
                mask[5:9, 6:10] = True
                (root / "masks" / cls).mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255).save(root / "masks" / cls / f"{image_id}.png")
    (root / "meta.json").write_text(json.dumps({"name": name, "classes": list(LESION_CLASSES), "granularity": "fine"}))
    (root / "splits.json").write_text(json.dumps(ids_by_split))


def _write_toy_maps(maps_dir, root, ids, inside=0.9, outside=0.1):
    import numpy as np

    from bench.common.io import LESION_CLASSES, read_fov, read_mask, write_prob

    for image_id in ids:
        fov = read_fov(root, image_id)
        for cls in LESION_CLASSES:
            mask = read_mask(root, cls, image_id, shape=fov.shape)
            prob = np.where(mask, inside, outside).astype(np.float32)
            prob[~fov] = 0.0
            write_prob(maps_dir, cls, image_id, prob)


def test_outputs_layout_pool_thresholds_and_evaluation(tmp_path, monkeypatch):
    import hashlib
    import json

    import pytest

    from bench.e5 import frozen as fz
    from bench.e5 import outputs as out
    from bench.eval.evaluate import load_protocol

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    proto = load_protocol()
    source = out.source_name("a", "SAM", "vith", "p64")
    assert source == "a_sam_vith_p64" and out.source_name("b", "SAM 2", "point") == "b_sam2_point"
    assert out.set_dirname("DDR") == "DDR" and out.set_dirname("TJDR_uwf", ["fov", "cap0.02"]) == "TJDR_uwf__fov__cap0.02"
    assert out.pool_set_dirname("DDR") == "DDR-val"
    assert out.out_dir("a", source, "TOY") == tmp_path / "runs" / "E5" / "a" / source / "TOY"
    assert out.pred_dir(tmp_path / "d", "val").name == "prob_val" and out.pred_dir(tmp_path / "d", "test").name == "prob_test"
    with pytest.raises(ValueError, match="tier"):
        out.source_name("c", "SAM")
    with pytest.raises(ValueError, match="slug"):
        out.slug("SAMed")
    with pytest.raises(ValueError, match="token"):
        out.source_name("a", "SAM", "ViT-H")
    with pytest.raises(ValueError, match="plain token"):
        out.set_dirname("DDR", ["a__b"])
    # the pool: two toy sets standing in for IDRiD val and DDR val; the test set is a third
    pool_a, pool_b, test_root = (tmp_path / "prepared" / n for n in ("PA", "PB", "TOY"))
    _toy_e5_source(pool_a, "PA", {"val": ["p0", "p1"]})
    _toy_e5_source(pool_b, "PB", {"val": ["q0"]})
    _toy_e5_source(test_root, "TOY", {"test": ["t0", "t1"]})
    pool = [("PA", pool_a, "val"), ("PB", pool_b, "val")]
    assert out.pool_thresholds("a", source, frozen, proto, pool=pool) is None
    for name, root, split in pool:
        directory = out.out_dir("a", source, out.pool_set_dirname(name))
        _write_toy_maps(out.pred_dir(directory, split), root, json.loads((root / "splits.json").read_text())[split])
        out.write_predicted(directory, {"tier": "a", "model": "SAM", "set": name, "split": split})
    thresholds = out.pool_thresholds("a", source, frozen, proto, pool=pool)
    assert set(thresholds) == {"MA", "HE", "EX", "SE"} and all(0.0 <= v <= 1.0 for v in thresholds.values())
    assert 0.1 < thresholds["MA"] <= 0.9 and 0.1 < thresholds["EX"] <= 0.9  # lesions at 0.9, background at 0.1
    # a test set evaluated with the pool thresholds
    directory = out.out_dir("a", source, out.set_dirname("TOY"))
    _write_toy_maps(out.pred_dir(directory, "test"), test_root, ["t0", "t1"])
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"weights")
    record = out.checkpoint_record(weights)
    assert record == {"file": "w.pth", "sha256": hashlib.sha256(b"weights").hexdigest(), "bytes": 7}
    out.write_predicted(directory, {"tier": "a", "model": "SAM", "checkpoint": record, "n_images": 2, **out.environment()})
    assert out.read_predicted(directory)["checkpoint"]["file"] == "w.pth" and not out.is_done(directory)
    result = out.evaluate_dir(directory, "TOY", "test", data_dir=test_root, proto=proto, val_thresholds=thresholds)
    assert out.is_done(directory)
    done = json.loads((directory / "done.json").read_text())
    assert done["thresholds"] == "fixed and pool val_opt" and done["val_thresholds"] == thresholds
    assert done["maps_dropped"] is False and done["model"] == "SAM" and "git" in done
    assert done["summary"]["AUPR_any_lesion"] == pytest.approx(1.0) and result["summary"]["mAUPR"] == pytest.approx(1.0)
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["per_class"]["MA"]["thresholds"]["val_opt"]["threshold"] == pytest.approx(thresholds["MA"])
    assert metrics["val_thresholds_given"]["EX"] == pytest.approx(thresholds["EX"]) and metrics["val_pred_dir"] is None
    assert (directory / "metrics.npz").is_file() and (directory / "prob_test").is_dir()
    # a variant directory with drop_maps loses its maps after the metrics; without thresholds it is fixed only
    other = out.out_dir("a", source, out.set_dirname("TOY", ["cap0.02"]))
    _write_toy_maps(out.pred_dir(other, "test"), test_root, ["t0", "t1"])
    out.write_predicted(other, {"tier": "a", "model": "SAM"})
    out.evaluate_dir(other, "TOY", "test", data_dir=test_root, proto=proto, drop_maps=["TOY"])
    done2 = json.loads((other / "done.json").read_text())
    assert done2["maps_dropped"] is True and done2["thresholds"] == "fixed only" and done2["val_thresholds"] is None
    assert not (other / "prob_test").exists() and (other / "metrics.npz").is_file()
    # a pool run evaluates its val split and keeps its maps even when its set is in drop_maps
    pool_dir = out.out_dir("a", source, out.pool_set_dirname("PA"))
    out.evaluate_dir(pool_dir, "PA", "val", data_dir=pool_a, proto=proto, drop_maps=["PA"])
    assert out.pred_dir(pool_dir, "val").is_dir() and json.loads((pool_dir / "done.json").read_text())["maps_dropped"] is False
    with pytest.raises(FileNotFoundError, match="predicted.json"):
        out.evaluate_dir(tmp_path / "nowhere", "TOY", "test", data_dir=test_root, proto=proto)


def test_evaluate_takes_given_validation_thresholds_and_needs_every_class(tmp_path):
    import pytest

    from bench.eval.evaluate import evaluate, load_protocol

    root = tmp_path / "prepared" / "TOY"
    _toy_e5_source(root, "TOY", {"test": ["t0"]})
    _write_toy_maps(tmp_path / "pred", root, ["t0"])
    proto = load_protocol()
    given = {"MA": 0.4, "HE": 0.5, "EX": 0.6, "SE": 0.7}
    result = evaluate(root, "test", tmp_path / "pred", proto, val_thresholds=given)
    assert result["per_class"]["EX"]["thresholds"]["val_opt"]["threshold"] == pytest.approx(0.6)
    assert "mDice_pooled_val_opt" in result["summary"] and result["val_thresholds_given"] == given
    plain = evaluate(root, "test", tmp_path / "pred", proto)
    assert "val_opt" not in plain["per_class"]["EX"]["thresholds"] and plain["val_thresholds_given"] is None
    with pytest.raises(ValueError, match="val_thresholds lacks"):
        evaluate(root, "test", tmp_path / "pred", proto, val_thresholds={"MA": 0.5})


# ---- step 7: bench/e5/amg.py ------------------------------------------------------------------


class _FakeGenerator:
    """Two masks per window: a 2 x 2 block at the window's top-left (IoU 0.9, stability 0.8) and the whole window (0.3, 0.6)."""

    def __init__(self):
        self.calls = []

    def generate(self, image):
        import numpy as np

        height, width = image.shape[:2]
        self.calls.append((height, width))
        small = np.zeros((height, width), bool)
        small[:2, :2] = True
        return [
            {"segmentation": small, "area": 4, "bbox": [0, 0, 2, 2], "predicted_iou": 0.9, "stability_score": 0.8},
            {"segmentation": np.ones((height, width), bool), "area": height * width, "bbox": [0, 0, width, height],
             "predicted_iou": 0.3, "stability_score": 0.6},
        ]


def _filled_frozen(tmp_path, density, cap, name="filled.yaml"):
    """The real protocol with the pool selections filled in."""
    doc = yaml.safe_load(FROZEN.read_text())
    doc["zero_shot"]["grid"]["selected"] = {"points_per_side": density, "area_cap_fraction": cap}
    doc["zero_shot"]["sam3"]["selected"] = {"MA": "red dot", "HE": "retinal hemorrhage", "EX": "hard exudate", "SE": "soft exudate"}
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return path


def test_amg_pool_run_writes_every_cap_and_kind_from_fake_masks(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pytest

    from bench.common.io import read_fov, read_prob_quantized
    from bench.e5 import amg
    from bench.e5 import frozen as fz
    from bench.e5 import geometry as geo
    from bench.e5 import outputs as out

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    root = tmp_path / "prepared" / "DDR"
    _toy_e5_source(root, "DDR", {"val": ["v0", "v1"]})
    assert amg.generator_params(frozen, "SAM", 64)["pred_iou_thresh"] == 0.88 and amg.generator_params(frozen, "SAM 2", 128)["use_m2m"] is False
    with pytest.raises(ValueError, match="registered grid"):
        amg.generator_params(frozen, "SAM", 32)
    with pytest.raises(ValueError, match="automatic masks run for"):
        amg.spec(frozen, "MedSAM")
    assert amg.source_name(frozen, "SAM", 64) == "a_sam_vith_p64" and amg.source_name(frozen, "SAM 2", 128) == "a_sam2_hieral21_p128"
    assert amg.cap_token(1.0) == "" and amg.cap_token(0.02) == "cap0.02" and amg.cap_token(0.005) == "cap0.005"
    fake = _FakeGenerator()
    summary = amg.run("SAM", 64, "DDR", "val", generator=fake, frozen=frozen, data_dir=root, long_side=20, window=8)
    names = sorted(Path(p).name for p in summary["directories"])
    assert names == sorted([
        "DDR-val", "DDR-val__oracle_class", "DDR-val__stability",
        "DDR-val__cap0.005", "DDR-val__cap0.005__oracle_class", "DDR-val__cap0.005__stability",
        "DDR-val__cap0.02", "DDR-val__cap0.02__oracle_class", "DDR-val__cap0.02__stability",
    ])
    source = tmp_path / "runs" / "E5" / "a" / "a_sam_vith_p64"
    main = source / "DDR-val"
    predicted = json.loads((main / "predicted.json").read_text())
    cv = geo.canvas(root, "v0", 20, masks=True)
    per_image = len(geo.windows(cv.hw, 8, 0.5))
    assert predicted["density"] == 64 and predicted["params"]["points_per_side"] == 64 and predicted["params"]["pred_iou_thresh"] == 0.88
    assert predicted["geometry"] == "G1" and predicted["window"] == 8 and predicted["overlap"] == 0.5 and predicted["canvas_long_side"] == 20
    assert predicted["n_images"] == 2 and predicted["n_windows"] == 2 * per_image and predicted["n_masks"] == 4 * per_image
    assert predicted["kind"] == "any_lesion" and predicted["area_cap_fraction"] == 1.0 and predicted["checkpoint"] is None
    assert predicted["generator"] == "_FakeGenerator" and predicted["seconds_per_window_mean"] is not None and "git" in predicted
    assert all(call == (8, 8) for call in fake.calls) and len(fake.calls) == 2 * per_image  # windows are padded to 8 x 8
    # the any-lesion map: 0.3 wherever a window covers, 0.9 at window origins, 0 outside the field of view
    fov = read_fov(root, "v0")
    q = read_prob_quantized(main / "prob_val", "MA", "v0")
    assert q.shape == fov.shape and (q[~fov] == 0).all() and q[fov].min() >= round(0.3 * 65535) - 1
    assert abs(int(q.max()) - round(0.9 * 65535)) <= 1
    assert np.array_equal(q, read_prob_quantized(main / "prob_val", "SE", "v0"))  # stored under all four names
    capped = read_prob_quantized(source / "DDR-val__cap0.02" / "prob_val", "MA", "v0")
    assert abs(int(capped.max()) - round(0.9 * 65535)) <= 1 and (capped == 0).sum() > (q == 0).sum()  # the whole-window masks are gone
    assert read_prob_quantized(source / "DDR-val__cap0.005" / "prob_val", "MA", "v0").max() == 0  # 1-pixel cap drops everything
    stability = read_prob_quantized(source / "DDR-val__stability" / "prob_val", "MA", "v0")
    assert abs(int(stability.max()) - round(0.8 * 65535)) <= 1 and stability[fov].min() >= round(0.6 * 65535) - 1
    oracle = {cls: read_prob_quantized(source / "DDR-val__oracle_class" / "prob_val", cls, "v0") for cls in ("MA", "HE", "EX", "SE")}
    assert np.array_equal(oracle["HE"], oracle["SE"]) and (oracle["MA"] >= oracle["HE"]).all()
    assert (oracle["MA"] > 0).sum() > (oracle["HE"] > 0).sum()  # ties MA / EX go to MA; no-overlap masks go to all four
    # evaluated with fixed thresholds only; the pool keeps its maps; a second call touches nothing
    for name in names:
        done = json.loads((source / name / "done.json").read_text())
        assert done["thresholds"] == "fixed only" and done["maps_dropped"] is False and "AUPR_any_lesion" in done["summary"]
    stamp = (main / "predicted.json").stat().st_mtime_ns
    amg.run("SAM", 64, "DDR", "val", generator=fake, frozen=frozen, data_dir=root, long_side=20, window=8)
    assert (main / "predicted.json").stat().st_mtime_ns == stamp and len(fake.calls) == 2 * per_image


def test_amg_test_run_needs_the_selection_uses_pool_thresholds_and_obeys_the_budget_rule(tmp_path, monkeypatch):
    import json

    import pytest

    from bench.e5 import amg
    from bench.e5 import frozen as fz
    from bench.e5 import outputs as out

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    root = tmp_path / "prepared" / "DDR"
    _toy_e5_source(root, "DDR", {"val": ["v0"], "test": ["t0", "t1"]})
    unfilled = fz.load_frozen()
    with pytest.raises(ValueError, match="grid.selected is empty"):
        amg.run("SAM", 64, "DDR", "test", generator=_FakeGenerator(), frozen=unfilled, data_dir=root, long_side=20, window=8)
    frozen = fz.load_frozen(_filled_frozen(tmp_path, 64, 0.02))
    with pytest.raises(ValueError, match="selected density 64"):
        amg.run("SAM", 128, "DDR", "test", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8)
    # the pool run first (one toy pool set stands in for both), then the test split with its thresholds
    pool = [("DDR", root, "val")]
    amg.run("SAM", 64, "DDR", "val", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8)
    summary = amg.run("SAM", 64, "DDR", "test", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8, pool=pool)
    names = sorted(Path(p).name for p in summary["directories"])
    assert names == ["DDR", "DDR__oracle_class", "DDR__stability"]
    source = tmp_path / "runs" / "E5" / "a" / "a_sam_vith_p64"
    done = json.loads((source / "DDR" / "done.json").read_text())
    assert done["area_cap_fraction"] == 0.02 and done["thresholds"] == "fixed and pool val_opt" and set(done["val_thresholds"]) == {"MA", "HE", "EX", "SE"}
    assert (source / "DDR" / "prob_test").is_dir() and done["n_images"] == 2
    oracle = json.loads((source / "DDR__oracle_class" / "done.json").read_text())
    assert oracle["kind"] == "oracle_class" and oracle["thresholds"] == "fixed and pool val_opt"
    # drop_maps and limit
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs2" / "E5")
    amg.run("SAM", 64, "DDR", "val", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8)
    amg.run("SAM", 64, "DDR", "test", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8, pool=pool, drop_maps=["DDR"], limit=1)
    dropped = tmp_path / "runs2" / "E5" / "a" / "a_sam_vith_p64" / "DDR"
    done = json.loads((dropped / "done.json").read_text())
    assert done["maps_dropped"] is True and not (dropped / "prob_test").exists() and done["n_images"] == 1 and done["limit"] == 1
    assert done["excluded"] == ["t1"] and json.loads((dropped / "metrics.json").read_text())["n_images"] == 1
    # a later run without --limit predicts the smoke directory again, in full
    amg.run("SAM", 64, "DDR", "test", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8, pool=pool, evaluate=False)
    assert json.loads((dropped / "predicted.json").read_text())["limit"] is None and json.loads((dropped / "predicted.json").read_text())["n_images"] == 2
    # the registered budget rule: the largest density skips the per-degree main variant of TJDR_uwf
    largest = fz.load_frozen(_filled_frozen(tmp_path, 128, 1.0, "largest.yaml"))
    assert amg.budget_rule_skips(largest, "test", 128, "TJDR_uwf", "main") is True
    assert amg.budget_rule_skips(largest, "test", 128, "TJDR_uwf", "fov") is False
    assert amg.budget_rule_skips(largest, "test", 128, "DDR", "main") is False
    assert amg.budget_rule_skips(largest, "val", 128, "TJDR_uwf", "main") is False
    skipped = amg.run("SAM", 128, "TJDR_uwf", "test", generator=_FakeGenerator(), frozen=largest, data_dir=root, long_side=20, window=8)
    assert skipped["skipped"] == "budget_rule" and skipped["directories"] == []
    assert not (tmp_path / "runs2" / "E5" / "a" / "a_sam_vith_p128").exists()
    with pytest.raises(ValueError, match="geometry"):
        amg.run("SAM", 64, "DDR", "val", geometry="G3", generator=_FakeGenerator(), frozen=frozen, data_dir=root)


# ---- step 8: bench/e5/prompted.py and bench/e5/same_info.py -----------------------------------


class _FakeDecoder:
    """A box fills its rectangle with 0.7; a point lights a 3 x 3 block with 0.5; every call is recorded."""

    def __init__(self):
        self.images, self.boxes, self.points = [], [], []

    def set_image(self, image):
        self.hw = tuple(image.shape[:2])
        self.images.append(self.hw)

    def box(self, box):
        import numpy as np

        self.boxes.append(tuple(int(v) for v in box))
        x0, y0, x1, y1 = (int(v) for v in box)
        prob = np.zeros(self.hw, np.float32)
        prob[max(y0, 0) : y1, max(x0, 0) : x1] = 0.7
        return prob

    def point(self, xy):
        import numpy as np

        self.points.append(tuple(int(v) for v in xy))
        x, y = (int(v) for v in xy)
        prob = np.zeros(self.hw, np.float32)
        prob[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = 0.5
        return prob


def _two_component_set(root, ids_by_split):
    """_toy_e5_source plus a second MA component in every image."""
    import numpy as np
    from PIL import Image

    _toy_e5_source(root, root.name, ids_by_split)
    for ids in ids_by_split.values():
        for image_id in ids:
            path = root / "masks" / "MA" / f"{image_id}.png"
            mask = np.asarray(Image.open(path)) > 0
            mask[12:16, 19:24] = True
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def test_choose_window_takes_the_window_whose_edges_are_farthest_from_the_centre():
    from bench.e5 import geometry as geo
    from bench.e5 import prompted

    boxes = geo.windows((17, 20), 8, 0.5)
    assert boxes[prompted.choose_window(boxes, 8, (5.0, 5.0))] == (0, 0)  # margin 2 beats the neighbours' 1
    assert boxes[prompted.choose_window(boxes, 8, (6.0, 6.0))] == (4, 4)
    assert boxes[prompted.choose_window(boxes, 8, (19.0, 16.0))] == (9, 12)
    assert prompted.choose_window([(0, 0)], 8, (30.0, 30.0)) == 0  # outside every window: the nearest
    assert prompted.box_center([2, 4, 6, 8]) == (3.5, 5.5)


def test_prompted_pool_run_decodes_once_per_window_and_fills_prefix_maxima(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pytest

    from bench.common.io import read_prob_quantized
    from bench.e5 import frozen as fz
    from bench.e5 import geometry as geo
    from bench.e5 import outputs as out
    from bench.e5 import prompted, prompts

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    root = tmp_path / "prepared" / "DDR"
    _two_component_set(root, {"val": ["v0", "v1"]})
    pdir = tmp_path / "prompts"
    with pytest.raises(FileNotFoundError, match="bench.e5.prompts"):
        prompted.run("SAM", "boxes", "DDR", "val", decoder=_FakeDecoder(), frozen=frozen, data_dir=root, long_side=20, window=8, prompts_dir=pdir)
    prompts.build("DDR", "val", frozen=frozen, out_dir=pdir, data_dir=root, long_side=20)
    with pytest.raises(ValueError, match="runs \\['boxes'\\]"):
        prompted.run("MedSAM", "points", "DDR", "val", decoder=_FakeDecoder(), frozen=frozen, data_dir=root, long_side=20, window=8, prompts_dir=pdir)
    with pytest.raises(ValueError, match="regenerate"):
        prompted.run("SAM", "boxes", "DDR", "val", decoder=_FakeDecoder(), frozen=frozen, data_dir=root, long_side=24, window=8, prompts_dir=pdir)
    assert prompted.source_name(frozen, "SAM", "boxes") == "b_sam_vitb_box"
    assert prompted.source_name(frozen, "SAM 2", "points") == "b_sam2_hieral21_point" and prompted.source_name(frozen, "MedSAM", "boxes") == "b_medsam_vitb_box"
    fake = _FakeDecoder()
    summary = prompted.run("SAM", "boxes", "DDR", "val", decoder=fake, frozen=frozen, data_dir=root, long_side=20, window=8, prompts_dir=pdir)
    names = sorted(Path(p).name for p in summary["directories"])
    assert names == ["DDR-val__k1", "DDR-val__k20", "DDR-val__k5", "DDR-val__kall"]
    source = tmp_path / "runs" / "E5" / "b" / "b_sam_vitb_box"
    predicted = json.loads((source / "DDR-val__kall" / "predicted.json").read_text())
    assert predicted["kind"] == "boxes" and predicted["k"] == "all" and predicted["n_prompts"] == 2 * 3  # MA 2 + EX 1 per image
    assert predicted["n_windows_encoded"] == len(fake.images) and all(hw == (8, 8) for hw in fake.images)
    assert predicted["n_windows_encoded"] <= predicted["n_windows"] and predicted["decoder"] == "_FakeDecoder"
    assert predicted["prompt_file_sha256"] and predicted["prompt_rule"]["box"].startswith("tight") and "git" in predicted
    doc = prompts.load(fz.prompt_file("DDR", "val", pdir))
    distinct = set()
    cv = geo.canvas(root, "v0", 20)
    boxes = geo.windows(cv.hw, 8, 0.5)
    for cls in ("MA", "EX"):
        for prompt in doc["images"]["v0"]["classes"][cls]["prompts"]:
            distinct.add(prompted.choose_window(boxes, 8, prompted.box_center(prompt["box"])))
    assert len(fake.images) == 2 * len(distinct) or len(fake.images) >= len(distinct)  # one encode per window with prompts, per image
    # k maps are prefix maxima: k1 has the first-ranked MA box only, kall both; EX has one component so k1 == kall
    k1 = read_prob_quantized(source / "DDR-val__k1" / "prob_val", "MA", "v0")
    kall = read_prob_quantized(source / "DDR-val__kall" / "prob_val", "MA", "v0")
    assert (k1 <= kall).all() and (kall > 0).sum() > (k1 > 0).sum() and abs(int(kall.max()) - round(0.7 * 65535)) <= 1
    assert np.array_equal(read_prob_quantized(source / "DDR-val__k1" / "prob_val", "EX", "v0"), read_prob_quantized(source / "DDR-val__kall" / "prob_val", "EX", "v0"))
    assert read_prob_quantized(source / "DDR-val__kall" / "prob_val", "HE", "v0").max() == 0
    # the lesion itself is covered by its own box prompt
    from bench.common.io import read_mask

    ma = read_mask(root, "MA", "v0")
    assert (kall[ma] > 0).mean() > 0.9
    assert json.loads((source / "DDR-val__k5" / "done.json").read_text())["thresholds"] == "fixed only"
    # points for SAM 2: the fake lights a 3 x 3 block at the component's farthest-from-boundary pixel
    point_fake = _FakeDecoder()
    prompted.run("SAM 2", "points", "DDR", "val", decoder=point_fake, frozen=frozen, data_dir=root, long_side=20, window=8, prompts_dir=pdir)
    assert len(point_fake.points) == 2 * 3 and not point_fake.boxes
    pk = read_prob_quantized(tmp_path / "runs" / "E5" / "b" / "b_sam2_hieral21_point" / "DDR-val__kall" / "prob_val", "MA", "v0")
    assert 0 < (pk > 0).sum() < (kall > 0).sum() and abs(int(pk.max()) - round(0.5 * 65535)) <= 1
    # a second call leaves the finished directories alone
    stamp = (source / "DDR-val__kall" / "predicted.json").stat().st_mtime_ns
    prompted.run("SAM", "boxes", "DDR", "val", decoder=fake, frozen=frozen, data_dir=root, long_side=20, window=8, prompts_dir=pdir)
    assert (source / "DDR-val__kall" / "predicted.json").stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="box prompts only"):
        prompted.MedSamDecoder(None, "cpu").point([1, 1])


def test_prompted_test_run_uses_pool_thresholds_and_same_info_restricts_reference_maps(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pytest

    from bench.common.io import read_mask, read_prob_quantized
    from bench.e5 import frozen as fz
    from bench.e5 import outputs as out
    from bench.e5 import prompted, prompts, same_info

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    root = tmp_path / "prepared" / "DDR"
    _two_component_set(root, {"val": ["v0"], "test": ["t0", "t1"]})
    pdir = tmp_path / "prompts"
    prompts.build("DDR", "val", frozen=frozen, out_dir=pdir, data_dir=root, long_side=20)
    prompts.build("DDR", "test", frozen=frozen, out_dir=pdir, data_dir=root, long_side=20)
    pool = [("DDR", root, "val")]
    prompted.run("SAM", "boxes", "DDR", "val", decoder=_FakeDecoder(), frozen=frozen, data_dir=root, long_side=20, window=8, prompts_dir=pdir)
    summary = prompted.run("SAM", "boxes", "DDR", "test", decoder=_FakeDecoder(), frozen=frozen, data_dir=root, long_side=20, window=8,
                           prompts_dir=pdir, pool=pool, limit=1)
    assert sorted(Path(p).name for p in summary["directories"]) == ["DDR__k1", "DDR__k20", "DDR__k5", "DDR__kall"]
    done = json.loads((tmp_path / "runs" / "E5" / "b" / "b_sam_vitb_box" / "DDR__k5" / "done.json").read_text())
    assert done["thresholds"] == "fixed and pool val_opt" and done["excluded"] == ["t1"] and done["n_images"] == 1
    # same-information reference: a supervised model's maps (0.9 on lesions, 0.1 elsewhere) restricted to the k boxes
    reference = tmp_path / "ref" / "prob_test"
    _write_toy_maps(reference, root, ["t0", "t1"])
    result = same_info.run("M2MRF", ["ddr", "seed0"], reference, "DDR", "test", frozen=frozen, data_dir=root, prompts_dir=pdir)
    assert result["source"] == "b_m2mrf_ref_ddr_seed0"
    ref_dir = tmp_path / "runs" / "E5" / "b" / "b_m2mrf_ref_ddr_seed0"
    assert sorted(p.name for p in ref_dir.iterdir()) == ["DDR__k1", "DDR__k20", "DDR__k5", "DDR__kall"]
    k1 = read_prob_quantized(ref_dir / "DDR__k1" / "prob_test", "MA", "t0")
    kall = read_prob_quantized(ref_dir / "DDR__kall" / "prob_test", "MA", "t0")
    ma = read_mask(root, "MA", "t0")
    assert (k1 <= kall).all() and (kall > 0).sum() > (k1 > 0).sum()
    assert abs(int(kall[ma].min()) - round(0.9 * 65535)) <= 1  # the lesion pixels keep the model's own value
    outside = ~ma
    assert (kall[outside] == 0).mean() > 0.9  # almost everything outside the boxes is 0 (the 0.1 background is masked)
    done = json.loads((ref_dir / "DDR__kall" / "done.json").read_text())
    assert done["kind"] == "same_information_reference" and done["thresholds"] == "fixed only" and done["thresholds_note"].startswith("reference")
    assert same_info.canvas_box_to_original([0, 0, 2, 2], (10, 10), (2, 22, 3, 23)) == [3, 2, 7, 6]
    with pytest.raises(FileNotFoundError, match="prediction directory"):
        same_info.run("M2MRF", ["x"], tmp_path / "nope", "DDR", "test", frozen=frozen, data_dir=root, prompts_dir=pdir)
    np.testing.assert_array_equal(same_info.box_union_mask([], "all", (10, 10), (0, 10, 0, 10), (10, 10)), np.zeros((10, 10), bool))


# ---- step 9: bench/e5/sam3_text.py and the retiseg-sam3 environment ---------------------------


class _FakeScorer:
    """Scores depend on the phrase's registered index: 0.1 x (index + 1) over the window, 0.5 + 0.1 x index at its top-left 2 x 2."""

    def __init__(self, phrase_ids):
        self.by_phrase = {phrase: key for key, phrase in phrase_ids.items()}
        self.images, self.phrases = [], []

    def set_image(self, image):
        from PIL import Image

        assert isinstance(image, Image.Image)
        self.hw = (image.size[1], image.size[0])
        self.images.append(self.hw)

    def score(self, phrase):
        import numpy as np

        _cls, index = self.by_phrase[phrase]
        self.phrases.append(phrase)
        prob = np.full(self.hw, 0.1 * (index + 1), np.float32)
        prob[:2, :2] = 0.5 + 0.1 * index
        return prob


def test_sam3_phrase_runs_selection_and_assembly(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pytest

    from bench.common.io import read_fov, read_prob_quantized
    from bench.e5 import frozen as fz
    from bench.e5 import geometry as geo
    from bench.e5 import outputs as out
    from bench.e5 import sam3_text
    from bench.eval.evaluate import load_protocol

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    proto = load_protocol()
    root = tmp_path / "prepared" / "DDR"
    _toy_e5_source(root, "DDR", {"val": ["v0", "v1"], "test": ["t0", "t1"]})
    pool = [("DDR", root, "val")]
    fake = _FakeScorer(fz.phrase_ids(frozen))
    assert sam3_text.phrases_for(frozen, 1) == {"MA": "red dot", "HE": "blot hemorrhage", "EX": "yellow deposit", "SE": "soft exudate"}
    with pytest.raises(ValueError, match="phrase index"):
        sam3_text.phrases_for(frozen, 3)
    assert sam3_text.source_name(2) == "a_sam3_ph2" and sam3_text.selected_source() == "a_sam3_selected"
    # selection needs every index: before any pool run it refuses
    with pytest.raises(FileNotFoundError, match="every phrase index"):
        sam3_text.select_phrases(frozen, proto, pool=pool)
    for index in (0, 1, 2):
        summary = sam3_text.run(index, "DDR", "val", scorer=fake, frozen=frozen, data_dir=root, long_side=20, window=8)
        assert summary["assembled"] is None  # nothing selected yet
    source = tmp_path / "runs" / "E5" / "a"
    predicted = json.loads((source / "a_sam3_ph1" / "DDR-val" / "predicted.json").read_text())
    cv = geo.canvas(root, "v0", 20)
    per_image = len(geo.windows(cv.hw, 8, 0.5))
    assert predicted["kind"] == "text" and predicted["phrase_index"] == 1 and predicted["phrases"]["EX"] == "yellow deposit"
    assert predicted["n_windows"] == 2 * per_image and predicted["confidence_threshold"] == 0.0 and predicted["scorer"] == "_FakeScorer"
    assert predicted["window"] == 8 and all(hw == (8, 8) for hw in fake.images) and len(fake.phrases) == 4 * 3 * 2 * per_image
    fov = read_fov(root, "v0")
    q = read_prob_quantized(source / "a_sam3_ph1" / "DDR-val" / "prob_val", "MA", "v0")
    assert (q[~fov] == 0).all() and q[fov].min() >= round(0.2 * 65535) - 1 and abs(int(q.max()) - round(0.6 * 65535)) <= 1
    assert json.loads((source / "a_sam3_ph0" / "DDR-val" / "done.json").read_text())["thresholds"] == "fixed only"
    # selection: per class the index with the highest pooled AUPR, consistent with the table
    chosen, table = sam3_text.select_phrases(frozen, proto, pool=pool)
    assert set(chosen) == {"MA", "HE", "EX", "SE"} and all(len(table[cls]) == 3 for cls in table)
    for cls in chosen:
        assert chosen[cls]["index"] == int(np.argmax(table[cls])) and chosen[cls]["phrase"] == frozen["zero_shot"]["sam3"]["phrases"][cls][chosen[cls]["index"]]
    snippet = sam3_text.format_selected_yaml(chosen, table)
    assert snippet.startswith("    selected:") and all(chosen[cls]["phrase"] in snippet for cls in chosen)
    # with the selection written, the test split runs; the third index assembles a_sam3_selected and evaluates it with pool thresholds
    with pytest.raises(ValueError, match="sam3.selected lacks"):
        sam3_text.run(0, "DDR", "test", scorer=fake, frozen=frozen, data_dir=root, long_side=20, window=8, pool=pool)
    doc = yaml.safe_load(FROZEN.read_text())
    doc["zero_shot"]["sam3"]["selected"] = {cls: chosen[cls]["phrase"] for cls in chosen}
    doc["zero_shot"]["grid"]["selected"] = {"points_per_side": 64, "area_cap_fraction": 1.0}
    filled_path = tmp_path / "filled.yaml"
    filled_path.write_text(yaml.safe_dump(doc))
    filled = fz.load_frozen(filled_path)
    assert sam3_text.selected_indices(filled) == {cls: chosen[cls]["index"] for cls in chosen}
    # the pool's selected source first (its thresholds feed the test split)
    pooled = sam3_text.assemble_selected("DDR", "val", frozen=filled, data_dir=root, pool=pool)
    assert pooled == source / "a_sam3_selected" / "DDR-val" and out.is_done(pooled)
    for index in (0, 1, 2):
        summary = sam3_text.run(index, "DDR", "test", scorer=fake, frozen=filled, data_dir=root, long_side=20, window=8, pool=pool)
    assert summary["assembled"] == str(source / "a_sam3_selected" / "DDR")
    selected = json.loads((source / "a_sam3_selected" / "DDR" / "done.json").read_text())
    assert selected["kind"] == "text_selected" and selected["phrase_index"] == {cls: chosen[cls]["index"] for cls in chosen}
    assert selected["thresholds"] == "fixed and pool val_opt" and selected["n_images"] == 2
    for cls in ("MA", "HE", "EX", "SE"):
        copied = read_prob_quantized(source / "a_sam3_selected" / "DDR" / "prob_test", cls, "t0")
        original = read_prob_quantized(source / f"a_sam3_ph{chosen[cls]['index']}" / "DDR" / "prob_test", cls, "t0")
        assert np.array_equal(copied, original)
    done = json.loads((source / "a_sam3_ph2" / "DDR" / "done.json").read_text())
    assert done["thresholds"] == "fixed and pool val_opt"
    # a second assembly leaves the finished directory alone; a run with a scorer rejects numpy windows
    stamp = (source / "a_sam3_selected" / "DDR" / "predicted.json").stat().st_mtime_ns
    sam3_text.assemble_selected("DDR", "test", frozen=filled, data_dir=root, pool=pool)
    assert (source / "a_sam3_selected" / "DDR" / "predicted.json").stat().st_mtime_ns == stamp
    with pytest.raises(TypeError, match="PIL image"):
        sam3_text.Sam3Scorer(None).set_image(np.zeros((8, 8, 3), np.uint8))


def test_sam3_environment_scripts_parse_and_name_the_env():
    import subprocess

    for script in ("bench/scripts/cluster_env.sh", "bench/scripts/server_run.sh"):
        subprocess.run(["bash", "-n", str(REPO / script)], check=True)
    text = (REPO / "bench/scripts/cluster_env.sh").read_text()
    assert "install_sam3" in text and "ensure_env retiseg-sam3 3.12" in text and "RETISEG_SKIP_SAM3" in text
    assert "official_code/sam3" in text and "(2, 7)" in text and "(12, 6)" in text
    assert "retiseg-sam3" in (REPO / "bench/scripts/server_run.sh").read_text()


# ---- review fix 2 (A5): a smoke run's done.json does not answer a full request ---------------------


def test_a_full_run_replaces_the_evaluation_of_a_smoke_run(tmp_path, monkeypatch):
    """A directory predicted again in full is evaluated again, and a smoke pool run feeds no selection or threshold."""
    import json

    from bench.e5 import amg
    from bench.e5 import frozen as fz
    from bench.e5 import outputs as out
    from bench.eval.evaluate import load_protocol

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    root = tmp_path / "prepared" / "DDR"
    _toy_e5_source(root, "DDR", {"val": ["v0", "v1", "v2"], "test": ["t0"]})
    directory = tmp_path / "runs" / "E5" / "a" / "a_sam_vith_p64" / "DDR-val"
    amg.run("SAM", 64, "DDR", "val", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8, limit=1)
    smoke = json.loads((directory / "done.json").read_text())
    assert smoke["n_images"] == 1 and smoke["limit"] == 1 and smoke["excluded"] == ["v1", "v2"]
    assert out.is_done(directory, limit=1) and not out.is_done(directory)
    proto = load_protocol(REPO / "bench/eval/protocol.yaml")
    pool = [("DDR", root, "val")]
    assert out.pool_class_aupr("a", "a_sam_vith_p64", frozen, proto, pool=pool) is None
    assert out.pool_thresholds("a", "a_sam_vith_p64", frozen, proto, pool=pool) is None
    amg.run("SAM", 64, "DDR", "val", generator=_FakeGenerator(), frozen=frozen, data_dir=root, long_side=20, window=8)
    full = json.loads((directory / "done.json").read_text())
    assert full["n_images"] == 3 and full["limit"] is None and full["excluded"] == [] and out.is_done(directory)
    assert json.loads((directory / "metrics.json").read_text())["n_images"] == 3
    assert out.pool_thresholds("a", "a_sam_vith_p64", frozen, proto, pool=pool) is not None
    # a later smoke request is answered by the full run: nothing is predicted or evaluated again
    again = _FakeGenerator()
    amg.run("SAM", 64, "DDR", "val", generator=again, frozen=frozen, data_dir=root, long_side=20, window=8, limit=1)
    assert again.calls == [] and json.loads((directory / "done.json").read_text())["n_images"] == 3


# ---- review fix 3 (A3): a test split is not evaluated without the pool thresholds -------------------


def test_a_test_split_is_not_evaluated_without_the_pool_thresholds(tmp_path, monkeypatch):
    """No silent fall-back to the fixed threshold: the prediction stays and the next call evaluates it."""
    import json

    import pytest

    from bench.e5 import frozen as fz
    from bench.e5 import outputs as out
    from bench.e5 import prompted, prompts

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    root = tmp_path / "prepared" / "DDR"
    _two_component_set(root, {"val": ["v0"], "test": ["t0", "t1"]})
    pdir = tmp_path / "prompts"
    for split in ("val", "test"):
        prompts.build("DDR", split, frozen=frozen, out_dir=pdir, data_dir=root, long_side=20)
    pool = [("DDR", root, "val")]
    kwargs = dict(frozen=frozen, data_dir=root, long_side=20, window=8, prompts_dir=pdir)
    with pytest.raises(RuntimeError, match="run the pool first"):
        prompted.run("SAM", "boxes", "DDR", "test", decoder=_FakeDecoder(), pool=pool, **kwargs)
    directory = tmp_path / "runs" / "E5" / "b" / "b_sam_vitb_box" / "DDR__kall"
    assert (directory / "predicted.json").is_file() and not (directory / "done.json").exists()
    # a smoke request on that full prediction does not evaluate it as a smoke either: the scope is the prediction's
    with pytest.raises(RuntimeError, match="run the pool first"):
        prompted.run("SAM", "boxes", "DDR", "test", decoder=_FakeDecoder(), pool=pool, limit=1, **kwargs)
    assert not (directory / "done.json").exists()
    # a smoke pool run is not a pool run; a smoke test prediction is the exception and says so
    prompted.run("SAM", "boxes", "DDR", "val", decoder=_FakeDecoder(), limit=1, **kwargs)
    with pytest.raises(RuntimeError, match="run the pool first"):
        prompted.run("SAM", "boxes", "DDR", "test", decoder=_FakeDecoder(), pool=pool, **kwargs)
    smoke_root = tmp_path / "prepared" / "IDRiD"
    _two_component_set(smoke_root, {"val": ["w0"], "test": ["s0", "s1"]})
    for split in ("val", "test"):
        prompts.build("IDRiD", split, frozen=frozen, out_dir=pdir, data_dir=smoke_root, long_side=20)
    prompted.run("SAM", "boxes", "IDRiD", "test", decoder=_FakeDecoder(), pool=[("IDRiD", smoke_root, "val")], limit=1,
                 frozen=frozen, data_dir=smoke_root, long_side=20, window=8, prompts_dir=pdir)
    smoke = json.loads((tmp_path / "runs" / "E5" / "b" / "b_sam_vitb_box" / "IDRiD__kall" / "done.json").read_text())
    assert smoke["thresholds"] == "fixed only" and smoke["limit"] == 1 and smoke["excluded"] == ["s1"]
    # the full pool run arrives: the same command evaluates only, with the pool thresholds
    prompted.run("SAM", "boxes", "DDR", "val", decoder=_FakeDecoder(), **kwargs)
    again = _FakeDecoder()
    prompted.run("SAM", "boxes", "DDR", "test", decoder=again, pool=pool, **kwargs)
    assert again.images == []
    done = json.loads((directory / "done.json").read_text())
    assert done["thresholds"] == "fixed and pool val_opt" and done["limit"] is None and done["excluded"] == []
    assert out.is_done(directory) and json.loads((directory / "metrics.json").read_text())["n_images"] == 2


# ---- review fix 4 (A2): the assembly is the only writer of a_sam3_selected and the last reader of the phrase maps


def test_sam3_assembly_on_a_drop_maps_set_and_without_a_prepared_pool_assembly(tmp_path, monkeypatch):
    """The selected source exists on a --drop-maps set and has pool thresholds; the phrase maps go only after it is evaluated."""
    import json

    import pytest

    from bench.e5 import frozen as fz
    from bench.e5 import outputs as out
    from bench.e5 import sam3_text

    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    frozen = fz.load_frozen()
    root = tmp_path / "prepared" / "FGADR"
    _toy_e5_source(root, "FGADR", {"val": ["v0"], "test": ["t0", "t1"]})
    pool = [("FGADR", root, "val")]
    fake = _FakeScorer(fz.phrase_ids(frozen))
    kwargs = dict(scorer=fake, data_dir=root, long_side=20, window=8)
    for index in (0, 1, 2):
        sam3_text.run(index, "FGADR", "val", frozen=frozen, **kwargs)
    filled = fz.load_frozen(_filled_frozen(tmp_path, 64, 1.0))  # MA and SE from index 1, HE and EX from index 0
    source = tmp_path / "runs" / "E5" / "a"
    with pytest.raises(FileNotFoundError, match="no finished prediction"):
        sam3_text.assemble_selected("FGADR", "test", frozen=filled, data_dir=root, pool=pool)
    # the queue's phrase items: --no-assemble, and --drop-maps is not passed to them; a drop-maps flag would be ignored anyway
    for index in (0, 1, 2):
        summary = sam3_text.run(index, "FGADR", "test", frozen=filled, pool=pool, assemble=False, drop_maps=("FGADR",), **kwargs)
        assert summary["assembled"] is None and (source / f"a_sam3_ph{index}" / "FGADR" / "prob_test").is_dir()
        assert json.loads((source / f"a_sam3_ph{index}" / "FGADR" / "done.json").read_text())["maps_dropped"] is False
    assert not (source / "a_sam3_selected" / "FGADR-val").exists()  # nobody assembled the pool beforehand
    target = sam3_text.assemble_selected("FGADR", "test", frozen=filled, data_dir=root, pool=pool, drop_maps=("FGADR",))
    done = json.loads((target / "done.json").read_text())
    assert done["thresholds"] == "fixed and pool val_opt" and done["maps_dropped"] is True and not (target / "prob_test").exists()
    assert not (target / "prob_test.partial").exists()
    assert out.is_done(source / "a_sam3_selected" / "FGADR-val")  # assembled on demand: its thresholds were needed
    for index in (0, 1, 2):
        phrase = source / f"a_sam3_ph{index}" / "FGADR"
        assert not (phrase / "prob_test").exists() and json.loads((phrase / "done.json").read_text())["maps_dropped"] is True
    assert (source / "a_sam3_ph0" / "FGADR-val" / "prob_val").is_dir()  # a pool run keeps its maps
    assert sam3_text.assemble_selected("FGADR", "test", frozen=filled, data_dir=root, pool=pool, drop_maps=("FGADR",)) == target  # idempotent
    # a full request refuses smoke phrase runs; a smoke request assembles the smoke
    smoke_root = tmp_path / "prepared" / "DiaRetDB1"
    _toy_e5_source(smoke_root, "DiaRetDB1", {"val": ["w0"], "test": ["s0", "s1"]})
    for index in (0, 1, 2):
        sam3_text.run(index, "DiaRetDB1", "test", frozen=filled, pool=pool, assemble=False, limit=1, scorer=fake, data_dir=smoke_root, long_side=20, window=8)
    with pytest.raises(ValueError, match="smoke runs"):
        sam3_text.assemble_selected("DiaRetDB1", "test", frozen=filled, data_dir=smoke_root, pool=pool)
    smoke = sam3_text.assemble_selected("DiaRetDB1", "test", frozen=filled, data_dir=smoke_root, pool=pool, limit=1)
    assert json.loads((smoke / "done.json").read_text())["limit"] == 1 and not out.is_done(smoke) and out.is_done(smoke, limit=1)
