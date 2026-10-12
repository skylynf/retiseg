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
