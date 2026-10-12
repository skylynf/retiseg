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
