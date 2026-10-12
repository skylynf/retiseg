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
