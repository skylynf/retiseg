"""The E5 inference protocol (bench/configs/e5_frozen.yaml) and what must be filled before a test split is read.

The file holds the inference tiers' own settings (docs/design/15 sections 4-6 under the
docs/paper/16 scope, aligned 2026-10-12). Two fields are "filled from validation":
``zero_shot.grid.selected`` (the automatic-mask density and area cap, one pair per
model: SAM and SAM 2 are selected separately, review fix 7 / B1) and
``zero_shot.sam3.selected`` (one phrase per class). Both are written once from the
validation pool and must be present before any test split is read; a run on the pool
itself needs neither. The oracle tier reads one prompt file per (test set, split)
written by bench/e5/prompts.py, for the pool as well as for the test splits.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FROZEN_PATH = REPO / "bench" / "configs" / "e5_frozen.yaml"
PROMPTS_DIR = REPO / "runs" / "E5_prompts"
CLASSES = ("MA", "HE", "EX", "SE")
TIERS = ("a", "b")
ZERO_SHOT_MODELS = ("SAM", "SAM 2", "SAM 3")
GRID_MODELS = ("SAM", "SAM 2")  # the automatic-mask models: each gets its own pair of zero_shot.grid.selected
POOL_SPLIT = "val"

_REQUIRED = {
    "validation_pool": ("IDRiD", "DDR"),
    "geometry": ("G1", "G2", "main"),
    "zero_shot": ("grid", "sam", "sam2", "sam3", "pixel_score", "outputs"),
    "oracle": ("models", "checkpoints", "k", "curve_sets", "order", "box", "point", "decode", "medsam",
               "same_information_reference"),
}
_GRID_KEYS = ("points_per_side", "area_cap_fraction", "select_on", "select_by", "select_rule", "selected")


def load_frozen(path=None):
    """The protocol as a dict, validated (see ``validate``)."""
    doc = yaml.safe_load(Path(path or FROZEN_PATH).read_text())
    validate(doc)
    return doc


def validate(frozen):
    """Raise ValueError when a section or key the inference code relies on is missing or malformed."""
    for section, keys in _REQUIRED.items():
        if not isinstance(frozen.get(section), dict):
            raise ValueError(f"e5_frozen.yaml has no {section} section")
        missing = [key for key in keys if key not in frozen[section]]
        if missing:
            raise ValueError(f"e5_frozen.yaml {section} lacks {missing}")
    grid = frozen["zero_shot"]["grid"]
    missing = [key for key in _GRID_KEYS if key not in grid]
    if missing:
        raise ValueError(f"e5_frozen.yaml zero_shot.grid lacks {missing}")
    if not grid["points_per_side"] or not grid["area_cap_fraction"]:
        raise ValueError("e5_frozen.yaml zero_shot.grid lists at least one density and one area cap")
    selected = grid.get("selected") or {}
    if not isinstance(selected, dict) or any(model not in GRID_MODELS for model in selected):
        raise ValueError(f"e5_frozen.yaml zero_shot.grid.selected is a mapping of model -> pair for {GRID_MODELS}, got {selected}")
    sam3 = frozen["zero_shot"]["sam3"]
    phrases = sam3.get("phrases") or {}
    if set(phrases) != set(CLASSES) or any(len(items) != 3 for items in phrases.values()):
        raise ValueError("e5_frozen.yaml zero_shot.sam3.phrases needs exactly three phrases for each of MA, HE, EX, SE")
    if "selected" not in sam3:
        raise ValueError("e5_frozen.yaml zero_shot.sam3 lacks selected")
    for name, item in frozen["validation_pool"].items():
        if isinstance(item, dict) and item.get("split") != POOL_SPLIT:
            raise ValueError(f"validation_pool.{name} must use the {POOL_SPLIT} split, got {item.get('split')!r}")
    if not frozen["oracle"]["k"] or frozen["oracle"]["k"][-1] != "all":
        raise ValueError("e5_frozen.yaml oracle.k ends with 'all'")


def pool_sets(frozen):
    """[(name, prepared directory, split)] of the validation pool, in file order."""
    return [
        (name, REPO / item["data"], item["split"])
        for name, item in frozen["validation_pool"].items()
        if isinstance(item, dict)
    ]


def is_pool_split(split):
    return split == POOL_SPLIT


def grid_pairs(frozen):
    """Every registered (points_per_side, area_cap_fraction) pair, densities outer, caps inner."""
    grid = frozen["zero_shot"]["grid"]
    return [(int(density), float(cap)) for density in grid["points_per_side"] for cap in grid["area_cap_fraction"]]


def selected_pair(frozen, model):
    """The (density, cap) chosen for ``model`` on the pool; ValueError while it is unfilled or outside the grid."""
    if model not in GRID_MODELS:
        raise ValueError(f"the grid is selected for {GRID_MODELS}, not {model!r}")
    grid = frozen["zero_shot"]["grid"]
    chosen = (grid.get("selected") or {}).get(model) or {}
    if not chosen:
        raise ValueError(f"zero_shot.grid.selected is empty for {model}: choose the density and area cap on the validation pool first "
                         "(python -m bench.e5.amg --select)")
    try:
        pair = (int(chosen["points_per_side"]), float(chosen["area_cap_fraction"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"zero_shot.grid.selected.{model} needs points_per_side and area_cap_fraction, got {chosen}") from error
    if pair not in grid_pairs(frozen):
        raise ValueError(f"zero_shot.grid.selected.{model} {pair} is not in the registered grid {grid_pairs(frozen)}")
    return pair


def selected_pairs(frozen):
    """{model: (density, cap)} for every automatic-mask model; ValueError while one is unfilled."""
    return {model: selected_pair(frozen, model) for model in GRID_MODELS}


def phrase_ids(frozen):
    """{(cls, index): phrase}: the SAM 3 phrases, index 0-2 in the registered order."""
    return {
        (cls, index): text
        for cls, items in frozen["zero_shot"]["sam3"]["phrases"].items()
        for index, text in enumerate(items)
    }


def selected_phrases(frozen):
    """{cls: phrase} chosen on the pool; ValueError while a class is unfilled or its phrase is not registered."""
    sam3 = frozen["zero_shot"]["sam3"]
    chosen = sam3.get("selected") or {}
    missing = [cls for cls in CLASSES if cls not in chosen]
    if missing:
        raise ValueError(f"zero_shot.sam3.selected lacks {missing}: choose the phrases on the validation pool first")
    out = {}
    for cls in CLASSES:
        if chosen[cls] not in sam3["phrases"][cls]:
            raise ValueError(f"{chosen[cls]!r} is not a registered {cls} phrase {sam3['phrases'][cls]}")
        out[cls] = chosen[cls]
    return out


def k_levels(frozen):
    """The oracle tier's prompt counts, ints and the final 'all'."""
    return [k if k == "all" else int(k) for k in frozen["oracle"]["k"]]


def oracle_models(frozen):
    """{model: [prompt kinds]} of the oracle tier."""
    return {name: list(kinds) for name, kinds in frozen["oracle"]["models"].items()}


MAIN_VARIANT = "main"


def prompt_file(set_name, split, prompts_dir=None, variant=MAIN_VARIANT):
    """runs/E5_prompts/<set>_<split>.json; a comparison variant (TJDR_uwf "fov") adds __<variant>."""
    suffix = "" if variant == MAIN_VARIANT else f"__{variant}"
    return Path(prompts_dir or PROMPTS_DIR) / f"{set_name}_{split}{suffix}.json"


def assert_ready(frozen, tier, split, model=None, set_name=None, prompts_dir=None, variant=MAIN_VARIANT):
    """Raise before a run that would read a test split with an unfilled selection, or has no prompt file.

    Tier "a" on a test split needs zero_shot.grid.selected for the model (SAM,
    SAM 2; each has its own pair) and zero_shot.sam3.selected (SAM 3); ``model``
    None checks every selection. A pool run
    (split "val") needs neither. Tier "b" needs the prompt file of
    (set_name, split), for the pool as well.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    if tier == "a":
        if model is not None and model not in ZERO_SHOT_MODELS:
            raise ValueError(f"zero-shot models are {ZERO_SHOT_MODELS}, got {model!r}")
        if is_pool_split(split):
            return
        if model is None:
            selected_pairs(frozen)
        elif model in GRID_MODELS:
            selected_pair(frozen, model)
        if model in (None, "SAM 3"):
            selected_phrases(frozen)
        return
    if set_name is None:
        raise ValueError("tier b needs set_name")
    path = prompt_file(set_name, split, prompts_dir, variant)
    if not path.is_file():
        flag = "" if variant == MAIN_VARIANT else f" --variant {variant}"
        raise FileNotFoundError(f"{path} is missing; write it with: python -m bench.e5.prompts --set {set_name} --split {split}{flag}")
