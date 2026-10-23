"""BF cells: the factor experiments E3b, E4, E7, E8 and E9 (docs/paper/16 section 3).

A BF cell trains one model with the B-std protocol (bench/configs/bs_frozen.yaml:
augmentation, patch geometry, validation spacing, checkpoint selection, inference
geometry, evaluation) and changes exactly one thing:

    E3b  the training source (TJDR_std or FGADR instead of IDRiD/DDR)
    E4   the training set (DDR plus mixed or coarse labels; Retinal-Lesions alone)
    E7   the canvas (long side 512, or 0 = native resolution, instead of 1440); the HRDecoder
         case study (docs/paper/26 N1c) is an E7 cell at the author's 2880 canvas with
         ``patch: 1920`` and ``budget_mode: equivalent_epochs``
    E8   the amount of DDR (a frozen subset) or the fine-grained sources merged
    E9   the loss terms of U-Net (loss_override: loss name and loss_params)
    REF  the reference model (docs/paper/26 section 5 item 2; 14 note 58): the two B-std models with the highest DDR
         validation mAUPR, trained on IDRiD + DDR + TJDR_std merged at the DDR pixel budget (the E8 merged-cell rule)

The training amount is the pixel budget of the reference B-std source
(``budget_reference``, DDR for every cell of 16): the frozen equivalent epochs
of that source times the canvas pixels of its training split at the frozen
diameter, so every row of a factor block sees as many patch pixels as the
main-table run it is compared with. The recipe (loss, optimizer, learning
rate, effective batch, recipe revisions) is the model's recipe for that
reference source.

Cell keys (bench/scripts/make_factor_jobs.py writes them):

    experiment: BF
    factor: E4
    model: M2MRF
    dataset: dataset/prepared/DDR         # evaluation source: split_val selects the checkpoint, split_test is scored
    split_val: val
    split_test: test
    fov_diameter: 1440                     # 0 = native resolution (E7)
    train:                                 # one or more prepared sources, optional frozen subset list
      - {dataset: dataset/prepared/DDR, split: train}
      - {dataset: dataset/prepared/FGADR, split: train, subset: bench/data/splits/subsets/FGADR_train_383.json}
    budget_reference: DDR
    loss_override: {loss: dice, loss_params: {batch_dice: true, smooth: 1.0e-5}}   # E9 only: the loss term(s) of the recipe
    patch: 1920                            # E7 case study only: a wrapper with a fixed window (HRDecoder.set_patch) trains and
                                           # infers at this window instead of its B-std one
    budget_mode: equivalent_epochs         # E7 case study only: the reference's equivalent epochs counted at THIS canvas
                                           # (default patch_pixels: the reference's pixels at the frozen diameter)
    seed: 0
    num_workers: 4
    paper: {experiment: E4, variant: "DDR + FGADR (383)", train_set: "DDR + FGADR (383)"}

Ignore regions of a training source (Retinal-Lesions) are background in the
training target and are recorded as a deviation; evaluation leaves them out as
always. Independent tests of BF checkpoints are a later addition to bench/e3a.py.
"""

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from bench import bstd
from bench.common.io import LESION_CLASSES, load_split
from bench.data.b1_input import normalize  # noqa: F401  (re-exported for the runners)
from bench.data.bs_input import PatchTrain
from bench.data.factor_input import MultiSource, PreparedSource, prepared_name
from bench.runtime import REPO, apply_matmul_precision, micro_batch_and_deviations, write_environment, write_recipe

FACTORS = ("E3b", "E4", "E7", "E8", "E9", "REF")
# 16 section 3: the models each factor uses. REF is the reference model of docs/paper/26 section 5 item 2 (14 note 58):
# the two B-std models with the highest DDR validation mAUPR, trained on the three fine-grained sources merged at the
# DDR pixel budget (the E8 merged-cell rule).
FACTOR_MODELS = {
    "E3b": ("U-Net", "M2MRF", "SAM2-UNet", "RETFound"),  # nnU-Net runs through its own bridge
    "E4": ("U-Net", "M2MRF"),
    "E7": ("M2MRF", "RETFound", "HRDecoder"),   # HRDecoder only in the 2880 case study (26 N1c)
    "E8": ("U-Net", "M2MRF"),
    "E9": ("U-Net",),
    "REF": ("RETFound", "SAMed"),
}


AUTHOR_CANVAS = {"HRDecoder": 2880}       # the author's IDRiD canvas long side (official_code/HRDecoder: 2880 x 1920)
BUDGET_MODES = ("patch_pixels", "equivalent_epochs")


def is_factor(config):
    return config.get("experiment") == "BF"


def apply_cell_geometry(model, config):
    """A cell's ``patch`` key (E7 case study) rebuilds a fixed-window wrapper at that window before training,
    scoring or an independent test, so every reader of the checkpoint sees the same geometry."""
    patch = config.get("patch") if isinstance(config, dict) else None
    if not patch:
        return model
    if not hasattr(model, "set_patch"):
        raise ValueError(f"{config.get('model')} has no set_patch; the patch key is for wrappers with a fixed window")
    model.set_patch(int(patch))
    return model


def assert_factor_cell(config, frozen):
    if config.get("factor") not in FACTORS:
        raise ValueError(f"factor must be one of {FACTORS}, got {config.get('factor')}")
    if config.get("model") not in FACTOR_MODELS[config["factor"]]:
        raise ValueError(f"{config['factor']} uses {FACTOR_MODELS[config['factor']]}, not {config.get('model')}")
    ref = config.get("budget_reference")
    if ref not in frozen["datasets"]:
        raise ValueError(f"budget_reference must be a B-std source {frozen['datasets']}, got {ref}")
    for key in ("epochs", "eval_every", "lr", "loss", "optimizer", "schedule", "lr_scale", "budget_probe"):
        if config.get(key) is not None:
            raise ValueError(f"a BF cell does not set {key}")
    train = config.get("train") or []
    if not train:
        raise ValueError("a BF cell lists at least one training source under train:")
    for item in train:
        prepared_name(REPO / item["dataset"] if not Path(item["dataset"]).is_absolute() else item["dataset"])
    prepared_name(REPO / config["dataset"] if not Path(config["dataset"]).is_absolute() else config["dataset"])
    diameter = int(config["fov_diameter"])
    if config["factor"] != "E7" and diameter != int(frozen["fov_diameter"]):
        raise ValueError(f"only E7 changes fov_diameter; {config['factor']} keeps {frozen['fov_diameter']}")
    canvases = {0, 512, int(frozen["fov_diameter"])}
    if config.get("model") == "HRDecoder":
        canvases.add(AUTHOR_CANVAS["HRDecoder"])
    if config["factor"] == "E7" and diameter not in canvases:
        raise ValueError(f"E7 canvases are 512, {frozen['fov_diameter']} or 0 (native), got {diameter}")
    if config.get("model") == "HRDecoder" and (config["factor"] != "E7" or diameter != AUTHOR_CANVAS["HRDecoder"]):
        raise ValueError("HRDecoder is a factor model only in the E7 case study at its author canvas (26 N1c)")
    if config.get("patch") is not None and config["factor"] != "E7":
        raise ValueError("patch is the E7 case study only")
    mode = config.get("budget_mode", "patch_pixels")
    if mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode is one of {BUDGET_MODES}, got {mode}")
    if mode == "equivalent_epochs" and config["factor"] != "E7":
        raise ValueError("budget_mode equivalent_epochs is the E7 case study only")
    if config["factor"] == "REF" and (len(config.get("train") or []) < 2 or config.get("budget_reference") != "DDR"):
        raise ValueError("a REF cell merges several sources at the DDR pixel budget (26 section 5 item 2: the E8 merged-cell rule)")
    if config.get("loss_override") and config["factor"] != "E9":
        raise ValueError("loss_override is the E9 factor only")
    if config.get("loss_override") and set(config["loss_override"]) - {"loss", "loss_params"}:
        raise ValueError("loss_override takes loss and loss_params only")
    if "paper" not in config:
        raise ValueError("a BF cell carries paper: {experiment, variant, train_set} for paper/scripts/collect_runs.py")


def geometry(card, frozen, diameter):
    """B-std geometry, with the training patch clipped to a smaller canvas (E7 at 512)."""
    patch, window = bstd.geometry(card, frozen)
    if diameter > 0:
        patch = (min(int(patch[0]), diameter), min(int(patch[1]), diameter))
    return patch, window


def inference_for(config, card, frozen=None):
    """What predict_image uses for a BF checkpoint: the B-std window rule."""
    frozen = frozen or bstd.load_frozen()
    _, window = geometry(card, frozen, int(config["fov_diameter"]))
    return {"window": window, "overlap": float(frozen["window_overlap"]), "batch": int(getattr(card, "window_batch", 8))}


def reference_pixels(ref, frozen, diameter=None):
    """Canvas pixels of the reference source's training split at the frozen diameter (or at ``diameter``)."""
    root = REPO / "dataset" / "prepared" / ref
    ids = load_split(root, frozen["splits"][ref]["train"])
    return bstd.canvas_pixels(root, ids, int(frozen["fov_diameter"]) if diameter is None else int(diameter))


def _abs(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def build_sources(config, mean, std):
    diameter = int(config["fov_diameter"])
    sources = []
    for item in config["train"]:
        subset = item.get("subset")
        sources.append(PreparedSource(_abs(item["dataset"]), item["split"], diameter, mean, std, subset=None if subset is None else _abs(subset)))
    return MultiSource(sources)


def has_ignore(source):
    return any((src.root / "ignore").is_dir() for src in source.sources)


def run_cell(config_path, config, model, declared, run_dir, predict_fn, diagnostic=False, resume=False, before_training=None, extra_deviations=()):
    from bench.train import run_training, score_run

    config_path = Path(config_path)
    run_dir = Path(run_dir)
    frozen = bstd.load_frozen()
    assert_factor_cell(config, frozen)
    card = model.card
    ref = config["budget_reference"]
    diameter = int(config["fov_diameter"])
    patch, window = geometry(card, frozen, diameter)
    mean, std = card.normalization["mean"], card.normalization["std"]
    source = build_sources(config, mean, std)
    val_split = PreparedSource(_abs(config["dataset"]), config["split_val"], diameter, mean, std)
    budget_mode = config.get("budget_mode", "patch_pixels")
    pixels = reference_pixels(ref, frozen, diameter if budget_mode == "equivalent_epochs" else None)
    margin = int(getattr(card, "output_margin", 0) or 0)
    resolved, deviations, per_epoch, samples, eval_every = bstd.budget(
        declared, ref, frozen, pixels, patch, 1, model_name=card.name, margin=margin
    )
    deviations = [
        f"BF {config['factor']}: trained on {source.describe()}; budget is the {ref} B-std amount "
        f"({frozen['equivalent_epochs'][ref]} equivalent epochs of {pixels} canvas pixels at diameter {frozen['fov_diameter']}); "
        f"recipe is the {ref} recipe of {card.name}"
    ] + deviations
    if diameter != int(frozen["fov_diameter"]):
        deviations.append(f"E7 canvas: long side {diameter if diameter else 'native (no resize)'} instead of {frozen['fov_diameter']}; patch {patch[0]}x{patch[1]}")
    if budget_mode == "equivalent_epochs":
        deviations.append(f"budget_mode equivalent_epochs: the {ref} equivalent epochs counted at canvas {diameter} ({pixels} pixels), "
                          f"the same step count as the B-std run with {patch[0]}x{patch[1]} patches (26 N1c), not the E7 fixed-patch-pixel rule")
    if config["factor"] == "REF":
        deviations.append("REF: the reference model of 26 section 5 item 2 (14 note 58); member chosen by DDR validation mAUPR among the B-std "
                          "models before any merged run; the DDR pixel budget over the merged sources (the E8 merged-cell rule); not a ranked row")
    if config.get("patch"):
        deviations.append(f"patch {config['patch']}: the wrapper's fixed window rebuilt at the cell's patch (set_patch); inference windows follow")
    if has_ignore(source):
        deviations.append("ignore regions of the training sources are background in the training target; evaluation leaves them out (14 note 38)")
    override = config.get("loss_override")
    if override:
        resolved = replace(resolved, loss=override.get("loss", resolved.loss), loss_params=dict(override.get("loss_params", {})))
        deviations.append(f"E9: loss {resolved.loss} with loss_params {resolved.loss_params} replace the recipe's {declared.loss} {declared.loss_params}")
    if apply_matmul_precision(model, config):
        deviations.append("matrix multiplies in TF32 (E5 decision 2026-10-12); convolutions in TF32 as in every B-std run")
    if config.get("activation_checkpointing"):
        deviations.append(model.enable_activation_checkpointing())
    micro, batch_deviations = micro_batch_and_deviations(config, resolved)
    deviations = deviations + batch_deviations + list(extra_deviations)
    train_set = PatchTrain(source, patch, samples, frozen["augment"], config["seed"])
    loader = bstd.train_loader(train_set, micro, config["num_workers"], card.pad_multiple, config["seed"])

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    write_recipe(run_dir / "recipe.json", ref, resolved, declared, samples, per_epoch, micro, deviations)
    payload = json.loads((run_dir / "recipe.json").read_text())
    payload["n_train"] = len(source.ids)
    payload["bstd"] = bstd.budget_record(ref, frozen, pixels, patch, window, per_epoch, samples, eval_every, resolved, 1)
    payload["factor"] = {
        "factor": config["factor"],
        "train": source.describe(),
        "evaluation": {"dataset": prepared_name(_abs(config["dataset"])), "val": config["split_val"], "test": config["split_test"]},
        "fov_diameter": diameter,
        "budget_reference": ref,
        "budget_mode": budget_mode,
        "patch": config.get("patch"),
        "loss_override": override or None,
        "paper": config["paper"],
    }
    (run_dir / "recipe.json").write_text(json.dumps(payload, indent=2) + "\n")

    best = None
    started = time.perf_counter()
    try:
        if before_training is not None:
            before_training()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = bstd.ValScorer(val_split, card, model.build_loss(resolved).to(device), window, frozen["window_overlap"])
        best = run_training(model, loader, None, resolved, config, run_dir, micro, per_epoch, device=device, eval_every=eval_every,
                            resume=resume, select_by=frozen["select_by"], val_scorer=scorer)
    finally:
        seconds = time.perf_counter() - started
        write_environment(run_dir / "environment.json", config["seed"], model.n_parameters(), seconds, best, best_key="best_val_maupr")
    if diagnostic:
        (run_dir / "diagnostic.json").write_text(json.dumps({"test_split_read": False, "look_at": ["history.jsonl", "recipe.json", "environment.json"]}, indent=2) + "\n")
        print("diagnostic: test split was not read", flush=True)
        return seconds
    score_run(run_dir / "config.yaml", run_dir, declared, predict_fn)
    print(f"trained in {seconds:.1f}s", flush=True)
    return seconds


def train_cell(config_path, config, run_dir, predict_fn, diagnostic=False, score_only=False, resume=False):
    """Entry for bench/train.py: build the model the shared runner's way and run the cell."""
    from bench.models.registry import build_model
    from bench.runtime import assert_not_finished_run, run_directory, set_seed
    from bench.train import score_run

    if tuple(LESION_CLASSES) != ("MA", "HE", "EX", "SE"):
        raise RuntimeError("lesion order must stay MA, HE, EX, SE")
    run_dir = Path(run_dir) if run_dir is not None else run_directory(config)
    assert_not_finished_run(run_dir)
    if score_only:
        missing = [name for name in ("best.pt", "last.pt", "recipe.json") if not (run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{run_dir} is missing {missing}; the run has not finished")
        model = apply_cell_geometry(build_model(config["model"]), config)
        score_run(config_path, run_dir, model.author_recipe(config["budget_reference"]), predict_fn)
        return 0.0
    set_seed(config["seed"])
    model = apply_cell_geometry(build_model(config["model"]), config)
    declared = model.author_recipe(config["budget_reference"])

    def _pretrained():
        if hasattr(model, "load_pretrained"):
            print(f"pretrained tensors copied: {model.load_pretrained()}", flush=True)

    return run_cell(config_path, config, model, declared, run_dir, predict_fn, diagnostic, resume, before_training=_pretrained)


def load_cell(path):
    return yaml.safe_load(Path(path).read_text())
