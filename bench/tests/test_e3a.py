import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from bench import e3a
from bench.common.io import LESION_CLASSES, read_prob_quantized
from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

FROZEN = e3a.load_frozen()


def test_targets_leave_out_the_training_source_and_add_the_uwf_variant():
    pairs = e3a.targets(["IDRiD"], FROZEN)
    names = [t for t, _ in pairs]
    assert "IDRiD" not in names and "DDR" in names
    assert ("TJDR_uwf", "main") in pairs and ("TJDR_uwf", "fov") in pairs
    assert names.count("FGADR") == 1
    assert e3a.targets(["IDRiD"], FROZEN, sets=["FGADR"]) == [("FGADR", "main")]
    with pytest.raises(ValueError, match="unknown test sets"):
        e3a.targets(["IDRiD"], FROZEN, sets=["NOPE"])


def test_scale_rule_per_degree_for_uwf_and_fov_size_elsewhere():
    assert e3a.target_long_side("TJDR_uwf", "main", "IDRiD", 1440, FROZEN) == round(1440 * 133 / 50)
    assert e3a.target_long_side("TJDR_uwf", "main", "DDR", 1440, FROZEN) == round(1440 * 133 / 45)
    assert e3a.target_long_side("TJDR_uwf", "fov", "DDR", 1440, FROZEN) == 1440
    assert e3a.target_long_side("FGADR", "main", "IDRiD", 3413, FROZEN) == 3413
    with pytest.raises(ValueError):
        e3a.target_long_side("FGADR", "fov", "IDRiD", 1440, FROZEN)
    assert e3a.set_dirname("TJDR_uwf", "fov") == "TJDR_uwf__fov"


class _Card:
    def __init__(self, forward_size=None):
        self.forward_size = forward_size
        self.window_batch = 8


def test_inference_geometry():
    bs = {"experiment": "BS"}
    b1 = {"experiment": "B1"}
    whole = e3a.inference_geometry(bs, _Card(), 1440, FROZEN)
    assert whole["window"] is None
    big = e3a.inference_geometry(bs, _Card(), 4256, FROZEN)
    assert big == {"window": (1440, 1440), "overlap": 0.5, "batch": 1}
    assert e3a.inference_geometry(b1, _Card(), 3830, FROZEN)["window"] == (1440, 1440)
    assert e3a.inference_geometry(b1, _Card((224, 224)), 3830, FROZEN) is None
    swin = e3a.inference_geometry(bs, _Card((224, 224)), 4256, FROZEN)
    assert swin["window"] == (224, 224) and swin["batch"] == 8


class _Tiny(nn.Module, SegmentationWrapper):
    card = ModelCard(name="tiny", family="general", track="B", repro_level="R0", source="test", pad_multiple=8)
    classes = tuple(LESION_CLASSES)

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0]))

    def forward(self, x):
        return self.conv(x)

    def author_recipe(self, dataset):
        return AuthorRecipe(
            loss="bce_with_logits", optimizer="adam", lr=1e-3, weight_decay=0.0, schedule="none",
            iterations=0, batch_size=1, crop_size=(8, 8), pretrained="",
        )


def _toy_source(root, name, ids_by_split, shape, ext):
    rng = np.random.default_rng(len(name))
    for split, ids in ids_by_split.items():
        for image_id in ids:
            image = rng.integers(0, 255, shape + (3,), dtype=np.uint8)
            (root / "images").mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(root / "images" / f"{image_id}.{ext}")
            fov = np.zeros(shape, bool)
            fov[2:-2, 3:-3] = True
            (root / "fov").mkdir(parents=True, exist_ok=True)
            Image.fromarray(fov.astype(np.uint8) * 255).save(root / "fov" / f"{image_id}.png")
            for cls in LESION_CLASSES:
                mask = np.zeros(shape, bool)
                mask[5:9, 6:10] = True
                (root / "masks" / cls).mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255).save(root / "masks" / cls / f"{image_id}.png")
    (root / "meta.json").write_text(json.dumps({"name": name, "classes": list(LESION_CLASSES), "granularity": "fine"}))
    (root / "splits.json").write_text(json.dumps(ids_by_split))


def test_run_source_end_to_end(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    train = repo / "dataset" / "prepared" / "IDRiD"
    indep = repo / "dataset" / "prepared" / "TOY"
    _toy_source(train, "IDRiD", {"train": ["a"], "val": ["v0", "v1"], "test": ["t0"]}, (24, 30), "jpg")
    _toy_source(indep, "TOY", {"train": [], "val": [], "test": ["x0", "x1", "x2"]}, (40, 36), "png")
    frozen = dict(FROZEN)
    frozen["test_sets"] = {
        "IDRiD": {"data": "dataset/prepared/IDRiD", "tier": "fine"},
        "TOY": {"data": "dataset/prepared/TOY", "tier": "fine"},
    }
    frozen["canvas_reference"] = 32
    frozen["exclude"] = {"TOY": ["x2"]}
    frozen["comparison_variants"] = {}
    frozen["per_degree_sets"] = []
    frozen["eval_protocol"] = str(Path(e3a.REPO) / "bench" / "eval" / "protocol.yaml")
    monkeypatch.setattr(e3a, "REPO", repo)
    monkeypatch.setattr(e3a, "OUT_ROOT", repo / "runs" / "E3a")
    monkeypatch.setattr(e3a, "load_frozen", lambda path=None: frozen)
    run = repo / "runs" / "BS_tiny_idrid_seed0"
    run.mkdir(parents=True)
    config = {"experiment": "BS", "model": "tiny", "dataset": "dataset/prepared/IDRiD", "seed": 0, "fov_diameter": 32}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "checkpoint_roles.json").write_text(json.dumps({"primary": "best.pt"}))
    torch.save({"model": _Tiny().state_dict(), "model_name": "tiny"}, run / "best.pt")
    # prob_val of the training source decides the val_opt thresholds.
    from bench.common.io import write_prob

    for image_id in ("v0", "v1"):
        for cls in LESION_CLASSES:
            write_prob(run / "prob_val", cls, image_id, np.full((24, 30), 0.3))
    monkeypatch.setattr(e3a, "load_run_model", lambda config, checkpoint, device: _Tiny().eval())
    monkeypatch.chdir(repo)
    e3a.run_source(run, device="cpu")
    out = repo / "runs" / "E3a" / "BS_tiny_idrid_seed0" / "TOY"
    done = json.loads((out / "done.json").read_text())
    assert done["n_images"] == 2 and done["excluded"] == ["x2"]
    assert done["train_source"] == "IDRiD" and done["thresholds"].startswith("fixed and training-source")
    assert done["canvas_long_side"] == 32 and done["inference"]["window"] is None
    q = read_prob_quantized(out / "prob_test", "SE", "x0")
    assert q.shape == (40, 36)
    fov = np.asarray(Image.open(indep / "fov" / "x0.png")) > 0
    assert (q[~fov] == 0).all() and (q[fov] > 0).all()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["val_dataset_dir"].endswith("IDRiD") and metrics["n_images"] == 2
    # A second call finds every set done and leaves the files alone.
    stamp = (out / "done.json").stat().st_mtime_ns
    e3a.run_source(run, device="cpu")
    assert (out / "done.json").stat().st_mtime_ns == stamp


def test_nnunet_helpers(tmp_path):
    from bench.scripts import nnunet_e3a

    root = tmp_path / "S"
    _toy_source(root, "S", {"test": ["x"]}, (20, 30), "png")
    rgb, fov, factor = nnunet_e3a.scaled_image(root, "x", 52)
    # The field-of-view box is 16 x 24, so its long side 24 becomes 52.
    assert factor == pytest.approx(52 / 24)
    assert rgb.shape == (round(20 * factor), round(30 * factor), 3)
    array = nnunet_e3a.nnunet_input(rgb)
    assert array.shape == (3, 1) + rgb.shape[:2] and array.dtype == np.float32
    probs = np.zeros((5, 1) + rgb.shape[:2], np.float32)
    probs[nnunet_e3a.LABELS["MA"]] = 0.25
    probs[nnunet_e3a.LABELS["EX"]] = 0.75
    planes = nnunet_e3a.lesion_planes(probs, 20, 30, fov)
    assert planes["MA"].shape == (20, 30)
    assert np.allclose(planes["MA"][fov], 0.25) and np.allclose(planes["EX"][fov], 0.75)
    assert (planes["HE"] == 0).all() and (planes["MA"][~fov] == 0).all()


def test_drop_maps_removes_the_maps_after_the_metrics(tmp_path, monkeypatch):
    test_run_source_end_to_end(tmp_path, monkeypatch)
    out = tmp_path / "repo" / "runs" / "E3a" / "BS_tiny_idrid_seed0" / "TOY"
    (out / "done.json").unlink()
    frozen = e3a.load_frozen()
    e3a.evaluate_set("BS_tiny_idrid_seed0", "TOY", "main", frozen, drop_maps=["TOY"])
    done = json.loads((out / "done.json").read_text())
    assert done["maps_dropped"] is True and not (out / "prob_test").exists()
    assert (out / "metrics.npz").is_file()


def test_leak_exclusions_follow_the_training_sources(tmp_path):
    table = {"by_training_source": {"FGADR": {"FGADR": ["a", "b"], "Retinal-Lesions": ["r"]}, "DDR": {"FGADR": []}}}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(table))
    assert e3a.leak_exclusions(["FGADR"], "FGADR", path) == {"a", "b"}
    assert e3a.leak_exclusions(["DDR", "FGADR"], "Retinal-Lesions", path) == {"r"}
    assert e3a.leak_exclusions(["DDR"], "FGADR", path) == set()


def test_frozen_exclusions_leave_the_main_table_sources_untouched():
    table = json.loads(e3a.LEAK_EXCLUSIONS.read_text())["by_training_source"]
    for source in ("IDRiD", "DDR"):
        assert all(not ids for ids in table[source].values())
    assert len(table["FGADR"]["FGADR"]) == 106 and table["TJDR_uwf"]["TJDR_uwf"] == ["TJDR_test_021", "TJDR_test_023"]
