"""bench/scripts/ensemble.py on a toy training source with three members (docs/paper/14 M15, 26 N2).

No torch, no real data: a 120 x 160 toy source with val and test splits, three member run directories
holding prob_val / prob_test maps, and one "independent" set under a toy E3a root. The frozen E3a file is
replaced by a toy one so the test never reads the real prepared directories.
"""

import json

import numpy as np
import pytest
import yaml
from PIL import Image

from bench import e3a
from bench.common import io
from bench.scripts import ensemble as ens


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _save_bin(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8) * 255).save(path)


def _source(root, name, ids, seed):
    """A toy prepared directory with MA/HE/EX masks; returns {id: {cls: mask}}."""
    shape = (120, 160)
    rng = np.random.default_rng(seed)
    truth = {}
    for split, split_ids in ids.items():
        for image_id in split_ids:
            fov = _disk(shape, 60, 80, 58)
            _save_bin(root / "fov" / f"{image_id}.png", fov)
            masks = {}
            for c in ("MA", "HE", "EX"):
                m = np.zeros(shape, bool)
                for _ in range(rng.integers(1, 4)):
                    m |= _disk(shape, rng.integers(20, 100), rng.integers(30, 130), rng.integers(1, 6))
                masks[c] = m & fov
                _save_bin(root / "masks" / c / f"{image_id}.png", masks[c])
            truth[image_id] = masks
    (root / "meta.json").write_text(json.dumps({"name": name, "classes": ["MA", "HE", "EX"], "granularity": "fine"}))
    (root / "splits.json").write_text(json.dumps(ids))
    return truth


def _maps(pred_dir, truth, noise, seed):
    rng = np.random.default_rng(seed)
    for image_id, masks in truth.items():
        for c, m in masks.items():
            p = np.clip(m.astype(float) * 0.8 + 0.1 + rng.normal(0, noise, m.shape), 0, 1)
            io.write_prob(pred_dir, c, image_id, p)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    data = tmp_path / "dataset" / "prepared"
    ids_train = {"train": ["a0"], "val": ["v0", "v1"], "test": ["t0", "t1", "t2"]}
    ids_other = {"train": [], "val": [], "test": ["o0", "o1"]}
    truth_ddr = _source(data / "DDR", "DDR", ids_train, 1)
    truth_other = _source(data / "TJDR_std", "TJDR_std", ids_other, 2)
    frozen = {
        "experiment": "E3a", "split": "test", "eval_protocol": "bench/eval/protocol.yaml",
        "test_sets": {"DDR": {"data": "dataset/prepared/DDR", "tier": "fine"}, "TJDR_std": {"data": "dataset/prepared/TJDR_std", "tier": "fine"}},
        "train_sources": ["DDR"], "canvas_reference": 1440,
    }
    frozen_path = tmp_path / "e3a_frozen.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen))
    # the toy E3a file and the toy dataset root replace the real ones
    monkeypatch.setattr(e3a, "REPO", tmp_path)
    monkeypatch.setattr(e3a, "FROZEN_PATH", frozen_path)
    monkeypatch.setattr(e3a, "LEAK_EXCLUSIONS", tmp_path / "none.json")
    monkeypatch.setattr(e3a, "load_frozen", lambda path=frozen_path: yaml.safe_load(frozen_path.read_text()))
    monkeypatch.setattr(ens, "REPO", tmp_path)
    runs = tmp_path / "runs"
    members = [("BS_unet_ddr_seed0", 0.35), ("BS_hrnet_ddr_seed0", 0.2), ("NN_default_ddr_fold0", 0.5)]
    for i, (name, noise) in enumerate(members):
        run = runs / name
        _maps(run / "prob_val", {k: truth_ddr[k] for k in ids_train["val"]}, noise, 10 + i)
        _maps(run / "prob_test", {k: truth_ddr[k] for k in ids_train["test"]}, noise, 20 + i)
        _maps(runs / "E3a" / name / "TJDR_std" / "prob_test", truth_other, noise, 30 + i)
        (run / "metrics_test.json").write_text("{}")
    # the two B-std members carry the number their checkpoint was selected with (the registered rule); nnU-Net has none
    (runs / "BS_unet_ddr_seed0" / "environment.json").write_text(json.dumps({"best_val_maupr": 0.55}))
    (runs / "BS_hrnet_ddr_seed0" / "history.jsonl").write_text("\n".join(json.dumps({"iteration": k, "val_maupr": v}) for k, v in ((1, 0.4), (2, 0.62), (3, 0.6))) + "\n")
    # the real protocol file is read through e3a.REPO: copy it next to the toy repo root
    real_proto = e3a.__file__.rsplit("/bench/", 1)[0] + "/bench/eval/protocol.yaml"
    (tmp_path / "bench" / "eval").mkdir(parents=True)
    (tmp_path / "bench" / "eval" / "protocol.yaml").write_text(open(real_proto).read())
    return tmp_path, runs, truth_ddr


def test_members_are_ranked_by_validation_maupr_and_compositions_follow_the_registered_rules(toy):
    tmp_path, runs, _truth = toy
    table = ens.members_table("DDR", runs_root=runs)
    assert set(table) == {"U-Net", "HRNet-W48", "nnU-Net"}
    assert table["U-Net"]["val_maupr"] == 0.55 and table["HRNet-W48"]["val_maupr"] == 0.62, "the checkpoint-selection numbers, as registered"
    assert table["nnU-Net"]["source"].startswith("bench.eval") and 0 < table["nnU-Net"]["val_maupr"] < 1, "no selection number: scored on prob_val"
    assert table["HRNet-W48"]["val_maupr"] > table["U-Net"]["val_maupr"] > table["nnU-Net"]["val_maupr"]
    assert (runs / "ENS" / "DDR" / "members.json").is_file()
    assert ens.compose("all", table) == ["U-Net", "HRNet-W48", "nnU-Net"]            # MEMBERS order, not rank order
    assert ens.compose("top3", table) == ["U-Net", "HRNet-W48", "nnU-Net"]
    assert ens.compose("fam:general", table) == ["U-Net", "HRNet-W48"]
    assert ens.compose("1perfam", table) == ["HRNet-W48", "nnU-Net"]                 # best general, the only auto member
    with pytest.raises(ValueError):
        ens.compose("fam:fundus", table)                                              # no member: not an ensemble
    with pytest.raises(ValueError):
        ens.compose("top9", table)
    # a model scored after the registration is not a member even when a run and a cached row exist (21 item 88)
    assert "MedSAM (full FT)" not in ens.REGISTERED_MEMBERS and len(ens.REGISTERED_MEMBERS) == 18
    path = runs / "ENS" / "DDR" / "members.json"
    blob = json.loads(path.read_text())
    blob["members"]["MedSAM (full FT)"] = {"family": "foundation", "run": "BS_medsam_ddr_seed0", "val_maupr": 0.9, "source": "x", "seconds": 0.0}
    path.write_text(json.dumps(blob))
    table = ens.members_table("DDR", runs_root=runs)
    assert "MedSAM (full FT)" not in table and "MedSAM (full FT)" not in json.loads(path.read_text())["members"]
    assert ens.compose("top3", table) == ["U-Net", "HRNet-W48", "nnU-Net"]


def test_run_averages_the_maps_scores_both_sets_and_records_the_members(toy):
    tmp_path, runs, truth = toy
    results = ens.run("DDR", "top3", runs_root=runs)
    assert set(results) == {"DDR", "TJDR_std"}
    out = runs / "ENS" / "DDR" / "top3"
    assert (out / "prob_val" / "done.json").is_file() and (out / "DDR" / "metrics.json").is_file()
    done = json.loads((out / "TJDR_std" / "done.json").read_text())
    assert done["members"] == ["U-Net", "HRNet-W48", "nnU-Net"] and done["n_images"] == 2 and done["experiment"] == "ENS"
    # the ensemble map is the mean of the member maps (quantization aside)
    q = [io.read_prob_quantized(runs / m / "prob_test", "MA", "t0") for m in ("BS_unet_ddr_seed0", "BS_hrnet_ddr_seed0", "NN_default_ddr_fold0")]
    e = io.read_prob_quantized(out / "DDR" / "prob_test", "MA", "t0")
    assert np.abs(e - np.mean(q, axis=0)).max() <= 1
    # scoring is the ordinary protocol: a per-class report with val_opt thresholds from the ensemble's own prob_val
    report = json.loads((out / "DDR" / "metrics.json").read_text())
    assert "val_opt" in report["per_class"]["MA"]["thresholds"] and 0 <= report["summary"]["mAUPR"] <= 1
    # a second run is a no-op
    assert ens.run("DDR", "top3", runs_root=runs) == {}


def test_missing_member_maps_are_reported_not_averaged(toy):
    tmp_path, runs, _truth = toy
    import shutil

    shutil.rmtree(runs / "E3a" / "BS_hrnet_ddr_seed0" / "TJDR_std" / "prob_test")
    with pytest.raises(FileNotFoundError, match="regenerate"):
        ens.run("DDR", "fam:general", sets=["TJDR_std"], runs_root=runs)
    assert not (runs / "ENS" / "DDR" / "fam_general" / "TJDR_std" / "done.json").exists()
