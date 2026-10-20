"""End-to-end check of collect_runs.py against a report written by the real evaluator.

    python paper/scripts/test_collect_runs.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
for _p in [PAPER, *PAPER.parents]:
    if (_p / "bench" / "configs" / "models.yaml").exists():
        sys.path.insert(0, str(_p))
        break
sys.path.insert(0, str(PAPER / "scripts"))

import collect_runs  # noqa: E402
from bench.common import io  # noqa: E402
from bench.eval.evaluate import evaluate, load_protocol  # noqa: E402

CLASSES = ["MA", "HE", "EX", "SE"]


def disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def toy(root):
    shape, rng = (120, 160), np.random.default_rng(0)
    ids = {"train": [], "val": ["v0", "v1"], "test": ["t0", "t1", "t2"]}
    truth = {}
    for image_id in ids["val"] + ids["test"]:
        fov = disk(shape, 60, 80, 58)
        (root / "fov").mkdir(parents=True, exist_ok=True)
        Image.fromarray(fov.astype(np.uint8) * 255).save(root / "fov" / f"{image_id}.png")
        truth[image_id] = {}
        for c in CLASSES:
            m = np.zeros(shape, bool)
            for _ in range(rng.integers(1, 4)):
                m |= disk(shape, rng.integers(20, 100), rng.integers(30, 130), rng.integers(1, 6))
            m &= fov
            (root / "masks" / c).mkdir(parents=True, exist_ok=True)
            Image.fromarray(m.astype(np.uint8) * 255).save(root / "masks" / c / f"{image_id}.png")
            truth[image_id][c] = m
    (root / "meta.json").write_text(json.dumps({"name": "TOY", "classes": CLASSES, "granularity": "fine"}))
    (root / "splits.json").write_text(json.dumps(ids))
    return ids, truth


def e5_report(values, any_aupr):
    """A minimal evaluate() report: per-class AUPR and fixed / val_opt thresholds, plus the any-lesion entry."""
    per_class = {}
    for cls, v in zip(CLASSES, values):
        per_class[cls] = {"aupr": v, "aupr_m2mrf_impl": v, "aupr_ci": [v - 0.05, v + 0.05],
                          "thresholds": {"fixed": {"pooled": {"dice": v / 2}},
                                         "val_opt": {"pooled": {"dice": v / 2, "iou": v / 3}, "lesion": {"precision": v, "by_size": {"small": {"recall": v}}}}}}
    return {"summary": {"mAUPR": sum(values) / 4, "AUPR_any_lesion": any_aupr}, "any_lesion": {"aupr": any_aupr}, "per_class": per_class}


def e5_toy(runs):
    """runs/E5 as bench/e5 writes it: one directory of every kind the collector reads, plus two it must skip."""
    tree = {
        "a/a_sam_vith_p64/DDR": ({"kind": "any_lesion", "area_cap_fraction": 0.02}, [0.1, 0.2, 0.3, 0.4], 0.5),
        "a/a_sam_vith_p64/DDR__oracle_class": ({"kind": "oracle_class", "area_cap_fraction": 0.02}, [0.2, 0.3, 0.4, 0.5], 0.5),
        "a/a_sam2_hieral21_p128/IDRiD-val__cap0.02__stability": ({"kind": "stability", "area_cap_fraction": 0.02}, [0.1, 0.1, 0.1, 0.1], 0.2),
        "a/a_sam3_ph1/IDRiD": ({"kind": "text", "phrases": {"MA": "red dot", "HE": "blot hemorrhage", "EX": "yellow deposit", "SE": "soft exudate"}}, [0.3, 0.3, 0.3, 0.3], 0.4),
        "a/a_sam3_selected/IDRiD": ({"kind": "text_selected"}, [0.35, 0.35, 0.35, 0.35], 0.45),
        "b/b_sam2_hieral21_point/IDRiD__k5": ({"kind": "points", "k": 5}, [0.5, 0.6, 0.7, 0.8], 0.8),
        "b/b_medsam_vitb_box/TJDR_std__kall": ({"kind": "boxes", "k": "all"}, [0.6, 0.6, 0.6, 0.6], 0.7),
        "b/b_m2mrf_ref_ddr_seed0/DDR__kall": ({"kind": "same_information_reference"}, [0.7, 0.7, 0.7, 0.7], 0.75),
        "b/b_sam_vitb_box/TJDR_uwf__fov__k1": ({"kind": "boxes"}, [0.9, 0.9, 0.9, 0.9], 0.9),  # canvas variant: skipped
        "a/a_sam_vith_p64/FGADR": None,  # no done.json: skipped
        # the budget rule (review fix 8, N1): SAM at 128 has no TJDR_uwf main variant and its fov run says so
        # (stands_in_for_main), so it stands in (canvas 1440); SAM 2 at 64 has both, the main one (4256) is the
        # result and its fov directory is skipped; SAM at 64 lost its main run and its fov run does not stand in
        "a/a_sam_vith_p128/TJDR_uwf__fov": ({"kind": "any_lesion", "area_cap_fraction": 1.0, "canvas_long_side": 1440, "stands_in_for_main": True}, [0.2, 0.2, 0.2, 0.2], 0.3),
        "a/a_sam2_hieral21_p64/TJDR_uwf": ({"kind": "any_lesion", "area_cap_fraction": 0.02, "canvas_long_side": 4256, "stands_in_for_main": False}, [0.4, 0.4, 0.4, 0.4], 0.6),
        "a/a_sam2_hieral21_p64/TJDR_uwf__fov": ({"kind": "any_lesion", "area_cap_fraction": 0.02, "canvas_long_side": 1440, "stands_in_for_main": False}, [0.1, 0.1, 0.1, 0.1], 0.1),
        "a/a_sam_vith_p64/TJDR_uwf__fov": ({"kind": "any_lesion", "area_cap_fraction": 0.02, "canvas_long_side": 1440, "stands_in_for_main": False}, [0.9, 0.9, 0.9, 0.9], 0.9),  # skipped: the rule did not apply
    }
    for rel, spec in tree.items():
        d = runs / "E5" / rel
        d.mkdir(parents=True)
        if spec is None:
            continue
        done, values, any_aupr = spec
        (d / "metrics.json").write_text(json.dumps(e5_report(values, any_aupr)))
        (d / "done.json").write_text(json.dumps(done))


def check_e5(runs):
    e5_toy(runs)
    new, seen = collect_runs.collect_e5(runs)
    assert len(seen) == 10, seen
    keys = new[["experiment", "model", "variant", "train_set", "seed", "test_set"]].drop_duplicates()
    got = {tuple(r) for r in keys.values.tolist()}
    expected = {
        ("E5", "SAM", "zero_shot:auto_any_lesion", "none", -1, "DDR"),
        ("E5", "SAM", "zero_shot:auto_oracle_class", "none", -1, "DDR"),
        ("E5", "SAM 2", "zero_shot:auto_stability:p128:cap0.02", "none", -1, "IDRiD-val"),
        ("E5", "SAM 3", "text:selected", "none", -1, "IDRiD"),
        ("E5", "SAM 2", "oracle:point:5", "none", -1, "IDRiD"),
        ("E5", "MedSAM", "oracle:box:all", "none", -1, "TJDR-std"),
        ("E5", "M2MRF", "oracle_ref:box:all", "DDR", 0, "DDR"),
    }
    assert expected <= got, expected - got
    assert ("E5", "SAM 3", "text:red dot", "none", -1, "IDRiD") in got and ("E5", "SAM 3", "text:yellow deposit", "none", -1, "IDRiD") in got
    assert not any("fov" in k[5] or k[5] == "FGADR" for k in got)
    uwf_sam = new[(new.model == "SAM") & (new.test_set == "TJDR-uwf")]
    assert set(uwf_sam.variant) == {"zero_shot:auto_any_lesion"} and uwf_sam[uwf_sam.metric == "canvas_long_side"]["value"].iloc[0] == 1440
    assert list(uwf_sam[uwf_sam.metric == "points_per_side"]["value"]) == [128] and abs(uwf_sam[uwf_sam.metric == "aupr_any_lesion"]["value"].iloc[0] - 0.3) < 1e-12
    assert not any(abs(v - 0.9) < 1e-12 for v in uwf_sam["value"])  # the SAM p64 fov run (0.9) never stands in
    uwf_sam2 = new[(new.model == "SAM 2") & (new.test_set == "TJDR-uwf")]
    assert uwf_sam2[uwf_sam2.metric == "canvas_long_side"]["value"].iloc[0] == 4256 and abs(uwf_sam2[uwf_sam2.metric == "aupr_any_lesion"]["value"].iloc[0] - 0.6) < 1e-12
    assert len(new[(new.model == "SAM") & (new.test_set == "DDR") & (new.metric == "canvas_long_side")]) == 0  # the toy DDR done.json has no canvas: no row
    assert set(new[new.metric == "aupr"]["cls"]) <= set(CLASSES)
    any_rows = new[(new.cls == "any") & (new.metric == collect_runs.ANY_LESION_METRIC)]
    assert len(any_rows) == 9 and abs(any_rows[any_rows.model == "SAM 2"]["value"].max() - 0.8) < 1e-12  # no any-lesion row for a phrase run
    extra = new[(new.model == "SAM") & (new.test_set == "DDR") & (new.cls == "")]
    assert set(extra.metric) == {"points_per_side", "area_cap_fraction"} and extra[extra.metric == "points_per_side"]["value"].iloc[0] == 64
    phrase_rows = new[(new.model == "SAM 3") & (new.variant == "text:red dot")]
    assert set(phrase_rows.cls) == {"MA"}
    assert "dice_valopt" in set(new.metric) and "lesion_recall_small" in set(new.metric)
    print(f"ok: {len(new)} E5 rows from {len(seen)} inference-tier directories")


def check_insource(runs, report_path):
    """runs/BF_<cell>/insource_<set>/metrics.json (17 section 4 item 27) becomes a row of the cell's variant on that set."""
    import yaml

    cell = runs / "BF_e8_unet_fine_seed0"
    (cell / "insource_TJDR_std").mkdir(parents=True)
    (cell / "config.yaml").write_text(yaml.safe_dump({
        "experiment": "BF", "factor": "E8", "model": "U-Net", "dataset": "dataset/prepared/DDR", "seed": 0,
        "train": [{"dataset": "dataset/prepared/IDRiD", "split": "train"}, {"dataset": "dataset/prepared/DDR", "split": "train"}, {"dataset": "dataset/prepared/TJDR_std", "split": "train"}],
        "paper": {"experiment": "E8", "variant": "size:fine merged", "train_set": "DDR"}}))
    shutil.copyfile(report_path, cell / "metrics_test.json")
    shutil.copyfile(report_path, cell / "insource_TJDR_std" / "metrics.json")
    new, seen = collect_runs.collect(runs)
    names = sorted(n for r, n, _ in seen if r == "BF_e8_unet_fine_seed0")
    assert names == ["insource_TJDR_std/metrics.json", "metrics_test.json"], names
    inside = new[(new.experiment == "E8") & (new.variant == "size:fine merged") & (new.test_set == "TJDR-std")]
    assert len(inside) and (inside.model == "U-Net").all() and (inside.train_set == "DDR").all()
    shutil.rmtree(cell)
    print("ok: an in-source test of a merged-source cell is read as that variant on its own set")


def check_references():
    """The factor blocks' reference rows are copies of the measured main-table / E3b rows (docs/ops/21 item 68)."""
    cols = ["experiment", "model", "variant", "train_set", "seed", "test_set", "cls", "metric", "value", "source"]
    rows = [
        ("E2", "U-Net", "bstd", "DDR", 0, "DDR", "MA", "aupr", 0.15, "measured"),
        ("E2", "U-Net", "bstd", "DDR", 0, "DDR", "MA", "lesion_recall_small", 0.25, "measured"),
        ("E2", "U-Net", "bstd", "DDR", 1, "DDR", "MA", "aupr", 0.99, "measured"),          # seed 1 is not the reference
        ("E2", "U-Net", "bstd", "DDR", 0, "IDRiD", "MA", "aupr", 0.40, "measured"),
        ("E3b", "U-Net", "bstd", "FGADR", 0, "IDRiD", "MA", "aupr", 0.36, "measured"),
        ("E8", "U-Net", "size:DDR 100%", "DDR", 0, "DDR", "MA", "aupr", 0.5, "placeholder"),
        ("E8", "U-Net", "size:DDR 10%", "DDR", 0, "DDR", "MA", "aupr", 0.3, "measured"),
        ("E4", "U-Net", "DDR", "DDR", 0, "IDRiD", "MA", "aupr", 0.5, "placeholder"),
        ("E4", "U-Net", "FGADR", "FGADR", 0, "IDRiD", "MA", "aupr", 0.5, "placeholder"),
        ("E7", "M2MRF", "res:1440", "DDR", 0, "DDR", "mean", "lesion_recall_small", 0.5, "placeholder"),  # no measured M2MRF: left alone
    ]
    table = pd.DataFrame(rows, columns=cols)
    out, n = collect_runs.derive_references(table)
    e8 = out[(out.experiment == "E8") & (out.variant == "size:DDR 100%")]
    assert set(e8.metric) == {"aupr", "lesion_recall_small"} and (e8.source == "measured").all() and set(e8.seed) == {0}
    assert float(e8[e8.metric == "aupr"].value.iloc[0]) == 0.15, "seed 0 of the main run, not seed 1"
    assert float(out[(out.experiment == "E4") & (out.variant == "DDR")].value.iloc[0]) == 0.40
    assert float(out[(out.experiment == "E4") & (out.variant == "FGADR")].value.iloc[0]) == 0.36
    e7 = out[(out.experiment == "E7")]
    assert len(e7) == 1 and e7.source.iloc[0] == "placeholder", "a reference without a measured source stays a placeholder"
    assert n == 4 and float(out[(out.experiment == "E8") & (out.variant == "size:DDR 10%")].value.iloc[0]) == 0.3
    print("ok: factor reference rows are copied from the measured seed-0 source rows")


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        ids, truth = toy(tmp / "TOY")
        rng = np.random.default_rng(1)
        run = tmp / "runs" / "BS_unet_ddr_seed0"
        for split, name in (("val", "prob_val"), ("test", "prob_test")):
            for image_id in ids[split]:
                for c in CLASSES:
                    p = np.clip(truth[image_id][c] * 0.8 + 0.05 + rng.normal(0, 0.05, truth[image_id][c].shape), 0, 1)
                    io.write_prob(run / name, c, image_id, p)
        proto = load_protocol()
        report = evaluate(tmp / "TOY", "test", run / "prob_test", proto, val_pred_dir=run / "prob_val", out_prefix=run / "metrics_test")
        (tmp / "runs" / "BSprobe_unet_ddr_seed0").mkdir()  # must be ignored
        new, seen = collect_runs.collect(tmp / "runs")
        assert len(seen) == 1 and seen[0][0] == "BS_unet_ddr_seed0", seen
        assert set(new["cls"]) == set(CLASSES) | {"any"}
        # metric "aupr" is the four classes and nothing else: their mean is the evaluator's mAUPR (the main-table number)
        aupr = new[new.metric == "aupr"]
        assert set(aupr["cls"]) == set(CLASSES) and abs(aupr["value"].mean() - report["summary"]["mAUPR"]) < 1e-12
        any_row = new[new.metric == collect_runs.ANY_LESION_METRIC]
        assert list(any_row["cls"]) == ["any"] and abs(any_row["value"].iloc[0] - report["summary"]["AUPR_any_lesion"]) < 1e-12
        assert (new[["experiment", "model", "variant", "train_set", "seed", "test_set"]].drop_duplicates().values.tolist()
                == [["E2", "U-Net", "bstd", "DDR", 0, "DDR"]])
        got = new[(new.cls == "MA") & (new.metric == "aupr")]["value"].iloc[0]
        assert abs(got - report["per_class"]["MA"]["aupr"]) < 1e-12
        needed = {"aupr", "aupr_11pt", "aupr_ci_lo", "aupr_ci_hi", "dice_fixed", "dice_valopt", "iou_valopt", "lesion_precision",
                  "lesion_recall_small", "lesion_recall_medium", "lesion_recall_large", "img_auc", "img_auc_top100"}
        assert needed <= set(new["metric"]), needed - set(new["metric"])
        table = pd.read_csv(PAPER / "data" / "results.csv", keep_default_na=False)
        # the test starts from a placeholder-only table so that the merge arithmetic is checked
        table = table[(table.experiment != "E2") | (table.source != "measured")]
        merged, replaced, added, stale = collect_runs.merge(table, new)
        key = (merged.experiment == "E2") & (merged.model == "U-Net") & (merged.train_set == "DDR") & (merged.seed == 0) & (merged.test_set == "DDR")
        cell = merged[key & merged.metric.isin(needed | {collect_runs.ANY_LESION_METRIC})]
        assert (cell["source"] == "measured").all() and len(cell) == len(new), (len(cell), len(new))
        assert replaced + added == len(new), (replaced, added)
        # placeholder seeds of the measured (U-Net, DDR, DDR) group are dropped; other models keep theirs
        other = merged[(merged.experiment == "E2") & (merged.model == "U-Net") & (merged.variant == "bstd") & (merged.train_set == "DDR") & (merged.test_set == "DDR")]
        assert (other["source"] == "measured").all() and set(other["seed"]) == {0}, set(other["seed"])
        assert len(merged) == len(table) - replaced - stale + len(new)
        assert new[new.metric == "lesion_recall_small"]["value"].isna().all()  # toy lesions are all large
        # nnU-Net run names: main-table sources are E2, the E3b training sources TJDR_std and FGADR are E3b
        assert collect_runs.nn_key(collect_runs.NN.match("NN_default_ddr_fold0_rep2")) == ("E2", "bstd", "nnU-Net", "DDR", 2)
        assert collect_runs.nn_key(collect_runs.NN.match("NN_default_fgadr_fold0")) == ("E3b", "bstd", "nnU-Net", "FGADR", 0)
        assert collect_runs.nn_key(collect_runs.NN.match("NN_resencm_tjdr_std_fold0")) == ("E3b", "bstd", "nnU-Net ResEnc-M", "TJDR-std", 0)
        print(f"ok: {len(new)} rows from one report replaced {replaced} placeholder rows; nothing else changed")
        check_insource(tmp / "runs", run / "metrics_test.json")
        check_e5(tmp / "runs")
        check_references()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
