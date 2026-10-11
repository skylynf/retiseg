import json

import numpy as np
import pytest
import yaml

from bench.scripts import summarize


def _metrics(maupr, image_auc=None, eleven=None):
    per_class = {c: {"aupr": maupr, "aupr_ci": [maupr - 0.05, maupr + 0.05]} for c in summarize.CLASSES}
    summary = {"mAUPR": maupr, "mAUPR_m2mrf_impl": eleven if eleven is not None else maupr + 0.02,
               "AUPR_any_lesion": maupr + 0.1, "mDice_pooled_fixed": maupr / 2}
    if image_auc is not None:
        summary["mAUC_image_max"] = image_auc
    return {"summary": summary, "per_class": per_class, "n_images": 10}


def _bs_run(runs, model, slug, train, seed, value, eleven=None):
    run = runs / f"BS_{slug}_{train.lower()}_seed{seed}"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"experiment": "BS", "model": model, "dataset": f"dataset/prepared/{train}", "seed": seed}))
    (run / "metrics_test.json").write_text(json.dumps(_metrics(value, eleven=eleven)))
    return run.name


def _e3a(runs, source, model, train, test, value, seed=0, kind="BS", variant="main", image_auc=None):
    directory = runs / "E3a" / source / (test if variant == "main" else f"{test}__{variant}")
    directory.mkdir(parents=True)
    done = {"model": model, "kind": kind, "seed": seed, "train_source": train, "test_set": test, "variant": variant}
    (directory / "done.json").write_text(json.dumps(done))
    (directory / "metrics.json").write_text(json.dumps(_metrics(value, image_auc=image_auc)))


def test_tables_and_ranking(tmp_path):
    runs = tmp_path / "runs"
    models = {"HRNet": ("hrnet", 0.7), "U-Net": ("unet", 0.5), "M2MRF": ("m2mrf", 0.6), "FCT": ("fct", 0.3)}
    for model, (slug, base) in models.items():
        for train, shift in (("IDRiD", 0.0), ("DDR", -0.2)):
            for seed in (0, 1):
                source = _bs_run(runs, model, slug, train, seed, base + shift + 0.01 * seed, eleven=base + 0.05 - 0.1 * (slug == "fct"))
                if seed:
                    continue
                for test in summarize.TEST_ORDER:
                    if test == train:
                        continue
                    _e3a(runs, source, model, train, test, base - 0.3, image_auc=base + 0.2 if test == "DiaRetDB1" else None)
                _e3a(runs, source, model, train, "TJDR_uwf", base - 0.35, variant="fov")
    out = tmp_path / "summary"
    summarize.main(["--runs", str(runs), "--out", str(out)])
    table = (out / "table_IDRiD.tsv").read_text().splitlines()
    header = table[0].split("\t")
    hrnet = dict(zip(header, [l for l in table if l.startswith("HRNet")][0].split("\t")))
    assert float(hrnet["Int"]) == pytest.approx(0.705)  # mean of seeds 0 and 1
    assert float(hrnet["TJDR_uwf__fov"]) == pytest.approx(0.35)
    assert float(hrnet["delta_fine_minus_int"]) == pytest.approx(0.4 - 0.705)
    report = json.loads((out / "ranking.json").read_text())
    ranks = report["all_seven"]["average_rank"]
    assert list(ranks)[0] == "HRNet" and list(ranks)[-1] == "FCT"
    assert report["all_seven"]["n_models"] == 4
    seed0 = tmp_path / "seed0"
    summarize.main(["--runs", str(runs), "--out", str(seed0), "--seed0-only"])
    row = [l for l in (seed0 / "table_IDRiD.tsv").read_text().splitlines() if l.startswith("HRNet")][0]
    assert row.split("\t")[1] == "0.7000"
    change = report["implementation_rank_change"]["IDRiD"]
    assert change["n_models"] == 4 and change["models_changing_rank"] >= 0


def test_e2c_table_reads_validation_history_only(tmp_path):
    runs = tmp_path / "runs"
    for name, values in (
        ("E2c_hrnet_ddr_seed0_lr1o3", [0.3, 0.5]),
        ("BS_hrnet_ddr_seed0", [0.4, 0.6]),
        ("E2c_hrnet_ddr_seed0_lr3", [0.2, 0.55]),
    ):
        (runs / name).mkdir(parents=True)
        (runs / name / "history.jsonl").write_text("".join(json.dumps({"val_maupr": v}) + "\n" for v in values))
    table = summarize.e2c_table(runs).splitlines()
    assert table[1].split("\t") == ["hrnet", "ddr", "0.5000", "0.6000", "0.5500", "x1"]
