"""bench/scripts/maupr_ci.py: the mAUPR interval from the stored per-image histograms, on the same resamples
as the per-class intervals (27 section 6 item 2)."""

import json

import numpy as np

from bench.eval import stats
from bench.scripts import maupr_ci as mc


def _histograms(rng, n_images, n_bins, scale):
    pos = rng.integers(0, 50, size=(n_images, n_bins)).astype(np.float64)
    neg = rng.integers(0, 500, size=(n_images, n_bins)).astype(np.float64)
    pos[:, -n_bins // 4:] *= scale   # the model puts positives at high scores
    return pos, neg


def _report(tmp_path, name="metrics_test", empty_class=None, with_ci=False):
    rng = np.random.default_rng(1)
    arrays = {"image_ids": np.array([f"i{k}" for k in range(12)])}
    per_class = {}
    for cls, scale in zip(mc.CLASSES, (3, 6, 9, 2)):
        pos, neg = _histograms(rng, 12, 64, scale)
        if cls == empty_class:
            pos[:] = 0
        arrays[f"{cls}_pos"], arrays[f"{cls}_neg"] = pos, neg
        ci = [None, None] if cls == empty_class else list(stats.bootstrap_aupr_ci(pos, neg, n_boot=200, seed=0, ci=0.95))
        per_class[cls] = {"aupr": float(stats.average_precision(pos.sum(0), neg.sum(0))) if cls != empty_class else None, "aupr_ci": ci}
    report = {"summary": {"mAUPR": 0.5}, "per_class": per_class, "protocol": {"bootstrap": {"n": 200, "seed": 0, "ci": 0.95, "unit": "test_image"}}}
    if with_ci:
        report["summary"]["mAUPR_ci"] = [0.1, 0.2]
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report))
    np.savez_compressed(tmp_path / f"{name}.npz", **arrays)
    return path, path.with_suffix(".npz"), arrays


def test_interval_uses_the_shared_resamples_and_reproduces_the_per_class_intervals(tmp_path):
    path, npz, arrays = _report(tmp_path)
    boot = {"n": 200, "seed": 0, "ci": 0.95}
    low, high, n_kept, per_class = mc.maupr_interval(arrays, boot)
    stored = json.loads(path.read_text())["per_class"]
    for cls in mc.CLASSES:
        assert np.allclose(per_class[cls], stored[cls]["aupr_ci"])
    point = np.mean([stored[cls]["aupr"] for cls in mc.CLASSES])
    assert low < point < high and n_kept == 200
    # the interval of a mean is narrower than the spread of the class intervals around it
    assert high - low < max(stored[cls]["aupr_ci"][1] - stored[cls]["aupr_ci"][0] for cls in mc.CLASSES)
    # a direct recomputation: the replicate means, from the same weights
    w = stats._bootstrap_weights(12, 200, 0)
    means = np.mean([stats._boot_ap(w, arrays[f"{cls}_pos"], arrays[f"{cls}_neg"]) for cls in mc.CLASSES], axis=0)
    assert np.isclose(low, np.quantile(means, 0.025)) and np.isclose(high, np.quantile(means, 0.975))


def test_process_writes_the_interval_skips_a_done_report_and_refuses_a_mismatch(tmp_path):
    path, npz, _arrays = _report(tmp_path)
    assert mc.process(path, npz) == "written"
    summary = json.loads(path.read_text())["summary"]
    assert len(summary["mAUPR_ci"]) == 2 and summary["mAUPR_ci"][0] < summary["mAUPR_ci"][1]
    assert summary["mAUPR_ci_n_replicates"] == 200
    assert mc.process(path, npz) == "skipped"
    assert mc.process(path, npz, force=True) == "written"
    report = json.loads(path.read_text())
    report["per_class"]["MA"]["aupr_ci"] = [0.0, 1.0]
    path.write_text(json.dumps(report))
    assert mc.process(path, npz, force=True) == "mismatch"


def test_a_class_without_positives_drops_every_replicate(tmp_path):
    path, npz, arrays = _report(tmp_path, empty_class="SE")
    low, high, n_kept, per_class = mc.maupr_interval(arrays, {"n": 200, "seed": 0, "ci": 0.95})
    assert np.isnan(low) and np.isnan(high) and n_kept == 0
    assert all(np.isnan(v) for v in per_class["SE"])
    assert mc.process(path, npz) == "written"
    assert json.loads(path.read_text())["summary"]["mAUPR_ci"] == [None, None]


def test_reports_finds_every_metrics_file_with_histograms(tmp_path):
    _report(tmp_path / "runs" / "BS_x", "metrics_test")
    _report(tmp_path / "runs" / "E3a" / "BS_x" / "IDRiD", "metrics")
    (tmp_path / "runs" / "BS_y").mkdir(parents=True)
    (tmp_path / "runs" / "BS_y" / "metrics_test.json").write_text("{}")   # no npz
    found = {str(p.relative_to(tmp_path / "runs")) for p, _npz in mc.reports(tmp_path / "runs")}
    assert found == {"BS_x/metrics_test.json", "E3a/BS_x/IDRiD/metrics.json"}
