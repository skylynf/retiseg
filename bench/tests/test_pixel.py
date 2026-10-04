from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from PIL import Image

from bench.common.io import PROB_LEVELS, _read_binary
from bench.eval import pixel

L = PROB_LEVELS
REPO = Path(__file__).resolve().parents[2]
M2MRF_METRICS = REPO / "official_code/M2MRF/mmseg/core/evaluation/my_metrics.py"


def _random_case(rng, n=20000, prevalence=0.02):
    gt = rng.random(n) < prevalence
    score = np.clip(rng.normal(0.25 + 0.4 * gt, 0.2), 0, 1)
    q = np.rint(score * L).astype(np.int64)
    return q, gt


@pytest.mark.parametrize("seed", range(5))
def test_average_precision_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    q, gt = _random_case(rng)
    pos, neg = pixel.histograms(q, gt, np.ones_like(gt), L)
    assert pixel.average_precision(pos, neg) == pytest.approx(average_precision_score(gt, q / L), abs=1e-12)


def test_average_precision_with_heavy_ties_matches_sklearn():
    rng = np.random.default_rng(1)
    gt = rng.random(5000) < 0.1
    q = rng.integers(0, 4, size=5000) * (L // 3)
    pos, neg = pixel.histograms(q, gt, np.ones_like(gt), L)
    assert pixel.average_precision(pos, neg) == pytest.approx(average_precision_score(gt, q / L), abs=1e-12)


def test_average_precision_no_positives_is_nan():
    pos = np.zeros(L + 1, np.int64)
    neg = np.zeros(L + 1, np.int64)
    neg[100] = 5
    assert np.isnan(pixel.average_precision(pos, neg))


def test_valid_mask_excludes_pixels():
    q = np.array([L, L, 0, 0])
    gt = np.array([True, False, True, False])
    valid = np.array([True, False, True, True])
    pos, neg = pixel.histograms(q, gt, valid, L)
    assert pos.sum() == 2 and neg.sum() == 1


def test_coarsen_preserves_counts():
    rng = np.random.default_rng(0)
    h = rng.integers(0, 10, size=L + 1)
    c = pixel.coarsen(h, 4096)
    assert c.shape == (4096,) and c.sum() == h.sum()
    assert c[0] == h[:16].sum() and c[-1] == h[-16:].sum()


def _load_m2mrf_sigmoid_metrics():
    src = M2MRF_METRICS.read_text().replace("dtype=np.float)", "dtype=np.float64)")
    ns = {}
    exec(compile(src, str(M2MRF_METRICS), "exec"), ns)
    return ns["sigmoid_metrics"]


@pytest.mark.skipif(not M2MRF_METRICS.exists(), reason="M2MRF repository not cloned")
@pytest.mark.parametrize("seed", range(3))
def test_m2mrf_replica_matches_original_code(seed):
    sigmoid_metrics = _load_m2mrf_sigmoid_metrics()
    rng = np.random.default_rng(seed)
    n_cls = 4
    results, gts = [], []
    for _ in range(3):
        gt = rng.integers(0, n_cls + 1, size=(40, 60))
        gt[rng.random(gt.shape) < 0.85] = 0
        logits = np.stack(
            [np.clip(rng.normal(0.3 + 0.4 * (gt == k), 0.25), 0, 1) for k in range(1, n_cls + 1)]
        )
        q = np.rint(logits * L).astype(np.int64)
        results.append(q / L)
        gts.append(gt)
    _, _, _, maupr = sigmoid_metrics(results, gts, n_cls + 1, compute_aupr=True)

    for k in range(1, n_cls + 1):
        pos = np.zeros(L + 1, np.int64)
        neg = np.zeros(L + 1, np.int64)
        for r, g in zip(results, gts):
            p, n = pixel.histograms(np.rint(r[k - 1] * L).astype(np.int64), g == k, np.ones(g.shape, bool), L)
            pos += p
            neg += n
        assert pixel.m2mrf_aupr(pos, neg, L, 11) == pytest.approx(maupr[k], abs=1e-12)


def test_m2mrf_anchor_inflates_low_precision_detector():
    # All positives and nine times as many negatives share one high score.
    # Exact AP is the precision at that score (0.1); the 11-threshold trapezoid
    # anchors the curve at (recall 0, precision 1) and reports 0.55.
    q = np.full(1000, int(0.95 * L))
    gt = np.zeros(1000, bool)
    gt[:100] = True
    pos, neg = pixel.histograms(q, gt, np.ones_like(gt), L)
    assert pixel.average_precision(pos, neg) == pytest.approx(0.1)
    assert pixel.m2mrf_aupr(pos, neg, L, 11) == pytest.approx(0.55)


def test_m2mrf_bias_sign_depends_on_curve_shape():
    rng = np.random.default_rng(0)
    diffs = []
    for shift in (0.2, 0.6):
        gt = rng.random(500000) < 0.001
        s = 1 / (1 + np.exp(-rng.normal(-3 + shift * 8 * gt, 1.5)))
        q = np.rint(s * L).astype(np.int64)
        pos, neg = pixel.histograms(q, gt, np.ones_like(gt), L)
        diffs.append(pixel.m2mrf_aupr(pos, neg, L, 11) - pixel.average_precision(pos, neg))
    weak, strong = diffs
    assert weak > 0 > strong


def test_threshold_level_and_confusion():
    assert pixel.level_for_threshold(0.5, L) == 32768
    assert pixel.level_for_threshold(0.0, L) == 0
    assert pixel.level_for_threshold(1.0, L) == L
    q = np.array([L, L, 0, int(0.6 * L)])
    gt = np.array([True, False, True, True])
    pos, neg = pixel.histograms(q, gt, np.ones_like(gt), L)
    tp, fp, fn = pixel.confusion_at(pos, neg, pixel.level_for_threshold(0.5, L))
    assert (tp, fp, fn) == (2, 1, 1)
    s = pixel.overlap_scores(tp, fp, fn)
    assert s["dice"] == pytest.approx(4 / 6) and s["iou"] == pytest.approx(2 / 4)


@pytest.mark.skipif(not M2MRF_METRICS.exists(), reason="M2MRF repository not cloned")
def test_streamed_author_scores_match_the_original_function():
    from bench.eval.m2mrf_stream import AuthorAccumulator

    sigmoid_metrics = _load_m2mrf_sigmoid_metrics()
    rng = np.random.default_rng(7)
    results, labels = [], []
    acc = AuthorAccumulator()
    for _ in range(2):
        label = rng.integers(0, 5, size=(30, 40))
        prob = np.clip(rng.random((4, 30, 40)), 0, 1)
        results.append((prob, True, True))
        labels.append(label)
        acc.add(prob, label)
    _, _, _, maupr = sigmoid_metrics(results, labels, 5, compute_aupr=True)
    _, _, aupr = acc.scores()
    assert aupr[1:] == pytest.approx(maupr[1:], abs=1e-12)


def test_best_dice_threshold_separable():
    q = np.array([int(0.2 * L)] * 50 + [int(0.7 * L)] * 50)
    gt = np.array([False] * 50 + [True] * 50)
    pos, neg = pixel.histograms(q, gt, np.ones_like(gt), L)
    t = pixel.best_dice_threshold(pos, neg, L)
    tp, fp, fn = pixel.confusion_at(pos, neg, pixel.level_for_threshold(t, L))
    assert (tp, fp, fn) == (50, 0, 0)


def test_rgba_mask_ignores_a_constant_alpha(tmp_path):
    image = np.zeros((4, 5, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[1, 2, 0] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(image, mode="RGBA").save(path)
    mask = _read_binary(path)
    assert mask.shape == (4, 5)
    assert int(mask.sum()) == 1
    assert bool(mask[1, 2])
