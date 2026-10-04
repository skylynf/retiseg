"""Stream the author's sigmoid metrics without keeping every probability map.

The sums are those in official_code/M2MRF/mmseg/core/evaluation/my_metrics.py.
Addition order does not change the totals. The curve is integrated with the
same trapezoid, including the threshold-0 and threshold-1 anchors.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import auc

_SOURCE = Path(__file__).resolve().parents[2] / "official_code/M2MRF/mmseg/core/evaluation/my_metrics.py"
_CONFUSED = None


def _confused():
    global _CONFUSED
    if _CONFUSED is None:
        src = _SOURCE.read_text().replace("dtype=np.float)", "dtype=np.float64)")
        namespace = {}
        exec(compile(src, str(_SOURCE), "exec"), namespace)
        _CONFUSED = namespace["sigmoid_confused_matrix"]
    return _CONFUSED


class AuthorAccumulator:
    def __init__(self, num_classes=5, n_thresholds=11):
        self.num_classes = num_classes
        self.threshs = np.linspace(0, 1, n_thresholds)
        self.total_p = np.zeros((n_thresholds, num_classes), dtype=np.float64)
        self.total_tp = np.zeros_like(self.total_p)
        self.total_fn = np.zeros_like(self.total_p)

    def add(self, prob, label):
        confused = _confused()
        for index, thresh in enumerate(self.threshs):
            pred, tp, fn = confused(prob, label, self.num_classes, float(thresh))
            self.total_p[index] += pred
            self.total_tp[index] += tp
            self.total_fn[index] += fn

    def scores(self):
        index = int(np.argmax(self.threshs == 0.5))
        tp = self.total_tp[index]
        pred = self.total_p[index]
        fn = self.total_fn[index]
        iou = tp / (pred + fn)
        f1 = 2 * tp / (pred + tp + fn)
        ppv = np.nan_to_num(self.total_tp / self.total_p, nan=1.0)
        sens = np.nan_to_num(self.total_tp / (self.total_tp + self.total_fn), nan=0.0)
        aupr = np.zeros(self.num_classes, dtype=np.float64)
        for cls in range(1, self.num_classes):
            aupr[cls] = auc(sens[:, cls], ppv[:, cls])
        return iou, f1, aupr
