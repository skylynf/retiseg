"""Connected-component (lesion-level) detection metrics.

A ground-truth lesion counts as detected if any predicted-positive pixel overlaps it;
a predicted component counts as a true positive if it overlaps any ground-truth pixel.
Lesion size is the equivalent diameter divided by the FOV equivalent diameter, so that
size strata are comparable across datasets with different image resolutions.
"""

import numpy as np
from scipy import ndimage

_STRUCTURE = {4: ndimage.generate_binary_structure(2, 1), 8: ndimage.generate_binary_structure(2, 2)}


def equivalent_diameter(area):
    return 2.0 * np.sqrt(np.asarray(area, dtype=np.float64) / np.pi)


def match_lesions(pred, gt, connectivity=8):
    structure = _STRUCTURE[connectivity]
    gt_lab, n_gt = ndimage.label(gt, structure=structure)
    pr_lab, n_pr = ndimage.label(pred, structure=structure)

    gt_area = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)[1:]
    gt_hit = np.zeros(n_gt, dtype=bool)
    hit_ids = np.unique(gt_lab[pred & (gt_lab > 0)])
    gt_hit[hit_ids - 1] = True

    n_pred_tp = int(np.unique(pr_lab[gt & (pr_lab > 0)]).size)
    return {"gt_area": gt_area, "gt_detected": gt_hit, "n_pred": int(n_pr), "n_pred_tp": n_pred_tp}


class LesionAccumulator:
    def __init__(self, size_edges, size_names):
        if len(size_edges) != len(size_names) + 1:
            raise ValueError("need exactly one more size edge than size name")
        self.edges = np.asarray(size_edges, dtype=np.float64)
        self.names = list(size_names)
        self.rel_diam = []
        self.detected = []
        self.n_pred = 0
        self.n_pred_tp = 0

    def add(self, match, fov_area):
        fov_diam = equivalent_diameter(fov_area)
        self.rel_diam.append(equivalent_diameter(match["gt_area"]) / fov_diam)
        self.detected.append(match["gt_detected"])
        self.n_pred += match["n_pred"]
        self.n_pred_tp += match["n_pred_tp"]

    def result(self):
        rel = np.concatenate(self.rel_diam) if self.rel_diam else np.zeros(0)
        det = np.concatenate(self.detected) if self.detected else np.zeros(0, dtype=bool)
        n_gt = int(det.size)
        recall = det.mean() if n_gt else float("nan")
        precision = self.n_pred_tp / self.n_pred if self.n_pred else float("nan")
        if n_gt and self.n_pred:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        else:
            f1 = float("nan")
        strata = {}
        bins = np.digitize(rel, self.edges[1:-1], right=False)
        for i, name in enumerate(self.names):
            sel = bins == i
            n = int(sel.sum())
            strata[name] = {
                "n_lesions": n,
                "recall": float(det[sel].mean()) if n else float("nan"),
                "rel_diameter_range": [float(self.edges[i]), float(self.edges[i + 1])],
            }
        return {
            "n_gt_lesions": n_gt,
            "n_pred_components": self.n_pred,
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "by_size": strata,
        }
