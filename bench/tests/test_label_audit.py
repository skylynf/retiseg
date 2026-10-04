import numpy as np
import torch

from bench.common.io import LESION_CLASSES
from bench.common.labels import exclusive_label, overwrite_removal_counts
from bench.data.b1_input import map_cropped_ids, resize_nearest_ids
from bench.eval.label_audit import component_fate
from bench.eval.evaluate import load_protocol


def test_resize_ids_match_a_single_component_mask():
    ids = np.zeros((8, 10), dtype=np.int32)
    ids[1, 1] = 7
    resized = resize_nearest_ids(ids, 4, 5)
    assert resized.dtype == np.int32 or resized.dtype == np.int64
    assert set(np.unique(resized).tolist()) <= {0, 7}


def test_nearest_resize_drops_one_of_two_corner_components():
    fov = np.ones((5, 5), dtype=bool)
    mask = np.zeros_like(fov)
    mask[0, 0] = True
    mask[4, 4] = True
    count, disappeared, resized = component_fate(mask, fov, 1, None)
    assert count == 2 and disappeared == 2
    assert int(resized.sum()) == 0


def test_overwrite_removes_earlier_classes_and_keeps_microaneurysms():
    target = torch.zeros(1, 4, 2, 2)
    target[0, 2, 0, 0] = 1  # EX
    target[0, 0, 0, 0] = 1  # MA on the same pixel
    target[0, 1, 0, 1] = 1  # HE alone
    counts = overwrite_removal_counts(target)
    assert counts["MA"]["removed"] == 0
    assert counts["MA"]["positive"] == 1
    assert counts["EX"]["positive"] == 1
    assert counts["EX"]["removed"] == 1
    assert counts["HE"]["removed"] == 0
    label = exclusive_label(target, (0, 1, 2, 3))
    assert label[0, 0, 0].item() == 0  # MA channel when class_index is identity
    assert counts["EX"]["removed"] + (counts["EX"]["positive"] - counts["EX"]["removed"]) == counts["EX"]["positive"]


def test_map_cropped_ids_uses_the_long_side_then_the_square():
    labeled = np.zeros((20, 40), dtype=np.int32)
    labeled[0, 0] = 1
    mapped = map_cropped_ids(labeled, 10, (4, 4))
    assert mapped.shape == (4, 4)


def test_protocol_evaluates_original_four_class_labels():
    proto = load_protocol()
    assert proto["label_policy"] == "multilabel"
    assert proto["evaluation_labels"] == "original_four_class"
    assert proto["missing_mask_file"] == "empty_negative"
    assert proto["lesion_match"] == "many_to_many_any_overlap"
    assert proto["negative_image"] == "no_ground_truth_component"
    assert proto["bootstrap"]["unit"] == "test_image"
    assert proto["bootstrap"]["empty_positive"] == "drop"
    assert proto["classes"] == list(LESION_CLASSES)
