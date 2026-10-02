"""Checks that do not need a fundus dataset."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clcnet.assumptions import (
    ASSUMPTIONS,
    BATCH_SIZE,
    CLASSIFICATION_CLASSES,
    DECODER_CHANNELS,
    EPOCHS,
    LR_INITIAL,
    SEGMENTATION_WEIGHTS,
    SEED,
    UP_IN_CHANNELS,
    learning_rate,
)
from clcnet.augment import _sampling_matrix
from clcnet.geometry import center_crop, context_origin, crop_with_pad, window_starts
from clcnet.labels import exclusive_label, presence_vector
from clcnet.losses import downsample_label
from clcnet.metrics import average_precision, dice_iou
from clcnet.preprocess import field_of_view, otsu_threshold
from clcnet.splits import eophtha_split


class LockedTests(unittest.TestCase):
    def test_learning_rate_steps(self):
        self.assertEqual(learning_rate(1), LR_INITIAL)
        self.assertAlmostEqual(learning_rate(19), 3e-4)
        self.assertAlmostEqual(learning_rate(20), 3e-5)
        self.assertAlmostEqual(learning_rate(49), 3e-5)
        self.assertAlmostEqual(learning_rate(50), 3e-6)
        self.assertAlmostEqual(learning_rate(60), 3e-6)
        self.assertEqual(EPOCHS, 60)
        self.assertEqual(BATCH_SIZE, 16)
        self.assertEqual(SEED, 20230108)

    def test_decoder_channels(self):
        self.assertEqual(UP_IN_CHANNELS, (2048, 2048, 1536, 1024))
        self.assertEqual(DECODER_CHANNELS, (2048, 1536, 1024, 576))
        for channels in list(UP_IN_CHANNELS) + list(DECODER_CHANNELS):
            self.assertEqual(channels % 32, 0)
            self.assertEqual((channels // 2) % 32, 0)

    def test_weights_and_label_order(self):
        self.assertEqual(SEGMENTATION_WEIGHTS, (1.0, 2.0, 2.0, 2.0, 2.0))
        self.assertEqual(CLASSIFICATION_CLASSES, ("MA", "EX", "HE", "SE"))

    def test_context_is_centered_on_local(self):
        self.assertEqual(context_origin(200, 300), (72, 172))
        image = np.arange(600 * 700, dtype=np.uint8).reshape(600, 700)
        local = crop_with_pad(image, 200, 300, 256)
        context = crop_with_pad(image, *context_origin(200, 300), 512)
        self.assertTrue(np.array_equal(center_crop(context, 256), local))

    def test_border_window_is_included(self):
        self.assertEqual(window_starts(256, 256, 128), [0])
        self.assertEqual(window_starts(400, 256, 128), [0, 128, 144])

    def test_overlap_priority_and_presence(self):
        shape = (4, 4)
        planes = {name: np.zeros(shape, dtype=np.uint8) for name in ("MA", "HE", "EX", "SE")}
        planes["HE"][1, 1] = 1
        planes["MA"][1, 1] = 1
        planes["EX"][2, 2] = 1
        label = exclusive_label(planes)
        self.assertEqual(int(label[1, 1]), 2)
        self.assertEqual(int(label[2, 2]), 3)
        vector = presence_vector(planes)
        self.assertTrue(np.array_equal(vector, np.array([1, 1, 1, 0], dtype=np.float32)))

    def test_context_label_keeps_a_single_pixel_lesion(self):
        import torch

        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 1, 1] = 2
        reduced = downsample_label(target)
        self.assertEqual(tuple(reduced.shape), (1, 2, 2))
        self.assertEqual(int(reduced[0, 0, 0]), 2)

    def test_otsu_disk(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:64, :64]
        disk = (yy - 32) ** 2 + (xx - 32) ** 2 <= 18**2
        image[disk] = 255
        fov = field_of_view(image)
        self.assertGreater(fov.sum(), 100)
        self.assertFalse(fov[0, 0])
        self.assertTrue(fov[32, 32])
        gray = np.zeros(32, dtype=np.uint8)
        gray[:16] = 0
        gray[16:] = 255
        self.assertGreaterEqual(otsu_threshold(gray), 0)

    def test_identity_warp_and_shift_direction(self):
        matrix, offset = _sampling_matrix(32, 32, angle_deg=0, scale=1, shift_y=0, shift_x=0)
        self.assertTrue(np.allclose(matrix, np.eye(2)))
        self.assertTrue(np.allclose(offset, 0))
        matrix, offset = _sampling_matrix(32, 32, angle_deg=0, scale=1, shift_y=0, shift_x=4)
        point = matrix @ np.array([10.0, 20.0]) + offset
        self.assertAlmostEqual(point[1], 16.0)

    def test_eophtha_split_is_stable(self):
        stems = [f"case_{index:02d}" for index in range(21)]
        first_train, first_test = eophtha_split(stems)
        second_train, second_test = eophtha_split(stems)
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_test, second_test)
        self.assertEqual(len(first_train), 15)
        self.assertEqual(len(first_test), 6)
        self.assertEqual(set(first_train) & set(first_test), set())
        with self.assertRaises(ValueError):
            eophtha_split(stems[:10])

    def test_average_precision_perfect_ranking(self):
        truth = np.array([1, 0, 1, 0])
        score = np.array([0.9, 0.1, 0.8, 0.2])
        self.assertAlmostEqual(average_precision(truth, score), 1.0)
        self.assertTrue(np.isnan(average_precision(np.zeros(4), np.zeros(4))))
        self.assertEqual(dice_iou(np.zeros(4), np.zeros(4)), (1.0, 1.0))

    def test_design_lists_every_assumption(self):
        text = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
        for item in ASSUMPTIONS:
            self.assertIn(item["id"], text)

    def test_manifest_roundtrip(self):
        from clcnet.discover import load_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.png"
            mask = root / "a_ma.png"
            _write_png(image, 8)
            _write_png(mask, 8)
            (root / "manifest.csv").write_text(
                "id,split,image,HE,MA,EX,SE\n"
                f"img1,test,{image},{''},{mask},{''},{''}\n",
                encoding="utf-8",
            )
            records = load_dataset(root, "idrid")
            self.assertEqual(records[0].id, "img1")
            self.assertEqual(records[0].split, "test")
            self.assertEqual(records[0].mask_paths["MA"], mask)
            self.assertIsNone(records[0].mask_paths["HE"])


def _write_png(path: Path, size: int) -> None:
    from PIL import Image

    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(path)


if __name__ == "__main__":
    unittest.main()
