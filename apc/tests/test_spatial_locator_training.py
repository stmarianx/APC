import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from apc.perception.card_locator import locator_targets
from apc.perception.train_card_locator import augment_spatial, frozen_spatial, train


class SpatialLocatorTrainingTests(unittest.TestCase):
    def data(self):
        image = torch.zeros(3, 184, 320)
        boxes = [[.2, .2, .3, .4], [.6, .6, .7, .8]]
        for x1, y1, x2, y2 in boxes:
            image[:, round(y1*184):round(y2*184), round(x1*320):round(x2*320)] = 1
        return image[None], locator_targets(boxes, 23, 40)[None], [[{"box_xyxy": b} for b in boxes]]

    def test_seeded_transform_moves_pixels_and_labels_without_clipping_cards(self):
        source = self.data()
        original = source[0].clone()
        bounds = [card["box_xyxy"][:] for card in source[2][0]]
        transformed = frozen_spatial(source, 123)
        self.assertEqual(int(transformed[1][0, 0].sum()), 2)
        for card in transformed[2][0]:
            x1, y1, x2, y2 = card["box_xyxy"]
            self.assertTrue(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)
            cx, cy = int((x1+x2)*160), int((y1+y2)*92)
            self.assertGreater(float(transformed[0][0, :, cy, cx].min()), .9)
        torch.testing.assert_close(source[0], original)
        self.assertEqual([card["box_xyxy"] for card in source[2][0]], bounds)
        self.assertNotEqual(transformed[2][0][0]["box_xyxy"], bounds[0])

    def test_frozen_validation_transform_preserves_training_rng(self):
        torch.manual_seed(31)
        before = torch.get_rng_state().clone()
        first = frozen_spatial(self.data(), 123)
        torch.testing.assert_close(torch.get_rng_state(), before)
        second = frozen_spatial(self.data(), 123)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_empty_scene_retains_no_positive_card_labels(self):
        images, labels, truths = augment_spatial((torch.zeros(1, 3, 184, 320), torch.zeros(1, 5, 23, 40), [[]]))
        self.assertEqual(float(images.abs().sum()), 0)
        self.assertEqual(float(labels.abs().sum()), 0)
        self.assertEqual(truths, [[]])

    def test_augmented_training_keeps_test_reporting_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            with patch("apc.perception.train_card_locator.dataset", return_value=self.data()):
                report = train(manifest, root / "run", epochs=1, relative_size=True, spatial_augmentation=True, hard_negative_ratio=3)
            self.assertEqual(report["history"][0]["validation"]["frames"], 2)
            self.assertEqual(report["test"]["frames"], 1)
            self.assertEqual(report["test_frozen_spatial"]["frames"], 1)
            self.assertEqual(report["spatial_augmentation"], "bounded_scale_translation_v1")
            self.assertEqual(report["hard_negative_ratio"], 3)
            self.assertEqual(report["hard_negative_minimum_cells_per_frame"], 32)
            self.assertFalse(report["full_table_readiness"])


if __name__ == "__main__":
    unittest.main()
