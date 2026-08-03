from __future__ import annotations

import tempfile
import unittest
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from apc.perception.evaluate_table_locator import box_iou
from apc.perception.evaluate_table_reference import evaluate_table_reference
from apc.perception.table_locator import detect_table_box
from apc.perception.viewport import NormalizedBox, ViewportCalibration


class TableLocatorTests(unittest.TestCase):
    def test_central_rounded_table_is_localized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.png"
            image = Image.new("RGB", (1280, 720), "#090d10")
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                (115, 130, 1165, 575), radius=210, fill="#1c5946"
            )
            image.save(path)
            result = detect_table_box(path)
            expected = NormalizedBox(115 / 1280, 130 / 720, 1050 / 1280, 445 / 720)
            predicted = NormalizedBox.from_dict(result["table_box"])
            self.assertEqual(result["status"], "detected_uncalibrated")
            self.assertGreaterEqual(box_iou(expected, predicted), 0.9)

    def test_low_contrast_screen_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            Image.new("RGB", (640, 480), "#111111").save(path)
            result = detect_table_box(path)
            self.assertEqual(result["status"], "abstain_low_table_contrast")
            self.assertIsNone(result["table_box"])

    def test_frozen_reference_evaluator_checks_image_and_profile_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "table.png"
            image = Image.new("RGB", (1280, 720), "#090d10")
            ImageDraw.Draw(image).rounded_rectangle(
                (115, 130, 1165, 575), radius=210, fill="#1c5946"
            )
            image.save(image_path)
            profile_path = root / "profile.json"
            ViewportCalibration(
                "fixture",
                (1280, 720),
                NormalizedBox(115 / 1280, 130 / 720, 1050 / 1280, 445 / 720),
                (1280, 720),
                NormalizedBox(115 / 1280, 130 / 720, 1050 / 1280, 445 / 720),
            ).save(profile_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            result = evaluate_table_reference(
                image_path, profile_path, expected_image_sha256=digest
            )
            self.assertTrue(result["comparison"]["iou_at_0_8"])
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                evaluate_table_reference(
                    image_path, profile_path, expected_image_sha256="0" * 64
                )


if __name__ == "__main__":
    unittest.main()
