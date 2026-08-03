from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.perception.viewport import (
    NormalizedBox,
    ViewportCalibration,
    normalize_viewport,
)


class ViewportCalibrationTests(unittest.TestCase):
    def calibration(self) -> ViewportCalibration:
        return ViewportCalibration(
            "test-profile",
            (1600, 900),
            NormalizedBox(0.2, 0.2, 0.6, 0.5),
            (1280, 720),
            NormalizedBox(0.1, 0.18, 0.82, 0.62),
        )

    def test_table_box_maps_exactly_to_canonical_anchor(self) -> None:
        calibration = self.calibration()
        mapped = calibration.map_source_box(calibration.observed_table_box)
        expected = calibration.canonical_table_box
        for field in ("x", "y", "width", "height"):
            self.assertAlmostEqual(getattr(mapped, field), getattr(expected, field))

    def test_fingerprinted_profile_round_trips_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            self.calibration().save(path)
            self.assertEqual(ViewportCalibration.load(path), self.calibration())
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["observed_table_box"]["x"] = 0.21
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                ViewportCalibration.load(path)

    def test_identity_calibration_preserves_image_bytes_visually(self) -> None:
        from PIL import Image, ImageChops

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "normalized.png"
            image = Image.new("RGB", (128, 64), "navy")
            image.putpixel((42, 23), (255, 0, 0))
            image.save(source, format="PNG", optimize=False)
            calibration = ViewportCalibration(
                "identity",
                (128, 64),
                NormalizedBox(0.1, 0.1, 0.8, 0.8),
                (128, 64),
                NormalizedBox(0.1, 0.1, 0.8, 0.8),
            )
            report = normalize_viewport(source, calibration, output)
            with Image.open(output) as normalized:
                difference = ImageChops.difference(image, normalized.convert("RGB"))
                self.assertIsNone(difference.getbbox())
            self.assertEqual(report["paste_offset"], [0, 0])


if __name__ == "__main__":
    unittest.main()
