from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from apc.capture import CapturePlan, CaptureRegion, capture_frames


class ScreenCaptureTests(unittest.TestCase):
    def test_explicit_region_capture_is_ordered_fingerprinted_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls: list[tuple[int, int, int, int]] = []
            sleeps: list[float] = []

            def grabber(bbox: tuple[int, int, int, int]) -> Image.Image:
                calls.append(bbox)
                return Image.new("RGB", (640, 480), (len(calls), 20, 30))

            report = capture_frames(
                Path(directory) / "capture",
                CapturePlan(
                    session_id="controlled-001",
                    region=CaptureRegion(100, 50, 740, 530),
                    frames=3,
                    interval_ms=200,
                ),
                grabber=grabber,
                sleeper=sleeps.append,
            )
            self.assertEqual(calls, [(100, 50, 740, 530)] * 3)
            self.assertEqual(sleeps, [0.2, 0.2])
            self.assertEqual(
                [row["timestamp_ms"] for row in report["frames"]],
                [0, 200, 400],
            )
            self.assertEqual(len({row["sha256"] for row in report["frames"]}), 3)
            self.assertTrue(report["policy"]["read_only_pixels"])
            self.assertFalse(report["policy"]["input_control"])
            saved = json.loads(
                (Path(directory) / "capture" / "capture_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["frames_sha256"], report["frames_sha256"])

    def test_capture_requires_new_output_and_minimum_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "320x240"):
            CaptureRegion(0, 0, 319, 240)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                capture_frames(
                    root,
                    CapturePlan(
                        session_id="session",
                        region=CaptureRegion(0, 0, 640, 480),
                        frames=1,
                        interval_ms=50,
                    ),
                    grabber=lambda _: Image.new("RGB", (640, 480)),
                )


if __name__ == "__main__":
    unittest.main()
