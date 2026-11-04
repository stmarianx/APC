from __future__ import annotations

import json
import hashlib
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
            elapsed_ns = 0

            def sleeper(seconds: float) -> None:
                nonlocal elapsed_ns
                sleeps.append(seconds)
                elapsed_ns += round(seconds * 1_000_000_000)

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
                sleeper=sleeper,
                clock_ns=lambda: elapsed_ns,
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
            self.assertEqual(saved["status"], "completed")

    def test_slow_capture_preserves_actual_timing_not_requested_interval(self) -> None:
        elapsed_ns = 0

        def grabber(_):
            nonlocal elapsed_ns
            elapsed_ns += 300_000_000
            return Image.new("RGB", (320, 240))

        with tempfile.TemporaryDirectory() as directory:
            report = capture_frames(
                directory, CapturePlan("slow", CaptureRegion(0, 0, 320, 240), 3, 200),
                grabber=grabber, clock_ns=lambda: elapsed_ns,
                sleeper=lambda _: self.fail("Overdue captures should not sleep"),
            )
        self.assertEqual([row["timestamp_ms"] for row in report["frames"]], [0, 300, 600])
        self.assertEqual([row["scheduled_timestamp_ms"] for row in report["frames"]], [0, 200, 400])
        self.assertEqual(report["frames"][0]["capture_end_ns"], 300_000_000)

    def test_failure_and_interrupt_preserve_completed_frame_manifest(self) -> None:
        for failure, status in ((RuntimeError("grab failed"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                calls = 0

                def grabber(_):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        # A durable report already exists before the next capture begins.
                        running = json.loads((Path(directory) / "capture_report.json").read_text())
                        self.assertEqual(running["status"], "running")
                        self.assertEqual(running["captured_frames"], 1)
                        raise failure
                    return Image.new("RGB", (320, 240))

                with self.assertRaises(type(failure)):
                    capture_frames(
                        directory, CapturePlan("partial", CaptureRegion(0, 0, 320, 240), 3, 50),
                        grabber=grabber, sleeper=lambda _: None, clock_ns=lambda: 0,
                    )
                saved = json.loads((Path(directory) / "capture_report.json").read_text())
                self.assertEqual(saved["status"], status)
                self.assertEqual(saved["captured_frames"], 1)
                row = saved["frames"][0]
                self.assertEqual(hashlib.sha256((Path(directory) / row["filename"]).read_bytes()).hexdigest(), row["sha256"])
                self.assertNotIn("completed_at", saved)

    def test_submillisecond_ordering_is_explicit_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = capture_frames(
                directory, CapturePlan("fast", CaptureRegion(0, 0, 320, 240), 2, 50),
                grabber=lambda _: Image.new("RGB", (320, 240)),
                sleeper=lambda _: None, clock_ns=lambda: 0,
            )
        self.assertEqual(report["frames"][1]["timestamp_ms"], 1)
        self.assertEqual(report["frames"][1]["capture_start_ns"], 0)
        self.assertTrue(report["frames"][1]["timestamp_resolution_adjusted"])

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
