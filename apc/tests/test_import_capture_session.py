import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from apc.annotator.import_capture_session import import_capture_session
from apc.annotator.project import AnnotationProject
from apc.capture import CapturePlan, CaptureRegion, capture_frames
from apc.tools.validate_dataset import canonical_sha256


class CaptureSessionImportTests(unittest.TestCase):
    def setup_capture(self, root, *, fail=False):
        project = AnnotationProject.create(root / "project", project_id="measured-test",
            source_kind="controlled_training_table", provider_id="apc-full-hand-browser",
            layout_id="heads-up", theme_id="test", max_seats=2, locale="en-US")
        elapsed_ns, calls = 0, 0

        def grab(_):
            nonlocal elapsed_ns, calls
            calls += 1
            if fail and calls == 3:
                raise RuntimeError("Interrupted source")
            elapsed_ns += 300_000_000
            return Image.new("RGB", (320, 240), (calls, 20, 30))

        try:
            capture_frames(root / "capture", CapturePlan("measured", CaptureRegion(0, 0, 320, 240), 3, 200),
                grabber=grab, clock_ns=lambda: elapsed_ns, sleeper=lambda _: None)
        except RuntimeError:
            if not fail:
                raise
        return project, root / "capture"

    def test_measured_timestamps_survive_import_and_reimport_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            project, capture = self.setup_capture(Path(directory))
            result = import_capture_session(project, capture)
            self.assertEqual(result["inserted_frames"], 3)
            self.assertEqual([record.timestamp_ms for record in project.records], [0, 300, 600])
            self.assertFalse(result["labels_created"])
            self.assertFalse(result["readiness_counts_added"])
            self.assertTrue(all(project.load_annotation(record.sample_id) is None for record in project.records))
            again = import_capture_session(project, capture)
            self.assertEqual(again["inserted_frames"], 0)
            self.assertEqual(again["duplicate_frames"], 3)

    def test_corrupt_last_image_rejects_before_importing_any_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            project, capture = self.setup_capture(Path(directory))
            report = json.loads((capture / "capture_report.json").read_text())
            Image.new("RGB", (320, 240), "red").save(capture / report["frames"][-1]["filename"])
            with self.assertRaisesRegex(ValueError, "image fingerprint"):
                import_capture_session(project, capture)
            self.assertEqual(len(project.records), 0)

    def test_partial_capture_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            project, capture = self.setup_capture(Path(directory), fail=True)
            with self.assertRaisesRegex(ValueError, "explicitly import"):
                import_capture_session(project, capture)
            self.assertEqual(len(project.records), 0)
            result = import_capture_session(project, capture, allow_partial=True)
            self.assertEqual(result["source_status"], "failed")
            self.assertEqual(result["inserted_frames"], 2)

    def test_manifest_integrity_paths_and_ordering_are_checked_before_mutation(self):
        for corruption in ("hash", "path", "ordering", "running", "legacy", "plan"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                project, capture = self.setup_capture(Path(directory))
                path = capture / "capture_report.json"
                report = json.loads(path.read_text())
                if corruption == "hash":
                    report["frames"][0]["sha256"] = "0" * 64
                elif corruption == "path":
                    report["frames"][0]["filename"] = "../outside.png"
                elif corruption == "ordering":
                    report["frames"][1]["timestamp_ms"] = 200
                elif corruption == "running":
                    report["status"] = "running"
                elif corruption == "legacy":
                    report["schema_version"] = "1.0.0"
                else:
                    report["plan_sha256"] = "0" * 64
                if corruption != "hash":
                    report["frames_sha256"] = canonical_sha256(report["frames"])
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    import_capture_session(project, capture, allow_partial=True)
                self.assertEqual(len(project.records), 0)


if __name__ == "__main__":
    unittest.main()
