from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from apc.annotator import AnnotationProject
from apc.annotator.suggestions import generate_project_suggestions
from apc.tests.test_annotation_project import complete_annotation, png_bytes


class SuggestionTests(unittest.TestCase):
    def project_with_frame(self, root: Path) -> tuple[AnnotationProject, str]:
        project = AnnotationProject.create(
            root / "project",
            project_id="suggestion-fixture",
            source_kind="controlled_training_table",
            provider_id="fixture",
            layout_id="six-max",
            theme_id="dark",
            locale="en-US",
            max_seats=6,
        )
        frame = root / "frame.png"
        frame.write_bytes(png_bytes(marker=b"suggest"))
        record, _ = project.import_frame(frame, capture_session_id="session", timestamp_ms=0)
        return project, record.sample_id

    @staticmethod
    def prediction(path: Path) -> dict[str, object]:
        import hashlib

        return {
            "status": "abstain_incomplete_state",
            "minimum_supported_confidence": 0.42,
            "frame": {"image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "checkpoint_provenance": {"base_sha256": "b" * 64},
            "visible_state": {"layout_id": "six-max", "pot_bb": "6"},
            "field_confidence": {"layout_id": 0.9, "pot_bb": 0.42},
            "perception_abstentions": [],
        }

    def test_suggestion_is_fingerprinted_separate_and_never_auto_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, sample = self.project_with_frame(Path(directory))
            report = generate_project_suggestions(project, predictor=self.prediction)
            suggestion = project.load_suggestion(sample)
            self.assertEqual(report["generated_suggestions"], 1)
            self.assertTrue(suggestion["review_required"])
            self.assertFalse(suggestion["auto_applied"])
            self.assertFalse(project.annotation_path(sample).exists())
            self.assertEqual(project.status()["model_suggestions"], 1)

    def test_corrupted_suggestion_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, sample = self.project_with_frame(Path(directory))
            generate_project_suggestions(project, predictor=self.prediction)
            path = project.suggestion_path(sample)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["minimum_supported_confidence"] = 1.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                project.load_suggestion(sample)

    def test_loading_valid_suggestion_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, sample = self.project_with_frame(Path(directory))
            generate_project_suggestions(project, predictor=self.prediction)
            with mock.patch(
                "apc.annotator.project._write_json",
                side_effect=AssertionError("load must not persist"),
            ):
                suggestion = project.load_suggestion(sample)
            self.assertEqual(suggestion["sample_id"], sample)

    def test_annotated_frames_are_skipped_unless_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, sample = self.project_with_frame(Path(directory))
            project.save_annotation(sample, complete_annotation(project.annotation_template(sample)))
            report = generate_project_suggestions(project, predictor=self.prediction)
            self.assertEqual(report["generated_suggestions"], 0)
            self.assertEqual(report["skipped_annotated"], 1)


if __name__ == "__main__":
    unittest.main()
