from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.annotator import AnnotationProject
from apc.annotator.propagation import propagate_layout_suggestions
from apc.tests.test_annotation_project import complete_annotation, png_bytes


class LayoutPropagationTests(unittest.TestCase):
    def create_sequence(self, root: Path) -> tuple[AnnotationProject, list[str]]:
        project = AnnotationProject.create(
            root / "project",
            project_id="propagation-fixture",
            source_kind="controlled_training_table",
            provider_id="fixture",
            layout_id="six-max",
            theme_id="dark",
            locale="en-US",
            max_seats=6,
        )
        samples = []
        for index in range(3):
            frame = root / f"frame-{index}.png"
            frame.write_bytes(png_bytes(marker=str(index).encode()))
            record, _ = project.import_frame(
                frame,
                capture_session_id="session-a",
                timestamp_ms=index * 100,
            )
            samples.append(record.sample_id)
        return project, samples

    def test_verified_layout_propagates_as_review_only_suggestions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, samples = self.create_sequence(Path(directory))
            source = complete_annotation(project.annotation_template(samples[0]))
            project.save_annotation(samples[0], source)

            report = propagate_layout_suggestions(project, source_sample_id=samples[0])

            self.assertEqual(report["generated_suggestions"], 2)
            suggestion = project.load_suggestion(samples[1])
            self.assertEqual(suggestion["kind"], "apc_layout_propagation_suggestion")
            self.assertTrue(suggestion["review_required"])
            self.assertFalse(suggestion["auto_applied"])
            self.assertEqual(suggestion["suggested_visible_state"]["hero_seat"], 1)
            self.assertIn("table", suggestion["suggested_objects"])
            self.assertNotIn("player_name", suggestion["suggested_objects"]["seats"][0])
            self.assertFalse(project.annotation_path(samples[1]).exists())

    def test_unverified_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, samples = self.create_sequence(Path(directory))
            source = complete_annotation(project.annotation_template(samples[0]))
            source["provenance"]["verified"] = False
            source["provenance"]["reviewer"] = None
            project.save_annotation(samples[0], source)

            with self.assertRaisesRegex(ValueError, "verified source"):
                propagate_layout_suggestions(project, source_sample_id=samples[0])

    def test_existing_annotation_and_suggestion_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, samples = self.create_sequence(Path(directory))
            project.save_annotation(
                samples[0], complete_annotation(project.annotation_template(samples[0]))
            )
            project.save_annotation(
                samples[1], complete_annotation(project.annotation_template(samples[1]))
            )
            first = propagate_layout_suggestions(project, source_sample_id=samples[0])
            second = propagate_layout_suggestions(project, source_sample_id=samples[0])

            self.assertEqual(first["generated_suggestions"], 1)
            self.assertEqual(first["skipped_annotated"], 1)
            self.assertEqual(second["generated_suggestions"], 0)
            self.assertEqual(second["skipped_existing_suggestion"], 1)

    def test_targets_from_other_sessions_are_never_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, samples = self.create_sequence(root)
            other = root / "other.png"
            other.write_bytes(png_bytes(marker=b"other"))
            other_record, _ = project.import_frame(
                other, capture_session_id="session-b", timestamp_ms=0
            )
            project.save_annotation(
                samples[0], complete_annotation(project.annotation_template(samples[0]))
            )

            report = propagate_layout_suggestions(
                project,
                source_sample_id=samples[0],
                target_sample_ids=[other_record.sample_id],
            )

            self.assertEqual(report["generated_suggestions"], 0)
            self.assertIsNone(project.load_suggestion(other_record.sample_id))


if __name__ == "__main__":
    unittest.main()
