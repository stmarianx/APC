from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.annotator import AnnotationProject
from apc.tests.test_annotation_project import complete_annotation, png_bytes
from apc.tools.merge_datasets import _source_namespace, merge_dataset_manifests


class MergeDatasetsTests(unittest.TestCase):
    @staticmethod
    def source(root: Path, dataset_id: str, marker: bytes) -> Path:
        project = AnnotationProject.create(
            root,
            project_id=dataset_id,
            source_kind="controlled_training_table",
            provider_id="merge-fixture",
            layout_id="heads-up",
            theme_id="dark",
            locale="en-US",
            max_seats=2,
        )
        for index in range(3):
            frame = root.parent / f"{dataset_id}-{index}.png"
            frame.write_bytes(png_bytes(marker=marker + bytes([index])))
            record, _ = project.import_frame(
                frame,
                capture_session_id=f"session-{index}",
                timestamp_ms=0,
            )
            project.save_annotation(
                record.sample_id,
                complete_annotation(project.annotation_template(record.sample_id)),
            )
        manifest, report = project.export_manifest(dataset_version="0.1.0")
        if not report["valid"]:
            raise AssertionError(report["errors"])
        return manifest

    def test_merge_namespaces_colliding_sessions_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.source(root / "first", "source-one", b"one")
            second = self.source(root / "second", "source-two", b"two")
            result = merge_dataset_manifests(
                root / "merged-a",
                [second, first],
                dataset_id="combined",
                dataset_version="1.0.0",
            )
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(result["project"]["frames"], 6)
            self.assertEqual(result["project"]["capture_sessions"], 6)
            prefixes = {row["namespace"] for row in result["source_datasets"]}
            self.assertEqual(
                prefixes,
                {_source_namespace("source-one"), _source_namespace("source-two")},
            )

            repeated = merge_dataset_manifests(
                root / "merged-b",
                [first, second],
                dataset_id="combined-copy",
                dataset_version="1.0.0",
            )
            self.assertEqual(
                result["validation"]["computed_fingerprints"],
                repeated["validation"]["computed_fingerprints"],
            )
            self.assertEqual(result["merged_rows_sha256"], repeated["merged_rows_sha256"])

    def test_merge_rejects_identical_frames_across_source_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.source(root / "first", "source-one", b"same")
            second = self.source(root / "second", "source-two", b"same")
            with self.assertRaisesRegex(ValueError, "Identical source-frame digest"):
                merge_dataset_manifests(
                    root / "merged",
                    [first, second],
                    dataset_id="combined",
                    dataset_version="1.0.0",
                )


if __name__ == "__main__":
    unittest.main()
