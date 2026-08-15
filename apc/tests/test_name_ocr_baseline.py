from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.perception.name_ocr_baseline import (
    evaluate_name_ocr_baseline,
    load_name_ocr_checkpoint,
    train_name_ocr_baseline,
)
from apc.synthetic.render_table import generate_dataset


class NameOcrBaselineTests(unittest.TestCase):
    def test_unseen_whole_names_are_decoded_character_by_character(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            generate_dataset(
                dataset,
                sessions=12,
                seed=2026081503,
                include_name_ocr=True,
            )
            checkpoint_path = root / "name-ocr.json"
            checkpoint = train_name_ocr_baseline(
                dataset / "dataset_manifest.json", checkpoint_path
            )
            loaded = load_name_ocr_checkpoint(checkpoint_path)
            report = evaluate_name_ocr_baseline(
                checkpoint_path,
                dataset / "dataset_manifest.json",
                split="test",
            )
            self.assertEqual(loaded["checkpoint_sha256"], checkpoint["checkpoint_sha256"])
            self.assertEqual(report["training_session_overlap"], [])
            self.assertEqual(report["metrics"]["unseen_whole_name_rate"], 1.0)
            self.assertEqual(report["metrics"]["exact_name_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["character_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["final_identity_resolution_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
