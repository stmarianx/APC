from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.perception.turn_clock_baseline import (
    evaluate_turn_clock_baseline,
    load_turn_clock_checkpoint,
    train_turn_clock_baseline,
)
from apc.synthetic.render_table import generate_dataset


class TurnClockBaselineTests(unittest.TestCase):
    def test_train_and_held_out_evaluate_turn_clock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            generate_dataset(
                dataset,
                sessions=12,
                seed=2026081501,
                include_turn_clock=True,
            )
            checkpoint_path = root / "clock.json"
            checkpoint = train_turn_clock_baseline(
                dataset / "dataset_manifest.json", checkpoint_path
            )
            loaded = load_turn_clock_checkpoint(checkpoint_path)
            report = evaluate_turn_clock_baseline(
                checkpoint_path,
                dataset / "dataset_manifest.json",
                split="test",
            )
            self.assertEqual(loaded["checkpoint_sha256"], checkpoint["checkpoint_sha256"])
            self.assertEqual(report["training_session_overlap"], [])
            self.assertEqual(report["metrics"]["exact_remaining_ms_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
