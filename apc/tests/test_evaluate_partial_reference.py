from __future__ import annotations

import unittest
from pathlib import Path

from apc.tools.evaluate_partial_reference import evaluate_partial_reference


ROOT = Path(__file__).resolve().parents[2]


class PartialReferenceEvaluatorTests(unittest.TestCase):
    def test_frozen_visible_reference_records_failure_and_safe_abstention(self) -> None:
        report = evaluate_partial_reference(
            ROOT / "apc" / "data" / "reference" / "controlled-reference-v1" / "ground_truth_partial.json",
            ROOT / "apc" / "runs" / "controlled-reference-v1" / "synthetic_checkpoint_prediction.json",
        )
        self.assertEqual(report["exact_fields"], 1)
        self.assertTrue(report["safe_abstention"])
        self.assertTrue(report["invalid_cards_rejected"])
        self.assertFalse(report["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
