from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path

from apc.evaluate_solver_coverage import audit_solver_coverage, evaluate_files


ROOT = Path(__file__).resolve().parents[2]
COACH_SRC = ROOT / "coach" / "src"
if str(COACH_SRC) not in sys.path:
    sys.path.insert(0, str(COACH_SRC))

from poker_coach import PokerStarsParser, SolverBundleImporter


class SolverCoverageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hand_path = ROOT / "coach" / "examples" / "sample_play_money_hand.txt"
        cls.solver_path = ROOT / "coach" / "examples" / "sample_solver_bundle.json"
        cls.hands = PokerStarsParser().parse_file(cls.hand_path)
        cls.bundle = SolverBundleImporter().parse_file(cls.solver_path)

    def test_sample_corpus_reports_real_low_coverage_without_promotion(self) -> None:
        report = audit_solver_coverage(self.hands, self.bundle.spots)
        self.assertTrue(report["audit_valid"])
        self.assertFalse(report["coverage_gate"]["passed"])
        self.assertEqual(report["corpus"], {"hands": 1, "hero_decisions": 3})
        self.assertEqual(report["metrics"]["matched_decisions"], 1)
        self.assertEqual(report["metrics"]["exact_matches"], 1)
        self.assertEqual(report["metrics"]["state_coverage"], "0.3333333333333333333333333333")
        self.assertEqual(report["metrics"]["observed_action_coverage"], "0.3333333333333333333333333333")
        self.assertEqual(sum(report["metrics"]["exclusion_counts"].values()), 2)

    def test_slices_reconcile_to_total_decisions(self) -> None:
        report = audit_solver_coverage(self.hands, self.bundle.spots)
        for dimension in ("street", "players", "hero_position"):
            rows = report["slices"][dimension]
            self.assertEqual(sum(row["decisions"] for row in rows.values()), 3)
            self.assertEqual(sum(row["matched"] for row in rows.values()), 1)

    def test_thresholds_are_explicit_and_cannot_pass_small_fixture(self) -> None:
        report = audit_solver_coverage(
            self.hands,
            self.bundle.spots,
            minimum_decisions=3,
            minimum_exact_coverage=Decimal("0.34"),
        )
        self.assertFalse(report["coverage_gate"]["passed"])
        with self.assertRaises(ValueError):
            audit_solver_coverage(self.hands, self.bundle.spots, minimum_decisions=0)
        with self.assertRaises(ValueError):
            audit_solver_coverage(
                self.hands,
                self.bundle.spots,
                minimum_exact_coverage=Decimal("1.01"),
            )

    def test_file_evaluator_fingerprints_both_inputs(self) -> None:
        report = evaluate_files(self.solver_path, self.hand_path)
        self.assertEqual(len(report["inputs"]["solver_export_sha256"]), 64)
        self.assertEqual(len(report["inputs"]["hand_history_corpus_sha256"]), 64)
        self.assertEqual(report["inputs"]["hand_history_files"], 1)


if __name__ == "__main__":
    unittest.main()
