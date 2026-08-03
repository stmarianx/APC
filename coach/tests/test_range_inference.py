from decimal import Decimal
from pathlib import Path
from dataclasses import replace
import unittest

from poker_coach.range_inference import condition_solution_range
from poker_coach.solver_adapters import SolverExportRegistry
from poker_coach.solver_import import SolverBundleImporter


ROOT = Path(__file__).resolve().parents[1]


class RangeInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        spots = SolverBundleImporter().parse_file(
            ROOT / "examples" / "sample_solver_bundle.json"
        ).spots
        self.shared = spots[:2]

    def test_check_updates_uniform_prior_toward_jto(self) -> None:
        result = condition_solution_range(self.shared, "check")
        self.assertEqual(
            Decimal(result["action_probability_under_prior"]), Decimal("0.33")
        )
        rows = {row["hand_class"]: row for row in result["combos"]}
        self.assertEqual(Decimal(rows["KQo"]["posterior"]), Decimal("0.24") / Decimal("0.66"))
        self.assertEqual(Decimal(rows["JTo"]["posterior"]), Decimal("0.42") / Decimal("0.66"))
        self.assertGreater(
            Decimal(result["information"]["entropy_reduction_bits"]), Decimal("0")
        )

    def test_bet_updates_uniform_prior_toward_kqo(self) -> None:
        result = condition_solution_range(self.shared, "bet:0.33")
        self.assertEqual(result["action_probability_under_prior"], "0.595")
        rows = {row["hand_class"]: row for row in result["combos"]}
        self.assertGreater(Decimal(rows["KQo"]["posterior"]), Decimal("0.5"))
        self.assertLess(Decimal(rows["JTo"]["posterior"]), Decimal("0.5"))

    def test_caller_prior_and_bayes_factor_are_audited(self) -> None:
        result = condition_solution_range(
            self.shared,
            "bet:0.75",
            prior_weights={"Kc Qd": "0.8", "Jc Td": "0.2"},
        )
        self.assertEqual(result["prior_source"], "caller_supplied_exact_combo_weights")
        self.assertEqual(result["action_probability_under_prior"], "0.090")
        rows = {row["hand_class"]: row for row in result["combos"]}
        self.assertEqual(Decimal(rows["KQo"]["posterior"]), Decimal("8") / Decimal("9"))
        self.assertEqual(Decimal(rows["JTo"]["posterior"]), Decimal("1") / Decimal("9"))

    def test_unknown_action_prior_combo_and_mixed_state_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "not covered"):
            condition_solution_range(self.shared, "jam")
        with self.assertRaisesRegex(ValueError, "unknown combos"):
            condition_solution_range(
                self.shared, "check", prior_weights={"As Kd": 1}
            )
        third = SolverBundleImporter().parse_file(
            ROOT / "examples" / "sample_solver_bundle.json"
        ).spots[2]
        with self.assertRaisesRegex(ValueError, "one public decision state"):
            condition_solution_range((*self.shared, third), "check")

    def test_zero_probability_action_is_rejected(self) -> None:
        preflop = SolverExportRegistry().parse_file(
            ROOT / "examples" / "sample_solver_export.csv"
        ).bundle.spots[0]
        self.assertEqual(preflop.action("fold").frequency, Decimal("0"))
        with self.assertRaisesRegex(ValueError, "zero probability"):
            condition_solution_range((preflop,), "fold")

    def test_private_cards_and_prior_mass_are_required(self) -> None:
        public_only = replace(
            self.shared[0], key=replace(self.shared[0].key, hero_cards=())
        )
        with self.assertRaisesRegex(ValueError, "exact two-card"):
            condition_solution_range((public_only,), "check")
        with self.assertRaisesRegex(ValueError, "positive mass"):
            condition_solution_range(
                self.shared,
                "check",
                prior_weights={"Kc Qd": 0, "Jc Td": 0},
            )


if __name__ == "__main__":
    unittest.main()
