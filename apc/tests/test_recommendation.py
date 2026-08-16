from __future__ import annotations

import sys
import unittest
from pathlib import Path

from apc.evaluate_recommendations import evaluate_recommendations
from apc.recommendation import action_command_from_solver_id, build_auditable_recommendation


ROOT = Path(__file__).resolve().parents[2]
COACH_SRC = ROOT / "coach" / "src"
if str(COACH_SRC) not in sys.path:
    sys.path.insert(0, str(COACH_SRC))

from poker_coach import LiveTableService, SolverExportRegistry


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = SolverExportRegistry().parse_file(
            ROOT / "coach" / "examples" / "sample_solver_export.csv"
        ).bundle

    def backend(self, spot_index: int = 0, *, legal_actions: list[str] | None = None) -> dict[str, object]:
        spot = self.bundle.spots[spot_index]
        key = spot.key
        service = LiveTableService()
        session = service.create_session(f"table-{spot_index}")
        return service.update_state(
            session["session_id"],
            {
                "schema_version": "1.0.0",
                "table_id": f"table-{spot_index}",
                "hand_id": f"hand-{spot_index}",
                "revision": 0,
                "game": key.game,
                "players": key.players,
                "hero_position": key.hero_position,
                "effective_stack_bb": format(key.effective_stack_bb, "f"),
                "pot_bb": format(key.pot_bb, "f"),
                "to_call_bb": "0",
                "board": [str(card) for card in key.board],
                "hero_cards": [str(card) for card in key.hero_cards],
                "action_history": list(key.action_history),
                "legal_actions": legal_actions or [row.action for row in spot.actions],
                "rake_model": key.rake_model,
                "utility_model": key.utility_model,
                "source": "test",
            },
            self.bundle.spots,
        )

    @staticmethod
    def plan() -> dict[str, object]:
        return {"status": "compute", "strategy_tier": "cached_exact_solver"}

    def recommend(self, backend: dict[str, object], **changes: object) -> dict[str, object]:
        values = {
            "recommendation_allowed": True,
            "perception_calibrated": True,
            "virtual_chip_environment": True,
            "decision_plan": self.plan(),
            "sampling_key": "stable-test-key",
        }
        values.update(changes)
        return build_auditable_recommendation(backend, **values)  # type: ignore[arg-type]

    def test_exact_solver_node_returns_deterministic_auditable_mix(self) -> None:
        backend = self.backend()
        first = self.recommend(backend)
        second = self.recommend(backend)
        self.assertEqual(first["status"], "recommendation_ready")
        self.assertEqual(first["recommendation_sha256"], second["recommendation_sha256"])
        mix = first["recommendation"]["mixed_strategy"]
        self.assertEqual(sum(float(row["conditional_frequency"]) for row in mix), 1.0)
        self.assertEqual(first["units"], "BB")
        self.assertFalse(first["actuation_authorized"])
        self.assertFalse(first["recommendation"]["gto_claim"])
        self.assertEqual(first["audit"]["solver_node_id"], "hu_btn_open_ako_preflop")

    def test_solver_action_sizing_is_explicitly_converted_to_bb(self) -> None:
        bet = action_command_from_solver_id(
            "bet:0.33", pot_bb="6.5", to_call_bb="0", effective_stack_bb="97"
        )
        self.assertEqual(
            bet.payload(),
            {"action": "bet", "amount_bb": "2.145", "to_amount_bb": "2.145"},
        )
        raised = action_command_from_solver_id(
            "raise_to:2.5", pot_bb="1.5", to_call_bb="0", effective_stack_bb="99"
        )
        self.assertEqual(raised.payload(), {"action": "raise", "to_amount_bb": "2.5"})

    def test_closed_or_uncalibrated_gate_abstains(self) -> None:
        result = self.recommend(
            self.backend(), recommendation_allowed=False, perception_calibrated=False
        )
        self.assertEqual(result["status"], "abstain_recommendation_gate")
        self.assertIsNone(result["recommendation"])
        self.assertEqual(
            set(result["reasons"]),
            {"recommendation_gate_closed", "perception_not_calibrated"},
        )

    def test_legal_action_subset_is_conditionally_renormalized(self) -> None:
        result = self.recommend(self.backend(0, legal_actions=["raise_to:3"]))
        self.assertEqual(result["status"], "recommendation_ready")
        self.assertEqual(result["recommendation"]["mixed_strategy"][0]["conditional_frequency"], "1")
        self.assertEqual(result["audit"]["legal_frequency_mass"], "0.25")

    def test_deadline_fallback_has_no_gto_claim(self) -> None:
        result = self.recommend(
            self.backend(),
            decision_plan={"status": "fallback_required", "fallback": {"action": "fold"}},
        )
        self.assertEqual(result["status"], "safe_fallback_only")
        self.assertFalse(result["recommendation"]["gto_claim"])
        self.assertFalse(result["actuation_authorized"])

    def test_unmatched_backend_state_abstains(self) -> None:
        backend = self.backend()
        backend["status"] = "unmatched"
        backend["match"] = None
        self.assertEqual(self.recommend(backend)["status"], "abstain_solver_uncovered")

    def test_solver_fixture_regression_covers_every_node(self) -> None:
        report = evaluate_recommendations(
            ROOT / "coach" / "examples" / "sample_solver_export.csv"
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["nodes"], 4)
        self.assertEqual(report["metrics"]["recommendations_ready"], 4)
        self.assertEqual(report["metrics"]["closed_gate_abstentions"], 4)


if __name__ == "__main__":
    unittest.main()
