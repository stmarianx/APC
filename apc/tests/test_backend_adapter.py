from __future__ import annotations

import sys
import unittest
from pathlib import Path

from apc.backend_adapter import BackendAdapterConfig, build_backend_observation, seat_positions


ROOT = Path(__file__).resolve().parents[2]
COACH_SRC = ROOT / "coach" / "src"
if str(COACH_SRC) not in sys.path:
    sys.path.insert(0, str(COACH_SRC))

from poker_coach.live_state import LiveTableService, LiveTableState
from poker_coach.solver_adapters import SolverExportRegistry
from poker_coach.visual_capture import VisualObservationAdapter


def tracked(*, seats: int = 2, raise_event: bool = False) -> dict[str, object]:
    seat_rows = [{"seat_no": seat, "stack_bb": str(100 - seat)} for seat in range(1, seats + 1)]
    identities = [
        {
            "seat_no": seat,
            "identity_id": f"identity-{seat}",
            "profile_key": f"training:p{seat}",
            "display_name": f"P{seat}",
            "status": "resolved",
            "posterior_probability": 1.0,
            "evidence_frames": 3,
        }
        for seat in range(1, seats + 1)
    ]
    event = {"actor_seat": 2, "action": "raise", "amount_bb": "3"} if raise_event else {"actor_seat": 2, "action": "call", "amount_bb": "1"}
    return {
        "track_id": "table-track",
        "status": "state_tracked_identity_resolved",
        "missing_critical_fields": [] if seats == 2 else ["effective_stack_bb"],
        "identity_gate": {"status": "passed"},
        "state": {
            "hand_id": "internal-hand",
            "history_complete": True,
            "hero_seat": 1,
            "dealer_seat": 1,
            "hero_cards": ["Ah", "Kd"],
            "board_cards": [],
            "pot_bb": "2.5",
            "to_call_bb": "1",
            "effective_stack_bb": "98" if seats == 2 else None,
            "seat_stacks_bb": seat_rows,
            "legal_actions": ["fold", "call", "raise"],
            "action_history": [event],
            "player_identities": identities,
        },
        "perception_evidence": {
            "minimum_supported_confidence": 0.72,
            "frames": {"after": {"image_sha256": "a" * 64}},
            "checkpoint_provenance": {"base_sha256": "b" * 64},
        },
    }


class BackendAdapterTests(unittest.TestCase):
    def test_position_map_matches_backend_clockwise_convention(self) -> None:
        self.assertEqual(
            seat_positions([1, 2, 3, 4, 5, 6], 4),
            {4: "BTN", 5: "SB", 6: "BB", 1: "UTG", 2: "HJ", 3: "CO"},
        )

    def test_heads_up_observation_round_trips_through_backend_contract(self) -> None:
        built = build_backend_observation(tracked())
        self.assertEqual(built["status"], "observation_ready_uncalibrated")
        adapter = VisualObservationAdapter(minimum_confidence="0", stable_frames=1)
        accepted = adapter.submit(built["payload"])
        self.assertEqual(accepted["status"], "state_ready")
        state = LiveTableState.from_dict(accepted["payload"])
        self.assertEqual(state.hero_position, "BTN")
        self.assertEqual(state.action_history, ("BB call",))
        self.assertEqual(state.effective_stack_bb, 98)

    def test_multiway_scalar_effective_stack_ambiguity_abstains(self) -> None:
        built = build_backend_observation(tracked(seats=6))
        self.assertEqual(built["status"], "abstain_incomplete_backend_state")
        self.assertIn("effective_stack_bb", built["missing"])

    def test_multiway_scalar_requires_explicit_solver_policy(self) -> None:
        sample = tracked(seats=6)
        sample["state"]["effective_stacks_by_opponent_bb"] = [
            {"opponent_seat": seat, "effective_stack_bb": str(100 - seat)}
            for seat in range(2, 7)
        ]
        strict = build_backend_observation(sample)
        self.assertIn("effective_stack_bb", strict["missing"])
        declared = build_backend_observation(
            sample,
            config=BackendAdapterConfig(
                multiway_effective_stack_policy="minimum_active_opponent"
            ),
        )
        self.assertEqual(declared["status"], "observation_ready_uncalibrated")
        self.assertEqual(declared["payload"]["fields"]["effective_stack_bb"]["value"], "94")
        self.assertEqual(
            declared["payload"]["apc_evidence"]["effective_stack_semantics"]["active_opponents"],
            [2, 3, 4, 5, 6],
        )

    def test_multiway_table_reduced_to_heads_up_has_unambiguous_effective_stack(self) -> None:
        sample = tracked(seats=6)
        sample["state"]["action_history"] = [
            {"actor_seat": seat, "action": "fold"} for seat in (2, 3, 4, 5)
        ]
        sample["state"]["effective_stacks_by_opponent_bb"] = [
            {"opponent_seat": 6, "effective_stack_bb": "94"}
        ]
        built = build_backend_observation(sample)
        self.assertEqual(built["status"], "observation_ready_uncalibrated")
        self.assertEqual(built["payload"]["fields"]["players"]["value"], 2)
        self.assertEqual(built["payload"]["fields"]["effective_stack_bb"]["value"], "94")

    def test_raise_requires_declared_to_amount_semantics(self) -> None:
        strict = build_backend_observation(tracked(raise_event=True))
        self.assertIn("canonical_action_history", strict["missing"])
        declared = build_backend_observation(
            tracked(raise_event=True),
            config=BackendAdapterConfig(raise_amount_semantics="amount_is_to"),
        )
        self.assertEqual(declared["audit"]["canonical_action_history"], ["BB raise_to:3"])

    def test_dry_run_reaches_exact_solver_node_but_keeps_recommendations_gated(self) -> None:
        sample = tracked()
        sample["state"].update(
            {
                "hero_cards": ["As", "Kd"],
                "pot_bb": "1.5",
                "to_call_bb": "0",
                "effective_stack_bb": "99",
                "legal_actions": ["fold", "raise_to:2.5", "raise_to:3"],
                "action_history": [],
            }
        )
        built = build_backend_observation(sample)
        self.assertFalse(built["audit"]["recommendation_allowed"])

        visual = VisualObservationAdapter(minimum_confidence="0", stable_frames=1)
        accepted = visual.submit(built["payload"])
        registry = SolverExportRegistry().parse_file(
            ROOT / "coach" / "examples" / "sample_solver_export.csv"
        )
        service = LiveTableService()
        session = service.create_session("table-track")
        result = service.update_state(
            session["session_id"], accepted["payload"], registry.bundle.spots
        )
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["match"]["confidence"], "exact")
        self.assertEqual(result["match"]["node_id"], "hu_btn_open_ako_preflop")


if __name__ == "__main__":
    unittest.main()
