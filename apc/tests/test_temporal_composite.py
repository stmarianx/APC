from __future__ import annotations

import unittest

from apc.perception.temporal_composite import (
    _history_rows,
    _temporal_perception_abstention,
    effective_stacks_by_opponent,
    heads_up_effective_stack,
    validate_visible_transition,
)


def state(*, seats: list[tuple[int, str]], pot: str = "10") -> dict[str, object]:
    return {
        "layout_id": "heads-up" if len(seats) == 2 else "six-max",
        "theme_id": "midnight",
        "street": "flop",
        "hero_seat": 1,
        "dealer_seat": 2,
        "hero_cards": ["Ah", "Kd"],
        "board_cards": ["2c", "7d", "Ts"],
        "pot_bb": pot,
        "seat_stacks_bb": [
            {"seat_no": seat_no, "stack_bb": stack_bb}
            for seat_no, stack_bb in seats
        ],
    }


class TemporalCompositeTests(unittest.TestCase):
    def test_heads_up_effective_stack_is_minimum_visible_stack(self) -> None:
        self.assertEqual(
            heads_up_effective_stack(state(seats=[(1, "93"), (2, "106")])),
            "93",
        )

    def test_multiway_does_not_invent_scalar_effective_stack(self) -> None:
        self.assertIsNone(
            heads_up_effective_stack(state(seats=[(1, "93"), (2, "106"), (3, "88")]))
        )

    def test_multiway_preserves_each_active_opponent_effective_stack(self) -> None:
        visible = state(seats=[(1, "93"), (2, "106"), (3, "88"), (4, "74")])
        self.assertEqual(
            effective_stacks_by_opponent(visible, folded_seats={3}),
            [
                {"opponent_seat": 2, "effective_stack_bb": "93"},
                {"opponent_seat": 4, "effective_stack_bb": "74"},
            ],
        )

    def test_chip_event_requires_actor_and_pot_conservation(self) -> None:
        before = state(seats=[(1, "93"), (2, "106")], pot="10")
        after = state(seats=[(1, "93"), (2, "102")], pot="14")
        audit = validate_visible_transition(
            before,
            after,
            {"actor_seat": 2, "action": "raise", "amount_bb": "4"},
        )
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(audit["deltas"]["pot_bb"], "4")

    def test_rejects_chip_delta_on_the_wrong_seat(self) -> None:
        before = state(seats=[(1, "93"), (2, "106")], pot="10")
        after = state(seats=[(1, "89"), (2, "106")], pot="14")
        audit = validate_visible_transition(
            before,
            after,
            {"actor_seat": 2, "action": "raise", "amount_bb": "4"},
        )
        self.assertEqual(audit["status"], "rejected")
        self.assertIn(
            "only_actor_stack_changes",
            {row["code"] for row in audit["violations"]},
        )

    def test_non_chip_event_requires_no_public_chip_change(self) -> None:
        before = state(seats=[(1, "93"), (2, "106")], pot="10")
        after = state(seats=[(1, "93"), (2, "106")], pot="10")
        audit = validate_visible_transition(
            before,
            after,
            {"actor_seat": 2, "action": "check"},
        )
        self.assertEqual(audit["status"], "accepted")

    def test_history_append_is_idempotent_for_duplicate_frame_pair(self) -> None:
        event = {"actor_seat": 2, "action": "check"}
        self.assertEqual(_history_rows([event], event), [event])

    def test_prerequisite_head_failure_becomes_temporal_abstention(self) -> None:
        import tempfile
        import time
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory) / "before.bin"
            after = Path(directory) / "after.bin"
            before.write_bytes(b"before")
            after.write_bytes(b"after")
            result = _temporal_perception_abstention(
                "frame_perception_rejected",
                before,
                after,
                {
                    "visible_state": {},
                    "minimum_supported_confidence": 0.1,
                    "missing_critical_fields": ["table_state_perception"],
                    "checkpoint_provenance": {"base_sha256": "b" * 64},
                },
                {
                    "visible_state": {},
                    "minimum_supported_confidence": 0.0,
                    "missing_critical_fields": ["visible_card_integrity"],
                    "checkpoint_provenance": {"base_sha256": "b" * 64},
                },
                started=time.perf_counter(),
                detail={"before": "failed", "after": "failed"},
            )
        self.assertEqual(result["status"], "abstain_frame_perception_rejected")
        self.assertEqual(result["transition_audit"]["status"], "rejected")
        self.assertIn("temporally_consistent_state", result["missing_critical_fields"])
        self.assertIsNone(result["recommendation"])


if __name__ == "__main__":
    unittest.main()
