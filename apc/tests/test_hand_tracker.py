from __future__ import annotations

import copy
import unittest

from apc.perception.hand_tracker import (
    TemporalHandTracker,
    attach_player_identities,
    resolve_ocr_player_identities,
    resolve_visual_player_identities,
)
from apc.player_identity import PlayerIdentityRegistry


def visible(*, street: str, board: list[str], pot: str, stacks: tuple[str, str]) -> dict[str, object]:
    return {
        "layout_id": "heads-up",
        "theme_id": "midnight",
        "street": street,
        "hero_seat": 1,
        "dealer_seat": 2,
        "hero_cards": ["Ah", "Kd"],
        "board_cards": board,
        "pot_bb": pot,
        "seat_stacks_bb": [
            {"seat_no": 1, "stack_bb": stacks[0]},
            {"seat_no": 2, "stack_bb": stacks[1]},
        ],
        "effective_stack_bb": min(stacks, key=float),
    }


def result(*, before: dict[str, object], after: dict[str, object], event: dict[str, object], frame: str) -> dict[str, object]:
    current = copy.deepcopy(after)
    current["observed_action"] = event
    return {
        "schema_version": "1.0.0",
        "previous_visible_state": before,
        "visible_state": current,
        "frames": {
            "before": {"image_sha256": "b" * 63 + frame},
            "after": {"image_sha256": "a" * 63 + frame},
        },
        "transition_audit": {"status": "accepted", "checks": [], "violations": []},
    }


class HandTrackerTests(unittest.TestCase):
    def test_tracks_complete_history_from_evidenced_hand_start(self) -> None:
        tracker = TemporalHandTracker()
        before = visible(street="preflop", board=[], pot="1.5", stacks=("100", "100"))
        after = visible(street="preflop", board=[], pot="2.5", stacks=("100", "99"))
        first = tracker.submit(
            "table",
            result(before=before, after=after, event={"actor_seat": 2, "action": "call", "amount_bb": "1"}, frame="1"),
            hand_start=True,
            boundary_confidence=1.0,
        )
        self.assertEqual(first["status"], "state_tracked_incomplete_identity")
        self.assertTrue(first["state"]["history_complete"])
        self.assertEqual(len(first["state"]["action_history"]), 1)

        flop_before = visible(street="flop", board=["2c", "7s", "Th"], pot="2.5", stacks=("100", "99"))
        flop_after = visible(street="flop", board=["2c", "7s", "Th"], pot="5", stacks=("100", "96.5"))
        second = tracker.submit(
            "table",
            result(before=flop_before, after=flop_after, event={"actor_seat": 2, "action": "bet", "amount_bb": "2.5"}, frame="2"),
            hand_start=False,
            boundary_confidence=0.0,
        )
        self.assertEqual(second["revision"], 1)
        self.assertEqual(len(second["state"]["action_history"]), 2)
        self.assertEqual(second["boundary_audit"]["status"], "accepted")

    def test_refuses_continuation_without_observed_hand_start(self) -> None:
        tracker = TemporalHandTracker()
        before = visible(street="flop", board=["2c", "7s", "Th"], pot="6", stacks=("100", "98"))
        after = copy.deepcopy(before)
        response = tracker.submit(
            "table",
            result(before=before, after=after, event={"actor_seat": 2, "action": "check"}, frame="3"),
            hand_start=False,
            boundary_confidence=0.0,
        )
        self.assertEqual(response["status"], "abstain_hand_boundary_unresolved")

    def test_rejects_cross_street_stack_discontinuity(self) -> None:
        tracker = TemporalHandTracker()
        before = visible(street="preflop", board=[], pot="1.5", stacks=("100", "100"))
        after = visible(street="preflop", board=[], pot="1.5", stacks=("100", "100"))
        tracker.submit(
            "table",
            result(before=before, after=after, event={"actor_seat": 2, "action": "check"}, frame="4"),
            hand_start=True,
            boundary_confidence=1.0,
        )
        broken = visible(street="flop", board=["2c", "7s", "Th"], pot="1.5", stacks=("100", "90"))
        response = tracker.submit(
            "table",
            result(before=broken, after=broken, event={"actor_seat": 2, "action": "check"}, frame="5"),
            hand_start=False,
            boundary_confidence=0.0,
        )
        self.assertEqual(response["status"], "abstain_cross_pair_transition_rejected")

    def test_duplicate_after_frame_does_not_append_history_twice(self) -> None:
        tracker = TemporalHandTracker()
        before = visible(street="preflop", board=[], pot="1.5", stacks=("100", "100"))
        after = visible(street="preflop", board=[], pot="1.5", stacks=("100", "100"))
        payload = result(before=before, after=after, event={"actor_seat": 2, "action": "check"}, frame="6")
        tracker.submit("table", payload, hand_start=True, boundary_confidence=1.0)
        duplicate = tracker.submit("table", payload, hand_start=False, boundary_confidence=0.0)
        self.assertEqual(duplicate["status"], "duplicate_frame_ignored")
        self.assertEqual(len(duplicate["state"]["action_history"]), 1)

    def test_resolved_unique_identities_clear_only_identity_missing_field(self) -> None:
        tracked = {
            "status": "state_tracked_incomplete_identity",
            "state": {
                "seat_stacks_bb": [
                    {"seat_no": 1, "stack_bb": "100"},
                    {"seat_no": 2, "stack_bb": "98"},
                ]
            },
            "missing_critical_fields": ["player_identities"],
            "recommendation": None,
        }
        enriched = attach_player_identities(
            tracked,
            [
                {"seat_no": 1, "status": "resolved", "identity_id": "hero-id", "profile_key": "p:hero", "display_name": "Hero", "posterior_probability": 1.0, "frames": 3},
                {"seat_no": 2, "status": "resolved", "identity_id": "villain-id", "profile_key": "p:villain", "display_name": "Villain", "posterior_probability": 1.0, "frames": 3},
            ],
        )
        self.assertEqual(enriched["status"], "state_tracked_identity_resolved")
        self.assertEqual(enriched["missing_critical_fields"], [])
        self.assertIsNone(enriched["recommendation"])

    def test_duplicate_identity_collision_keeps_gate_closed(self) -> None:
        tracked = {
            "status": "state_tracked_incomplete_identity",
            "state": {"seat_stacks_bb": [{"seat_no": 1, "stack_bb": "100"}, {"seat_no": 2, "stack_bb": "98"}]},
            "missing_critical_fields": ["player_identities"],
            "recommendation": None,
        }
        enriched = attach_player_identities(
            tracked,
            [
                {"seat_no": 1, "status": "resolved", "identity_id": "same"},
                {"seat_no": 2, "status": "resolved", "identity_id": "same"},
            ],
        )
        self.assertEqual(enriched["identity_gate"]["status"], "unresolved")
        self.assertIn("player_identities", enriched["missing_critical_fields"])

    def test_repeated_visual_tokens_resolve_pseudonymous_profiles_without_ocr_claim(self) -> None:
        registry = PlayerIdentityRegistry("visual-training")
        enriched = None
        for frame in range(1, 4):
            tracked = {
                "track_id": "table-visual",
                "status": "state_tracked_incomplete_identity",
                "state": {
                    "seat_stacks_bb": [
                        {"seat_no": 1, "stack_bb": "100"},
                        {"seat_no": 2, "stack_bb": "98"},
                    ],
                    "visual_identity_signatures": [
                        {
                            "seat_no": 1,
                            "signature_sha256": "1" * 64,
                            "visual_token": f"visual:{'1' * 32}",
                            "quality_score": 0.99,
                            "frame_sha256": f"{frame:064x}",
                        },
                        {
                            "seat_no": 2,
                            "signature_sha256": "2" * 64,
                            "visual_token": f"visual:{'2' * 32}",
                            "quality_score": 0.99,
                            "frame_sha256": f"{frame + 10:064x}",
                        },
                    ],
                },
                "missing_critical_fields": ["player_identities"],
                "recommendation": None,
            }
            enriched = resolve_visual_player_identities(
                tracked,
                registry,
                observed_at_ms=frame * 100,
            )
        self.assertEqual(enriched["identity_gate"]["status"], "passed")
        self.assertEqual(
            enriched["identity_gate"]["evidence_kind"],
            "visual_name_band_signature_not_ocr",
        )
        self.assertFalse(enriched["identity_gate"]["human_readable_names"])

    def test_repeated_ocr_names_resolve_human_readable_profiles(self) -> None:
        registry = PlayerIdentityRegistry("ocr-training")
        enriched = None
        for frame in range(1, 4):
            tracked = {
                "track_id": "table-ocr",
                "status": "state_tracked_incomplete_identity",
                "state": {
                    "seat_stacks_bb": [
                        {"seat_no": 1, "stack_bb": "100"},
                        {"seat_no": 2, "stack_bb": "98"},
                    ],
                    "recognized_player_names": [
                        {
                            "seat_no": 1,
                            "player_name": "PLAYER01",
                            "confidence": 0.99,
                            "frame_sha256": f"{frame:064x}",
                        },
                        {
                            "seat_no": 2,
                            "player_name": "VILLAIN2",
                            "confidence": 0.99,
                            "frame_sha256": f"{frame + 10:064x}",
                        },
                    ],
                },
                "missing_critical_fields": ["player_identities"],
                "recommendation": None,
            }
            enriched = resolve_ocr_player_identities(
                tracked,
                registry,
                observed_at_ms=frame * 100,
            )
        self.assertEqual(enriched["identity_gate"]["status"], "passed")
        self.assertEqual(enriched["identity_gate"]["evidence_kind"], "human_readable_name_ocr")
        self.assertTrue(enriched["identity_gate"]["human_readable_names"])
        self.assertEqual(
            [row["display_name"] for row in enriched["state"]["player_identities"]],
            ["PLAYER01", "VILLAIN2"],
        )


if __name__ == "__main__":
    unittest.main()
