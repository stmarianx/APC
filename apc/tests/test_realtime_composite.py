from __future__ import annotations

import unittest

from apc.perception.evaluate_realtime_composite import supported_state_checks


class RealtimeCompositeAuditTests(unittest.TestCase):
    def test_supported_state_audit_covers_names_clock_and_bb_values(self) -> None:
        annotation = {
            "environment": {"layout_id": "heads-up", "theme_id": "midnight"},
            "state": {
                "street": "flop",
                "legal_actions": ["check", "bet"],
                "hero_seat": 1,
                "dealer_seat": 2,
                "pot_bb": "6.0",
                "to_call_bb": "0",
            },
            "objects": {
                "hero_cards": [{"rank": "A", "suit": "h"}, {"rank": "Q", "suit": "h"}],
                "board_cards": [
                    {"rank": "2", "suit": "s"},
                    {"rank": "8", "suit": "d"},
                    {"rank": "5", "suit": "s"},
                ],
                "seats": [
                    {"seat_no": 1, "player_name": "PLAYER01", "stack_bb": "100.0"},
                    {"seat_no": 2, "player_name": "VILLAIN2", "stack_bb": "98"},
                ],
                "turn_clock": {"remaining_ms": 12_000},
            },
        }
        visible = {
            "layout_id": "heads-up",
            "theme_id": "midnight",
            "street": "flop",
            "legal_actions": ["check", "bet"],
            "hero_seat": 1,
            "dealer_seat": 2,
            "hero_cards": ["Ah", "Qh"],
            "board_cards": ["2s", "8d", "5s"],
            "pot_bb": "6",
            "to_call_bb": "0.0",
            "seat_stacks_bb": [
                {"seat_no": 1, "stack_bb": "100"},
                {"seat_no": 2, "stack_bb": "98.0"},
            ],
            "recognized_player_names": [
                {"seat_no": 1, "player_name": "PLAYER01"},
                {"seat_no": 2, "player_name": "VILLAIN2"},
            ],
            "decision_time_remaining_ms": 12_000,
        }
        checks = supported_state_checks(annotation, visible)
        self.assertTrue(all(checks.values()), checks)

    def test_supported_state_audit_rejects_wrong_clock(self) -> None:
        annotation = {
            "environment": {"layout_id": "heads-up", "theme_id": "midnight"},
            "state": {
                "street": "preflop",
                "legal_actions": ["fold"],
                "hero_seat": 1,
                "dealer_seat": 1,
                "pot_bb": "1.5",
                "to_call_bb": "0.5",
            },
            "objects": {
                "hero_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "d"}],
                "board_cards": [],
                "seats": [{"seat_no": 1, "player_name": "PLAYER01", "stack_bb": "100"}],
                "turn_clock": {"remaining_ms": 5_000},
            },
        }
        visible = {
            "layout_id": "heads-up",
            "theme_id": "midnight",
            "street": "preflop",
            "legal_actions": ["fold"],
            "hero_seat": 1,
            "dealer_seat": 1,
            "hero_cards": ["Ah", "Kd"],
            "board_cards": [],
            "pot_bb": "1.5",
            "to_call_bb": "0.5",
            "seat_stacks_bb": [{"seat_no": 1, "stack_bb": "100"}],
            "recognized_player_names": [{"seat_no": 1, "player_name": "PLAYER01"}],
            "decision_time_remaining_ms": 8_000,
        }
        checks = supported_state_checks(annotation, visible)
        self.assertFalse(checks["decision_time_remaining_ms"])
        self.assertFalse(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
