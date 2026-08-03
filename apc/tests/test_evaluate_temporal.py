from __future__ import annotations

import unittest

from apc.perception.evaluate_temporal import _expected_after


class TemporalEvaluatorTests(unittest.TestCase):
    def test_expected_multiway_state_keeps_scalar_effective_stack_null(self) -> None:
        annotation = {
            "environment": {"layout_id": "six-max", "theme_id": "midnight"},
            "state": {
                "street": "preflop",
                "legal_actions": ["fold", "call", "raise"],
                "hero_seat": 1,
                "dealer_seat": 2,
                "pot_bb": "2.5",
                "to_call_bb": "1",
            },
            "objects": {
                "hero_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "d"}],
                "board_cards": [],
                "seats": [
                    {"seat_no": seat, "stack_bb": str(100 - seat)}
                    for seat in range(1, 7)
                ],
                "observed_action": {"actor_seat": 3, "action": "call", "amount_bb": "1"},
            },
        }
        expected = _expected_after(annotation)
        self.assertIsNone(expected["effective_stack_bb"])
        self.assertEqual(expected["hero_cards"], ["Ah", "Kd"])
        self.assertEqual(
            expected["effective_stacks_by_opponent_bb"],
            [
                {"opponent_seat": 2, "effective_stack_bb": "98"},
                {"opponent_seat": 3, "effective_stack_bb": "97"},
                {"opponent_seat": 4, "effective_stack_bb": "96"},
                {"opponent_seat": 5, "effective_stack_bb": "95"},
                {"opponent_seat": 6, "effective_stack_bb": "94"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
