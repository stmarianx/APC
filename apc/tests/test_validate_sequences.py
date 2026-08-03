from __future__ import annotations

import copy
import unittest

from apc.tools.validate_sequences import _audit_pair


def frame(*, pot: str, stacks: tuple[str, str], history: list[dict[str, object]], event: dict[str, object] | None) -> dict[str, object]:
    return {
        "state": {
            "table_id": "table-1",
            "hand_id": "hand-1",
            "street": "flop",
            "hero_seat": 1,
            "dealer_seat": 2,
            "pot_bb": pot,
            "action_history": history,
        },
        "objects": {
            "hero_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "d"}],
            "board_cards": [
                {"rank": "2", "suit": "c"},
                {"rank": "7", "suit": "s"},
                {"rank": "T", "suit": "h"},
            ],
            "seats": [
                {"seat_no": 1, "stack_bb": stacks[0]},
                {"seat_no": 2, "stack_bb": stacks[1]},
            ],
            "observed_action": event,
        },
    }


class SequenceValidatorTests(unittest.TestCase):
    def test_pair_accepts_exact_history_stack_and_pot_conservation(self) -> None:
        event = {"actor_seat": 2, "action": "bet", "amount_bb": "2.5"}
        before = frame(pot="6", stacks=("100", "98"), history=[], event=None)
        after = frame(pot="8.5", stacks=("100", "95.5"), history=[event], event=event)
        errors: list[str] = []
        _audit_pair(before, after, pair_label="pair", errors=errors)
        self.assertEqual(errors, [])

    def test_pair_rejects_history_that_does_not_append_observed_event(self) -> None:
        event = {"actor_seat": 2, "action": "check"}
        before = frame(pot="6", stacks=("100", "98"), history=[], event=None)
        after = frame(pot="6", stacks=("100", "98"), history=[], event=event)
        errors: list[str] = []
        _audit_pair(before, after, pair_label="pair", errors=errors)
        self.assertTrue(any("append exactly" in error for error in errors))

    def test_pair_rejects_card_mutation(self) -> None:
        event = {"actor_seat": 2, "action": "check"}
        before = frame(pot="6", stacks=("100", "98"), history=[], event=None)
        after = frame(pot="6", stacks=("100", "98"), history=[event], event=event)
        after = copy.deepcopy(after)
        after["objects"]["board_cards"][0]["rank"] = "3"
        errors: list[str] = []
        _audit_pair(before, after, pair_label="pair", errors=errors)
        self.assertTrue(any("visible cards change" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
