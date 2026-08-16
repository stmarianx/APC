from __future__ import annotations

import unittest
from decimal import Decimal

from apc.deadline import ActionCommand
from apc.evaluate_full_hand_table import evaluate_full_hand_table
from apc.full_hand_table import HeadsUpVirtualHand


class HeadsUpVirtualHandTests(unittest.TestCase):
    def assert_conserved(self, hand: HeadsUpVirtualHand) -> None:
        observation = hand.observation()
        stacks = sum((Decimal(value) for value in observation["stacks_bb"].values()), Decimal("0"))
        self.assertEqual(stacks + Decimal(observation["pot_bb"]), Decimal("200"))

    def test_seeded_deal_is_deterministic_and_private(self) -> None:
        first = HeadsUpVirtualHand(seed=7).observation()
        second = HeadsUpVirtualHand(seed=7).observation()
        different = HeadsUpVirtualHand(seed=8).observation()
        self.assertEqual(first["state_fingerprint"], second["state_fingerprint"])
        self.assertNotEqual(first["hero_cards"], different["hero_cards"])
        self.assertIsNone(first["opponent_cards"])
        self.assertFalse(first["provider"]["external_actuation"])

    def test_checkdown_completes_full_hand_with_zero_sum_showdown(self) -> None:
        hand = HeadsUpVirtualHand(seed=17)
        actions = 0
        final = None
        while not hand.terminal:
            self.assert_conserved(hand)
            buttons = {row["action"]: row for row in hand.legal_action_buttons()}
            command = ActionCommand("check") if "check" in buttons else ActionCommand("call")
            final = hand.step(command)
            actions += 1
        assert final is not None
        self.assertEqual(actions, 8)
        feedback = final["completed_hand_feedback"]
        self.assertTrue(feedback["full_hand_completed"])
        self.assertEqual(feedback["terminal_reason"], "showdown")
        self.assertEqual(len(feedback["board"]), 5)
        self.assertTrue(feedback["showdown_revealed"])
        self.assertEqual(sum((Decimal(value) for value in feedback["rewards_bb"].values()), Decimal("0")), Decimal("0"))
        self.assert_conserved(hand)

    def test_preflop_fold_settles_blinds_exactly(self) -> None:
        hand = HeadsUpVirtualHand(seed=1, button_player=0)
        feedback = hand.step(ActionCommand("fold"))["completed_hand_feedback"]
        self.assertEqual(feedback["winner"], "Villain")
        self.assertEqual(feedback["final_pot_bb"], "1.5")
        self.assertEqual(feedback["rewards_bb"], {"Hero": "-0.5", "Villain": "0.5"})
        self.assertFalse(feedback["showdown_revealed"])

    def test_all_in_call_runs_board_and_rejects_future_steps(self) -> None:
        hand = HeadsUpVirtualHand(seed=22)
        first = hand.step(ActionCommand("all_in"))
        self.assertFalse(first["terminal"])
        final = hand.step(ActionCommand("call"))
        self.assertTrue(final["terminal"])
        self.assertEqual(final["completed_hand_feedback"]["final_pot_bb"], "200")
        self.assertEqual(final["command"]["amount_bb"], "99")
        with self.assertRaisesRegex(ValueError, "already terminal"):
            hand.step(ActionCommand("check"))

    def test_raise_bounds_and_visible_call_price_are_enforced(self) -> None:
        hand = HeadsUpVirtualHand(seed=3)
        with self.assertRaisesRegex(ValueError, "outside the legal BB range"):
            hand.step(ActionCommand("raise", to_amount_bb="1.5"))
        raised = hand.step(ActionCommand("raise", to_amount_bb="3"))
        self.assertEqual(raised["command"]["amount_bb"], "2.5")
        self.assertEqual(hand.observation()["to_call_bb"], "2")
        with self.assertRaisesRegex(ValueError, "call amount does not match"):
            hand.step(ActionCommand("call", amount_bb="1"))
        called = hand.step(ActionCommand("call", amount_bb="2"))
        self.assertEqual(called["state"]["street"], "flop")
        self.assertEqual(called["state"]["pot_bb"], "6")

    def test_multi_hand_audit_is_deterministic_and_complete(self) -> None:
        first = evaluate_full_hand_table(hands=10, seed_start=50)
        second = evaluate_full_hand_table(hands=10, seed_start=50)
        self.assertTrue(first["passed"])
        self.assertEqual(first["metrics"]["replay_mismatches"], 0)
        self.assertEqual(first["metrics"]["conservation_failures"], 0)
        self.assertEqual(first["metrics"]["card_uniqueness_failures"], 0)
        self.assertEqual(first["metrics"]["zero_sum_failures"], 0)
        self.assertEqual(first["metrics"]["external_actuation_violations"], 0)
        self.assertEqual(first["metrics"]["action_kinds_covered"], [
            "all_in", "bet", "call", "check", "fold", "raise"
        ])
        self.assertEqual(
            [row["terminal_fingerprint"] for row in first["rows"]],
            [row["terminal_fingerprint"] for row in second["rows"]],
        )


if __name__ == "__main__":
    unittest.main()
