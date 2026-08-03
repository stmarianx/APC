from __future__ import annotations

import unittest

from apc.synthetic.render_events import EVENT_ACTIONS, plan_event


class SyntheticEventTests(unittest.TestCase):
    def test_plans_cover_all_action_classes(self) -> None:
        actions = {
            plan_event(session_index=index, street_index=0, seats=6, street="flop").action
            for index in range(len(EVENT_ACTIONS))
        }
        self.assertEqual(actions, set(EVENT_ACTIONS))

    def test_bet_event_conserves_visible_stack_and_pot_delta(self) -> None:
        event = plan_event(session_index=3, street_index=0, seats=6, street="flop")
        self.assertEqual(event.action, "bet")
        self.assertEqual(event.amount_bb, "2.5")
        self.assertEqual(float(event.stack_before_bb) - float(event.stack_after_bb), 2.5)
        self.assertEqual(float(event.pot_after_bb) - float(event.pot_before_bb), 2.5)
        self.assertEqual(event.hero_to_call_after_bb, "2.5")

    def test_fold_event_changes_status_without_chip_delta(self) -> None:
        event = plan_event(session_index=0, street_index=0, seats=2, street="preflop")
        self.assertEqual(event.action, "fold")
        self.assertEqual(event.stack_before_bb, event.stack_after_bb)
        self.assertEqual(event.pot_before_bb, event.pot_after_bb)
        self.assertEqual(event.actor_status_after, "folded")


if __name__ == "__main__":
    unittest.main()
