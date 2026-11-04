import unittest
from apc.practice_session import PracticeTableSession
from apc.neural.contract import validate_completed_hand_replay


class PracticeSessionTests(unittest.TestCase):
    def test_stale_and_illegal_commands_do_not_advance(self):
        session = PracticeTableSession(81)
        state = session.snapshot()["state"]
        for action, fingerprint in (("fold", "stale"), ("check", state["state_fingerprint"])):
            with self.assertRaises(ValueError):
                session.act(action, fingerprint)
            self.assertEqual(session.snapshot()["state"], state)
        with self.assertRaises(ValueError):
            session.completed_replay("session")

    def test_full_hand_reaches_terminal_with_eligible_replay(self):
        session = PracticeTableSession(82)
        while not session.hand.terminal:
            state = session.snapshot()["state"]
            legal = {row["action"] for row in state["action_buttons"]}
            session.act("check" if "check" in legal else "call", state["state_fingerprint"])
        report = validate_completed_hand_replay(session.completed_replay("session"))
        self.assertTrue(report["valid"], report)
        self.assertIsNotNone(session.snapshot()["feedback"])
        self.assertFalse(session.snapshot()["external_actuation"])
