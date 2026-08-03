from pathlib import Path
import unittest

from poker_coach import SolverExportRegistry, SolverPracticeService
from poker_coach.range_strategy import public_node_fingerprint


ROOT = Path(__file__).resolve().parents[1]


class SolverPracticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parsed = SolverExportRegistry().parse_file(
            ROOT / "examples" / "sample_solver_export.csv"
        )
        cls.spots = parsed.bundle.spots

    def setUp(self) -> None:
        self.service = SolverPracticeService()

    def test_challenge_hides_solver_frequencies_and_evs(self) -> None:
        target = self.spots[1]
        challenge = self.service.create(
            self.spots, public_fingerprint=public_node_fingerprint(target)
        )
        self.assertTrue(challenge["strategy_hidden"])
        self.assertEqual(challenge["node_id"], target.node_id)
        encoded = repr(challenge)
        self.assertNotIn("frequency", encoded)
        self.assertNotIn("ev_bb", encoded)
        self.assertEqual(
            {action["action_id"] for action in challenge["actions"]},
            {action.action for action in target.actions},
        )

    def test_submit_reveals_strategy_and_exact_ev_loss(self) -> None:
        target = self.spots[-1]
        challenge = self.service.create(
            self.spots, public_fingerprint=public_node_fingerprint(target)
        )
        chosen = min(target.actions, key=lambda action: action.ev)
        result = self.service.submit(challenge["session_id"], chosen.action)
        self.assertEqual(result["chosen_action"], chosen.action)
        self.assertEqual(result["ev_loss_bb"], format(target.best_ev - chosen.ev, "f"))
        self.assertEqual(len(result["strategy"]), len(target.actions))
        self.assertEqual(result["source"], target.source)

    def test_illegal_action_does_not_consume_attempt_then_duplicate_fails(self) -> None:
        target = self.spots[0]
        challenge = self.service.create(
            self.spots, public_fingerprint=public_node_fingerprint(target)
        )
        with self.assertRaisesRegex(ValueError, "Illegal solved action"):
            self.service.submit(challenge["session_id"], "not-legal")
        self.service.submit(challenge["session_id"], target.actions[0].action)
        with self.assertRaisesRegex(ValueError, "already answered"):
            self.service.submit(challenge["session_id"], target.actions[0].action)

    def test_unknown_public_node_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "No imported private-hand solution"):
            self.service.create(self.spots, public_fingerprint="0" * 64)


if __name__ == "__main__":
    unittest.main()
