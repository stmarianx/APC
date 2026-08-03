from decimal import Decimal
import unittest

from poker_coach.trainer import ScenarioLibrary, TrainingService


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = ScenarioLibrary.bundled()
        self.service = TrainingService(self.library)

    def test_bundled_scenarios_are_valid_and_provenance_is_explicit(self) -> None:
        self.assertEqual(len(self.library.all()), 6)
        for scenario in self.library.all():
            self.assertEqual(sum(action.frequency for action in scenario.actions), Decimal("1"))
            self.assertFalse(scenario.provenance.solver_verified)
            self.assertEqual(scenario.provenance.tier, "educational_baseline")
            self.assertGreater(scenario.mathematical_context()["spr"], 0)

    def test_strategy_is_hidden_until_decision(self) -> None:
        scenario = self.library.all()[0]
        hidden = scenario.to_dict()
        revealed = scenario.to_dict(reveal_strategy=True)
        self.assertNotIn("frequency", hidden["actions"][0])
        self.assertNotIn("explanation", hidden)
        self.assertIn("frequency", revealed["actions"][0])
        self.assertIn("explanation", revealed)

    def test_seeded_session_and_feedback(self) -> None:
        first = self.service.create_session(count=2, seed=11)
        second = self.service.create_session(count=2, seed=11)
        self.assertEqual(first["scenario"]["scenario_id"], second["scenario"]["scenario_id"])
        action_id = first["scenario"]["actions"][0]["action_id"]
        result = self.service.submit_decision(first["session_id"], action_id)
        self.assertIn(result["feedback"]["grade"], {"excellent", "good", "review", "major_leak"})
        self.assertEqual(result["progress"]["answered"], 1)
        self.assertIn("provenance", result["feedback"]["scenario"])

    def test_illegal_and_duplicate_decisions_fail(self) -> None:
        session = self.service.create_session(count=1, seed=2)
        with self.assertRaises(ValueError):
            self.service.submit_decision(session["session_id"], "teleport")
        valid = session["scenario"]["actions"][0]["action_id"]
        self.service.submit_decision(session["session_id"], valid)
        with self.assertRaises(ValueError):
            self.service.submit_decision(session["session_id"], valid)


if __name__ == "__main__":
    unittest.main()
