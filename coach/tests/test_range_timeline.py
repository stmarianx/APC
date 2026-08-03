from decimal import Decimal
from pathlib import Path
import unittest

from poker_coach.models import ActionKind, Card, Street
from poker_coach.pokerstars import PokerStarsParser
from poker_coach.range_timeline import (
    build_opponent_range_timelines,
    opponent_decision_contexts,
)
from poker_coach.solutions import ActionSolution, SolutionKey, SolvedSpot


ROOT = Path(__file__).resolve().parents[1]
D = Decimal


class OpponentRangeTimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hand = PokerStarsParser().parse_file(
            ROOT / "examples" / "sample_play_money_hand.txt"
        )[0]
        contexts = opponent_decision_contexts(self.hand)
        self.flop = next(
            row
            for row in contexts
            if row.actor == "VillainB"
            and row.street == Street.FLOP
            and row.action.kind == ActionKind.CHECK
        )
        self.turn = next(
            row
            for row in contexts
            if row.actor == "VillainB"
            and row.street == Street.TURN
            and row.action.kind == ActionKind.CHECK
        )
        self.combos = (
            (Card.parse("Kc"), Card.parse("Qh")),
            (Card.parse("Js"), Card.parse("Td")),
        )

    @staticmethod
    def spot(context, cards, check, *, node_id):
        return SolvedSpot(
            key=SolutionKey(
                game=context.game,
                players=context.players,
                hero_position=context.actor_position,
                effective_stack_bb=context.effective_stack_bb,
                pot_bb=context.pot_bb,
                board=context.board,
                action_history=context.action_history,
                rake_model="play_money_fixture",
                allowed_sizes=(D("0.5"),),
                hero_cards=cards,
            ),
            actions=(
                ActionSolution("check", D(check), D("1")),
                ActionSolution("bet:0.5", D("1") - D(check), D("0.9")),
            ),
            source="timeline_fixture",
            source_version="1",
            node_id=node_id,
        )

    def two_street_solutions(self):
        return (
            self.spot(self.flop, self.combos[0], "0.8", node_id="flop_kq"),
            self.spot(self.flop, self.combos[1], "0.2", node_id="flop_jt"),
            self.spot(self.turn, self.combos[0], "0.25", node_id="turn_kq"),
            self.spot(self.turn, self.combos[1], "0.75", node_id="turn_jt"),
        )

    def test_saved_actions_create_sequential_bayesian_timeline(self) -> None:
        result = build_opponent_range_timelines(
            (self.hand,), self.two_street_solutions()
        )
        self.assertEqual(result["public_state_matches"], 2)
        self.assertEqual(result["conditioned_actions"], 2)
        timeline = result["timelines"][0]
        self.assertEqual(timeline["opponent"], "VillainB")
        first, second = timeline["events"]
        self.assertEqual(first["street"], "flop")
        self.assertEqual(first["observed_action"], "check")
        self.assertEqual(
            D(first["posterior"]["action_probability_under_prior"]), D("0.5")
        )
        first_weights = {
            row["combo"]: D(row["posterior"])
            for row in first["posterior"]["combos"]
        }
        self.assertEqual(first_weights["Kc Qh"], D("0.8"))
        self.assertEqual(second["prior_transition"]["mode"], "posterior_carried")
        self.assertEqual(second["prior_transition"]["unmatched_actions_skipped"], 1)
        self.assertEqual(
            D(second["posterior"]["action_probability_under_prior"]), D("0.35")
        )
        second_weights = {
            row["combo"]: D(row["posterior"])
            for row in second["posterior"]["combos"]
        }
        self.assertAlmostEqual(float(second_weights["Kc Qh"]), 4 / 7)
        self.assertAlmostEqual(float(second_weights["Js Td"]), 3 / 7)

    def test_changed_combo_coverage_resets_instead_of_inventing_mass(self) -> None:
        solutions = self.two_street_solutions()[:2] + (
            self.spot(self.turn, self.combos[0], "0.25", node_id="turn_kq"),
        )
        result = build_opponent_range_timelines((self.hand,), solutions)
        second = result["timelines"][0]["events"][1]
        self.assertEqual(second["prior_transition"]["mode"], "uniform_reset")
        self.assertEqual(
            second["prior_transition"]["reason"], "exact_combo_coverage_changed"
        )
        self.assertEqual(second["posterior"]["combos"][0]["posterior"], "1")

    def test_public_match_with_uncovered_action_is_reported_not_conditioned(self) -> None:
        call_context = next(
            row
            for row in opponent_decision_contexts(self.hand)
            if row.actor == "VillainB"
            and row.street == Street.FLOP
            and row.action.kind == ActionKind.CALL
        )
        solutions = (
            self.spot(call_context, self.combos[0], "0.6", node_id="call_kq"),
            self.spot(call_context, self.combos[1], "0.4", node_id="call_jt"),
        )
        result = build_opponent_range_timelines((self.hand,), solutions)
        event = result["timelines"][0]["events"][0]
        self.assertEqual(event["status"], "action_not_covered")
        self.assertIsNone(event["observed_action"])
        self.assertEqual(result["conditioning_failures"], 1)
        self.assertEqual(result["conditioned_actions"], 0)

    def test_no_solutions_has_explicit_zero_coverage(self) -> None:
        result = build_opponent_range_timelines((self.hand,), ())
        self.assertGreater(result["opponent_decisions"], 0)
        self.assertEqual(result["public_state_matches"], 0)
        self.assertEqual(result["coverage"], "0")
        self.assertEqual(result["timelines"], [])


if __name__ == "__main__":
    unittest.main()
