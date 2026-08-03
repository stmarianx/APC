from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import unittest

from poker_coach.models import Card, HoleCards
from poker_coach.pokerstars import PokerStarsParser
from poker_coach.range_calibration import score_opponent_range_timelines
from poker_coach.range_timeline import build_opponent_range_timelines
from poker_coach.solver_import import SolverBundleImporter
from poker_coach.solutions import ActionSolution


ROOT = Path(__file__).resolve().parents[1]
D = Decimal


class RangeCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hand = PokerStarsParser().parse_file(
            ROOT / "examples" / "sample_play_money_hand.txt"
        )[0]
        self.solutions = SolverBundleImporter().parse_file(
            ROOT / "examples" / "sample_solver_bundle.json"
        ).spots

    def revealed_hand(self, cards=("Kc", "Qh")):
        return replace(
            self.hand,
            hole_cards=self.hand.hole_cards
            + (
                HoleCards(
                    "VillainB",
                    tuple(Card.parse(card) for card in cards),
                    shown=True,
                ),
            ),
        )

    def score(self, hand, solutions=None):
        spots = self.solutions if solutions is None else tuple(solutions)
        timeline = build_opponent_range_timelines((hand,), spots)
        return score_opponent_range_timelines((hand,), spots, timeline)

    def test_revealed_combo_scores_probability_trajectory_and_proper_losses(self) -> None:
        result = self.score(self.revealed_hand())
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["scored_predictions"], 2)
        self.assertEqual(aggregate["support_coverage"], "1")
        self.assertEqual(aggregate["infinite_log_loss_predictions"], 0)
        predictions = result["timelines"][0]["predictions"]
        self.assertEqual([row["street"] for row in predictions], ["flop", "turn"])
        self.assertEqual(predictions[0]["actual_combo"], "Kc Qh")
        self.assertEqual(D(predictions[0]["actual_prior_probability"]), D("1") / 3)
        self.assertGreater(
            D(predictions[0]["actual_posterior_probability"]), D("0.5")
        )
        self.assertEqual(predictions[0]["posterior_rank"], 1)
        self.assertGreater(D(predictions[0]["log_loss_bits"]), 0)
        self.assertGreater(D(predictions[0]["multiclass_brier_score"]), 0)
        self.assertEqual(
            sum(bucket["observations"] for bucket in result["calibration_buckets"]),
            6,
        )

    def test_unrevealed_and_out_of_support_hands_are_not_fabricated(self) -> None:
        unshown = self.score(self.hand)
        self.assertEqual(unshown["aggregate"]["scored_predictions"], 0)
        self.assertEqual(unshown["aggregate"]["unrevealed_timelines"], 1)
        self.assertEqual(unshown["timelines"][0]["status"], "not_revealed")

        unsupported = self.score(self.revealed_hand(("5c", "4d")))
        self.assertEqual(unsupported["aggregate"]["scored_predictions"], 0)
        self.assertEqual(unsupported["aggregate"]["support_misses"], 2)
        self.assertTrue(
            all(
                row["mapping_status"] == "revealed_combo_outside_imported_support"
                for row in unsupported["timelines"][0]["predictions"]
            )
        )

    def test_revealed_combo_maps_through_global_suit_renaming(self) -> None:
        suit_map = {"c": "s", "s": "c", "d": "h", "h": "d"}

        def rename(card):
            return Card(card.rank, suit_map[card.suit])

        renamed = tuple(
            replace(
                spot,
                key=replace(
                    spot.key,
                    board=tuple(rename(card) for card in spot.key.board),
                    hero_cards=tuple(rename(card) for card in spot.key.hero_cards),
                ),
            )
            for spot in self.solutions
        )
        result = self.score(self.revealed_hand(), renamed)
        self.assertEqual(result["aggregate"]["scored_predictions"], 2)
        first = result["timelines"][0]["predictions"][0]
        self.assertEqual(first["mapping_status"], "mapped")
        self.assertEqual(first["actual_combo"], "Ks Qd")

    def test_zero_probability_reveal_reports_infinite_log_loss(self) -> None:
        adjusted = []
        for spot in self.solutions:
            if spot.node_id == "sample_hand_flop_bb_kq":
                spot = replace(
                    spot,
                    actions=(
                        ActionSolution("check", D("0"), D("1.18")),
                        ActionSolution("bet:0.5", D("1"), D("1.09")),
                    ),
                )
            adjusted.append(spot)
        result = self.score(self.revealed_hand(), adjusted)
        self.assertEqual(
            result["aggregate"]["infinite_log_loss_predictions"], 2
        )
        self.assertEqual(result["aggregate"]["mean_log_loss_bits"], "infinite")
        self.assertTrue(
            all(
                row["log_loss_bits"] == "infinite"
                for row in result["timelines"][0]["predictions"]
            )
        )


if __name__ == "__main__":
    unittest.main()
