from decimal import Decimal
import unittest

from poker_coach import PokerStarsParser, ProfileBook, position_map
from poker_coach.profiles import Tendency
from test_pokerstars import HAND


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hand = PokerStarsParser().parse(HAND)
        self.book = ProfileBook()
        self.book.observe_hand(self.hand)

    def test_six_max_positions(self) -> None:
        positions = position_map(self.hand)
        self.assertEqual(positions["zawaaa"], "BTN")
        self.assertEqual(positions["13_Xerxes_13"], "SB")
        self.assertEqual(positions["thecooler992"], "BB")
        self.assertEqual(positions["Senecady"], "HJ")
        self.assertEqual(positions["Joe Hahn"], "CO")

    def test_vpip_and_pfr_are_opportunity_based(self) -> None:
        hero = self.book.profile("Senecady")
        self.assertEqual(hero.estimate(Tendency.VPIP, position="HJ").mean, Decimal(2) / Decimal(3))
        self.assertEqual(hero.estimate(Tendency.PFR, position="HJ").mean, Decimal(2) / Decimal(3))
        folded_utg = self.book.profile("jeckjeck")
        self.assertEqual(folded_utg.estimate(Tendency.VPIP, position="UTG").mean, Decimal(1) / Decimal(3))

    def test_three_bet_and_fold_to_three_bet(self) -> None:
        joe = self.book.profile("Joe Hahn")
        self.assertEqual(joe.estimate(Tendency.THREE_BET, position="CO").mean, Decimal(2) / Decimal(3))
        button = self.book.profile("zawaaa")
        self.assertEqual(button.estimate(Tendency.FOLD_TO_THREE_BET, position="BTN").mean, Decimal(2) / Decimal(3))
        big_blind = self.book.profile("thecooler992")
        self.assertEqual(big_blind.estimate(Tendency.FOLD_TO_THREE_BET, position="BB").mean, Decimal(1) / Decimal(3))

    def test_showdown_is_conditioned_on_seeing_the_flop(self) -> None:
        hero = self.book.profile("Senecady")
        self.assertEqual(hero.estimate(Tendency.WENT_TO_SHOWDOWN, position="HJ").mean, Decimal(2) / Decimal(3))
        folded_preflop = self.book.profile("jeckjeck")
        self.assertEqual(folded_preflop.estimate(Tendency.WENT_TO_SHOWDOWN, position="UTG").mean, Decimal("0.5"))


if __name__ == "__main__":
    unittest.main()
