from decimal import Decimal
import unittest

from poker_coach import HandReplayer, PokerStarsParser
from test_pokerstars import HAND


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hand = PokerStarsParser().parse(HAND)
        self.result = HandReplayer().replay(self.hand)

    def test_pre_action_state_and_call_price(self) -> None:
        first = self.result.decisions[0]
        self.assertEqual(first.actor, "jeckjeck")
        self.assertEqual(first.pot, Decimal("6"))
        self.assertEqual(first.current_bet, Decimal("4"))
        self.assertEqual(first.to_call, Decimal("4"))
        self.assertEqual(first.call_amount, Decimal("4"))
        self.assertEqual(first.board, ())

    def test_raise_targets_and_return_reconcile_pot(self) -> None:
        joe = next(snapshot for snapshot in self.result.decisions if snapshot.actor == "Joe Hahn")
        self.assertEqual(joe.pot, Decimal("123"))
        self.assertEqual(joe.to_call, Decimal("117"))
        self.assertEqual(self.result.committed_pot, Decimal("404.10"))
        self.assertEqual(self.result.awarded_total, Decimal("401.10"))
        self.assertEqual(self.result.rake, Decimal("3"))
        self.assertEqual(self.result.reconciliation_error, Decimal("0.00"))

    def test_folded_players_are_marked(self) -> None:
        players = {player.player: player for player in self.result.players}
        self.assertTrue(players["jeckjeck"].folded)
        self.assertTrue(players["zawaaa"].folded)
        self.assertFalse(players["Senecady"].folded)


if __name__ == "__main__":
    unittest.main()

