from decimal import Decimal
from pathlib import Path
import unittest

from poker_coach import ActionKind, PokerStarsParser, Street


HAND = """PokerStars Hand #40000000000: Hold'em No Limit ($2/$4 USD) - 2010/02/19 12:43:11 ET
Table 'Naef III' 6-max Seat #3 is the button
Seat 1: Senecady ($117 in chips)
Seat 2: Joe Hahn ($412 in chips)
Seat 3: zawaaa ($266.45 in chips)
Seat 4: 13_Xerxes_13 ($400 in chips)
Seat 5: thecooler992 ($142.55 in chips)
Seat 6: jeckjeck ($515.40 in chips)
13_Xerxes_13: posts small blind $2
thecooler992: posts big blind $4
*** HOLE CARDS ***
Dealt to Senecady [Kc Ts]
jeckjeck: folds
Senecady: raises $113 to $117 and is all-in
Joe Hahn: raises $295 to $412 and is all-in
zawaaa: folds
13_Xerxes_13: folds
thecooler992: calls $138.55 and is all-in
Uncalled bet ($269.45) returned to Joe Hahn
*** FLOP *** [7s Qh 7d]
*** TURN *** [7s Qh 7d] [Tc]
*** RIVER *** [7s Qh 7d Tc] [7c]
*** SHOW DOWN ***
thecooler992: shows [5d Kh] (three of a kind, Sevens)
Joe Hahn: shows [Jd Kd] (three of a kind, Sevens)
thecooler992 collected $25.55 from side pot
Joe Hahn collected $25.55 from side pot
Senecady: shows [Kc Ts] (a full house, Sevens full of Tens)
Senecady collected $350 from main pot
*** SUMMARY ***
Total pot $401.10 | Rake $3
Board [7s Qh 7d Tc 7c]
"""

RO_TOURNAMENT_HAND = """PokerStars Hand #261605993521: Tournament #4020308189, $0.46+$0.04 USD Hold'em No Limit - Level I (10/20) - 2026/08/01 1:21:11 EET [2026/07/31 18:21:11 ET]
Table '4020308189 1' 3-max Seat #1 is the button
Seat 1: Diegho89 (500 in chips) 
Seat 2: st_marian_x (500 in chips) 
Seat 3: alpoemo (500 in chips) 
st_marian_x: posts small blind 10
alpoemo: posts big blind 20
*** HOLE CARDS ***
Dealt to st_marian_x [6s 5c]
Diegho89: raises 480 to 500 and is all-in
st_marian_x: folds 
alpoemo: calls 480 and is all-in
*** FLOP *** [Qh Kc 4d]
*** TURN *** [Qh Kc 4d] [Qc]
*** RIVER *** [Qh Kc 4d Qc] [3h]
*** SHOW DOWN ***
alpoemo: shows [9c 9h] (two pair, Queens and Nines)
Diegho89: shows [Ks 5d] (two pair, Kings and Queens)
Diegho89 collected 1010 from pot
*** SUMMARY ***
Total pot 1010 | Rake 0 
Board [Qh Kc 4d Qc 3h]
Seat 1: Diegho89 (button) showed [Ks 5d] and won (1010) with two pair, Kings and Queens
Seat 2: st_marian_x (small blind) folded before Flop
Seat 3: alpoemo (big blind) showed [9c 9h] and lost with two pair, Queens and Nines
"""


class PokerStarsParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hand = PokerStarsParser().parse(HAND)

    def test_header_table_players_and_cards(self) -> None:
        self.assertEqual(self.hand.hand_id, "40000000000")
        self.assertEqual(self.hand.small_blind, Decimal("2"))
        self.assertEqual(self.hand.big_blind, Decimal("4"))
        self.assertEqual(self.hand.currency, "USD")
        self.assertEqual(self.hand.button_seat, 3)
        self.assertEqual(len(self.hand.players), 6)
        self.assertEqual(self.hand.hero, "Senecady")
        self.assertEqual([str(card) for card in self.hand.board], ["7s", "Qh", "7d", "Tc", "7c"])

    def test_actions_and_awards(self) -> None:
        kinds = [action.kind for action in self.hand.actions]
        self.assertEqual(kinds[:3], [ActionKind.POST_SMALL_BLIND, ActionKind.POST_BIG_BLIND, ActionKind.FOLD])
        raises = [action for action in self.hand.actions if action.kind == ActionKind.RAISE]
        self.assertEqual(raises[0].to_amount, Decimal("117"))
        self.assertTrue(raises[0].all_in)
        returned = [action for action in self.hand.actions if action.kind == ActionKind.RETURN]
        self.assertEqual(returned[0].amount, Decimal("269.45"))
        self.assertEqual(sum(award.amount for award in self.hand.awards), Decimal("401.10"))
        self.assertEqual(self.hand.total_pot, Decimal("401.10"))
        self.assertEqual(self.hand.rake, Decimal("3"))
        self.assertEqual(self.hand.actions[-1].street, Street.SHOWDOWN)

    def test_parse_many(self) -> None:
        hands = PokerStarsParser().parse_many(HAND + "\n\n" + HAND.replace("#40000000000", "#40000000001", 1))
        self.assertEqual([hand.hand_id for hand in hands], ["40000000000", "40000000001"])

    def test_play_money_saved_history(self) -> None:
        path = Path(__file__).resolve().parent.parent / "examples" / "sample_play_money_hand.txt"
        hand = PokerStarsParser().parse_file(path)[0]
        self.assertEqual(hand.currency, "Play Money")
        self.assertEqual(hand.small_blind, Decimal("10"))
        self.assertEqual(hand.big_blind, Decimal("20"))
        self.assertEqual(hand.hero, "Hero")
        self.assertEqual(hand.total_pot, Decimal("290"))

    def test_ro_tournament_header_and_trailing_spaces(self) -> None:
        hand = PokerStarsParser().parse(RO_TOURNAMENT_HAND)
        self.assertEqual(hand.hand_id, "261605993521")
        self.assertEqual((hand.small_blind, hand.big_blind), (Decimal("10"), Decimal("20")))
        self.assertEqual(hand.currency, "USD")
        self.assertEqual(
            [player.name for player in hand.players],
            ["Diegho89", "st_marian_x", "alpoemo"],
        )
        self.assertEqual(hand.hero, "st_marian_x")
        self.assertEqual(hand.total_pot, Decimal("1010"))


if __name__ == "__main__":
    unittest.main()
