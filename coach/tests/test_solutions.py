from decimal import Decimal
import unittest

from poker_coach import ActionSolution, InMemorySolutionStore, SolutionKey, SolvedSpot
from poker_coach.models import Card


class SolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = SolutionKey(
            game="holdem_no_limit",
            players=2,
            hero_position="BTN",
            effective_stack_bb=Decimal("100"),
            pot_bb=Decimal("6.5"),
            board=(Card.parse("Ah"), Card.parse("7c"), Card.parse("2d")),
            action_history=("BTN raise_to:2.5", "BB call"),
            rake_model="5pct_cap_3bb",
            allowed_sizes=(Decimal("0.33"), Decimal("0.75")),
        )
        self.spot = SolvedSpot(
            self.key,
            (
                ActionSolution("check", Decimal("0.3"), Decimal("18")),
                ActionSolution("bet:80", Decimal("0.7"), Decimal("20")),
            ),
            source="fixture_solver",
            source_version="1",
        )

    def test_fingerprint_store_and_ev_loss(self) -> None:
        self.assertEqual(self.key.fingerprint, self.key.fingerprint)
        self.assertEqual(len(self.key.fingerprint), 64)
        self.assertEqual(self.spot.best_ev, Decimal("20"))
        self.assertEqual(self.spot.ev_loss("check"), Decimal("2"))
        store = InMemorySolutionStore()
        store.put(self.spot)
        self.assertEqual(store.get(self.key), self.spot)

    def test_invalid_frequency_sum_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SolvedSpot(
                self.key,
                (ActionSolution("check", Decimal("0.2"), Decimal("1")),),
                source="bad",
            )

    def test_private_hand_is_part_of_fingerprint(self) -> None:
        first = SolutionKey(
            **{**self.key.__dict__, "hero_cards": (Card.parse("Kc"), Card.parse("Qd"))}
        )
        second = SolutionKey(
            **{**self.key.__dict__, "hero_cards": (Card.parse("Jc"), Card.parse("Td"))}
        )
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.canonical()["hero_cards"], ["Kc", "Qd"])

    def test_fingerprint_collapses_only_suit_isomorphic_states(self) -> None:
        base = SolutionKey(
            **{
                **self.key.__dict__,
                "board": (Card.parse("Ah"), Card.parse("7h"), Card.parse("2c")),
                "hero_cards": (Card.parse("Kh"), Card.parse("Qd")),
            }
        )
        renamed = SolutionKey(
            **{
                **self.key.__dict__,
                "board": (Card.parse("As"), Card.parse("7s"), Card.parse("2d")),
                "hero_cards": (Card.parse("Ks"), Card.parse("Qc")),
            }
        )
        different_blocker = SolutionKey(
            **{
                **self.key.__dict__,
                "board": (Card.parse("Ah"), Card.parse("7h"), Card.parse("2c")),
                "hero_cards": (Card.parse("Ks"), Card.parse("Qd")),
            }
        )
        reversed_private = SolutionKey(
            **{**base.__dict__, "hero_cards": tuple(reversed(base.hero_cards))}
        )
        self.assertEqual(base.fingerprint, renamed.fingerprint)
        self.assertEqual(base.fingerprint, reversed_private.fingerprint)
        self.assertNotEqual(base.fingerprint, different_blocker.fingerprint)
        self.assertEqual(base.canonical()["board"], ["Ah", "7h", "2c"])
        self.assertEqual(
            base.canonical_isomorphic()["card_normalization"], "suit_isomorphism_v1"
        )


if __name__ == "__main__":
    unittest.main()
