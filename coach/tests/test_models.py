from decimal import Decimal
import unittest

from poker_coach.models import Card, HandHistory, Player


class CardTests(unittest.TestCase):
    def test_ascii_and_unicode_cards_normalize(self) -> None:
        self.assertEqual(str(Card.parse("10♥")), "Th")
        self.assertEqual(str(Card.parse("As")), "As")

    def test_invalid_card_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Card.parse("1x")


class HandValidationTests(unittest.TestCase):
    def test_duplicate_board_card_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HandHistory(
                hand_id="1",
                game="Hold'em No Limit",
                limit="no_limit",
                stakes_raw="$1/$2 USD",
                currency="USD",
                small_blind=Decimal("1"),
                big_blind=Decimal("2"),
                played_at_raw="now",
                table_name="T",
                max_seats=2,
                button_seat=1,
                players=(Player(1, "A", Decimal("100")), Player(2, "B", Decimal("100"))),
                hole_cards=(),
                board=(Card.parse("As"), Card.parse("As"), Card.parse("Kd")),
                actions=(),
                awards=(),
            )


if __name__ == "__main__":
    unittest.main()

