from __future__ import annotations

import unittest

from apc.perception.event_baseline import _card_tokens


class EventBaselineTests(unittest.TestCase):
    def test_card_token_order_is_hero_then_board(self) -> None:
        annotation = {
            "objects": {
                "hero_cards": [{"rank": "A", "suit": "s"}, {"rank": "K", "suit": "h"}],
                "board_cards": [{"rank": "2", "suit": "c"}],
            }
        }
        self.assertEqual(_card_tokens(annotation), ("As", "Kh", "2c"))


if __name__ == "__main__":
    unittest.main()
