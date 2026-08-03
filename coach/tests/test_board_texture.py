import unittest

from poker_coach.board_texture import RANGE_CAVEAT, analyze_board_texture
from poker_coach.models import Card


def cards(*tokens: str) -> tuple[Card, ...]:
    return tuple(Card.parse(token) for token in tokens)


class BoardTextureTests(unittest.TestCase):
    def test_preflop_holding_structure_has_no_board_claim(self) -> None:
        result = analyze_board_texture((), cards("As", "Ks")).to_dict()
        self.assertEqual(result["street"], "preflop")
        self.assertEqual(result["pairing"], "no_board")
        self.assertEqual(result["hero"]["pair_relation"], "suited")
        self.assertEqual(result["range_caveat"], RANGE_CAVEAT)

    def test_dry_flop_and_top_pair_are_classified(self) -> None:
        result = analyze_board_texture(
            cards("Ac", "7d", "2h"), cards("As", "Kd")
        ).to_dict()
        self.assertEqual(result["pairing"], "unpaired")
        self.assertEqual(result["suit_texture"], "rainbow")
        self.assertEqual(result["straight_texture"], "disconnected")
        self.assertEqual(result["hero"]["made_hand"], "one_pair")
        self.assertEqual(result["hero"]["pair_relation"], "top_pair")
        self.assertEqual(result["hero"]["flush_draw"], "none")

    def test_made_straight_retains_flush_draw(self) -> None:
        result = analyze_board_texture(
            cards("9h", "8h", "7c"), cards("Jh", "Th")
        ).to_dict()
        self.assertEqual(result["straight_texture"], "connected")
        self.assertEqual(result["hero"]["made_hand"], "straight")
        self.assertEqual(result["hero"]["straight_draw"], "none")
        self.assertEqual(result["hero"]["flush_draw"], "flush_draw")

    def test_paired_three_flush_turn_finds_nut_draw_and_blocker(self) -> None:
        result = analyze_board_texture(
            cards("Ah", "Kh", "7h", "7c"), cards("Qh", "Qs")
        ).to_dict()
        self.assertEqual(result["pairing"], "paired")
        self.assertEqual(result["suit_texture"], "three_flush")
        self.assertEqual(result["hero"]["made_hand"], "two_pair")
        self.assertEqual(result["hero"]["flush_draw"], "nut_flush_draw")
        self.assertEqual(result["hero"]["nut_flush_blockers"], ["Qh"])

    def test_four_connected_board_reports_exact_completion_ranks(self) -> None:
        result = analyze_board_texture(
            cards("9s", "8d", "7c", "6h"), cards("As", "Kd")
        ).to_dict()
        self.assertEqual(result["straight_texture"], "four_connected")
        self.assertEqual(result["max_straight_window_coverage"], 4)
        self.assertEqual(result["board_straight_completion_ranks"], ["T", "5"])
        self.assertEqual(result["hero"]["straight_draw"], "none")

    def test_open_ended_draw_lists_rank_outs_without_double_counting_suits(self) -> None:
        result = analyze_board_texture(
            cards("9s", "8d", "2c"), cards("7h", "6h")
        ).to_dict()
        self.assertEqual(result["hero"]["straight_draw"], "open_ended_or_double_gutshot")
        self.assertEqual(result["hero"]["straight_completion_ranks"], ["T", "5"])

    def test_invalid_or_conflicting_cards_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero, three, four, or five"):
            analyze_board_texture(cards("As"), cards("Kh", "Qh"))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            analyze_board_texture(cards("As", "7d", "2h"), cards("As", "Kd"))


if __name__ == "__main__":
    unittest.main()
