from dataclasses import replace
from pathlib import Path
import unittest

from poker_coach import (
    SolverBundleImporter,
    aggregate_range_strategies,
    hand_class,
    public_node_fingerprint,
)
from poker_coach.models import Card


ROOT = Path(__file__).resolve().parents[1]


class RangeStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spots = SolverBundleImporter().parse_file(
            ROOT / "examples" / "sample_solver_bundle.json"
        ).spots

    def test_exact_private_nodes_group_by_public_state_and_hand_class(self) -> None:
        report = aggregate_range_strategies(self.spots)
        self.assertEqual(report["group_count"], 4)
        shared = next(group for group in report["groups"] if group["private_nodes"] == 2)
        self.assertEqual(shared["covered_classes"], 2)
        cells = {cell["hand_class"]: cell for cell in shared["cells"]}
        self.assertEqual(set(cells), {"KQo", "JTo"})
        kqo = {action["action"]: action for action in cells["KQo"]["actions"]}
        self.assertEqual(kqo["bet:0.33"]["frequency"], "0.66")
        self.assertEqual(cells["KQo"]["exact_combos"], ["Kc Qd"])

    def test_hand_class_distinguishes_pair_suited_and_offsuit(self) -> None:
        base = self.spots[0]
        self.assertEqual(hand_class(base), "KQo")
        self.assertEqual(
            hand_class(replace(base, key=replace(base.key, hero_cards=(Card.parse("Kc"), Card.parse("Qc"))))),
            "KQs",
        )
        self.assertEqual(
            hand_class(replace(base, key=replace(base.key, hero_cards=(Card.parse("Kc"), Card.parse("Kd"))))),
            "KK",
        )

    def test_public_fingerprint_ignores_private_cards_and_literal_suits(self) -> None:
        first = self.spots[0]
        other_private = replace(
            first,
            key=replace(first.key, hero_cards=(Card.parse("Jc"), Card.parse("Td"))),
        )
        renamed = replace(
            first,
            key=replace(
                first.key,
                board=(Card.parse("As"), Card.parse("7c"), Card.parse("2d")),
                hero_cards=(Card.parse("Kd"), Card.parse("Qh")),
            ),
        )
        self.assertEqual(public_node_fingerprint(first), public_node_fingerprint(other_private))
        self.assertEqual(public_node_fingerprint(first), public_node_fingerprint(renamed))


if __name__ == "__main__":
    unittest.main()
