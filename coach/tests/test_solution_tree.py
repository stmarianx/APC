from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import unittest

from poker_coach import SolutionForest, SolverExportRegistry
from poker_coach.models import Card


ROOT = Path(__file__).resolve().parents[1]


class SolutionTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spots = SolverExportRegistry().parse_file(
            ROOT / "examples" / "sample_solver_export.csv"
        ).bundle.spots

    def test_multistreet_nodes_form_one_traversable_path(self) -> None:
        forest = SolutionForest(self.spots)
        self.assertEqual(len(forest.nodes), 4)
        self.assertEqual(forest.linked_edges, 3)
        self.assertEqual(len(forest.roots), 1)
        self.assertEqual(forest.ambiguous_nodes, 0)
        self.assertEqual(forest.max_depth, 3)
        path = forest.path_to(self.spots[-1].key.fingerprint)
        self.assertEqual(
            [node.node_id for node in path],
            [spot.node_id for spot in self.spots],
        )

    def test_branching_rivers_share_the_turn_parent(self) -> None:
        alternate = replace(
            self.spots[-1],
            node_id="hu_ako_a72r95_river",
            key=replace(
                self.spots[-1].key,
                board=(
                    Card.parse("Ah"),
                    Card.parse("7c"),
                    Card.parse("2d"),
                    Card.parse("9s"),
                    Card.parse("5h"),
                ),
            ),
        )
        forest = SolutionForest(self.spots + (alternate,))
        turn = forest.nodes[self.spots[2].key.fingerprint]
        self.assertEqual(len(turn.children), 2)
        self.assertEqual(forest.linked_edges, 4)

    def test_independently_renamed_suits_still_link_across_streets(self) -> None:
        renamed_turn = replace(
            self.spots[2],
            key=replace(
                self.spots[2].key,
                board=(
                    Card.parse("As"),
                    Card.parse("7d"),
                    Card.parse("2h"),
                    Card.parse("9c"),
                ),
                hero_cards=(Card.parse("Ac"), Card.parse("Kh")),
            ),
        )
        forest = SolutionForest((self.spots[0], self.spots[1], renamed_turn))
        self.assertEqual(forest.linked_edges, 2)
        self.assertEqual(forest.max_depth, 2)

    def test_equal_progress_parents_are_reported_as_ambiguous(self) -> None:
        alternate_flop = replace(
            self.spots[1],
            node_id="alternate_flop_abstraction",
            key=replace(self.spots[1].key, pot_bb=Decimal("7.0")),
        )
        forest = SolutionForest(
            (self.spots[0], self.spots[1], alternate_flop, self.spots[2])
        )
        turn = forest.nodes[self.spots[2].key.fingerprint]
        self.assertIsNone(turn.parent)
        self.assertEqual(len(turn.ambiguous_parents), 2)
        self.assertEqual(forest.ambiguous_nodes, 1)

    def test_different_private_ranks_remain_an_independent_tree(self) -> None:
        different_hand = replace(
            self.spots[1],
            node_id="aq_flop",
            key=replace(
                self.spots[1].key,
                hero_cards=(Card.parse("Ac"), Card.parse("Qd")),
            ),
        )
        forest = SolutionForest((self.spots[0], different_hand))
        self.assertEqual(forest.linked_edges, 0)
        self.assertEqual(len(forest.roots), 2)


if __name__ == "__main__":
    unittest.main()
