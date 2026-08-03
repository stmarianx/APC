from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .isomorphism import suit_isomorphic
from .solutions import SolvedSpot


def _private_ranks(spot: SolvedSpot) -> tuple[str, ...]:
    return tuple(sorted(card.rank for card in spot.key.hero_cards))


def _can_precede(parent: SolvedSpot, child: SolvedSpot) -> bool:
    first = parent.key
    second = child.key
    if parent.key.fingerprint == child.key.fingerprint:
        return False
    if (
        parent.source != child.source
        or parent.source_version != child.source_version
        or first.game != second.game
        or first.hero_position != second.hero_position
        or first.rake_model != second.rake_model
        or first.utility_model != second.utility_model
        or _private_ranks(parent) != _private_ranks(child)
        or first.players < second.players
        or first.effective_stack_bb < second.effective_stack_bb
        or len(first.board) > len(second.board)
        or len(first.action_history) > len(second.action_history)
    ):
        return False
    if second.action_history[: len(first.action_history)] != first.action_history:
        return False
    if not suit_isomorphic(
        first.board,
        first.hero_cards,
        second.board[: len(first.board)],
        second.hero_cards,
    ):
        return False
    return (
        len(first.board) < len(second.board)
        or len(first.action_history) < len(second.action_history)
    )


def _progress(spot: SolvedSpot) -> tuple[int, int]:
    return len(spot.key.action_history), len(spot.key.board)


@dataclass(frozen=True)
class SolutionTreeNode:
    solution: SolvedSpot
    parent: str | None
    children: tuple[str, ...]
    ambiguous_parents: tuple[str, ...]
    depth: int

    @property
    def fingerprint(self) -> str:
        return self.solution.key.fingerprint

    @property
    def node_id(self) -> str:
        return self.solution.node_id or self.fingerprint[:16]

    def to_dict(self) -> dict[str, object]:
        key = self.solution.key
        return {
            "fingerprint": self.fingerprint,
            "node_id": self.node_id,
            "parent": self.parent,
            "children": list(self.children),
            "ambiguous_parents": list(self.ambiguous_parents),
            "depth": self.depth,
            "street": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(key.board)],
            "board": [str(card) for card in key.board],
            "hero_cards": [str(card) for card in key.hero_cards],
            "hero_position": key.hero_position,
            "players": key.players,
            "pot_bb": format(key.pot_bb, "f"),
            "effective_stack_bb": format(key.effective_stack_bb, "f"),
            "history_length": len(key.action_history),
            "source": self.solution.source,
            "source_version": self.solution.source_version,
            "actions": [
                {
                    "action": action.action,
                    "frequency": format(action.frequency, "f"),
                    "ev": format(action.ev, "f"),
                }
                for action in self.solution.actions
            ],
        }


class SolutionForest:
    def __init__(self, solutions: Iterable[SolvedSpot]) -> None:
        rows = tuple(solutions)
        by_fingerprint = {spot.key.fingerprint: spot for spot in rows}
        if len(by_fingerprint) != len(rows):
            raise ValueError("Solution forest requires unique fingerprints")

        parents: dict[str, str | None] = {}
        ambiguities: dict[str, tuple[str, ...]] = {}
        for child_fingerprint, child in by_fingerprint.items():
            candidates = [
                (fingerprint, spot)
                for fingerprint, spot in by_fingerprint.items()
                if _can_precede(spot, child)
            ]
            maximal: list[tuple[str, SolvedSpot]] = []
            for fingerprint, candidate in candidates:
                history, board = _progress(candidate)
                dominated = any(
                    other_fingerprint != fingerprint
                    and _progress(other)[0] >= history
                    and _progress(other)[1] >= board
                    and _progress(other) != (history, board)
                    for other_fingerprint, other in candidates
                )
                if not dominated:
                    maximal.append((fingerprint, candidate))
            if len(maximal) == 1:
                parents[child_fingerprint] = maximal[0][0]
                ambiguities[child_fingerprint] = ()
            else:
                parents[child_fingerprint] = None
                ambiguities[child_fingerprint] = tuple(
                    sorted(fingerprint for fingerprint, _ in maximal)
                )

        child_rows: dict[str, list[str]] = {fingerprint: [] for fingerprint in by_fingerprint}
        for child, parent in parents.items():
            if parent is not None:
                child_rows[parent].append(child)

        depths: dict[str, int] = {}

        def depth(fingerprint: str) -> int:
            if fingerprint in depths:
                return depths[fingerprint]
            parent = parents[fingerprint]
            value = 0 if parent is None else depth(parent) + 1
            depths[fingerprint] = value
            return value

        self.nodes = {
            fingerprint: SolutionTreeNode(
                solution=spot,
                parent=parents[fingerprint],
                children=tuple(sorted(child_rows[fingerprint])),
                ambiguous_parents=ambiguities[fingerprint],
                depth=depth(fingerprint),
            )
            for fingerprint, spot in by_fingerprint.items()
        }

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(sorted(fingerprint for fingerprint, node in self.nodes.items() if node.parent is None))

    @property
    def linked_edges(self) -> int:
        return sum(1 for node in self.nodes.values() if node.parent is not None)

    @property
    def ambiguous_nodes(self) -> int:
        return sum(1 for node in self.nodes.values() if node.ambiguous_parents)

    @property
    def max_depth(self) -> int:
        return max((node.depth for node in self.nodes.values()), default=0)

    def path_to(self, fingerprint: str) -> tuple[SolutionTreeNode, ...]:
        if fingerprint not in self.nodes:
            raise KeyError(fingerprint)
        path: list[SolutionTreeNode] = []
        current: str | None = fingerprint
        while current is not None:
            node = self.nodes[current]
            path.append(node)
            current = node.parent
        path.reverse()
        return tuple(path)

    def to_dict(self) -> dict[str, object]:
        ordered = sorted(
            self.nodes.values(), key=lambda node: (node.depth, node.node_id, node.fingerprint)
        )
        return {
            "schema_version": "0.1.0",
            "nodes": len(self.nodes),
            "linked_edges": self.linked_edges,
            "roots": list(self.roots),
            "root_count": len(self.roots),
            "ambiguous_nodes": self.ambiguous_nodes,
            "max_depth": self.max_depth,
            "node_rows": [node.to_dict() for node in ordered],
        }
