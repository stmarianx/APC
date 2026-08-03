from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from .isomorphism import canonicalize_suit_state
from .solutions import SolvedSpot


RANKS = "AKQJT98765432"
_RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}


def hand_class(spot: SolvedSpot) -> str | None:
    cards = spot.key.hero_cards
    if len(cards) != 2:
        return None
    first, second = sorted(cards, key=lambda card: _RANK_INDEX[card.rank])
    if first.rank == second.rank:
        return first.rank * 2
    return first.rank + second.rank + ("s" if first.suit == second.suit else "o")


def public_node_payload(spot: SolvedSpot) -> dict[str, object]:
    key = spot.key
    board, _ = canonicalize_suit_state(key.board, ())
    return {
        "source": spot.source,
        "source_version": spot.source_version,
        "game": key.game,
        "players": key.players,
        "hero_position": key.hero_position,
        "effective_stack_bb": format(key.effective_stack_bb, "f"),
        "pot_bb": format(key.pot_bb, "f"),
        "board": [str(card) for card in board],
        "action_history": list(key.action_history),
        "rake_model": key.rake_model,
        "utility_model": key.utility_model,
        "allowed_sizes": [format(size, "f") for size in key.allowed_sizes],
    }


def public_node_fingerprint(spot: SolvedSpot) -> str:
    encoded = json.dumps(
        public_node_payload(spot), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values))


def aggregate_range_strategies(
    solutions: Iterable[SolvedSpot],
) -> dict[str, object]:
    grouped: dict[str, list[SolvedSpot]] = defaultdict(list)
    for spot in solutions:
        if hand_class(spot) is not None:
            grouped[public_node_fingerprint(spot)].append(spot)

    groups: list[dict[str, object]] = []
    for fingerprint, spots in grouped.items():
        by_class: dict[str, list[SolvedSpot]] = defaultdict(list)
        for spot in spots:
            label = hand_class(spot)
            assert label is not None
            by_class[label].append(spot)
        cells: list[dict[str, object]] = []
        for label, cell_spots in sorted(
            by_class.items(), key=lambda row: (_RANK_INDEX[row[0][0]], row[0])
        ):
            action_ids = sorted(
                {action.action for spot in cell_spots for action in spot.actions}
            )
            actions = []
            for action_id in action_ids:
                frequencies = []
                evs = []
                for spot in cell_spots:
                    try:
                        action = spot.action(action_id)
                    except KeyError:
                        frequencies.append(Decimal("0"))
                        continue
                    frequencies.append(action.frequency)
                    evs.append(action.ev)
                actions.append(
                    {
                        "action": action_id,
                        "frequency": format(_mean(frequencies), "f"),
                        "ev": None if not evs else format(_mean(evs), "f"),
                    }
                )
            cells.append(
                {
                    "hand_class": label,
                    "samples": len(cell_spots),
                    "exact_combos": [
                        " ".join(str(card) for card in spot.key.hero_cards)
                        for spot in sorted(
                            cell_spots,
                            key=lambda row: tuple(str(card) for card in row.key.hero_cards),
                        )
                    ],
                    "actions": actions,
                }
            )
        representative = spots[0]
        payload = public_node_payload(representative)
        groups.append(
            {
                "public_fingerprint": fingerprint,
                "label": (
                    f"{representative.key.hero_position} · "
                    f"{' '.join(str(card) for card in representative.key.board) or 'preflop'} · "
                    f"{format(representative.key.pot_bb, 'f')}bb pot"
                ),
                "source": representative.source,
                "source_version": representative.source_version,
                "state": payload,
                "private_nodes": len(spots),
                "covered_classes": len(cells),
                "cells": cells,
            }
        )
    groups.sort(key=lambda group: (str(group["source"]), str(group["label"]), str(group["public_fingerprint"])))
    return {
        "schema_version": "0.1.0",
        "ranks": list(RANKS),
        "groups": groups,
        "group_count": len(groups),
        "private_nodes": sum(int(group["private_nodes"]) for group in groups),
    }
