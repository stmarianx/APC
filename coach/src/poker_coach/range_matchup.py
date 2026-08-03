from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import Iterable

from .equity import CATEGORY_NAMES, HandRank, WeightedCombo, best_hand_rank, equity_vs_hand, full_deck
from .models import Card


D = Decimal
MAX_MONTE_CARLO_SAMPLES = 200_000
MIN_MONTE_CARLO_SAMPLES = 100


def _fmt(value: Decimal) -> str:
    return format(value, "f")


def _normalize_range(
    combos: Iterable[WeightedCombo], board: tuple[Card, ...], label: str
) -> tuple[tuple[WeightedCombo, ...], int]:
    merged: dict[tuple[Card, Card], Decimal] = {}
    excluded = 0
    board_set = set(board)
    for combo in combos:
        if len(combo.cards) != 2 or len(set(combo.cards)) != 2:
            raise ValueError(f"{label} contains an invalid two-card combination")
        if not combo.weight.is_finite() or combo.weight < 0:
            raise ValueError(f"{label} weights must be finite and non-negative")
        if combo.weight == 0 or any(card in board_set for card in combo.cards):
            excluded += 1
            continue
        cards = tuple(sorted(combo.cards))
        merged[cards] = max(merged.get(cards, D("0")), combo.weight)
    if not merged:
        raise ValueError(f"{label} has no positive-weight combinations after board blockers")
    return (
        tuple(WeightedCombo(cards, weight) for cards, weight in sorted(merged.items())),
        excluded,
    )


def _rank_payload(rank: HandRank | None) -> dict[str, object] | None:
    if rank is None:
        return None
    return {"category": rank.name, "kickers": list(rank.kickers)}


def _range_summary(
    combos: tuple[WeightedCombo, ...],
    effective_weights: dict[tuple[Card, Card], Decimal],
    board: tuple[Card, ...],
    global_best: HandRank | None,
    global_top_category: int | None,
    excluded: int,
) -> dict[str, object]:
    total = sum(effective_weights.values(), D("0"))
    categories = {name: D("0") for name in CATEGORY_NAMES}
    nut_weight = D("0")
    top_category_weight = D("0")
    strongest: HandRank | None = None
    for combo in combos:
        weight = effective_weights.get(combo.cards, D("0"))
        if weight == 0 or len(board) < 3:
            continue
        rank = best_hand_rank((*board, *combo.cards))
        categories[rank.name] += weight
        strongest = rank if strongest is None else max(strongest, rank)
        if global_best is not None and rank == global_best:
            nut_weight += weight
        if global_top_category is not None and rank.category == global_top_category:
            top_category_weight += weight
    distribution = {
        name: _fmt(weight / total)
        for name, weight in categories.items()
        if weight > 0 and total > 0
    }
    return {
        "input_combos": len(combos) + excluded,
        "active_combos": len(combos),
        "matchup_compatible_combos": sum(
            effective_weights.get(combo.cards, D("0")) > 0 for combo in combos
        ),
        "board_blocked_or_zero_combos": excluded,
        "matchup_weight": _fmt(total),
        "category_distribution": distribution,
        "strongest_current_hand": _rank_payload(strongest),
        "current_nut_share": None
        if global_best is None
        else _fmt(nut_weight / total),
        "top_category_share": None
        if global_top_category is None
        else _fmt(top_category_weight / total),
    }


def _weighted_selector(
    combos: tuple[WeightedCombo, ...]
) -> tuple[list[float], float]:
    cumulative: list[float] = []
    total = 0.0
    for combo in combos:
        total += float(combo.weight)
        cumulative.append(total)
    return cumulative, total


def _select(
    combos: tuple[WeightedCombo, ...], cumulative: list[float], total: float, rng: random.Random
) -> WeightedCombo:
    target = rng.random() * total
    index = bisect.bisect_left(cumulative, target)
    return combos[min(index, len(combos) - 1)]


@dataclass(frozen=True)
class RangeMatchupResult:
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return self.payload


def analyze_range_matchup(
    hero_range: Iterable[WeightedCombo],
    villain_range: Iterable[WeightedCombo],
    board: Iterable[Card] = (),
    *,
    samples: int = 20_000,
    seed: int = 1,
    max_exact_outcomes: int = 250_000,
) -> RangeMatchupResult:
    board_cards = tuple(board)
    if len(board_cards) not in (0, 3, 4, 5):
        raise ValueError("board must contain zero, three, four, or five cards")
    if len(set(board_cards)) != len(board_cards):
        raise ValueError("board contains duplicate cards")
    if isinstance(samples, bool) or not MIN_MONTE_CARLO_SAMPLES <= samples <= MAX_MONTE_CARLO_SAMPLES:
        raise ValueError(
            f"samples must be between {MIN_MONTE_CARLO_SAMPLES} and {MAX_MONTE_CARLO_SAMPLES}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(max_exact_outcomes, bool) or not isinstance(max_exact_outcomes, int) or max_exact_outcomes < 1:
        raise ValueError("max_exact_outcomes must be a positive integer")

    hero, hero_excluded = _normalize_range(hero_range, board_cards, "hero_range")
    villain, villain_excluded = _normalize_range(
        villain_range, board_cards, "villain_range"
    )
    hero_effective: dict[tuple[Card, Card], Decimal] = defaultdict(lambda: D("0"))
    villain_effective: dict[tuple[Card, Card], Decimal] = defaultdict(lambda: D("0"))
    compatible_matchups = 0
    blocked_matchups = 0
    pair_weight_total = D("0")
    for hero_combo in hero:
        hero_set = set(hero_combo.cards)
        for villain_combo in villain:
            if hero_set.intersection(villain_combo.cards):
                blocked_matchups += 1
                continue
            pair_weight = hero_combo.weight * villain_combo.weight
            if pair_weight == 0:
                continue
            compatible_matchups += 1
            pair_weight_total += pair_weight
            hero_effective[hero_combo.cards] += pair_weight
            villain_effective[villain_combo.cards] += pair_weight
    if compatible_matchups == 0 or pair_weight_total == 0:
        raise ValueError("The supplied ranges have no card-compatible matchups")

    ranks: list[HandRank] = []
    if len(board_cards) >= 3:
        ranks.extend(
            best_hand_rank((*board_cards, *combo.cards))
            for combo in hero
            if hero_effective.get(combo.cards, D("0")) > 0
        )
        ranks.extend(
            best_hand_rank((*board_cards, *combo.cards))
            for combo in villain
            if villain_effective.get(combo.cards, D("0")) > 0
        )
    global_best = max(ranks) if ranks else None
    global_top_category = max((rank.category for rank in ranks), default=None)

    runout_size = 5 - len(board_cards)
    runouts_per_matchup = math.comb(48 - len(board_cards), runout_size)
    estimated_outcomes = compatible_matchups * runouts_per_matchup
    method = (
        "exact_enumeration"
        if estimated_outcomes <= max_exact_outcomes
        else "deterministic_monte_carlo"
    )

    if method == "exact_enumeration":
        weighted_wins = weighted_ties = weighted_losses = D("0")
        for hero_combo in hero:
            hero_set = set(hero_combo.cards)
            for villain_combo in villain:
                if hero_set.intersection(villain_combo.cards):
                    continue
                pair_weight = hero_combo.weight * villain_combo.weight
                result = equity_vs_hand(
                    hero_combo.cards, villain_combo.cards, board_cards
                )
                weighted_wins += pair_weight * D(result.wins)
                weighted_ties += pair_weight * D(result.ties)
                weighted_losses += pair_weight * D(result.losses)
        outcome_weight = weighted_wins + weighted_ties + weighted_losses
        win_probability = weighted_wins / outcome_weight
        tie_probability = weighted_ties / outcome_weight
        loss_probability = weighted_losses / outcome_weight
        hero_equity = win_probability + tie_probability / 2
        outcomes_evaluated = estimated_outcomes
        confidence_half_width = D("0")
    else:
        hero_cumulative, hero_total = _weighted_selector(hero)
        villain_cumulative, villain_total = _weighted_selector(villain)
        rng = random.Random(seed)
        deck = full_deck()
        wins = ties = losses = 0
        for _ in range(samples):
            while True:
                hero_combo = _select(hero, hero_cumulative, hero_total, rng)
                villain_combo = _select(
                    villain, villain_cumulative, villain_total, rng
                )
                if not set(hero_combo.cards).intersection(villain_combo.cards):
                    break
            known = set((*board_cards, *hero_combo.cards, *villain_combo.cards))
            remaining = [card for card in deck if card not in known]
            runout = tuple(rng.sample(remaining, runout_size))
            complete_board = (*board_cards, *runout)
            hero_rank = best_hand_rank((*hero_combo.cards, *complete_board))
            villain_rank = best_hand_rank((*villain_combo.cards, *complete_board))
            if hero_rank > villain_rank:
                wins += 1
            elif hero_rank == villain_rank:
                ties += 1
            else:
                losses += 1
        win_probability = D(wins) / D(samples)
        tie_probability = D(ties) / D(samples)
        loss_probability = D(losses) / D(samples)
        hero_equity = win_probability + tie_probability / 2
        score_square_mean = (D(wins) + D(ties) * D("0.25")) / D(samples)
        variance = max(D("0"), score_square_mean - hero_equity * hero_equity)
        confidence_half_width = D(
            str(1.96 * math.sqrt(float(variance) / samples))
        )
        outcomes_evaluated = samples

    villain_equity = D("1") - hero_equity
    hero_summary = _range_summary(
        hero, hero_effective, board_cards, global_best, global_top_category, hero_excluded
    )
    villain_summary = _range_summary(
        villain,
        villain_effective,
        board_cards,
        global_best,
        global_top_category,
        villain_excluded,
    )
    hero_nut_share = hero_summary["current_nut_share"]
    villain_nut_share = villain_summary["current_nut_share"]
    nut_edge = None
    nut_leader = "not_available_preflop"
    if isinstance(hero_nut_share, str) and isinstance(villain_nut_share, str):
        nut_edge_value = D(hero_nut_share) - D(villain_nut_share)
        nut_edge = _fmt(nut_edge_value)
        nut_leader = (
            "hero" if nut_edge_value > 0 else "villain" if nut_edge_value < 0 else "even"
        )
    lower = max(D("0"), hero_equity - confidence_half_width)
    upper = min(D("1"), hero_equity + confidence_half_width)
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "board": [str(card) for card in board_cards],
        "street": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[
            len(board_cards)
        ],
        "method": method,
        "seed": seed if method == "deterministic_monte_carlo" else None,
        "samples_requested": samples,
        "outcomes_evaluated": outcomes_evaluated,
        "estimated_exact_outcomes": estimated_outcomes,
        "runouts_per_matchup": runouts_per_matchup,
        "compatible_matchups": compatible_matchups,
        "blocked_matchups": blocked_matchups,
        "equity": {
            "hero": _fmt(hero_equity),
            "villain": _fmt(villain_equity),
            "hero_edge": _fmt(hero_equity - villain_equity),
            "win": _fmt(win_probability),
            "tie": _fmt(tie_probability),
            "loss": _fmt(loss_probability),
            "confidence_95": {
                "lower": _fmt(lower),
                "upper": _fmt(upper),
                "half_width": _fmt(confidence_half_width),
            },
        },
        "current_range_relative_nuts": {
            "strongest_hand": _rank_payload(global_best),
            "top_category": None
            if global_top_category is None
            else CATEGORY_NAMES[global_top_category],
            "hero_nut_share": hero_nut_share,
            "villain_nut_share": villain_nut_share,
            "nut_share_edge": nut_edge,
            "leader": nut_leader,
        },
        "hero_range": hero_summary,
        "villain_range": villain_summary,
        "provenance": {
            "equity_definition": "showdown equity with ties split equally",
            "weighting": "combination weights adjusted for board and cross-range card removal",
            "nut_definition": "share of matchup weight holding the strongest current exact hand rank present in either supplied range",
            "caveat": "Current nut share is range-relative and uses the board as dealt; it is not a universal nut claim and does not replace solver EV.",
        },
    }
    return RangeMatchupResult(payload)
