from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Iterable

from .equity import RANK_VALUE, WeightedCombo
from .models import Card


RANKS = "23456789TJQKA"
SUITS = "cdhs"
_CLASS_RE = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so])?(\+)?$", re.IGNORECASE)
_EXACT_RE = re.compile(r"^(10|[2-9TJQKA])([cdhs])(10|[2-9TJQKA])([cdhs])$", re.IGNORECASE)


def _normalize_hole(cards: Iterable[Card]) -> tuple[Card, Card]:
    result = tuple(sorted(cards, key=lambda card: (RANK_VALUE[card.rank], card.suit), reverse=True))
    if len(result) != 2 or len(set(result)) != 2:
        raise ValueError("A combo must contain two unique cards")
    return result  # type: ignore[return-value]


def _classes_for(token: str) -> tuple[tuple[str, str, str | None], ...]:
    match = _CLASS_RE.match(token)
    if not match:
        raise ValueError(f"Unsupported range token: {token!r}")
    high, low, suitedness, plus = match.groups()
    high = high.upper()
    low = low.upper()
    suitedness = suitedness.lower() if suitedness else None
    if high == low:
        if suitedness:
            raise ValueError("Pocket pairs cannot have suitedness suffix")
        ranks = RANKS[RANKS.index(high) :] if plus else high
        return tuple((rank, rank, None) for rank in ranks)
    if RANK_VALUE[high] < RANK_VALUE[low]:
        raise ValueError("Range classes must put the higher rank first")
    lows = RANKS[RANKS.index(low) : RANKS.index(high)] if plus else low
    return tuple((high, candidate, suitedness) for candidate in lows)


def expand_class(token: str, *, dead: Iterable[Card] = ()) -> tuple[tuple[Card, Card], ...]:
    cleaned = token.strip().replace("10", "T")
    exact = _EXACT_RE.match(cleaned)
    dead_set = set(dead)
    if exact:
        combo = _normalize_hole((Card(exact.group(1), exact.group(2)), Card(exact.group(3), exact.group(4))))
        return () if any(card in dead_set for card in combo) else (combo,)

    combos: set[tuple[Card, Card]] = set()
    for high, low, suitedness in _classes_for(cleaned):
        if high == low:
            candidates = combinations((Card(high, suit) for suit in SUITS), 2)
        elif suitedness == "s":
            candidates = ((Card(high, suit), Card(low, suit)) for suit in SUITS)
        elif suitedness == "o":
            candidates = (
                (Card(high, high_suit), Card(low, low_suit))
                for high_suit in SUITS
                for low_suit in SUITS
                if high_suit != low_suit
            )
        else:
            candidates = (
                (Card(high, high_suit), Card(low, low_suit))
                for high_suit in SUITS
                for low_suit in SUITS
            )
        for candidate in candidates:
            combo = _normalize_hole(candidate)
            if not any(card in dead_set for card in combo):
                combos.add(combo)
    return tuple(sorted(combos, key=lambda combo: tuple(str(card) for card in combo)))


def parse_range(notation: str, *, dead: Iterable[Card] = ()) -> tuple[WeightedCombo, ...]:
    """Parse comma-separated classes with optional `:weight` in [0, 1]."""
    weighted: dict[tuple[Card, Card], Decimal] = {}
    for item in notation.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            token, weight_text = item.rsplit(":", 1)
            try:
                weight = Decimal(weight_text.strip())
            except InvalidOperation as exc:
                raise ValueError(f"Invalid range weight: {weight_text!r}") from exc
        else:
            token, weight = item, Decimal("1")
        if not Decimal("0") <= weight <= Decimal("1"):
            raise ValueError("Range weights must be between zero and one")
        for combo in expand_class(token.strip(), dead=dead):
            weighted[combo] = max(weighted.get(combo, Decimal("0")), weight)
    if not weighted:
        raise ValueError("Range notation produced no compatible combinations")
    return tuple(WeightedCombo(combo, weight) for combo, weight in sorted(weighted.items(), key=lambda item: tuple(str(card) for card in item[0])))

