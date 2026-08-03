from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable

from .features import position_map
from .models import ActionKind, HandHistory, Street
from .replay import HandReplayer


DEFAULT_SESSION_GAP = timedelta(minutes=60)
_TIMESTAMP_RE = re.compile(
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})"
)
_VOLUNTARY = {ActionKind.CALL, ActionKind.BET, ActionKind.RAISE}
_DECISIONS = {
    ActionKind.FOLD,
    ActionKind.CHECK,
    ActionKind.CALL,
    ActionKind.BET,
    ActionKind.RAISE,
}
_AGGRESSIVE = {ActionKind.BET, ActionKind.RAISE}


def _decimal(value: Decimal) -> str:
    return format(value, "f")


def _rate(numerator: int, denominator: int) -> str | None:
    if denominator == 0:
        return None
    return _decimal(Decimal(numerator) / Decimal(denominator))


def _timestamp(raw: str) -> datetime | None:
    match = _TIMESTAMP_RE.search(raw)
    if match is None:
        return None
    try:
        return datetime(**{key: int(value) for key, value in match.groupdict().items()})
    except ValueError:
        return None


def _new_counter() -> dict[str, object]:
    return {
        "hands": 0,
        "net_bb": Decimal("0"),
        "vpip_count": 0,
        "pfr_count": 0,
        "streets": defaultdict(
            lambda: {
                "decisions": 0,
                "aggressive_actions": 0,
                "calls": 0,
                "checks": 0,
                "folds": 0,
            }
        ),
    }


def _record_hand(counter: dict[str, object], row: dict[str, object]) -> None:
    counter["hands"] = int(counter["hands"]) + 1
    counter["net_bb"] = Decimal(counter["net_bb"]) + Decimal(row["net_bb"])
    counter["vpip_count"] = int(counter["vpip_count"]) + int(row["vpip"])
    counter["pfr_count"] = int(counter["pfr_count"]) + int(row["pfr"])
    streets = counter["streets"]
    assert isinstance(streets, defaultdict)
    for street, action_counts in row["streets"].items():
        for key, value in action_counts.items():
            streets[street][key] += int(value)


def _street_rows(streets: dict[str, dict[str, int]]) -> dict[str, dict[str, object]]:
    return {
        street: {
            **counts,
            "aggression_rate": _rate(
                counts["aggressive_actions"], counts["decisions"]
            ),
        }
        for street, counts in sorted(
            streets.items(), key=lambda item: Street[item[0].upper()].value
        )
    }


def _finish_counter(counter: dict[str, object]) -> dict[str, object]:
    hands = int(counter["hands"])
    net_bb = Decimal(counter["net_bb"])
    streets = counter["streets"]
    assert isinstance(streets, defaultdict)
    return {
        "hands": hands,
        "net_bb": _decimal(net_bb),
        "bb_per_100": _decimal(net_bb * Decimal("100") / Decimal(hands))
        if hands
        else None,
        "vpip": {
            "count": int(counter["vpip_count"]),
            "opportunities": hands,
            "observed_rate": _rate(int(counter["vpip_count"]), hands),
        },
        "pfr": {
            "count": int(counter["pfr_count"]),
            "opportunities": hands,
            "observed_rate": _rate(int(counter["pfr_count"]), hands),
        },
        "streets": _street_rows(streets),
    }


def _hand_row(hand: HandHistory, index: int) -> dict[str, object] | None:
    hero = hand.hero
    if hero is None or hand.big_blind is None or hand.big_blind <= 0:
        return None
    replay = HandReplayer().replay(hand)
    ledger = next(player for player in replay.players if player.player == hero)
    net_bb = (ledger.awarded - ledger.committed_total) / hand.big_blind
    preflop = [
        action
        for action in hand.actions
        if action.player == hero
        and action.street == Street.PREFLOP
        and action.kind in _DECISIONS
    ]
    street_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "decisions": 0,
            "aggressive_actions": 0,
            "calls": 0,
            "checks": 0,
            "folds": 0,
        }
    )
    for action in hand.actions:
        if action.player != hero or action.kind not in _DECISIONS:
            continue
        street = action.street.name.lower()
        street_counts[street]["decisions"] += 1
        if action.kind in _AGGRESSIVE:
            street_counts[street]["aggressive_actions"] += 1
        elif action.kind == ActionKind.CALL:
            street_counts[street]["calls"] += 1
        elif action.kind == ActionKind.CHECK:
            street_counts[street]["checks"] += 1
        elif action.kind == ActionKind.FOLD:
            street_counts[street]["folds"] += 1
    return {
        "index": index,
        "hand_id": hand.hand_id,
        "hero": hero,
        "table": hand.table_name,
        "timestamp": _timestamp(hand.played_at_raw),
        "played_at": hand.played_at_raw,
        "position": position_map(hand)[hero],
        "net_bb": net_bb,
        "vpip": any(action.kind in _VOLUNTARY for action in preflop),
        "pfr": any(action.kind == ActionKind.RAISE for action in preflop),
        "streets": street_counts,
    }


def _session_id(hero: str, table: str, rows: list[dict[str, object]]) -> str:
    material = "|".join(
        (hero, table, str(rows[0]["hand_id"]), str(rows[-1]["hand_id"]))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _session_rows(
    hero: str,
    rows: list[dict[str, object]],
    session_gap: timedelta,
) -> list[dict[str, object]]:
    streams: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        streams[str(row["table"])].append(row)
    sessions: list[dict[str, object]] = []
    for table, stream in sorted(streams.items()):
        stream.sort(
            key=lambda row: (
                row["timestamp"] is None,
                row["timestamp"] or datetime.max,
                int(row["index"]),
            )
        )
        groups: list[list[dict[str, object]]] = []
        for row in stream:
            if not groups:
                groups.append([row])
                continue
            previous = groups[-1][-1]["timestamp"]
            current = row["timestamp"]
            if (
                previous is None
                or current is None
                or current - previous > session_gap
                or current < previous
            ):
                groups.append([row])
            else:
                groups[-1].append(row)
        for group in groups:
            counter = _new_counter()
            for row in group:
                _record_hand(counter, row)
            summary = _finish_counter(counter)
            started = group[0]["timestamp"]
            ended = group[-1]["timestamp"]
            sessions.append(
                {
                    "session_id": _session_id(hero, table, group),
                    "table": table,
                    "started_at": started.isoformat() if started else group[0]["played_at"],
                    "ended_at": ended.isoformat() if ended else group[-1]["played_at"],
                    **summary,
                }
            )
    sessions.sort(key=lambda row: (str(row["started_at"]), str(row["table"])))
    return sessions


def analyze_hero_trends(
    hands: Iterable[HandHistory],
    *,
    session_gap: timedelta = DEFAULT_SESSION_GAP,
) -> dict[str, object]:
    """Aggregate descriptive hero trends from completed hands in BB.

    These are observed action rates and results, not solver-derived leak claims.
    Sessions split by table after a configurable inactivity gap.
    """

    if session_gap <= timedelta(0):
        raise ValueError("session_gap must be positive")
    hand_rows: list[dict[str, object]] = []
    hands_without_usable_hero = 0
    for index, hand in enumerate(hands):
        row = _hand_row(hand, index)
        if row is None:
            hands_without_usable_hero += 1
        else:
            hand_rows.append(row)

    rows_by_hero: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in hand_rows:
        rows_by_hero[str(row["hero"])].append(row)

    heroes: dict[str, dict[str, object]] = {}
    for hero, rows in sorted(rows_by_hero.items()):
        overall = _new_counter()
        positions: dict[str, dict[str, object]] = defaultdict(_new_counter)
        for row in rows:
            _record_hand(overall, row)
            _record_hand(positions[str(row["position"])], row)
        finished = _finish_counter(overall)
        heroes[hero] = {
            **finished,
            "positions": {
                position: _finish_counter(counter)
                for position, counter in sorted(positions.items())
            },
            "sessions": _session_rows(hero, rows, session_gap),
        }

    primary_hero = (
        sorted(heroes, key=lambda hero: (-int(heroes[hero]["hands"]), hero))[0]
        if heroes
        else None
    )
    return {
        "schema_version": "1.0.0",
        "units": "BB",
        "primary_hero": primary_hero,
        "heroes": heroes,
        "hands_without_usable_hero": hands_without_usable_hero,
        "session_gap_minutes": int(session_gap.total_seconds() // 60),
        "caveats": [
            "Observed action rates are descriptive and are not solver-derived leak claims.",
            "BB/100 and short-session results are high-variance; use them with sample size and solver coverage.",
        ],
    }
