from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .features import position_map
from .ingest import DEFAULT_MAX_FILE_BYTES, _read_history_file
from .models import ActionKind, HandHistory, Street
from .pokerstars import PokerStarsParseError, PokerStarsParser
from .replay import HandReplayer, PlayerLedger


D = Decimal
_HAND_START_RE = re.compile(r"(?=^PokerStars (?:Hand|Game) #)", re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\*\*\* SUMMARY \*\*\*\r?$", re.MULTILINE)
_DECISIONS = {
    ActionKind.FOLD,
    ActionKind.CHECK,
    ActionKind.CALL,
    ActionKind.BET,
    ActionKind.RAISE,
}
_ACTION_PREFIXES = (
    "posts ",
    "folds",
    "checks",
    "calls",
    "bets",
    "raises",
    "shows",
    "mucks",
    "doesn't show",
)


def _compact(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.to_integral())
    return format(value.normalize(), "f")


def _game_id(hand: HandHistory) -> str:
    if "hold" in hand.game.lower() and hand.limit == "no_limit":
        return "holdem_no_limit"
    return f"{hand.game.lower().replace(' ', '_')}_{hand.limit}"


def _street(hand: HandHistory) -> Street:
    return {
        0: Street.PREFLOP,
        3: Street.FLOP,
        4: Street.TURN,
        5: Street.RIVER,
    }[len(hand.board)]


def _latest_hand_block(source: str) -> str | None:
    blocks = [
        block.strip()
        for block in _HAND_START_RE.split(source)
        if block.strip().lower().startswith(("pokerstars hand #", "pokerstars game #"))
    ]
    return blocks[-1] if blocks else None


def _has_unparsed_action_tail(block: str, hand: HandHistory) -> bool:
    parsed = {action.raw for action in hand.actions}
    for line in block.splitlines()[1:]:
        if ":" not in line:
            continue
        _, description = line.split(":", 1)
        if description.strip().lower().startswith(_ACTION_PREFIXES) and line not in parsed:
            return True
    return False


def _normalized_position(value: str) -> str:
    return "BTN" if value == "BTN/SB" else value


def _normalized_history(
    hand: HandHistory, positions: dict[str, str]
) -> tuple[str, ...]:
    assert hand.big_blind is not None
    rows: list[str] = []
    for action in hand.actions:
        if action.kind not in _DECISIONS or action.player is None:
            continue
        actor = _normalized_position(positions[action.player])
        if action.kind == ActionKind.BET:
            rows.append(f"{actor} bet:{_compact(action.amount / hand.big_blind)}")
        elif action.kind == ActionKind.RAISE:
            assert action.to_amount is not None
            rows.append(
                f"{actor} raise_to:{_compact(action.to_amount / hand.big_blind)}"
            )
        else:
            rows.append(f"{actor} {action.kind.value}")
    return tuple(rows)


@dataclass(frozen=True)
class ProjectedDecision:
    status: str
    message: str
    hand_id: str
    hero: str | None
    next_actor: str | None
    payload: dict[str, object] | None = None


@dataclass(frozen=True)
class LiveCapturePoll:
    status: str
    message: str
    path: str
    changed: bool
    revision: int | None
    hand_id: str | None
    hero: str | None
    next_actor: str | None
    payload: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "message": self.message,
            "path": self.path,
            "changed": self.changed,
            "revision": self.revision,
            "hand_id": self.hand_id,
            "hero": self.hero,
            "next_actor": self.next_actor,
            "payload": self.payload,
        }


class PokerStarsDecisionProjector:
    def __init__(self, *, rake_model: str = "training_no_rake") -> None:
        self.rake_model = rake_model

    def project(self, hand: HandHistory, *, table_id: str) -> ProjectedDecision:
        hero = hand.hero
        if hero is None:
            return ProjectedDecision(
                "waiting_for_hero",
                "The latest hand does not yet contain dealt hero cards.",
                hand.hand_id,
                None,
                None,
            )
        if hand.big_blind is None or hand.big_blind <= 0:
            return ProjectedDecision(
                "unsupported_stakes",
                "A positive big blind is required for normalized live analysis.",
                hand.hand_id,
                hero,
                None,
            )
        if hand.awards or any(
            action.street >= Street.SHOWDOWN for action in hand.actions
        ):
            return ProjectedDecision(
                "waiting_for_new_hand",
                "The latest hand is complete or at showdown.",
                hand.hand_id,
                hero,
                None,
            )

        replay = HandReplayer().replay(hand)
        ledgers = {row.player: row for row in replay.players}
        active = {
            player
            for player, ledger in ledgers.items()
            if not ledger.folded
        }
        if hero not in active or len(active) < 2:
            return ProjectedDecision(
                "waiting_for_new_hand",
                "No further hero decision exists in the latest hand.",
                hand.hand_id,
                hero,
                None,
            )

        street = _street(hand)
        street_actions = [
            action
            for action in hand.actions
            if action.street == street and action.kind in _DECISIONS
        ]
        can_act = {
            player
            for player in active
            if ledgers[player].stack > 0
        }
        pending = set(can_act)
        for action in street_actions:
            assert action.player is not None
            if action.kind in {ActionKind.BET, ActionKind.RAISE}:
                pending = {
                    player
                    for player in can_act
                    if player != action.player
                }
            else:
                pending.discard(action.player)
            if action.kind == ActionKind.FOLD:
                can_act.discard(action.player)
                pending.discard(action.player)

        if not pending:
            return ProjectedDecision(
                "waiting_for_next_street",
                "The betting round is closed; waiting for the next board update.",
                hand.hand_id,
                hero,
                None,
            )
        next_actor = self._next_actor(hand, street, street_actions, pending)
        if next_actor != hero:
            return ProjectedDecision(
                "waiting_for_player",
                f"Waiting for {next_actor or 'the next player'} to act.",
                hand.hand_id,
                hero,
                next_actor,
            )

        positions = position_map(hand)
        hero_ledger = ledgers[hero]
        current_commitments = self._current_commitments(
            hand, street, ledgers
        )
        current_bet = max(current_commitments.values(), default=D("0"))
        to_call = max(D("0"), current_bet - current_commitments.get(hero, D("0")))
        opponent_stacks = [
            ledger.stack
            for player, ledger in ledgers.items()
            if player != hero and not ledger.folded
        ]
        effective_stack = max(
            (min(hero_ledger.stack, stack) for stack in opponent_stacks),
            default=D("0"),
        )
        holding = next(
            row.cards for row in hand.hole_cards if row.player == hero
        )
        payload = {
            "schema_version": "1.0.0",
            "table_id": table_id,
            "hand_id": hand.hand_id,
            "revision": 0,
            "game": _game_id(hand),
            "players": len(active),
            "hero_position": _normalized_position(positions[hero]),
            "effective_stack_bb": format(effective_stack / hand.big_blind, "f"),
            "pot_bb": format(replay.committed_pot / hand.big_blind, "f"),
            "to_call_bb": format(to_call / hand.big_blind, "f"),
            "board": [str(card) for card in hand.board],
            "hero_cards": [str(card) for card in holding],
            "action_history": list(_normalized_history(hand, positions)),
            "legal_actions": [],
            "rake_model": self.rake_model,
            "utility_model": "chip_ev",
            "source": f"pokerstars_text_tail:{hand.table_name}",
        }
        return ProjectedDecision(
            "state_ready",
            f"Hero is next to act on the {_street(hand).name.lower()}.",
            hand.hand_id,
            hero,
            hero,
            payload,
        )

    @staticmethod
    def _current_commitments(
        hand: HandHistory,
        street: Street,
        ledgers: dict[str, PlayerLedger],
    ) -> dict[str, Decimal]:
        last_action_street = hand.actions[-1].street if hand.actions else Street.PREFLOP
        if last_action_street < street:
            return {player: D("0") for player in ledgers}
        return {
            player: ledger.committed_street
            for player, ledger in ledgers.items()
        }

    @staticmethod
    def _next_actor(
        hand: HandHistory,
        street: Street,
        street_actions: list[object],
        pending: set[str],
    ) -> str | None:
        players_by_seat = {player.seat: player.name for player in hand.players}
        seats = sorted(players_by_seat)
        if street_actions:
            last_player = street_actions[-1].player
            reference = hand.player(last_player).seat
        elif street == Street.PREFLOP:
            big_blind = next(
                (
                    action.player
                    for action in hand.actions
                    if action.kind == ActionKind.POST_BIG_BLIND
                ),
                None,
            )
            reference = hand.player(big_blind).seat if big_blind else hand.button_seat
        else:
            reference = hand.button_seat
        start = seats.index(reference)
        for offset in range(1, len(seats) + 1):
            player = players_by_seat[seats[(start + offset) % len(seats)]]
            if player in pending:
                return player
        return None


class PokerStarsLiveTailAdapter:
    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        projector: PokerStarsDecisionProjector | None = None,
    ) -> None:
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        self.max_file_bytes = max_file_bytes
        self.projector = projector or PokerStarsDecisionProjector()
        self.parser = PokerStarsParser()
        self._revisions: dict[str, tuple[str, int]] = {}
        self._lock = threading.RLock()

    def poll(self, source_path: str | Path, *, table_id: str) -> LiveCapturePoll:
        path = self._resolve_source(source_path)
        before = path.stat()
        if before.st_size > self.max_file_bytes:
            raise ValueError(f"File exceeds {self.max_file_bytes} byte live-tail limit")
        try:
            source = _read_history_file(path)
        except (OSError, UnicodeError) as error:
            return LiveCapturePoll(
                "pending_write", str(error), str(path), False, None, None, None, None, None
            )
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return LiveCapturePoll(
                "unstable_file",
                "The hand-history file changed while it was being read; retrying is safe.",
                str(path),
                False,
                None,
                None,
                None,
                None,
                None,
            )
        block = _latest_hand_block(source)
        if block is None:
            return LiveCapturePoll(
                "waiting_for_hand", "No PokerStars hand header was found.", str(path), False, None, None, None, None, None
            )
        if _SUMMARY_RE.search(block):
            return LiveCapturePoll(
                "waiting_for_new_hand",
                "The latest hand is complete; waiting for the next header.",
                str(path),
                False,
                None,
                None,
                None,
                None,
                None,
            )
        try:
            hand = self.parser.parse(block)
            if _has_unparsed_action_tail(block, hand):
                raise PokerStarsParseError(
                    "an action-like line is incomplete or unsupported"
                )
            projected = self.projector.project(hand, table_id=table_id)
        except (PokerStarsParseError, ValueError) as error:
            return LiveCapturePoll(
                "pending_write",
                f"The latest hand is not parseable yet: {error}",
                str(path),
                False,
                None,
                None,
                None,
                None,
                None,
            )
        if projected.payload is None:
            return LiveCapturePoll(
                projected.status,
                projected.message,
                str(path),
                False,
                None,
                projected.hand_id,
                projected.hero,
                projected.next_actor,
                None,
            )

        signature_payload = dict(projected.payload)
        signature_payload.pop("revision", None)
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        revision_key = f"{path}:{table_id}"
        with self._lock:
            previous = self._revisions.get(revision_key)
            if previous is not None and previous[0] == signature:
                revision = previous[1]
                changed = False
            else:
                revision = 0 if previous is None else previous[1] + 1
                self._revisions[revision_key] = (signature, revision)
                changed = True
        payload = dict(projected.payload)
        payload["revision"] = revision
        return LiveCapturePoll(
            projected.status,
            projected.message,
            str(path),
            changed,
            revision,
            projected.hand_id,
            projected.hero,
            projected.next_actor,
            payload,
        )

    @staticmethod
    def _resolve_source(source_path: str | Path) -> Path:
        root = Path(source_path).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Live hand-history path does not exist: {root}")
        if root.is_dir():
            candidates = [
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            ]
            if not candidates:
                raise ValueError(f"No .txt hand-history files found in: {root}")
            return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        if not root.is_file() or root.suffix.lower() != ".txt":
            raise ValueError("Live hand-history source must be a .txt file or folder")
        return root
