from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import ActionKind, Card, HandAction, HandHistory, HoleCards, Player, PotAward, Street


class PokerStarsParseError(ValueError):
    pass


_HEADER_RE = re.compile(
    r"^PokerStars (?:Hand|Game) #(?P<id>\d+):\s*(?P<game>.+?)\s+\((?P<stakes>[^)]+)\)\s+-\s+(?P<date>.+)$",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(r"^Table ['‘](?P<name>.+?)['’](?:\s+(?P<max>\d+)-max)?\s+Seat #(?P<button>\d+) is the button", re.IGNORECASE)
_SEAT_RE = re.compile(r"^Seat (?P<seat>\d+): (?P<name>.+?) \((?P<stack>.+?)(?: in chips)?\)$", re.IGNORECASE)
_DEALT_RE = re.compile(r"^Dealt to (?P<name>.+?) \[(?P<cards>[^]]+)\]$", re.IGNORECASE)
_COLLECTED_RE = re.compile(r"^(?P<name>.+?)(?::)? collected (?P<amount>\S+) from (?P<pot>.+)$", re.IGNORECASE)
_RETURN_RE = re.compile(r"^Uncalled bet \((?P<amount>[^)]+)\) returned to (?P<name>.+)$", re.IGNORECASE)
_TOTAL_RE = re.compile(r"^Total pot (?P<pot>\S+).*?\| Rake (?P<rake>\S+)", re.IGNORECASE)
_STAKES_RE = re.compile(r"(?P<sb>[$€£]?[\d,.]+)\s*/\s*(?P<bb>[$€£]?[\d,.]+)(?:\s+(?P<currency>Play Money|[A-Z]{3}))?", re.IGNORECASE)


def _amount(text: str) -> Decimal:
    cleaned = text.strip().replace(",", "")
    cleaned = re.sub(r"^[\$€£]", "", cleaned)
    cleaned = re.sub(r"[^0-9.\-].*$", "", cleaned)
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError) as exc:
        raise PokerStarsParseError(f"Invalid amount: {text!r}") from exc


def _cards(text: str) -> tuple[Card, ...]:
    return tuple(Card.parse(token) for token in text.split() if token.strip())


class PokerStarsParser:
    """Parse saved English PokerStars-style hold'em hand histories.

    Unknown informational lines are retained in the hand's source text but do not
    become invented events. Structurally invalid required fields raise explicitly.
    """

    def parse_file(self, path: str | Path) -> tuple[HandHistory, ...]:
        source = Path(path).read_text(encoding="utf-8-sig")
        return self.parse_many(source)

    def parse_many(self, source: str) -> tuple[HandHistory, ...]:
        blocks = [block.strip() for block in re.split(r"(?=^PokerStars (?:Hand|Game) #)", source, flags=re.MULTILINE) if block.strip()]
        return tuple(self.parse(block) for block in blocks)

    def parse(self, source: str) -> HandHistory:
        # Regional desktop clients commonly emit trailing spaces after seat and
        # action lines. They are formatting noise, not part of player identity.
        lines = [line.strip("\ufeff\r ") for line in source.splitlines() if line.strip()]
        if not lines:
            raise PokerStarsParseError("Empty hand history")
        header = _HEADER_RE.match(lines[0])
        if not header:
            raise PokerStarsParseError(f"Unsupported PokerStars header: {lines[0]!r}")

        game = header.group("game").strip()
        limit_name = "no_limit" if "no limit" in game.lower() else "unknown"
        stakes_raw = header.group("stakes").strip()
        stakes = _STAKES_RE.search(stakes_raw)
        small_blind = _amount(stakes.group("sb")) if stakes else None
        big_blind = _amount(stakes.group("bb")) if stakes else None
        currency = stakes.group("currency") if stakes and stakes.group("currency") else None
        if currency is None:
            currency_context = f"{header.group('game')} {stakes_raw}"
            if " USD" in currency_context or "$" in currency_context:
                currency = "USD"
            elif " EUR" in currency_context or "€" in currency_context:
                currency = "EUR"
            elif " GBP" in currency_context or "£" in currency_context:
                currency = "GBP"

        table_name = ""
        max_seats: int | None = None
        button_seat: int | None = None
        players: list[Player] = []
        actions: list[HandAction] = []
        awards: list[PotAward] = []
        holdings: dict[str, HoleCards] = {}
        board: tuple[Card, ...] = ()
        street = Street.PREFLOP
        total_pot: Decimal | None = None
        rake: Decimal | None = None

        def add_action(kind: ActionKind, raw: str, player: str | None = None, amount: Decimal = Decimal("0"), to_amount: Decimal | None = None, all_in: bool = False, cards: tuple[Card, ...] = ()) -> None:
            actions.append(HandAction(len(actions), street, kind, player, amount, to_amount, all_in, cards, raw))

        for line in lines[1:]:
            table = _TABLE_RE.match(line)
            if table:
                table_name = table.group("name")
                max_seats = int(table.group("max")) if table.group("max") else None
                button_seat = int(table.group("button"))
                continue
            seat = _SEAT_RE.match(line)
            if seat and street == Street.PREFLOP and not line.startswith("Seat #"):
                players.append(Player(int(seat.group("seat")), seat.group("name"), _amount(seat.group("stack"))))
                continue

            if line.startswith("*** HOLE CARDS ***"):
                street = Street.PREFLOP
                continue
            if line.startswith("*** FLOP ***"):
                street = Street.FLOP
                groups = re.findall(r"\[([^]]+)\]", line)
                board = _cards(groups[0]) if groups else ()
                continue
            if line.startswith("*** TURN ***"):
                street = Street.TURN
                groups = re.findall(r"\[([^]]+)\]", line)
                board = _cards(" ".join(groups)) if groups else board
                continue
            if line.startswith("*** RIVER ***"):
                street = Street.RIVER
                groups = re.findall(r"\[([^]]+)\]", line)
                board = _cards(" ".join(groups)) if groups else board
                continue
            if line.startswith("*** SHOW DOWN ***"):
                street = Street.SHOWDOWN
                continue
            if line.startswith("*** SUMMARY ***"):
                street = Street.SUMMARY
                continue

            dealt = _DEALT_RE.match(line)
            if dealt:
                holdings[dealt.group("name")] = HoleCards(dealt.group("name"), _cards(dealt.group("cards")), shown=False)
                continue
            returned = _RETURN_RE.match(line)
            if returned:
                add_action(ActionKind.RETURN, line, returned.group("name"), _amount(returned.group("amount")))
                continue
            collected = _COLLECTED_RE.match(line)
            if collected:
                awards.append(PotAward(collected.group("name"), _amount(collected.group("amount")), collected.group("pot")))
                continue
            total = _TOTAL_RE.match(line)
            if total:
                total_pot = _amount(total.group("pot"))
                rake = _amount(total.group("rake"))
                continue
            if line.startswith("Board [") and street == Street.SUMMARY:
                groups = re.findall(r"\[([^]]+)\]", line)
                if groups:
                    board = _cards(groups[0])
                continue

            if ":" not in line:
                continue
            player_name, description = line.split(":", 1)
            description = description.strip()
            lower = description.lower()
            all_in = "all-in" in lower or "all in" in lower
            if lower.startswith("posts small blind "):
                add_action(ActionKind.POST_SMALL_BLIND, line, player_name, _amount(description.split()[-1]))
            elif lower.startswith("posts big blind "):
                add_action(ActionKind.POST_BIG_BLIND, line, player_name, _amount(description.split()[-1]))
            elif lower.startswith("posts the ante ") or lower.startswith("posts ante "):
                add_action(ActionKind.POST_ANTE, line, player_name, _amount(description.split()[-1]))
            elif lower.startswith("posts") and "blind" in lower:
                add_action(ActionKind.POST_DEAD_BLIND, line, player_name, _amount(description.split()[-1]))
            elif lower.startswith("folds"):
                add_action(ActionKind.FOLD, line, player_name)
            elif lower.startswith("checks"):
                add_action(ActionKind.CHECK, line, player_name)
            elif lower.startswith("calls "):
                match = re.search(r"calls\s+([^ ]+)", description, re.IGNORECASE)
                if not match:
                    raise PokerStarsParseError(f"Cannot parse call: {line}")
                add_action(ActionKind.CALL, line, player_name, _amount(match.group(1)), all_in=all_in)
            elif lower.startswith("bets "):
                match = re.search(r"bets\s+([^ ]+)", description, re.IGNORECASE)
                if not match:
                    raise PokerStarsParseError(f"Cannot parse bet: {line}")
                add_action(ActionKind.BET, line, player_name, _amount(match.group(1)), all_in=all_in)
            elif lower.startswith("raises "):
                match = re.search(r"raises\s+([^ ]+)\s+to\s+([^ ]+)", description, re.IGNORECASE)
                if not match:
                    raise PokerStarsParseError(f"Cannot parse raise: {line}")
                add_action(ActionKind.RAISE, line, player_name, _amount(match.group(1)), _amount(match.group(2)), all_in)
            elif lower.startswith("shows "):
                groups = re.findall(r"\[([^]]+)\]", description)
                cards = _cards(groups[0]) if groups else ()
                if len(cards) == 2 and player_name not in holdings:
                    holdings[player_name] = HoleCards(player_name, cards, shown=True)
                add_action(ActionKind.SHOW, line, player_name, cards=cards)
            elif lower.startswith("mucks") or lower.startswith("doesn't show"):
                add_action(ActionKind.MUCK, line, player_name)

        if not table_name or button_seat is None:
            raise PokerStarsParseError("Missing table/button line")
        if not players:
            raise PokerStarsParseError("Missing player seats")

        return HandHistory(
            hand_id=header.group("id"),
            game=game,
            limit=limit_name,
            stakes_raw=stakes_raw,
            currency=currency,
            small_blind=small_blind,
            big_blind=big_blind,
            played_at_raw=header.group("date").strip(),
            table_name=table_name,
            max_seats=max_seats,
            button_seat=button_seat,
            players=tuple(players),
            hole_cards=tuple(holdings.values()),
            board=board,
            actions=tuple(actions),
            awards=tuple(awards),
            total_pot=total_pot,
            rake=rake,
            source=source,
        )
