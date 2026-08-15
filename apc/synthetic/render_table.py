from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from apc.annotator import AnnotationProject


WIDTH = 1280
HEIGHT = 720
RANKS = tuple("23456789TJQKA")
SUITS = tuple("cdhs")
STREETS = ("preflop", "flop", "turn", "river")
BOARD_COUNTS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
THEMES = (
    {
        "id": "midnight",
        "background": "#090d18",
        "felt": "#183b34",
        "rail": "#3c4b48",
        "panel": "#101816",
        "text": "#edf7f2",
        "accent": "#b8f171",
    },
    {
        "id": "ocean",
        "background": "#07131d",
        "felt": "#134f63",
        "rail": "#1f7183",
        "panel": "#0c2630",
        "text": "#eefaff",
        "accent": "#72e6ff",
    },
)
LAYOUTS = ((2, "heads-up"), (6, "six-max"), (9, "nine-max"))
SUIT_COLORS = {"c": "#1f9d55", "d": "#2675ff", "h": "#e33d4f", "s": "#151515"}
PROVIDER_ID = "apc-synthetic-renderer-v2"
CLOCK_VALUES_MS = (
    5000,
    8000,
    9000,
    10000,
    12000,
    18000,
    20000,
    28000,
    30000,
    38000,
    45000,
    48000,
    58000,
    60000,
    68000,
    75000,
    78000,
    88000,
    90000,
    98000,
    99000,
)
NAME_OCR_CHARSET = "ABEFJKLPRSTVXY0123456789"
NAME_OCR_LENGTH = 8


def _pil() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError(
            "Synthetic rendering requires Pillow; use the bundled workspace Python or install apc/requirements-vision.txt"
        ) from error
    return Image, ImageDraw, ImageFont


def normalized_box(box: tuple[int, int, int, int]) -> dict[str, float]:
    left, top, right, bottom = box
    return {
        "x": left / WIDTH,
        "y": top / HEIGHT,
        "width": (right - left) / WIDTH,
        "height": (bottom - top) / HEIGHT,
    }


def seat_boxes(seats: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    center_x, center_y = WIDTH / 2, HEIGHT / 2 - 20
    radius_x, radius_y = 475, 205
    for index in range(seats):
        angle = math.pi / 2 + (2 * math.pi * index / seats)
        x = center_x + radius_x * math.cos(angle)
        y = center_y + radius_y * math.sin(angle)
        boxes.append((round(x - 80), round(y - 34), round(x + 80), round(y + 34)))
    return boxes


def _card_deck(rng: random.Random) -> list[tuple[str, str]]:
    deck = [(rank, suit) for rank in RANKS for suit in SUITS]
    rng.shuffle(deck)
    return deck


def _draw_centered(draw: Any, box: tuple[int, int, int, int], text: str, *, fill: str, font: Any) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(((left + right - width) / 2, (top + bottom - height) / 2), text, fill=fill, font=font)


def _card_object(box: tuple[int, int, int, int], card: tuple[str, str]) -> dict[str, object]:
    return {
        "box": normalized_box(box),
        "rank": card[0],
        "suit": card[1],
        "visibility": "clear",
    }


@dataclass(frozen=True)
class RenderedFrame:
    image_path: Path
    annotation: dict[str, object]


def render_frame(
    output: Path,
    *,
    rng: random.Random,
    session_id: str,
    sequence_index: int,
    seats: int,
    layout_id: str,
    theme: dict[str, str],
    street: str,
    seat_stack_overrides: dict[int, str] | None = None,
    seat_status_overrides: dict[int, str] | None = None,
    pot_bb_override: str | None = None,
    to_call_bb_override: str | None = None,
    observed_action: dict[str, object] | None = None,
    action_history: list[dict[str, object]] | None = None,
    provider_id: str = PROVIDER_ID,
    hero_cards_override: list[tuple[str, str]] | None = None,
    board_cards_override: list[tuple[str, str]] | None = None,
    dealer_seat_override: int | None = None,
    hand_id_override: str | None = None,
    seat_name_overrides: dict[int, str] | None = None,
    hand_start: bool | None = None,
    decision_time_remaining_ms: int | None = None,
) -> RenderedFrame:
    Image, ImageDraw, ImageFont = _pil()
    image = Image.new("RGB", (WIDTH, HEIGHT), theme["background"])
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bold = font
    draw.rounded_rectangle((90, 105, 1190, 600), radius=230, fill=theme["rail"])
    table_box = (115, 130, 1165, 575)
    draw.rounded_rectangle(table_box, radius=210, fill=theme["felt"])
    deck = _card_deck(rng)
    hero_cards = list(hero_cards_override) if hero_cards_override is not None else [deck.pop(), deck.pop()]
    board = list(board_cards_override) if board_cards_override is not None else [deck.pop() for _ in range(BOARD_COUNTS[street])]
    if len(hero_cards) != 2:
        raise ValueError("hero card override must contain exactly two cards")
    if len(board) != BOARD_COUNTS[street]:
        raise ValueError(f"{street} board override must contain {BOARD_COUNTS[street]} cards")
    visible_cards = hero_cards + board
    if len(set(visible_cards)) != len(visible_cards):
        raise ValueError("visible card overrides must be unique")
    dealer_seat = dealer_seat_override or (1 + (sequence_index % seats))
    if not 1 <= dealer_seat <= seats:
        raise ValueError("dealer seat override is outside the table")
    hero_seat = 1
    pot_bb = pot_bb_override or {"preflop": "1.5", "flop": "6", "turn": "10", "river": "15"}[street]
    to_call_bb = to_call_bb_override or {"preflop": "0.5", "flop": "0", "turn": "2.5", "river": "0"}[street]
    legal_actions = ["fold", "call", "raise"] if DecimalLike(to_call_bb) > 0 else ["check", "bet"]
    hand_id = hand_id_override or f"synthetic-{session_id}"

    seats_payload = []
    for index, box in enumerate(seat_boxes(seats), start=1):
        draw.rounded_rectangle(box, radius=12, fill=theme["panel"], outline=theme["accent"], width=2 if index == hero_seat else 1)
        name = (seat_name_overrides or {}).get(index, "Hero" if index == hero_seat else f"Player {index}")
        stack = (seat_stack_overrides or {}).get(index, synthetic_stack(index, sequence_index))
        status = (seat_status_overrides or {}).get(index, "active")
        _draw_centered(draw, (box[0], box[1] + 4, box[2], box[1] + 30), name, fill=theme["text"], font=bold)
        _draw_centered(draw, (box[0], box[1] + 31, box[2], box[3] - 3), f"{stack} BB", fill=theme["accent"], font=font)
        if index == dealer_seat:
            draw.ellipse((box[2] - 20, box[1] - 8, box[2] + 4, box[1] + 16), fill="#f7f7f2")
            _draw_centered(draw, (box[2] - 20, box[1] - 8, box[2] + 4, box[1] + 16), "D", fill="#202020", font=bold)
        seats_payload.append(
            {
                "seat_no": index,
                "box": normalized_box(box),
                "occupied": True,
                "is_hero": index == hero_seat,
                "has_dealer_button": index == dealer_seat,
                "player_name": name,
                "stack_bb": stack,
                "raw_stack_text": f"{stack} BB",
                "status": status,
                "visibility": "clear",
            }
        )

    card_width, card_height, gap = 56, 78, 10
    board_left = int(WIDTH / 2 - (5 * card_width + 4 * gap) / 2)
    board_payload = []
    for index, card in enumerate(board):
        box = (board_left + index * (card_width + gap), 295, board_left + index * (card_width + gap) + card_width, 295 + card_height)
        draw.rounded_rectangle(box, radius=6, fill="#f7f3e8", outline="#d7d1c5")
        _draw_centered(draw, box, f"{card[0]}{card[1].upper()}", fill=SUIT_COLORS[card[1]], font=bold)
        board_payload.append(_card_object(box, card))

    hero_left = WIDTH // 2 - card_width - gap // 2
    hero_payload = []
    for index, card in enumerate(hero_cards):
        box = (hero_left + index * (card_width + gap), 585, hero_left + index * (card_width + gap) + card_width, 663)
        draw.rounded_rectangle(box, radius=6, fill="#f7f3e8", outline="#d7d1c5")
        _draw_centered(draw, box, f"{card[0]}{card[1].upper()}", fill=SUIT_COLORS[card[1]], font=bold)
        hero_payload.append(_card_object(box, card))

    pot_box = (550, 235, 730, 270)
    draw.rounded_rectangle(pot_box, radius=10, fill=theme["panel"])
    _draw_centered(draw, pot_box, f"Pot {pot_bb} BB", fill=theme["text"], font=bold)

    if observed_action is not None:
        banner_box = (475, 402, 805, 440)
        action = str(observed_action["action"])
        amount = observed_action.get("amount_bb") or observed_action.get("to_amount_bb")
        draw.rounded_rectangle(banner_box, radius=10, fill=theme["panel"], outline=theme["accent"])
        _draw_centered(draw, (485, 402, 575, 440), f"Seat {observed_action['actor_seat']}", fill=theme["text"], font=bold)
        _draw_centered(draw, (575, 402, 690, 440), action.replace("_", " ").title(), fill=theme["text"], font=bold)
        _draw_centered(draw, (690, 402, 795, 440), f"{amount} BB" if amount is not None else "-", fill=theme["text"], font=bold)

    buttons_payload = []
    button_width = 145
    total_width = len(legal_actions) * button_width + (len(legal_actions) - 1) * 12
    button_left = WIDTH - total_width - 28
    for index, action in enumerate(legal_actions):
        box = (button_left + index * (button_width + 12), 665, button_left + index * (button_width + 12) + button_width, 708)
        draw.rounded_rectangle(box, radius=9, fill=theme["accent"])
        label, amount_bb = action_display(action, to_call_bb)
        _draw_centered(draw, box, label, fill="#10200d", font=bold)
        button_payload = {
            "box": normalized_box(box),
            "action": action,
            "enabled": True,
            "raw_text": label,
            "visibility": "clear",
        }
        if amount_bb is not None:
            button_payload["amount_bb"] = amount_bb
        buttons_payload.append(button_payload)

    turn_clock_payload: dict[str, object] | None = None
    if decision_time_remaining_ms is not None:
        if (
            not isinstance(decision_time_remaining_ms, int)
            or isinstance(decision_time_remaining_ms, bool)
            or decision_time_remaining_ms <= 0
            or decision_time_remaining_ms % 1000
        ):
            raise ValueError("synthetic decision time must be a positive whole number of seconds")
        seconds = decision_time_remaining_ms // 1000
        clock_box = (28, 665, 128, 708)
        draw.rounded_rectangle(clock_box, radius=9, fill=theme["panel"], outline=theme["accent"])
        raw_clock = f"T {seconds} s"
        _draw_centered(draw, clock_box, raw_clock, fill=theme["accent"], font=bold)
        turn_clock_payload = {
            "box": normalized_box(clock_box),
            "remaining_ms": decision_time_remaining_ms,
            "raw_text": raw_clock,
            "visibility": "clear",
        }

    image.save(output, format="PNG", optimize=False)
    state_payload: dict[str, object] = {
            "game": "holdem_no_limit",
            "table_id": f"synthetic-{layout_id}-{theme['id']}",
            "hand_id": hand_id,
            "street": street,
            "hero_seat": hero_seat,
            "dealer_seat": dealer_seat,
            "pot_bb": pot_bb,
            "to_call_bb": to_call_bb,
            "legal_actions": legal_actions,
            "action_history": list(action_history or []),
    }
    if hand_start is not None:
        state_payload["hand_start"] = hand_start
    if turn_clock_payload is not None:
        state_payload.update(
            {
                "hero_to_act": True,
                "decision_time_remaining_ms": decision_time_remaining_ms,
                "decision_deadline_source": "training_table_clock",
            }
        )
    annotation = {
        "state": state_payload,
        "objects": {
            "table": normalized_box(table_box),
            "seats": seats_payload,
            "hero_cards": hero_payload,
            "board_cards": board_payload,
            "pot": {"box": normalized_box(pot_box), "amount_bb": pot_bb, "raw_text": f"Pot {pot_bb} BB", "visibility": "clear"},
            "action_buttons": buttons_payload,
            "observed_action": observed_action,
            "turn_clock": turn_clock_payload,
        },
        "provenance": {
            "annotator": provider_id,
            "annotation_version": 1,
            "verified": True,
            "reviewer": "deterministic-ground-truth",
            "created_at": "2026-08-02T00:00:00Z",
            "notes": "Synthetic bootstrap example; not a substitute for permitted real-table calibration.",
        },
    }
    return RenderedFrame(output, annotation)


def DecimalLike(value: str) -> float:
    return float(value)


def synthetic_stack(seat_no: int, sequence_index: int) -> str:
    return str(80 + ((seat_no * 13 + sequence_index * 7) % 70))


def bb_add(left: str, right: str) -> str:
    value = Decimal(left) + Decimal(right)
    return format(value.normalize(), "f")


def bb_subtract(left: str, right: str) -> str:
    value = Decimal(left) - Decimal(right)
    if value < 0:
        raise ValueError("synthetic BB subtraction cannot become negative")
    return format(value.normalize(), "f")


def action_display(action: str, to_call_bb: str) -> tuple[str, str | None]:
    if action == "call":
        return f"Call {to_call_bb} BB", to_call_bb
    return action.replace("_", " ").title(), None


def synthetic_ocr_player_names(
    rng: random.Random,
    seats: int,
) -> dict[int, str]:
    """Create stable, distinct fixed-advance names for the character OCR smoke task."""
    names: dict[int, str] = {}
    used: set[str] = set()
    for seat_no in range(1, seats + 1):
        while True:
            candidate = "".join(
                rng.choice(NAME_OCR_CHARSET) for _ in range(NAME_OCR_LENGTH)
            )
            if candidate not in used:
                names[seat_no] = candidate
                used.add(candidate)
                break
    return names


def generate_dataset(
    root: Path,
    *,
    sessions: int,
    seed: int,
    include_turn_clock: bool = False,
    include_name_ocr: bool = False,
) -> dict[str, object]:
    if sessions < 3:
        raise ValueError("Synthetic dataset generation requires at least three sessions")
    rng = random.Random(seed)
    project = AnnotationProject.create(
        root,
        project_id=f"apc-synthetic-{seed}",
        source_kind="synthetic_render",
        provider_id=PROVIDER_ID,
        layout_id="mixed",
        theme_id="mixed",
        locale="en-US",
        max_seats=9,
    )
    render_root = root / "rendered"
    render_root.mkdir(exist_ok=True)
    for session_index in range(sessions):
        seats, layout_id = LAYOUTS[session_index % len(LAYOUTS)]
        theme = THEMES[(session_index // len(LAYOUTS)) % len(THEMES)]
        session_id = f"synthetic-{session_index:04d}"
        session_names = synthetic_ocr_player_names(rng, seats) if include_name_ocr else None
        environment = {
            "source_kind": "synthetic_render",
            "provider_id": PROVIDER_ID,
            "layout_id": layout_id,
            "theme_id": theme["id"],
            "locale": "en-US",
            "max_seats": seats,
            "virtual_chips": True,
        }
        for sequence_index, street in enumerate(STREETS):
            rendered_path = render_root / f"{session_id}-{sequence_index:02d}.png"
            rendered = render_frame(
                rendered_path,
                rng=rng,
                session_id=session_id,
                sequence_index=sequence_index,
                seats=seats,
                layout_id=layout_id,
                theme=theme,
                street=street,
                seat_name_overrides=session_names,
                decision_time_remaining_ms=(
                    CLOCK_VALUES_MS[(session_index * len(STREETS) + sequence_index) % len(CLOCK_VALUES_MS)]
                    if include_turn_clock
                    else None
                ),
            )
            record, _ = project.import_frame(
                rendered.image_path,
                capture_session_id=session_id,
                timestamp_ms=sequence_index * 1000,
                environment=environment,
            )
            annotation = project.annotation_template(record.sample_id)
            annotation.update(rendered.annotation)
            project.save_annotation(record.sample_id, annotation)
    manifest_path, report = project.export_manifest(dataset_version="0.1.0")
    return {
        "project": project.status(),
        "manifest": str(manifest_path),
        "validation": report,
        "seed": seed,
        "include_turn_clock": include_turn_clock,
        "include_name_ocr": include_name_ocr,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deterministic synthetic APC visual dataset.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--include-turn-clock", action="store_true")
    parser.add_argument("--include-name-ocr", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    import json

    args = build_parser().parse_args(argv)
    result = generate_dataset(
        args.output.expanduser().resolve(),
        sessions=args.sessions,
        seed=args.seed,
        include_turn_clock=args.include_turn_clock,
        include_name_ocr=args.include_name_ocr,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
