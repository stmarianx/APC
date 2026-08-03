from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from apc.annotator import AnnotationProject
from apc.synthetic.render_table import (
    BOARD_COUNTS,
    LAYOUTS,
    RANKS,
    STREETS,
    SUITS,
    THEMES,
    bb_add,
    bb_subtract,
    render_frame,
)


PROVIDER_ID = "apc-synthetic-hand-sequence-renderer-v2"
GENERATION_ALGORITHM = "session-isolated-resumable-v2"
EARLY_ACTIONS = ("check", "call", "bet", "raise")
RIVER_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
FIXED_AMOUNTS = {"call": "1", "bet": "2.5", "raise": "4"}


def _session_seed(seed: int, session_index: int) -> int:
    material = f"{GENERATION_ALGORITHM}:{seed}:{session_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _generation_plan(*, sessions: int, hands_per_session: int, seed: int) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "algorithm": GENERATION_ALGORITHM,
        "provider_id": PROVIDER_ID,
        "sessions": sessions,
        "hands_per_session": hands_per_session,
        "seed": seed,
        "frames_per_hand": len(STREETS) * 2,
    }


def _write_plan(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class SequenceEvent:
    actor_seat: int
    action: str
    amount_bb: str | None

    def annotation(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "actor_seat": self.actor_seat,
            "action": self.action,
        }
        if self.amount_bb is not None:
            payload["amount_bb"] = self.amount_bb
        return payload


def _deck(rng: random.Random) -> list[tuple[str, str]]:
    cards = [(rank, suit) for rank in RANKS for suit in SUITS]
    rng.shuffle(cards)
    return cards


def _event_for_street(
    *,
    global_hand_index: int,
    street_index: int,
    seats: int,
    stacks: dict[int, str],
) -> SequenceEvent:
    actor = 2 + ((global_hand_index + street_index) % (seats - 1))
    if street_index < 3:
        action = EARLY_ACTIONS[(global_hand_index + street_index) % len(EARLY_ACTIONS)]
    else:
        action = RIVER_ACTIONS[global_hand_index % len(RIVER_ACTIONS)]
    amount = FIXED_AMOUNTS.get(action)
    if action == "all_in":
        amount = stacks[actor]
    return SequenceEvent(actor, action, amount)


def _initial_stacks(*, seats: int, global_hand_index: int) -> dict[int, str]:
    return {
        seat: str(90 + ((seat * 11 + global_hand_index * 7) % 35))
        for seat in range(1, seats + 1)
    }


def generate_hand_sequence_dataset(
    root: Path,
    *,
    sessions: int,
    hands_per_session: int,
    seed: int,
    resume: bool = False,
    session_limit: int | None = None,
) -> dict[str, object]:
    if sessions < 3:
        raise ValueError("Synthetic hand sequences require at least three capture sessions")
    if hands_per_session < 2:
        raise ValueError("At least two hands per session are required to label hand boundaries")
    if session_limit is not None and session_limit <= 0:
        raise ValueError("session_limit must be positive")
    root = root.expanduser().resolve()
    plan = _generation_plan(
        sessions=sessions,
        hands_per_session=hands_per_session,
        seed=seed,
    )
    plan_path = root / "generation_plan.json"
    if resume:
        if not plan_path.is_file():
            raise ValueError(f"Cannot resume without generation plan: {plan_path}")
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise ValueError("Resume parameters do not match the immutable generation plan")
        project = AnnotationProject(root)
        expected_project_id = f"apc-synthetic-hand-sequences-{seed}"
        if project.config.get("project_id") != expected_project_id:
            raise ValueError("Resume project_id does not match the generation seed")
        if project.config.get("environment", {}).get("provider_id") != PROVIDER_ID:
            raise ValueError("Resume provider does not match the renderer version")
    else:
        project = AnnotationProject.create(
            root,
            project_id=f"apc-synthetic-hand-sequences-{seed}",
            source_kind="synthetic_render",
            provider_id=PROVIDER_ID,
            layout_id="mixed",
            theme_id="mixed",
            locale="en-US",
            max_seats=9,
        )
        _write_plan(plan_path, plan)
    rendered_root = root / "rendered"
    rendered_root.mkdir(exist_ok=True)
    action_counts = {action: 0 for action in RIVER_ACTIONS}
    for global_hand_index in range(sessions * hands_per_session):
        seats, _ = LAYOUTS[(global_hand_index // hands_per_session) % len(LAYOUTS)]
        stacks = _initial_stacks(seats=seats, global_hand_index=global_hand_index)
        for street_index in range(len(STREETS)):
            event = _event_for_street(
                global_hand_index=global_hand_index,
                street_index=street_index,
                seats=seats,
                stacks=stacks,
            )
            action_counts[event.action] += 1
    boundary_frames = sessions * hands_per_session
    expected_frames_per_session = hands_per_session * len(STREETS) * 2
    expected_session_ids = {f"handseq-{index:04d}" for index in range(sessions)}
    initial_counts = Counter(record.capture_session_id for record in project.records)
    unexpected = sorted(set(initial_counts) - expected_session_ids)
    if unexpected:
        raise ValueError(f"Resume project contains unexpected sessions: {unexpected}")
    if any(count > expected_frames_per_session for count in initial_counts.values()):
        raise ValueError("Resume project contains more frames than the generation plan")
    skipped_complete_sessions = 0
    rendered_sessions = 0
    for session_index in range(sessions):
        seats, layout_id = LAYOUTS[session_index % len(LAYOUTS)]
        theme = THEMES[(session_index // len(LAYOUTS)) % len(THEMES)]
        session_id = f"handseq-{session_index:04d}"
        if initial_counts.get(session_id, 0) == expected_frames_per_session:
            skipped_complete_sessions += 1
            continue
        if session_limit is not None and rendered_sessions >= session_limit:
            continue
        rendered_sessions += 1
        rng = random.Random(_session_seed(seed, session_index))
        environment = {
            "source_kind": "synthetic_render",
            "provider_id": PROVIDER_ID,
            "layout_id": layout_id,
            "theme_id": theme["id"],
            "locale": "en-US",
            "max_seats": seats,
            "virtual_chips": True,
        }
        player_names = {
            seat: "Hero" if seat == 1 else f"Bot_{session_index:02d}_{seat:02d}"
            for seat in range(1, seats + 1)
        }
        timestamp_ms = 0
        for hand_index in range(hands_per_session):
            global_hand_index = session_index * hands_per_session + hand_index
            hand_id = f"{session_id}-hand-{hand_index:03d}"
            cards = _deck(rng)
            hero_cards = [cards.pop(), cards.pop()]
            full_board = [cards.pop() for _ in range(5)]
            dealer_seat = 1 + (hand_index % seats)
            stacks = _initial_stacks(seats=seats, global_hand_index=global_hand_index)
            statuses = {seat: "active" for seat in range(1, seats + 1)}
            pot = "1.5"
            history: list[dict[str, object]] = []
            for street_index, street in enumerate(STREETS):
                event = _event_for_street(
                    global_hand_index=global_hand_index,
                    street_index=street_index,
                    seats=seats,
                    stacks=stacks,
                )
                board = full_board[: BOARD_COUNTS[street]]
                stem = f"{session_id}-h{hand_index:03d}-s{street_index:02d}"
                before_path = rendered_root / f"{stem}-before.png"
                before = render_frame(
                    before_path,
                    rng=rng,
                    session_id=session_id,
                    sequence_index=street_index,
                    seats=seats,
                    layout_id=layout_id,
                    theme=theme,
                    street=street,
                    seat_stack_overrides=stacks,
                    seat_status_overrides=statuses,
                    pot_bb_override=pot,
                    to_call_bb_override="0",
                    action_history=history,
                    provider_id=PROVIDER_ID,
                    hero_cards_override=hero_cards,
                    board_cards_override=board,
                    dealer_seat_override=dealer_seat,
                    hand_id_override=hand_id,
                    seat_name_overrides=player_names,
                    hand_start=street_index == 0,
                )
                event_annotation = event.annotation()
                after_stacks = dict(stacks)
                after_statuses = dict(statuses)
                after_pot = pot
                if event.amount_bb is not None:
                    after_stacks[event.actor_seat] = bb_subtract(
                        stacks[event.actor_seat], event.amount_bb
                    )
                    after_pot = bb_add(pot, event.amount_bb)
                if event.action == "fold":
                    after_statuses[event.actor_seat] = "folded"
                elif event.action == "all_in":
                    after_statuses[event.actor_seat] = "all_in"
                after_history = [*history, event_annotation]
                after_path = rendered_root / f"{stem}-after.png"
                after = render_frame(
                    after_path,
                    rng=rng,
                    session_id=session_id,
                    sequence_index=street_index,
                    seats=seats,
                    layout_id=layout_id,
                    theme=theme,
                    street=street,
                    seat_stack_overrides=after_stacks,
                    seat_status_overrides=after_statuses,
                    pot_bb_override=after_pot,
                    to_call_bb_override=(
                        event.amount_bb
                        if event.action in {"bet", "raise", "all_in"}
                        and event.amount_bb is not None
                        else "0"
                    ),
                    observed_action=event_annotation,
                    action_history=after_history,
                    provider_id=PROVIDER_ID,
                    hero_cards_override=hero_cards,
                    board_cards_override=board,
                    dealer_seat_override=dealer_seat,
                    hand_id_override=hand_id,
                    seat_name_overrides=player_names,
                    hand_start=False,
                )
                for rendered in (before, after):
                    record, imported = project.import_frame(
                        rendered.image_path,
                        capture_session_id=session_id,
                        timestamp_ms=timestamp_ms,
                        environment=environment,
                    )
                    annotation = project.annotation_template(record.sample_id)
                    annotation.update(rendered.annotation)
                    if imported:
                        project.save_annotation(record.sample_id, annotation)
                    else:
                        if (
                            record.capture_session_id != session_id
                            or record.timestamp_ms != timestamp_ms
                            or record.environment != environment
                        ):
                            raise RuntimeError(
                                f"resume frame identity mismatch: {rendered.image_path}"
                            )
                        if project.load_annotation(record.sample_id) != annotation:
                            raise RuntimeError(
                                f"resume annotation mismatch: {rendered.image_path}"
                            )
                    timestamp_ms += 350
                stacks, statuses, pot, history = (
                    after_stacks,
                    after_statuses,
                    after_pot,
                    after_history,
                )
    final_counts = Counter(record.capture_session_id for record in project.records)
    complete = (
        set(final_counts) == expected_session_ids
        and all(
            final_counts.get(session_id, 0) == expected_frames_per_session
            for session_id in expected_session_ids
        )
    )
    manifest_path: Path | None = None
    report: dict[str, object] | None = None
    if complete:
        manifest_path, report = project.export_manifest(dataset_version="0.1.0")
    return {
        "project": project.status(),
        "manifest": str(manifest_path) if manifest_path else None,
        "validation": report,
        "seed": seed,
        "hands": sessions * hands_per_session,
        "hand_start_frames": boundary_frames,
        "event_action_counts": action_counts,
        "generation_algorithm": GENERATION_ALGORITHM,
        "complete": complete,
        "resumed": resume,
        "skipped_complete_sessions": skipped_complete_sessions,
        "rendered_sessions": rendered_sessions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate complete multi-street APC synthetic hand sequences.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--hands-per-session", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--session-limit",
        type=int,
        help="Render at most this many incomplete sessions in the current invocation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_hand_sequence_dataset(
        args.output.expanduser().resolve(),
        sessions=args.sessions,
        hands_per_session=args.hands_per_session,
        seed=args.seed,
        resume=args.resume,
        session_limit=args.session_limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
