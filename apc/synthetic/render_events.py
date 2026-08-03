from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apc.annotator import AnnotationProject
from apc.synthetic.render_table import (
    LAYOUTS,
    STREETS,
    THEMES,
    bb_add,
    bb_subtract,
    render_frame,
    synthetic_stack,
)


PROVIDER_ID = "apc-synthetic-event-renderer-v1"
EVENT_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
EVENT_AMOUNTS = {"call": "1", "bet": "2.5", "raise": "4"}
BASE_POTS = {"preflop": "1.5", "flop": "6", "turn": "10", "river": "15"}


@dataclass(frozen=True)
class PlannedEvent:
    actor_seat: int
    action: str
    amount_bb: str | None
    stack_before_bb: str
    stack_after_bb: str
    pot_before_bb: str
    pot_after_bb: str
    hero_to_call_after_bb: str
    actor_status_after: str

    def annotation(self) -> dict[str, object]:
        result: dict[str, object] = {
            "actor_seat": self.actor_seat,
            "action": self.action,
        }
        if self.amount_bb is not None:
            result["amount_bb"] = self.amount_bb
        return result


def plan_event(*, session_index: int, street_index: int, seats: int, street: str) -> PlannedEvent:
    if seats < 2:
        raise ValueError("an event sequence requires at least two seats")
    action = EVENT_ACTIONS[(session_index + street_index) % len(EVENT_ACTIONS)]
    actor_seat = 2 + ((session_index + street_index) % (seats - 1))
    stack_before = synthetic_stack(actor_seat, street_index)
    amount = EVENT_AMOUNTS.get(action)
    if action == "all_in":
        amount = stack_before
    stack_after = bb_subtract(stack_before, amount) if amount is not None else stack_before
    pot_before = BASE_POTS[street]
    pot_after = bb_add(pot_before, amount) if amount is not None else pot_before
    to_call = amount if action in {"bet", "raise", "all_in"} and amount is not None else "0"
    status = "folded" if action == "fold" else "all_in" if action == "all_in" else "active"
    return PlannedEvent(
        actor_seat,
        action,
        amount,
        stack_before,
        stack_after,
        pot_before,
        pot_after,
        to_call,
        status,
    )


def generate_event_dataset(root: Path, *, sessions: int, seed: int) -> dict[str, object]:
    if sessions < 3:
        raise ValueError("Synthetic event generation requires at least three sessions")
    rng = random.Random(seed)
    project = AnnotationProject.create(
        root,
        project_id=f"apc-synthetic-events-{seed}",
        source_kind="synthetic_render",
        provider_id=PROVIDER_ID,
        layout_id="mixed",
        theme_id="mixed",
        locale="en-US",
        max_seats=9,
    )
    rendered_root = root / "rendered"
    rendered_root.mkdir(exist_ok=True)
    action_counts = {action: 0 for action in EVENT_ACTIONS}
    for session_index in range(sessions):
        seats, layout_id = LAYOUTS[session_index % len(LAYOUTS)]
        theme = THEMES[(session_index // len(LAYOUTS)) % len(THEMES)]
        session_id = f"event-{session_index:04d}"
        environment = {
            "source_kind": "synthetic_render",
            "provider_id": PROVIDER_ID,
            "layout_id": layout_id,
            "theme_id": theme["id"],
            "locale": "en-US",
            "max_seats": seats,
            "virtual_chips": True,
        }
        for street_index, street in enumerate(STREETS):
            event = plan_event(
                session_index=session_index,
                street_index=street_index,
                seats=seats,
                street=street,
            )
            action_counts[event.action] += 1
            rng_state = rng.getstate()
            before_path = rendered_root / f"{session_id}-{street_index:02d}-before.png"
            before = render_frame(
                before_path,
                rng=rng,
                session_id=session_id,
                sequence_index=street_index,
                seats=seats,
                layout_id=layout_id,
                theme=theme,
                street=street,
                seat_stack_overrides={event.actor_seat: event.stack_before_bb},
                pot_bb_override=event.pot_before_bb,
                to_call_bb_override="0",
                provider_id=PROVIDER_ID,
            )
            rng.setstate(rng_state)
            after_path = rendered_root / f"{session_id}-{street_index:02d}-after.png"
            event_annotation = event.annotation()
            after = render_frame(
                after_path,
                rng=rng,
                session_id=session_id,
                sequence_index=street_index,
                seats=seats,
                layout_id=layout_id,
                theme=theme,
                street=street,
                seat_stack_overrides={event.actor_seat: event.stack_after_bb},
                seat_status_overrides={event.actor_seat: event.actor_status_after},
                pot_bb_override=event.pot_after_bb,
                to_call_bb_override=event.hero_to_call_after_bb,
                observed_action=event_annotation,
                action_history=[event_annotation],
                provider_id=PROVIDER_ID,
            )
            for pair_index, rendered in enumerate((before, after)):
                record, _ = project.import_frame(
                    rendered.image_path,
                    capture_session_id=session_id,
                    timestamp_ms=(street_index * 2 + pair_index) * 500,
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
        "event_action_counts": action_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paired APC synthetic action-event frames.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=48151623)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_event_dataset(args.output.expanduser().resolve(), sessions=args.sessions, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
