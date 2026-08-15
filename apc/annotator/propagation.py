from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from apc.annotator.project import AnnotationProject, utc_now
from apc.tools.validate_dataset import canonical_sha256


def _box(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    required = {"x", "y", "width", "height"}
    return copy.deepcopy(value) if required <= set(value) else None


def _layout_objects(annotation: dict[str, Any]) -> dict[str, object]:
    objects = annotation.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Source annotation has no objects")
    layout: dict[str, object] = {}
    table = _box(objects.get("table"))
    if table:
        layout["table"] = table
    seats = []
    for seat in objects.get("seats", []):
        if not isinstance(seat, dict) or not isinstance(seat.get("seat_no"), int):
            continue
        box = _box(seat.get("box"))
        if box:
            seats.append({"seat_no": seat["seat_no"], "box": box})
    layout["seats"] = seats
    for name in ("hero_cards", "board_cards", "action_buttons"):
        layout[name] = [
            {"box": box}
            for row in objects.get(name, [])
            if isinstance(row, dict) and (box := _box(row.get("box"))) is not None
        ]
    pot = objects.get("pot")
    if isinstance(pot, dict) and (pot_box := _box(pot.get("box"))) is not None:
        layout["pot"] = {"box": pot_box}
    turn_clock = objects.get("turn_clock")
    if isinstance(turn_clock, dict) and (clock_box := _box(turn_clock.get("box"))) is not None:
        layout["turn_clock"] = {"box": clock_box}
    return layout


def propagate_layout_suggestions(
    project: AnnotationProject,
    *,
    source_sample_id: str,
    target_sample_ids: Iterable[str] | None = None,
    replace_existing: bool = False,
) -> dict[str, object]:
    source_record = project.record(source_sample_id)
    source = project.load_annotation(source_sample_id)
    if source is None:
        raise ValueError("Layout propagation requires a saved source annotation")
    if source.get("provenance", {}).get("verified") is not True:
        raise ValueError("Layout propagation requires a verified source annotation")
    source_fingerprint = canonical_sha256(source)
    suggested_objects = _layout_objects(source)
    source_state = source.get("state") if isinstance(source.get("state"), dict) else {}
    selected = set(target_sample_ids) if target_sample_ids is not None else None
    known = {record.sample_id for record in project.records}
    if selected is not None and not selected <= known:
        raise ValueError(f"Unknown APC samples: {sorted(selected - known)}")

    rows: list[dict[str, object]] = []
    skipped_annotated = 0
    skipped_existing = 0
    for record in project.records:
        if record.capture_session_id != source_record.capture_session_id:
            continue
        if record.sequence_index <= source_record.sequence_index:
            continue
        if selected is not None and record.sample_id not in selected:
            continue
        if project.load_annotation(record.sample_id) is not None:
            skipped_annotated += 1
            continue
        if project.load_suggestion(record.sample_id) is not None and not replace_existing:
            skipped_existing += 1
            continue
        visible_state: dict[str, object] = {
            "layout_id": record.environment.get("layout_id")
            if record.environment
            else project.config["environment"]["layout_id"],
        }
        if isinstance(source_state.get("hero_seat"), int):
            visible_state["hero_seat"] = source_state["hero_seat"]
        payload: dict[str, object] = {
            "schema_version": "1.0.0",
            "kind": "apc_layout_propagation_suggestion",
            "sample_id": record.sample_id,
            "capture_session_id": record.capture_session_id,
            "sequence_index": record.sequence_index,
            "image": {
                "path": f"../{record.frame_path}",
                "sha256": record.sha256,
                "width": record.width,
                "height": record.height,
                "timestamp_ms": record.timestamp_ms,
            },
            "created_at": utc_now(),
            "review_required": True,
            "auto_applied": False,
            "model_status": "review_required_layout_propagation",
            "minimum_supported_confidence": None,
            "checkpoint_provenance": {},
            "source_annotation": {
                "sample_id": source_record.sample_id,
                "annotation_sha256": source_fingerprint,
            },
            "suggested_visible_state": visible_state,
            "suggested_objects": copy.deepcopy(suggested_objects),
            "field_confidence": {},
            "perception_abstentions": [
                "Only normalized layout geometry and the stable Hero seat were propagated.",
                "Cards, stacks, pot, timer value, dealer, actions, occupancy and player names require frame review.",
            ],
            "prediction_frame_sha256": record.sha256,
        }
        path = project.save_suggestion(record.sample_id, payload)
        saved = project.load_suggestion(record.sample_id)
        rows.append(
            {
                "sample_id": record.sample_id,
                "path": str(path),
                "suggestion_sha256": saved["suggestion_sha256"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "project_id": project.config["project_id"],
        "source_sample_id": source_record.sample_id,
        "source_annotation_sha256": source_fingerprint,
        "generated_suggestions": len(rows),
        "skipped_annotated": skipped_annotated,
        "skipped_existing_suggestion": skipped_existing,
        "rows": rows,
        "suggestions_sha256": canonical_sha256(rows),
        "policy": {
            "review_required": True,
            "auto_applied": False,
            "opens_training_gate": False,
            "same_session_only": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Propagate verified layout geometry as review-only APC suggestions.")
    parser.add_argument("project", type=Path)
    parser.add_argument("source_sample_id")
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = propagate_layout_suggestions(
            AnnotationProject(args.project),
            source_sample_id=args.source_sample_id,
            target_sample_ids=args.targets,
            replace_existing=args.replace_existing,
        )
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
