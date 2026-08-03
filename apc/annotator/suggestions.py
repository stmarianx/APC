from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from apc.annotator.project import AnnotationProject, utc_now
from apc.perception.baseline import BaselineCheckpoint
from apc.perception.card_baseline import load_card_checkpoint
from apc.perception.composite import infer_visible_state
from apc.perception.stack_baseline import load_stack_checkpoint
from apc.perception.table_state_baseline import load_table_state_checkpoint
from apc.tools.validate_dataset import canonical_sha256


def generate_project_suggestions(
    project: AnnotationProject,
    *,
    sample_ids: Iterable[str] | None = None,
    include_annotated: bool = False,
    predictor: Callable[[Path], dict[str, object]] | None = None,
    base_checkpoint_path: str | Path | None = None,
    card_checkpoint_path: str | Path | None = None,
    table_state_checkpoint_path: str | Path | None = None,
    stack_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    if predictor is None:
        required = (
            base_checkpoint_path,
            card_checkpoint_path,
            table_state_checkpoint_path,
            stack_checkpoint_path,
        )
        if any(path is None for path in required):
            raise ValueError("All four perception checkpoints are required")
        base = BaselineCheckpoint.load(base_checkpoint_path)
        card = load_card_checkpoint(card_checkpoint_path)
        table = load_table_state_checkpoint(table_state_checkpoint_path)
        stack = load_stack_checkpoint(stack_checkpoint_path)

        def predictor(path: Path) -> dict[str, object]:
            return infer_visible_state(
                path,
                base_checkpoint=base,
                card_checkpoint=card,
                table_state_checkpoint=table,
                stack_checkpoint=stack,
            )

    selected_ids = set(sample_ids) if sample_ids is not None else None
    unknown = selected_ids - {record.sample_id for record in project.records} if selected_ids is not None else set()
    if unknown:
        raise ValueError(f"Unknown APC samples: {sorted(unknown)}")
    rows: list[dict[str, object]] = []
    skipped_annotated = 0
    for record in project.records:
        if selected_ids is not None and record.sample_id not in selected_ids:
            continue
        if not include_annotated and project.load_annotation(record.sample_id) is not None:
            skipped_annotated += 1
            continue
        frame = project.root / record.frame_path
        prediction = predictor(frame)
        if not isinstance(prediction, dict):
            raise ValueError("Suggestion predictor must return a JSON object")
        payload: dict[str, object] = {
            "schema_version": "1.0.0",
            "kind": "apc_perception_suggestion",
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
            "model_status": prediction.get("status"),
            "minimum_supported_confidence": prediction.get("minimum_supported_confidence"),
            "checkpoint_provenance": prediction.get("checkpoint_provenance", {}),
            "suggested_visible_state": prediction.get("visible_state"),
            "field_confidence": prediction.get("field_confidence", {}),
            "perception_abstentions": prediction.get("perception_abstentions", []),
            "prediction_frame_sha256": prediction.get("frame", {}).get("image_sha256")
            if isinstance(prediction.get("frame"), dict)
            else None,
        }
        path = project.save_suggestion(record.sample_id, payload)
        saved = project.load_suggestion(record.sample_id)
        rows.append(
            {
                "sample_id": record.sample_id,
                "path": str(path),
                "model_status": payload["model_status"],
                "minimum_supported_confidence": payload["minimum_supported_confidence"],
                "suggestion_sha256": saved["suggestion_sha256"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "project_id": project.config["project_id"],
        "generated_suggestions": len(rows),
        "skipped_annotated": skipped_annotated,
        "rows": rows,
        "suggestions_sha256": canonical_sha256(rows),
        "status": project.status(),
        "policy": {
            "review_required": True,
            "auto_applied": False,
            "opens_training_gate": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate non-destructive APC model suggestions for captured frames.")
    parser.add_argument("project", type=Path)
    parser.add_argument("--sample", action="append", dest="samples")
    parser.add_argument("--include-annotated", action="store_true")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--table-state-checkpoint", type=Path, required=True)
    parser.add_argument("--stack-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = generate_project_suggestions(
            AnnotationProject(args.project),
            sample_ids=args.samples,
            include_annotated=args.include_annotated,
            base_checkpoint_path=args.base_checkpoint,
            card_checkpoint_path=args.card_checkpoint,
            table_state_checkpoint_path=args.table_state_checkpoint,
            stack_checkpoint_path=args.stack_checkpoint,
        )
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
