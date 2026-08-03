from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apc.backend_adapter import BackendAdapterConfig, build_backend_observation
from apc.perception.baseline import BaselineCheckpoint, _image_path, _manifest_annotations, _percentile
from apc.perception.boundary_baseline import load_boundary_checkpoint, predict_boundary
from apc.perception.card_baseline import load_card_checkpoint
from apc.perception.event_baseline import load_event_checkpoint
from apc.perception.hand_tracker import TemporalHandTracker, resolve_visual_player_identities
from apc.perception.stack_baseline import load_stack_checkpoint
from apc.perception.table_state_baseline import load_table_state_checkpoint
from apc.perception.temporal_composite import infer_temporal_state
from apc.player_identity import PlayerIdentityRegistry
from apc.tools.validate_dataset import canonical_sha256, validate_manifest
from apc.tools.validate_sequences import audit_sequence_manifest


def _coach_types() -> tuple[Any, Any, Any]:
    try:
        from poker_coach import LiveTableService, SolverExportRegistry, VisualObservationAdapter
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach import LiveTableService, SolverExportRegistry, VisualObservationAdapter
    return LiveTableService, SolverExportRegistry, VisualObservationAdapter


def evaluate_coaching_bridge(
    *,
    manifest_path: str | Path,
    base_checkpoint_path: str | Path,
    card_checkpoint_path: str | Path,
    table_state_checkpoint_path: str | Path,
    stack_checkpoint_path: str | Path,
    event_checkpoint_path: str | Path,
    boundary_checkpoint_path: str | Path,
    solver_export_path: str | Path,
    split: str = "validation",
    boundary_threshold: float = 0.20,
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out coaching bridge split must be validation or test")
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not validate_manifest(manifest_file)["valid"] or not audit_sequence_manifest(manifest_file)["valid"]:
        raise ValueError("coaching bridge dataset validation failed")
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    sessions: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for row in annotations:
        if str(row[1]["capture_session_id"]) in eval_sessions:
            sessions[str(row[1]["capture_session_id"])].append(row)

    base = BaselineCheckpoint.load(base_checkpoint_path)
    card = load_card_checkpoint(card_checkpoint_path)
    table = load_table_state_checkpoint(table_state_checkpoint_path)
    stack = load_stack_checkpoint(stack_checkpoint_path)
    event = load_event_checkpoint(event_checkpoint_path)
    boundary = load_boundary_checkpoint(boundary_checkpoint_path)
    training_sessions = {
        "base": set(base.training_sessions),
        "card": {str(value) for value in card["training"]["capture_sessions"]},
        "table": {str(value) for value in table["training"]["capture_sessions"]},
        "stack": {str(value) for value in stack["training"]["capture_sessions"]},
        "event": {str(value) for value in event["training"]["capture_sessions"]},
        "boundary": {str(value) for value in boundary["training"]["capture_sessions"]},
    }
    overlap = {
        name: sorted(eval_sessions & source_sessions)
        for name, source_sessions in training_sessions.items()
        if eval_sessions & source_sessions
    }
    if overlap:
        raise ValueError(f"held-out coaching bridge evaluation leaks training sessions: {overlap}")

    LiveTableService, SolverExportRegistry, VisualObservationAdapter = _coach_types()
    solver_bundle = SolverExportRegistry().parse_file(
        Path(solver_export_path).expanduser().resolve()
    ).bundle
    tracker = TemporalHandTracker(minimum_boundary_confidence=boundary_threshold)
    identities = PlayerIdentityRegistry("apc-visual-signature")
    visual_adapter = VisualObservationAdapter(minimum_confidence="0", stable_frames=1)
    service = LiveTableService()
    live_sessions: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    latencies: list[float] = []

    for session, raw_rows in sorted(sessions.items()):
        ordered = sorted(raw_rows, key=lambda row: int(row[1]["sequence_index"]))
        previous_after_image: Path | None = None
        for pair_index in range(0, len(ordered), 2):
            before_path, before_annotation = ordered[pair_index]
            after_path, after_annotation = ordered[pair_index + 1]
            before_image = _image_path(before_path, before_annotation)
            after_image = _image_path(after_path, after_annotation)
            expected_start = bool(before_annotation["state"].get("hand_start"))
            if previous_after_image is None:
                predicted_start, boundary_confidence = True, 1.0
            else:
                boundary_prediction = predict_boundary(boundary, previous_after_image, before_image)
                predicted_start = bool(boundary_prediction["hand_start"])
                boundary_confidence = float(boundary_prediction["confidence"])
            prior_history = [] if predicted_start else tracker.prior_history(session)
            started = time.perf_counter()
            temporal = infer_temporal_state(
                before_image,
                after_image,
                base_checkpoint=base,
                card_checkpoint=card,
                table_state_checkpoint=table,
                stack_checkpoint=stack,
                event_checkpoint=event,
                prior_action_history=prior_history,
                history_complete=True,
            )
            tracked = tracker.submit(
                session,
                temporal,
                hand_start=predicted_start,
                boundary_confidence=boundary_confidence,
            )
            status_counts[str(tracked.get("status"))] += 1
            if isinstance(tracked.get("state"), dict):
                tracked = resolve_visual_player_identities(
                    tracked,
                    identities,
                    observed_at_ms=int(after_annotation["image"]["timestamp_ms"]),
                )
                status_counts[str(tracked.get("status"))] += 1
            built = build_backend_observation(
                tracked,
                config=BackendAdapterConfig(
                    multiway_effective_stack_policy="minimum_active_opponent"
                ),
            )
            bridge_status = str(built["status"])
            status_counts[bridge_status] += 1
            for missing in built.get("missing", []):
                missing_counts[str(missing)] += 1
            visual_status = None
            backend_status = None
            if built.get("payload") is not None:
                accepted = visual_adapter.submit(built["payload"])
                visual_status = str(accepted["status"])
                status_counts[f"visual:{visual_status}"] += 1
                if visual_status == "state_ready":
                    live_session = live_sessions.get(session)
                    if live_session is None:
                        live_session = service.create_session(session)["session_id"]
                        live_sessions[session] = live_session
                    backend = service.update_state(
                        live_session,
                        accepted["payload"],
                        solver_bundle.spots,
                    )
                    backend_status = str(backend["status"])
                    backend_counts[backend_status] += 1
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed)
            rows.append(
                {
                    "pair": f"{before_annotation['sample_id']}->{after_annotation['sample_id']}",
                    "expected_hand_start": expected_start,
                    "predicted_hand_start": predicted_start,
                    "tracker_status": tracked.get("status"),
                    "identity_status": tracked.get("identity_gate", {}).get("status"),
                    "human_readable_names": tracked.get("identity_gate", {}).get("human_readable_names"),
                    "bridge_status": bridge_status,
                    "missing": built.get("missing", []),
                    "visual_status": visual_status,
                    "backend_status": backend_status,
                    "recommendation_allowed": built.get("audit", {}).get("recommendation_allowed"),
                }
            )
            previous_after_image = after_image

    if not rows:
        raise ValueError("coaching bridge split has no temporal pairs")
    unsafe = [row for row in rows if row["recommendation_allowed"] is True]
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "pixel_to_coaching_backend_bridge_smoke",
        "promotion_eligible": False,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": {},
        "pairs": len(rows),
        "metrics": {
            "identity_gate_passed": sum(row["identity_status"] == "passed" for row in rows),
            "backend_observations_ready": sum(row["bridge_status"] == "observation_ready_uncalibrated" for row in rows),
            "visual_contract_states_ready": sum(row["visual_status"] == "state_ready" for row in rows),
            "backend_state_status_counts": dict(sorted(backend_counts.items())),
            "strict_gate_violations": len(unsafe),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "status_counts": dict(sorted(status_counts.items())),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "rows": rows,
        "identity_registry_snapshot_sha256": identities.snapshot()["snapshot_sha256"],
        "prediction_sha256": canonical_sha256(rows),
        "limitations": [
            "Synthetic development validation only; no promotion or controlled-visible claim.",
            "Visual identities are pseudonymous name-band signatures and human-readable_names is always false.",
            "The sample solver export has heads-up coverage only, so nine-seat backend states are expected to be unmatched.",
            "Raise contribution is not converted to raise-to sizing; strict semantics intentionally abstain on those histories.",
            "The visual confidence threshold is zero only to audit software interoperability; recommendation_allowed remains false for every row.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate APC pixels through the coaching backend contract.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--table-state-checkpoint", type=Path, required=True)
    parser.add_argument("--stack-checkpoint", type=Path, required=True)
    parser.add_argument("--event-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-checkpoint", type=Path, required=True)
    parser.add_argument("--solver-export", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--boundary-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_coaching_bridge(
            manifest_path=args.manifest,
            base_checkpoint_path=args.base_checkpoint,
            card_checkpoint_path=args.card_checkpoint,
            table_state_checkpoint_path=args.table_state_checkpoint,
            stack_checkpoint_path=args.stack_checkpoint,
            event_checkpoint_path=args.event_checkpoint,
            boundary_checkpoint_path=args.boundary_checkpoint,
            solver_export_path=args.solver_export,
            split=args.split,
            boundary_threshold=args.boundary_threshold,
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
