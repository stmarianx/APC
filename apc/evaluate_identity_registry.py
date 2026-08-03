from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from apc.perception.baseline import _manifest_annotations
from apc.player_identity import NameObservation, PlayerIdentityRegistry, normalize_player_name
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


def evaluate_identity_registry(
    manifest_path: str | Path,
    *,
    split: str = "validation",
    provider_namespace: str = "apc-synthetic-identity",
) -> tuple[dict[str, object], PlayerIdentityRegistry]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out identity evaluation split must be validation or test")
    manifest_file = Path(manifest_path).expanduser().resolve()
    validation = validate_manifest(manifest_file)
    if not validation["valid"]:
        raise ValueError("identity dataset validation failed")
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    sessions: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for row in annotations:
        if str(row[1]["capture_session_id"]) in eval_sessions:
            sessions[str(row[1]["capture_session_id"])].append(row)
    registry = PlayerIdentityRegistry(provider_namespace)
    resolution_rows: list[dict[str, object]] = []
    total_observations = 0
    developing_observations = 0
    resolved_observations = 0
    final_exact = 0
    final_seats = 0
    for session, raw_rows in sorted(sessions.items()):
        rows = sorted(raw_rows, key=lambda row: int(row[1]["sequence_index"]))
        latest: dict[int, dict[str, object]] = {}
        expected: dict[int, str] = {}
        for _, annotation in rows:
            frame_sha = str(annotation["image"]["sha256"])
            timestamp = int(annotation["image"]["timestamp_ms"])
            observations = []
            for seat in annotation["objects"]["seats"]:
                seat_no = int(seat["seat_no"])
                name = str(seat["player_name"])
                expected.setdefault(seat_no, normalize_player_name(name))
                observations.append(NameObservation(seat_no, name, 1.0, frame_sha, timestamp))
            results = registry.observe_batch(session, observations)
            total_observations += len(results)
            developing_observations += sum(row["status"] == "developing" for row in results)
            resolved_observations += sum(row["status"] == "resolved" for row in results)
            for result in results:
                latest[int(result["seat_no"])] = result
        final_seats += len(expected)
        for seat_no, expected_name in expected.items():
            result = latest[seat_no]
            exact = result["status"] == "resolved" and result["normalized_name"] == expected_name
            final_exact += int(exact)
            resolution_rows.append(
                {
                    "session": session,
                    "seat_no": seat_no,
                    "expected_normalized_name": expected_name,
                    "status": result["status"],
                    "identity_id": result["identity_id"],
                    "profile_key": result["profile_key"],
                    "posterior_probability": result["posterior_probability"],
                    "frames": result["frames"],
                    "exact": exact,
                }
            )
    identity_ids = [str(row["identity_id"]) for row in resolution_rows if row["identity_id"] is not None]
    report = {
        "schema_version": "1.0.0",
        "evaluation_kind": "labeled_name_identity_association_smoke",
        "promotion_eligible": False,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "frames": sum(len(rows) for rows in sessions.values()),
        "name_observations": total_observations,
        "metrics": {
            "final_seat_resolution_accuracy": final_exact / final_seats,
            "unique_identity_rate": len(set(identity_ids)) / len(identity_ids),
            "developing_observations": developing_observations,
            "resolved_observations": resolved_observations,
            "final_seats": final_seats,
        },
        "resolutions": resolution_rows,
        "registry_snapshot_sha256": registry.snapshot()["snapshot_sha256"],
        "prediction_sha256": canonical_sha256(resolution_rows),
        "limitations": [
            "Uses verified annotation player-name strings with confidence 1.0; no pixels or OCR are evaluated.",
            "Proves uncertainty-aware association, collision handling and persistence only.",
            "A visual player-name OCR checkpoint and held-out controlled-table evaluation remain required."
        ],
    }
    return report, registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate APC's persistent player identity association layer.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--provider-namespace", default="apc-synthetic-identity")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registry-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, registry = evaluate_identity_registry(
            args.manifest,
            split=args.split,
            provider_namespace=args.provider_namespace,
        )
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.registry_output:
            registry.save(args.registry_output)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
