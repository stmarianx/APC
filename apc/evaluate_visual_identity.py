from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from apc.perception.baseline import _manifest_annotations
from apc.player_identity import PlayerIdentityRegistry
from apc.tools.validate_dataset import canonical_sha256, validate_manifest
from apc.visual_identity_signature import extract_frame_signatures


def evaluate_visual_identity(
    manifest_path: str | Path,
    *,
    split: str = "validation",
    provider_namespace: str = "apc-visual-signature",
) -> tuple[dict[str, object], PlayerIdentityRegistry]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out visual identity split must be validation or test")
    manifest_file = Path(manifest_path).expanduser().resolve()
    validation = validate_manifest(manifest_file)
    if not validation["valid"]:
        raise ValueError("visual identity dataset validation failed")
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    registry = PlayerIdentityRegistry(provider_namespace)
    expected_tokens: dict[str, set[str]] = defaultdict(set)
    token_names: dict[str, set[str]] = defaultdict(set)
    final_by_seat: dict[tuple[str, int], dict[str, object]] = {}
    predictions: list[dict[str, object]] = []
    frame_count = 0

    for annotation_path, annotation in sorted(
        annotations,
        key=lambda row: (str(row[1]["capture_session_id"]), int(row[1]["sequence_index"])),
    ):
        session = str(annotation["capture_session_id"])
        if session not in eval_sessions:
            continue
        image_path = (annotation_path.parent / str(annotation["image"]["path"])).resolve()
        signatures = extract_frame_signatures(image_path, annotation["objects"]["seats"])
        timestamp = int(annotation["image"]["timestamp_ms"])
        resolutions = registry.observe_batch(
            session,
            [row.registry_observation(observed_at_ms=timestamp) for row in signatures],
        )
        resolution_by_seat = {int(row["seat_no"]): row for row in resolutions}
        frame_count += 1
        for seat, signature in zip(annotation["objects"]["seats"], signatures):
            expected_name = str(seat["player_name"])
            token = signature.visual_token
            expected_tokens[expected_name].add(token)
            token_names[token].add(expected_name)
            final_by_seat[(session, signature.seat_no)] = resolution_by_seat[signature.seat_no]
            predictions.append(
                {
                    "capture_session_id": session,
                    "sequence_index": int(annotation["sequence_index"]),
                    "seat_no": signature.seat_no,
                    "expected_player_name": expected_name,
                    "visual_token": token,
                    "quality_score": signature.quality_score,
                    "foreground_pixels": signature.foreground_pixels,
                }
            )

    stable_names = sum(len(tokens) == 1 for tokens in expected_tokens.values())
    collisions = {token: sorted(names) for token, names in token_names.items() if len(names) > 1}
    resolved_seats = sum(row.get("status") == "resolved" for row in final_by_seat.values())
    report = {
        "schema_version": "1.0.0",
        "evaluation_kind": "pixel_visual_identity_signature_smoke",
        "promotion_eligible": False,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "frames": frame_count,
        "signature_observations": len(predictions),
        "metrics": {
            "expected_player_names": len(expected_tokens),
            "stable_signature_rate": stable_names / len(expected_tokens),
            "token_collision_count": len(collisions),
            "final_seat_resolution_rate": resolved_seats / len(final_by_seat),
            "minimum_quality_score": min(row["quality_score"] for row in predictions),
        },
        "collisions": collisions,
        "registry_snapshot_sha256": registry.snapshot()["snapshot_sha256"],
        "prediction_sha256": canonical_sha256(predictions),
        "limitations": [
            "Reads pixels inside an annotation-provided seat box but does not detect seats itself.",
            "Produces a stable pseudonymous visual token, not human-readable OCR text.",
            "The signature quality score is uncalibrated and the synthetic bitmap font is much narrower than controlled visible-table variation.",
            "A controlled-visible audit, OCR aliasing and appearance-change handling remain required before profile promotion.",
        ],
    }
    return report, registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate APC visual player signatures from seat-name pixels.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--provider-namespace", default="apc-visual-signature")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registry-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, registry = evaluate_visual_identity(
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
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
