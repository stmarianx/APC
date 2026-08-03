from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from apc.perception.evaluate_table_locator import box_iou
from apc.perception.table_locator import detect_table_box
from apc.perception.viewport import NormalizedBox, ViewportCalibration
from apc.tools.validate_dataset import canonical_sha256


def evaluate_table_reference(
    image_path: str | Path,
    profile_path: str | Path,
    *,
    expected_image_sha256: str,
) -> dict[str, object]:
    image = Path(image_path).expanduser().resolve()
    actual_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    if actual_sha256 != expected_image_sha256:
        raise ValueError("Reference image fingerprint does not match")
    profile = ViewportCalibration.load(profile_path)
    if profile.source_kind != "verified_manual_table_box":
        raise ValueError("Reference audit requires a verified manual table box")
    prediction = detect_table_box(image)
    predicted_payload = prediction["table_box"]
    iou = (
        box_iou(
            profile.observed_table_box,
            NormalizedBox.from_dict(predicted_payload),
        )
        if predicted_payload is not None
        else 0.0
    )
    comparison = {
        "expected_table_box": profile.observed_table_box.to_dict(),
        "predicted_table_box": predicted_payload,
        "iou": iou,
        "iou_at_0_8": iou >= 0.8,
        "detected": predicted_payload is not None,
    }
    return {
        "schema_version": "1.0.0",
        "use_policy": "frozen_reference_audit_only_not_training_tuning_or_gate_count",
        "image_sha256": actual_sha256,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.to_dict()["profile_sha256"],
        "prediction_status": prediction["status"],
        "prediction_confidence": prediction["confidence"],
        "prediction_latency_ms": prediction["latency_ms"],
        "comparison": comparison,
        "comparison_sha256": canonical_sha256(comparison),
        "prediction_sha256": canonical_sha256(prediction),
        "limitations": [
            "This is a single frozen OOD frame and cannot establish calibration.",
            "The result is not used to tune the detector or open any training/coaching gate.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-pass audit of table localization on a frozen reference.")
    parser.add_argument("image", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--expected-image-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_table_reference(
            args.image,
            args.profile,
            expected_image_sha256=args.expected_image_sha256,
        )
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
