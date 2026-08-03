from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from apc.tools.validate_dataset import canonical_sha256


def _decimal_equal(left: object, right: object) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return False


def evaluate_partial_reference(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
) -> dict[str, object]:
    truth_file = Path(ground_truth_path).expanduser().resolve()
    prediction_file = Path(prediction_path).expanduser().resolve()
    truth = json.loads(truth_file.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_file.read_text(encoding="utf-8"))
    if truth.get("use_policy") != "frozen_out_of_domain_audit_only_not_training_or_gate_count":
        raise ValueError("partial reference must be explicitly frozen outside training")
    image_file = (truth_file.parent / str(truth["image"]["path"])).resolve()
    image_sha = hashlib.sha256(image_file.read_bytes()).hexdigest()
    if image_sha != truth["image"]["sha256"]:
        raise ValueError("partial reference image fingerprint is invalid")
    prediction_frame = prediction.get("frame")
    if not isinstance(prediction_frame, dict) or prediction_frame.get("image_sha256") != image_sha:
        raise ValueError("prediction does not belong to the frozen reference image")
    expected = truth["verified_partial_state"]
    visible = prediction.get("visible_state")
    if not isinstance(visible, dict):
        raise ValueError("prediction has no visible_state")
    checks = {
        "layout_id": visible.get("layout_id") == expected["layout_id"],
        "street": visible.get("street") == expected["street"],
        "hero_cards": visible.get("hero_cards") == expected["hero_cards"],
        "board_cards": visible.get("board_cards") == expected["board_cards"],
        "pot_bb": _decimal_equal(visible.get("pot_bb"), expected["pot_bb"]),
        "visible_seat_slots": (
            isinstance(visible.get("seat_stacks_bb"), list)
            and len(visible["seat_stacks_bb"]) == int(expected["visible_seat_slots"])
        ),
    }
    confidence = prediction.get("minimum_supported_confidence")
    safe_abstention = (
        isinstance(prediction.get("status"), str)
        and str(prediction["status"]).startswith("abstain_")
        and prediction.get("recommendation") is None
        and isinstance(confidence, (int, float))
        and float(confidence) < 0.90
    )
    card_audit = prediction.get("card_integrity_audit")
    invalid_cards_rejected = (
        isinstance(card_audit, dict)
        and card_audit.get("status") == "rejected"
        and "visible_card_integrity" in prediction.get("missing_critical_fields", [])
    )
    comparison = {
        "expected": {
            key: expected[key]
            for key in ("layout_id", "street", "hero_cards", "board_cards", "pot_bb", "visible_seat_slots")
        },
        "observed": {
            "layout_id": visible.get("layout_id"),
            "street": visible.get("street"),
            "hero_cards": visible.get("hero_cards"),
            "board_cards": visible.get("board_cards"),
            "pot_bb": visible.get("pot_bb"),
            "visible_seat_slots": len(visible.get("seat_stacks_bb", [])),
        },
        "checks": checks,
    }
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "frozen_partial_controlled_visible_ood_audit",
        "promotion_eligible": False,
        "reference_id": truth["reference_id"],
        "image_sha256": truth["image"]["sha256"],
        "image_integrity_verified": True,
        "verified_fields": len(checks),
        "exact_fields": sum(checks.values()),
        "exact_field_accuracy": sum(checks.values()) / len(checks),
        "safe_abstention": safe_abstention,
        "invalid_cards_rejected": invalid_cards_rejected,
        "comparison": comparison,
        "minimum_supported_confidence": confidence,
        "prediction_status": prediction.get("status"),
        "comparison_sha256": canonical_sha256(comparison),
        "limitations": [
            "One user-provided virtual-chip screenshot with partial ground truth only.",
            "The frame is frozen for OOD audit and cannot be used for tuning, training, or gate counts.",
            "Seat identities, action history, dealer, call price and legal actions are not verified.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare APC output with a frozen partial visible-table reference.")
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_partial_reference(args.ground_truth, args.prediction)
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
