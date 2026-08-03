from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


ALLOWED_SOURCES = {
    "controlled_training_table",
    "explicitly_permitted_virtual_table",
    "synthetic_render",
}
STREET_BOARD_COUNTS = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
    "showdown": 5,
}
MINIMUM_DATASET = {
    "verified_frames": 2000,
    "capture_sessions": 8,
    "layouts": 3,
    "themes": 2,
    "temporal_sequence_frames": 500,
    "controlled_visible_frames": 500,
    "controlled_visible_sessions": 2,
}
DECLARED_REQUIRED_STATS = (
    "captured_frames",
    "labeled_frames",
    "verified_frames",
    "double_audited_frames",
    "capture_sessions",
    "layouts",
    "themes",
    "temporal_sequence_frames",
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _issue(issues: list[str], path: str, message: str) -> None:
    issues.append(f"{path}: {message}")


def _object(value: object, path: str, issues: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        _issue(issues, path, "must be an object")
        return None
    return value


def _required(payload: dict[str, Any], keys: Iterable[str], path: str, issues: list[str]) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        _issue(issues, path, f"missing fields: {', '.join(missing)}")


def _bb(value: object, path: str, issues: list[str]) -> Decimal | None:
    if not isinstance(value, str) or not value:
        _issue(issues, path, "must be a nonnegative BB string")
        return None
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        _issue(issues, path, "must be a nonnegative BB string")
        return None
    if not result.is_finite() or result < 0:
        _issue(issues, path, "must be a nonnegative finite BB amount")
        return None
    return result


def _box(value: object, path: str, issues: list[str]) -> None:
    box = _object(value, path, issues)
    if box is None:
        return
    _required(box, ("x", "y", "width", "height"), path, issues)
    if any(key not in box for key in ("x", "y", "width", "height")):
        return
    try:
        x, y, width, height = (
            float(box["x"]),
            float(box["y"]),
            float(box["width"]),
            float(box["height"]),
        )
    except (TypeError, ValueError):
        _issue(issues, path, "coordinates must be numeric")
        return
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        _issue(issues, path, "must stay inside normalized image bounds")


def validate_annotation(
    payload: object,
    *,
    annotation_path: Path | None = None,
    require_image: bool = True,
) -> list[str]:
    issues: list[str] = []
    root = _object(payload, "$", issues)
    if root is None:
        return issues
    _required(
        root,
        (
            "schema_version",
            "sample_id",
            "capture_session_id",
            "sequence_index",
            "image",
            "environment",
            "state",
            "objects",
            "provenance",
        ),
        "$",
        issues,
    )
    if root.get("schema_version") != "1.0.0":
        _issue(issues, "$.schema_version", "must equal 1.0.0")

    image = _object(root.get("image"), "$.image", issues)
    if image is not None:
        _required(image, ("path", "sha256", "width", "height", "timestamp_ms"), "$.image", issues)
        digest = image.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            _issue(issues, "$.image.sha256", "must be a lowercase SHA-256 digest")
        raw_path = image.get("path")
        if require_image and isinstance(raw_path, str) and raw_path:
            base = annotation_path.parent if annotation_path is not None else Path.cwd()
            path = (base / raw_path).resolve()
            if not path.is_file():
                _issue(issues, "$.image.path", f"image does not exist: {path}")
            else:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != digest:
                    _issue(issues, "$.image.sha256", "does not match image bytes")

    environment = _object(root.get("environment"), "$.environment", issues)
    if environment is not None:
        _required(environment, ("source_kind", "provider_id", "layout_id", "theme_id", "locale", "max_seats", "virtual_chips"), "$.environment", issues)
        if environment.get("source_kind") not in ALLOWED_SOURCES:
            _issue(issues, "$.environment.source_kind", "is not an allowed controlled virtual-chip source")
        if environment.get("virtual_chips") is not True:
            _issue(issues, "$.environment.virtual_chips", "must be true")

    state = _object(root.get("state"), "$.state", issues)
    objects = _object(root.get("objects"), "$.objects", issues)
    if state is None or objects is None:
        return issues
    _required(state, ("game", "table_id", "hand_id", "street", "hero_seat", "dealer_seat", "pot_bb", "to_call_bb", "legal_actions", "action_history"), "$.state", issues)
    _required(objects, ("table", "seats", "hero_cards", "board_cards", "pot", "action_buttons"), "$.objects", issues)
    _bb(state.get("pot_bb"), "$.state.pot_bb", issues)
    _bb(state.get("to_call_bb"), "$.state.to_call_bb", issues)
    _box(objects.get("table"), "$.objects.table", issues)

    seats = objects.get("seats")
    if not isinstance(seats, list) or not 2 <= len(seats) <= 10:
        _issue(issues, "$.objects.seats", "must contain 2 to 10 seats")
        seats = []
    seat_numbers: list[int] = []
    hero_seats: list[int] = []
    dealer_seats: list[int] = []
    for index, raw_seat in enumerate(seats):
        path = f"$.objects.seats[{index}]"
        seat = _object(raw_seat, path, issues)
        if seat is None:
            continue
        _required(seat, ("seat_no", "box", "occupied", "is_hero", "has_dealer_button", "status", "visibility"), path, issues)
        _box(seat.get("box"), f"{path}.box", issues)
        seat_no = seat.get("seat_no")
        if isinstance(seat_no, int):
            seat_numbers.append(seat_no)
            if seat.get("is_hero") is True:
                hero_seats.append(seat_no)
            if seat.get("has_dealer_button") is True:
                dealer_seats.append(seat_no)
        if seat.get("stack_bb") is not None:
            _bb(seat.get("stack_bb"), f"{path}.stack_bb", issues)
        if seat.get("occupied") is False and seat.get("status") != "empty":
            _issue(issues, f"{path}.status", "an unoccupied seat must be empty")
    if len(seat_numbers) != len(set(seat_numbers)):
        _issue(issues, "$.objects.seats", "seat numbers must be unique")
    if hero_seats != [state.get("hero_seat")]:
        _issue(issues, "$.state.hero_seat", "must match exactly one hero seat annotation")
    if dealer_seats != [state.get("dealer_seat")]:
        _issue(issues, "$.state.dealer_seat", "must match exactly one dealer-button annotation")

    cards: list[str] = []
    for collection_name in ("hero_cards", "board_cards"):
        raw_cards = objects.get(collection_name)
        if not isinstance(raw_cards, list):
            _issue(issues, f"$.objects.{collection_name}", "must be an array")
            continue
        if collection_name == "hero_cards" and len(raw_cards) not in (0, 2):
            _issue(issues, "$.objects.hero_cards", "must contain zero or two cards")
        for index, raw_card in enumerate(raw_cards):
            path = f"$.objects.{collection_name}[{index}]"
            card = _object(raw_card, path, issues)
            if card is None:
                continue
            _required(card, ("box", "rank", "suit", "visibility"), path, issues)
            _box(card.get("box"), f"{path}.box", issues)
            rank, suit = card.get("rank"), card.get("suit")
            if rank not in ("back", "unknown") and suit not in ("none", "unknown", None):
                cards.append(f"{rank}{suit}")
    if len(cards) != len(set(cards)):
        _issue(issues, "$.objects", "visible cards must be unique")
    street = state.get("street")
    board_cards = objects.get("board_cards")
    if street in STREET_BOARD_COUNTS and isinstance(board_cards, list) and len(board_cards) != STREET_BOARD_COUNTS[street]:
        _issue(issues, "$.objects.board_cards", f"{street} requires {STREET_BOARD_COUNTS[street]} board cards")

    pot = _object(objects.get("pot"), "$.objects.pot", issues)
    if pot is not None:
        _required(pot, ("box", "amount_bb", "raw_text", "visibility"), "$.objects.pot", issues)
        _box(pot.get("box"), "$.objects.pot.box", issues)
        _bb(pot.get("amount_bb"), "$.objects.pot.amount_bb", issues)
        if pot.get("amount_bb") != state.get("pot_bb"):
            _issue(issues, "$.objects.pot.amount_bb", "must equal state.pot_bb")

    buttons = objects.get("action_buttons")
    enabled_actions: set[str] = set()
    synthetic_v2 = environment is not None and environment.get("provider_id") == "apc-synthetic-renderer-v2"
    if not isinstance(buttons, list):
        _issue(issues, "$.objects.action_buttons", "must be an array")
    else:
        for index, raw_button in enumerate(buttons):
            path = f"$.objects.action_buttons[{index}]"
            button = _object(raw_button, path, issues)
            if button is None:
                continue
            _required(button, ("box", "action", "enabled", "raw_text", "visibility"), path, issues)
            _box(button.get("box"), f"{path}.box", issues)
            if button.get("enabled") is True and isinstance(button.get("action"), str):
                enabled_actions.add(button["action"])
            if button.get("amount_bb") is not None:
                _bb(button.get("amount_bb"), f"{path}.amount_bb", issues)
            if synthetic_v2 and button.get("enabled") is True and button.get("action") == "call":
                if button.get("amount_bb") != state.get("to_call_bb"):
                    _issue(issues, f"{path}.amount_bb", "synthetic v2 call price must be visible and equal state.to_call_bb")
    legal_actions = state.get("legal_actions")
    if isinstance(legal_actions, list) and set(legal_actions) != enabled_actions:
        _issue(issues, "$.state.legal_actions", "must equal enabled visible action buttons")
    return issues


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: str | Path, *, require_images: bool = True) -> dict[str, object]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _load_json(manifest_path)
    issues: list[str] = []
    root = _object(manifest, "$", issues)
    if root is None:
        return {"valid": False, "errors": issues}
    _required(root, ("schema_version", "dataset_id", "dataset_version", "created_at", "annotation_schema", "source_policy", "annotation_files", "splits", "statistics", "fingerprints"), "$", issues)
    if root.get("schema_version") != "1.0.0":
        _issue(issues, "$.schema_version", "must equal 1.0.0")
    annotation_files = root.get("annotation_files")
    if not isinstance(annotation_files, list) or not annotation_files:
        _issue(issues, "$.annotation_files", "must contain at least one annotation path")
        annotation_files = []
    annotations: list[tuple[Path, dict[str, Any]]] = []
    for index, raw_path in enumerate(annotation_files):
        annotation_path = (manifest_path.parent / str(raw_path)).resolve()
        if not annotation_path.is_file():
            _issue(issues, f"$.annotation_files[{index}]", f"does not exist: {annotation_path}")
            continue
        try:
            annotation = _load_json(annotation_path)
        except (OSError, json.JSONDecodeError) as error:
            _issue(issues, f"$.annotation_files[{index}]", str(error))
            continue
        annotation_issues = validate_annotation(annotation, annotation_path=annotation_path, require_image=require_images)
        issues.extend(f"{annotation_path.name}{issue[1:]}" for issue in annotation_issues)
        if isinstance(annotation, dict):
            annotations.append((annotation_path, annotation))

    sample_ids = [str(annotation.get("sample_id")) for _, annotation in annotations]
    if len(sample_ids) != len(set(sample_ids)):
        _issue(issues, "$.annotation_files", "sample_id values must be unique")
    sessions = {str(annotation.get("capture_session_id")) for _, annotation in annotations}
    split = _object(root.get("splits"), "$.splits", issues) or {}
    split_sets = {
        name: set(split.get(name, [])) if isinstance(split.get(name), list) else set()
        for name in ("train", "validation", "test")
    }
    if any(split_sets[left] & split_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        _issue(issues, "$.splits", "capture sessions must be group-exclusive")
    if set().union(*split_sets.values()) != sessions:
        _issue(issues, "$.splits", "split capture sessions must exactly cover annotated sessions")

    session_counts = Counter(str(annotation.get("capture_session_id")) for _, annotation in annotations)
    controlled_annotations = [
        annotation
        for _, annotation in annotations
        if annotation.get("environment", {}).get("source_kind") != "synthetic_render"
        and bool(annotation.get("provenance", {}).get("verified"))
    ]
    computed_stats = {
        "labeled_frames": len(annotations),
        "verified_frames": sum(bool(annotation.get("provenance", {}).get("verified")) for _, annotation in annotations),
        "double_audited_frames": sum(bool(annotation.get("provenance", {}).get("reviewer")) for _, annotation in annotations),
        "capture_sessions": len(sessions),
        "layouts": len({str(annotation.get("environment", {}).get("layout_id")) for _, annotation in annotations}),
        "themes": len({str(annotation.get("environment", {}).get("theme_id")) for _, annotation in annotations}),
        "temporal_sequence_frames": sum(count for count in session_counts.values() if count > 1),
        "controlled_visible_frames": len(controlled_annotations),
        "controlled_visible_sessions": len(
            {str(annotation.get("capture_session_id")) for annotation in controlled_annotations}
        ),
        "synthetic_frames": sum(
            annotation.get("environment", {}).get("source_kind") == "synthetic_render"
            for _, annotation in annotations
        ),
    }
    declared_stats = _object(root.get("statistics"), "$.statistics", issues) or {}
    _required(declared_stats, DECLARED_REQUIRED_STATS, "$.statistics", issues)
    for name, actual in computed_stats.items():
        if name in declared_stats and declared_stats.get(name) != actual:
            _issue(issues, f"$.statistics.{name}", f"declares {declared_stats.get(name)!r}, computed {actual}")
    if isinstance(declared_stats.get("captured_frames"), int) and declared_stats["captured_frames"] < computed_stats["labeled_frames"]:
        _issue(issues, "$.statistics.captured_frames", "cannot be smaller than labeled_frames")

    ordered_annotations = [annotation for _, annotation in sorted(annotations, key=lambda row: str(row[1].get("sample_id")))]
    duplicate_material = sorted(
        (
            str(annotation.get("image", {}).get("sha256")),
            str(annotation.get("image", {}).get("perceptual_hash", "")),
            str(annotation.get("capture_session_id")),
        )
        for _, annotation in annotations
    )
    expected_fingerprints = {
        "annotations_sha256": canonical_sha256(ordered_annotations),
        "split_sha256": canonical_sha256(split),
        "duplicate_audit_sha256": canonical_sha256(duplicate_material),
    }
    declared_fingerprints = _object(root.get("fingerprints"), "$.fingerprints", issues) or {}
    for name, expected in expected_fingerprints.items():
        if declared_fingerprints.get(name) != expected:
            _issue(issues, f"$.fingerprints.{name}", "does not match computed fingerprint")

    digest_splits: dict[str, set[str]] = {}
    for split_name, split_sessions in split_sets.items():
        digest_splits[split_name] = {
            str(annotation.get("image", {}).get("sha256"))
            for _, annotation in annotations
            if str(annotation.get("capture_session_id")) in split_sessions
        }
    if any(digest_splits[left] & digest_splits[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        _issue(issues, "$.splits", "identical image digests cross split boundaries")

    minimum_checks = {
        name: {
            "required": required,
            "actual": computed_stats.get(name, 0),
            "passed": computed_stats.get(name, 0) >= required,
        }
        for name, required in MINIMUM_DATASET.items()
    }
    minimum_checks["double_audit_rate"] = {
        "required": "0.10",
        "actual": format(
            Decimal(computed_stats["double_audited_frames"])
            / Decimal(max(1, computed_stats["verified_frames"])),
            "f",
        ),
        "passed": computed_stats["double_audited_frames"] * 10 >= computed_stats["verified_frames"],
    }
    return {
        "schema_version": "1.0.0",
        "dataset_id": root.get("dataset_id"),
        "valid": not issues,
        "errors": issues,
        "annotations": len(annotations),
        "computed_statistics": computed_stats,
        "computed_fingerprints": expected_fingerprints,
        "minimum_dataset": {
            "ready": not issues and all(check["passed"] for check in minimum_checks.values()),
            "checks": minimum_checks,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an APC visual dataset manifest and its annotations.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-images", action="store_true", help="Do not require or hash source image files")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return exit code 3 when the manifest is valid but Gate T minimums are not met",
    )
    parser.add_argument("--output", type=Path, help="Write the validation report as JSON")
    return parser


def validation_exit_code(report: dict[str, Any], *, require_ready: bool = False) -> int:
    if report.get("valid") is not True:
        return 1
    ready = bool(report.get("minimum_dataset", {}).get("ready"))
    if require_ready and not ready:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_manifest(args.manifest, require_images=not args.skip_images)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return validation_exit_code(report, require_ready=args.require_ready)


if __name__ == "__main__":
    raise SystemExit(main())
