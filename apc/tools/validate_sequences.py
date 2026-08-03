from __future__ import annotations

import argparse
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from apc.perception.baseline import _manifest_annotations
from apc.synthetic.render_table import STREETS
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


def _cards(annotation: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        f"{card['rank']}{card['suit']}"
        for collection in ("hero_cards", "board_cards")
        for card in annotation["objects"][collection]
    )


def _hero_cards(annotation: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{card['rank']}{card['suit']}" for card in annotation["objects"]["hero_cards"])


def _board(annotation: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{card['rank']}{card['suit']}" for card in annotation["objects"]["board_cards"])


def _stacks(annotation: dict[str, Any]) -> dict[int, Decimal]:
    try:
        return {
            int(seat["seat_no"]): Decimal(str(seat["stack_bb"]))
            for seat in annotation["objects"]["seats"]
        }
    except (InvalidOperation, TypeError, ValueError, KeyError) as error:
        raise ValueError("sequence contains an invalid seat stack") from error


def _names(annotation: dict[str, Any]) -> dict[int, str]:
    return {
        int(seat["seat_no"]): str(seat.get("player_name") or "")
        for seat in annotation["objects"]["seats"]
    }


def _audit_pair(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    pair_label: str,
    errors: list[str],
) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{pair_label}: {message}")

    event = after["objects"].get("observed_action")
    require(before["objects"].get("observed_action") is None, "before frame must not contain observed_action")
    require(isinstance(event, dict), "after frame must contain observed_action")
    for key in ("table_id", "hand_id", "street", "hero_seat", "dealer_seat"):
        require(before["state"].get(key) == after["state"].get(key), f"{key} changes inside adjacent pair")
    require(_cards(before) == _cards(after), "visible cards change inside adjacent pair")
    before_history = before["state"].get("action_history")
    after_history = after["state"].get("action_history")
    require(isinstance(before_history, list), "before history must be an array")
    require(isinstance(after_history, list), "after history must be an array")
    if isinstance(before_history, list) and isinstance(after_history, list) and isinstance(event, dict):
        require(after_history == [*before_history, event], "after history must append exactly the observed event")

    before_stacks, after_stacks = _stacks(before), _stacks(after)
    require(set(before_stacks) == set(after_stacks), "seat set changes inside adjacent pair")
    before_pot = Decimal(str(before["state"]["pot_bb"]))
    after_pot = Decimal(str(after["state"]["pot_bb"]))
    if isinstance(event, dict):
        actor = int(event["actor_seat"])
        action = str(event["action"])
        changed = sorted(seat for seat in before_stacks if before_stacks[seat] != after_stacks[seat])
        if action in {"fold", "check"}:
            require(not changed, "non-chip action changes a stack")
            require(after_pot == before_pot, "non-chip action changes the pot")
        else:
            amount = Decimal(str(event.get("amount_bb")))
            require(changed == [actor], "chip action must change only the actor stack")
            require(before_stacks[actor] - after_stacks[actor] == amount, "actor stack delta mismatches amount")
            require(after_pot - before_pot == amount, "pot delta mismatches amount")
            if action == "all_in":
                require(after_stacks[actor] == 0, "all-in must exhaust the actor stack")


def _audit_hand(
    rows: list[tuple[Path, dict[str, Any]]],
    *,
    session: str,
    errors: list[str],
) -> None:
    hand_id = str(rows[0][1]["state"]["hand_id"])
    label = f"{session}/{hand_id}"
    if len(rows) != len(STREETS) * 2:
        errors.append(f"{label}: complete hand must contain {len(STREETS) * 2} frames, got {len(rows)}")
        return
    if rows[0][1]["state"].get("hand_start") is not True:
        errors.append(f"{label}: first frame must be labeled hand_start=true")
    if any(row[1]["state"].get("hand_start") is not False for row in rows[1:]):
        errors.append(f"{label}: only the first frame may be hand_start=true")
    if rows[0][1]["state"].get("action_history") != []:
        errors.append(f"{label}: initial action history must be empty")

    previous_after: dict[str, Any] | None = None
    for street_index, expected_street in enumerate(STREETS):
        before = rows[street_index * 2][1]
        after = rows[street_index * 2 + 1][1]
        pair_label = f"{label}/{expected_street}"
        if before["state"].get("street") != expected_street or after["state"].get("street") != expected_street:
            errors.append(f"{pair_label}: frames are not in canonical street order")
        _audit_pair(before, after, pair_label=pair_label, errors=errors)
        if previous_after is not None:
            if _hero_cards(previous_after) != _hero_cards(before):
                errors.append(f"{pair_label}: hero cards change across streets")
            prior_board, current_board = _board(previous_after), _board(before)
            if current_board[: len(prior_board)] != prior_board:
                errors.append(f"{pair_label}: board is not a prefix-preserving progression")
            if previous_after["state"].get("action_history") != before["state"].get("action_history"):
                errors.append(f"{pair_label}: cumulative history is not carried into the next street")
            if _stacks(previous_after) != _stacks(before):
                errors.append(f"{pair_label}: stacks are not carried into the next street")
            if previous_after["state"].get("pot_bb") != before["state"].get("pot_bb"):
                errors.append(f"{pair_label}: pot is not carried into the next street")
            for key in ("table_id", "hand_id", "hero_seat", "dealer_seat"):
                if previous_after["state"].get(key) != before["state"].get(key):
                    errors.append(f"{pair_label}: {key} changes across streets")
        previous_after = after


def audit_sequence_manifest(manifest_path: str | Path) -> dict[str, object]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    frame_report = validate_manifest(manifest_file)
    errors = list(frame_report["errors"])
    if not frame_report["valid"]:
        return {
            "schema_version": "1.0.0",
            "valid": False,
            "errors": errors,
            "frame_validation": frame_report,
        }
    manifest, annotations = _manifest_annotations(manifest_file)
    sessions: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for row in annotations:
        sessions[str(row[1]["capture_session_id"])].append(row)

    hands = 0
    hand_starts = 0
    identity_rows: list[dict[str, object]] = []
    for session, raw_rows in sorted(sessions.items()):
        rows = sorted(raw_rows, key=lambda row: int(row[1]["sequence_index"]))
        table_ids = {str(row[1]["state"]["table_id"]) for row in rows}
        if len(table_ids) != 1:
            errors.append(f"{session}: table_id changes inside a capture session")
        expected_names = _names(rows[0][1])
        if any(not name for name in expected_names.values()):
            errors.append(f"{session}: all occupied seats require visible player names")
        for _, annotation in rows[1:]:
            if _names(annotation) != expected_names:
                errors.append(f"{session}: seat-to-player identity mapping changes inside the session")
                break
        identity_rows.append({"session": session, "seat_names": expected_names})

        grouped: list[list[tuple[Path, dict[str, Any]]]] = []
        seen_hand_ids: set[str] = set()
        for row in rows:
            hand_id = str(row[1]["state"]["hand_id"])
            if not grouped or str(grouped[-1][0][1]["state"]["hand_id"]) != hand_id:
                if hand_id in seen_hand_ids:
                    errors.append(f"{session}: hand_id {hand_id!r} recurs after a later hand began")
                seen_hand_ids.add(hand_id)
                grouped.append([])
            grouped[-1].append(row)
        for hand_rows in grouped:
            hands += 1
            hand_starts += int(hand_rows[0][1]["state"].get("hand_start") is True)
            _audit_hand(hand_rows, session=session, errors=errors)

    audit_material = {
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "capture_sessions": sorted(sessions),
        "hands": hands,
        "hand_start_frames": hand_starts,
        "identity_rows": identity_rows,
    }
    return {
        "schema_version": "1.0.0",
        "valid": not errors,
        "errors": errors,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "statistics": {
            "frames": len(annotations),
            "capture_sessions": len(sessions),
            "hands": hands,
            "hand_start_frames": hand_starts,
        },
        "sequence_audit_sha256": canonical_sha256(audit_material),
        "frame_validation": frame_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit APC complete-hand temporal continuity.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_sequence_manifest(args.manifest)
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["valid"] else 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
