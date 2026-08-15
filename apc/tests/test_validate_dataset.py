from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from apc.tools.validate_dataset import (
    canonical_sha256,
    validate_annotation,
    validate_manifest,
    validation_exit_code,
)


BOX = {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}


def annotation(session: str, sample: str, image_name: str, digest: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "sample_id": sample,
        "capture_session_id": session,
        "sequence_index": 0,
        "image": {
            "path": image_name,
            "sha256": digest,
            "perceptual_hash": digest[:16],
            "width": 1280,
            "height": 720,
            "timestamp_ms": 0,
        },
        "environment": {
            "source_kind": "controlled_training_table",
            "provider_id": "fixture",
            "layout_id": "heads-up",
            "theme_id": "dark",
            "locale": "en-US",
            "max_seats": 2,
            "virtual_chips": True,
        },
        "state": {
            "game": "holdem_no_limit",
            "table_id": "fixture-table",
            "hand_id": "fixture-hand",
            "street": "flop",
            "hero_seat": 1,
            "dealer_seat": 1,
            "pot_bb": "6",
            "to_call_bb": "0",
            "legal_actions": ["check", "bet"],
            "action_history": [],
        },
        "objects": {
            "table": {"x": 0, "y": 0, "width": 1, "height": 1},
            "seats": [
                {"seat_no": 1, "box": BOX, "occupied": True, "is_hero": True, "has_dealer_button": True, "player_name": "Hero", "stack_bb": "97", "raw_stack_text": "97 BB", "status": "active", "visibility": "clear"},
                {"seat_no": 2, "box": {"x": 0.8, "y": 0.1, "width": 0.1, "height": 0.1}, "occupied": True, "is_hero": False, "has_dealer_button": False, "player_name": "Villain", "stack_bb": "97", "raw_stack_text": "97 BB", "status": "active", "visibility": "clear"},
            ],
            "hero_cards": [
                {"box": BOX, "rank": "A", "suit": "s", "visibility": "clear"},
                {"box": BOX, "rank": "K", "suit": "d", "visibility": "clear"},
            ],
            "board_cards": [
                {"box": BOX, "rank": "Q", "suit": "c", "visibility": "clear"},
                {"box": BOX, "rank": "7", "suit": "h", "visibility": "clear"},
                {"box": BOX, "rank": "2", "suit": "s", "visibility": "clear"},
            ],
            "pot": {"box": BOX, "amount_bb": "6", "raw_text": "6 BB", "visibility": "clear"},
            "action_buttons": [
                {"box": BOX, "action": "check", "enabled": True, "raw_text": "Check", "visibility": "clear"},
                {"box": BOX, "action": "bet", "enabled": True, "raw_text": "Bet", "visibility": "clear"},
            ],
            "observed_action": None,
        },
        "provenance": {
            "annotator": "fixture",
            "annotation_version": 1,
            "verified": True,
            "reviewer": "reviewer",
            "created_at": "2026-08-03T00:00:00Z",
            "notes": "",
        },
    }


class DatasetValidatorTests(unittest.TestCase):
    def test_require_ready_exit_code_distinguishes_valid_from_trainable(self) -> None:
        not_ready = {"valid": True, "minimum_dataset": {"ready": False}}
        ready = {"valid": True, "minimum_dataset": {"ready": True}}
        invalid = {"valid": False, "minimum_dataset": {"ready": False}}
        self.assertEqual(validation_exit_code(not_ready), 0)
        self.assertEqual(validation_exit_code(not_ready, require_ready=True), 3)
        self.assertEqual(validation_exit_code(ready, require_ready=True), 0)
        self.assertEqual(validation_exit_code(invalid, require_ready=True), 1)

    def test_valid_annotation_and_manifest_produce_machine_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.bin"
            image.write_bytes(b"controlled virtual table frame")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            item = annotation("session-train", "sample-1", image.name, digest)
            annotation_path = root / "annotation.json"
            annotation_path.write_text(json.dumps(item), encoding="utf-8")
            splits = {"group_key": "capture_session_id", "group_exclusive": True, "train": ["session-train"], "validation": [], "test": []}
            duplicate_material = [(digest, digest[:16], "session-train")]
            manifest = {
                "schema_version": "1.0.0",
                "dataset_id": "fixture",
                "dataset_version": "0.1.0",
                "created_at": "2026-08-03T00:00:00Z",
                "annotation_schema": "schemas/frame_annotation.schema.json",
                "source_policy": {"virtual_chips_only": True, "allowed_source_kinds": ["controlled_training_table"], "player_identity_policy": "pseudonymized"},
                "annotation_files": [annotation_path.name],
                "splits": splits,
                "statistics": {"captured_frames": 1, "labeled_frames": 1, "verified_frames": 1, "double_audited_frames": 1, "capture_sessions": 1, "layouts": 1, "themes": 1, "temporal_sequence_frames": 0},
                "fingerprints": {
                    "annotations_sha256": canonical_sha256([item]),
                    "split_sha256": canonical_sha256(splits),
                    "duplicate_audit_sha256": canonical_sha256(duplicate_material),
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = validate_manifest(manifest_path)

            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["annotations"], 1)
            self.assertFalse(report["minimum_dataset"]["ready"])
            self.assertEqual(report["computed_statistics"]["verified_frames"], 1)
            self.assertEqual(report["computed_statistics"]["controlled_visible_frames"], 1)
            self.assertEqual(report["computed_statistics"]["controlled_visible_sessions"], 1)
            self.assertEqual(report["computed_statistics"]["synthetic_frames"], 0)

    def test_annotation_rejects_nonvirtual_and_inconsistent_state(self) -> None:
        item = annotation("session", "sample", "missing.bin", "0" * 64)
        item["environment"]["virtual_chips"] = False
        item["state"]["dealer_seat"] = 2
        item["state"]["pot_bb"] = "7"
        item["state"]["legal_actions"] = ["fold"]
        item["objects"]["board_cards"][2] = item["objects"]["hero_cards"][0]

        issues = validate_annotation(item, require_image=False)

        self.assertTrue(any("virtual_chips" in issue for issue in issues))
        self.assertTrue(any("dealer_seat" in issue for issue in issues))
        self.assertTrue(any("amount_bb" in issue for issue in issues))
        self.assertTrue(any("legal_actions" in issue for issue in issues))
        self.assertTrue(any("visible cards must be unique" in issue for issue in issues))

    def test_synthetic_v2_requires_visible_call_price(self) -> None:
        item = annotation("session", "sample", "missing.bin", "0" * 64)
        item["environment"]["provider_id"] = "apc-synthetic-renderer-v2"
        item["state"]["to_call_bb"] = "2.5"
        item["state"]["legal_actions"] = ["fold", "call"]
        item["objects"]["action_buttons"] = [
            {"box": BOX, "action": "fold", "enabled": True, "raw_text": "Fold", "visibility": "clear"},
            {"box": BOX, "action": "call", "enabled": True, "raw_text": "Call 2.5 BB", "visibility": "clear"},
        ]
        issues = validate_annotation(item, require_image=False)
        self.assertTrue(any("call price" in issue for issue in issues))
        item["objects"]["action_buttons"][1]["amount_bb"] = "2.5"
        issues = validate_annotation(item, require_image=False)
        self.assertFalse(any("call price" in issue for issue in issues))

    def test_turn_clock_must_match_canonical_remaining_time(self) -> None:
        item = annotation("session", "sample", "missing.bin", "0" * 64)
        item["state"].update(
            {
                "hero_to_act": True,
                "decision_time_remaining_ms": 12_500,
                "decision_deadline_source": "visible_timer",
            }
        )
        item["objects"]["turn_clock"] = {
            "box": BOX,
            "remaining_ms": 12_500,
            "raw_text": "12.5",
            "visibility": "clear",
        }
        self.assertEqual(validate_annotation(item, require_image=False), [])
        item["objects"]["turn_clock"]["remaining_ms"] = 11_000
        issues = validate_annotation(item, require_image=False)
        self.assertTrue(any("turn_clock.remaining_ms" in issue for issue in issues))

    def test_manifest_rejects_cross_split_session_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.bin"
            image.write_bytes(b"frame")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            item = annotation("shared", "sample", image.name, digest)
            annotation_path = root / "annotation.json"
            annotation_path.write_text(json.dumps(item), encoding="utf-8")
            splits = {"group_key": "capture_session_id", "group_exclusive": True, "train": ["shared"], "validation": ["shared"], "test": []}
            manifest = {
                "schema_version": "1.0.0",
                "dataset_id": "leak",
                "dataset_version": "0.1.0",
                "created_at": "2026-08-03T00:00:00Z",
                "annotation_schema": "schemas/frame_annotation.schema.json",
                "source_policy": {"virtual_chips_only": True, "allowed_source_kinds": ["controlled_training_table"], "player_identity_policy": "pseudonymized"},
                "annotation_files": [annotation_path.name],
                "splits": splits,
                "statistics": {"captured_frames": 1, "labeled_frames": 1, "verified_frames": 1, "double_audited_frames": 1, "capture_sessions": 1, "layouts": 1, "themes": 1, "temporal_sequence_frames": 0},
                "fingerprints": {
                    "annotations_sha256": canonical_sha256([item]),
                    "split_sha256": canonical_sha256(splits),
                    "duplicate_audit_sha256": canonical_sha256([(digest, digest[:16], "shared")]),
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = validate_manifest(manifest_path)

            self.assertFalse(report["valid"])
            self.assertTrue(any("group-exclusive" in issue for issue in report["errors"]))


if __name__ == "__main__":
    unittest.main()
