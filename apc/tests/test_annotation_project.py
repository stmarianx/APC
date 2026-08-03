from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from apc.annotator import AnnotationProject


BOX = {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}


def png_bytes(width: int = 640, height: int = 480, marker: bytes = b"") -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00" + marker


def complete_annotation(template: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(template)
    payload["state"] = {
        "game": "holdem_no_limit",
        "table_id": "table",
        "hand_id": "hand",
        "street": "flop",
        "hero_seat": 1,
        "dealer_seat": 1,
        "pot_bb": "6",
        "to_call_bb": "0",
        "legal_actions": ["check", "bet"],
        "action_history": [],
    }
    payload["objects"] = {
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
    }
    payload["provenance"] = {
        "annotator": "fixture",
        "annotation_version": 1,
        "verified": True,
        "reviewer": "reviewer",
        "created_at": "2026-08-03T00:00:00Z",
        "notes": "",
    }
    return payload


class AnnotationProjectTests(unittest.TestCase):
    def create_project(self, root: Path) -> AnnotationProject:
        return AnnotationProject.create(
            root,
            project_id="fixture-project",
            source_kind="controlled_training_table",
            provider_id="fixture",
            layout_id="heads-up",
            theme_id="dark",
            locale="en-US",
            max_seats=2,
        )

    def test_import_hashes_dimensions_and_deduplicates_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.create_project(root / "project")
            frame = root / "frame.png"
            frame.write_bytes(png_bytes(marker=b"one"))

            first, inserted = project.import_frame(frame, capture_session_id="session-1", timestamp_ms=100)
            duplicate, duplicate_inserted = project.import_frame(frame, capture_session_id="session-1", timestamp_ms=101)

            self.assertTrue(inserted)
            self.assertFalse(duplicate_inserted)
            self.assertEqual(first, duplicate)
            self.assertEqual((first.width, first.height), (640, 480))
            self.assertEqual(first.sha256, hashlib.sha256(frame.read_bytes()).hexdigest())
            self.assertEqual(project.status()["frames"], 1)

    def test_save_annotation_validates_against_immutable_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.create_project(root / "project")
            frame = root / "frame.png"
            frame.write_bytes(png_bytes())
            record, _ = project.import_frame(frame, capture_session_id="session", timestamp_ms=0)
            annotation = complete_annotation(project.annotation_template(record.sample_id))

            path = project.save_annotation(record.sample_id, annotation)

            self.assertTrue(path.is_file())
            self.assertEqual(project.status()["verified_annotations"], 1)
            invalid = copy.deepcopy(annotation)
            invalid["image"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "does not match image bytes"):
                project.save_annotation(record.sample_id, invalid)

    def test_export_builds_group_exclusive_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.create_project(root / "project")
            for index, session in enumerate(("session-a", "session-b", "session-c")):
                frame = root / f"frame-{index}.png"
                frame.write_bytes(png_bytes(marker=str(index).encode()))
                record, _ = project.import_frame(frame, capture_session_id=session, timestamp_ms=index)
                project.save_annotation(
                    record.sample_id,
                    complete_annotation(project.annotation_template(record.sample_id)),
                )

            manifest_path, report = project.export_manifest(dataset_version="0.1.0")

            self.assertTrue(report["valid"], report["errors"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            splits = [set(manifest["splits"][name]) for name in ("train", "validation", "test")]
            self.assertTrue(all(splits))
            self.assertFalse(splits[0] & splits[1])
            self.assertFalse(splits[0] & splits[2])
            self.assertFalse(splits[1] & splits[2])
            self.assertEqual(set().union(*splits), {"session-a", "session-b", "session-c"})

    def test_folder_import_uses_natural_order_sampling_and_monotonic_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.create_project(root / "project")
            capture = root / "capture"
            capture.mkdir()
            markers = {"frame-10.png": b"ten", "frame-2.png": b"two", "frame-1.png": b"one"}
            for name, marker in markers.items():
                (capture / name).write_bytes(png_bytes(marker=marker))

            report = project.import_folder(
                capture,
                capture_session_id="session-folder",
                timestamp_ms=100,
                interval_ms=50,
            )

            self.assertEqual(report["inserted_frames"], 3)
            self.assertEqual(
                [Path(row["source"]).name for row in report["frames"]],
                ["frame-1.png", "frame-2.png", "frame-10.png"],
            )
            self.assertEqual(
                [row["timestamp_ms"] for row in report["frames"]],
                [100, 150, 200],
            )

    def test_new_frame_timestamp_must_advance_inside_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.create_project(root / "project")
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(png_bytes(marker=b"first"))
            second.write_bytes(png_bytes(marker=b"second"))
            project.import_frame(first, capture_session_id="session", timestamp_ms=100)
            with self.assertRaisesRegex(ValueError, "timestamp must advance"):
                project.import_frame(second, capture_session_id="session", timestamp_ms=100)


if __name__ == "__main__":
    unittest.main()
