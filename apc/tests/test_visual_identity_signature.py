from __future__ import annotations

import json
import unittest
from pathlib import Path

from apc.evaluate_visual_identity import evaluate_visual_identity
from apc.visual_identity_signature import extract_frame_signatures


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "apc" / "data" / "processed" / "synthetic-handseq-dev-v1"


def annotation(relative: str) -> tuple[Path, dict[str, object]]:
    path = DATASET / "annotations" / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (path.parent / payload["image"]["path"]).resolve(), payload


class VisualIdentitySignatureTests(unittest.TestCase):
    def test_repeated_name_pixels_have_stable_signature_and_distinct_players_do_not_collide(self) -> None:
        first_path, first = annotation("handseq-0000-000000-a035a81f6bc5.json")
        second_path, second = annotation("handseq-0000-000001-08e935fd1138.json")
        first_rows = extract_frame_signatures(first_path, first["objects"]["seats"])
        second_rows = extract_frame_signatures(second_path, second["objects"]["seats"])
        self.assertEqual(
            [row.signature_sha256 for row in first_rows],
            [row.signature_sha256 for row in second_rows],
        )
        self.assertNotEqual(first_rows[0].signature_sha256, first_rows[1].signature_sha256)
        self.assertTrue(all(row.quality_score >= 0.85 for row in first_rows))

    def test_held_out_validation_uses_pixel_tokens_and_resolves_profiles(self) -> None:
        report, registry = evaluate_visual_identity(DATASET / "dataset_manifest.json")
        self.assertEqual(report["evaluation_kind"], "pixel_visual_identity_signature_smoke")
        self.assertEqual(report["metrics"]["stable_signature_rate"], 1.0)
        self.assertEqual(report["metrics"]["token_collision_count"], 0)
        self.assertEqual(report["metrics"]["final_seat_resolution_rate"], 1.0)
        self.assertTrue(registry.snapshot()["identities"])


if __name__ == "__main__":
    unittest.main()
