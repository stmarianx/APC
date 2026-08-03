import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from poker_coach import LiveTableService, SolverExportRegistry, VisualObservationAdapter


ROOT = Path(__file__).resolve().parents[1]


class VisualObservationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spot = SolverExportRegistry().parse_file(
            ROOT / "examples" / "sample_solver_export.csv"
        ).bundle.spots[-1]

    @classmethod
    def manifest(cls, frame_id="frame-1", confidence="0.99"):
        key = cls.spot.key
        values = {
            "table_id": "Visual Play Table",
            "hand_id": "visual-hand-1",
            "game": key.game,
            "players": key.players,
            "hero_position": key.hero_position,
            "effective_stack_bb": format(key.effective_stack_bb, "f"),
            "pot_bb": format(key.pot_bb, "f"),
            "to_call_bb": "0",
            "board": [str(card) for card in key.board],
            "hero_cards": [str(card) for card in key.hero_cards],
            "action_history": list(key.action_history),
            "legal_actions": [action.action for action in cls.spot.actions],
            "rake_model": key.rake_model,
            "utility_model": key.utility_model,
        }
        region_for = {
            "effective_stack_bb": "stack",
            "pot_bb": "pot",
            "to_call_bb": "actions",
            "board": "board",
            "hero_cards": "hero_cards",
            "legal_actions": "actions",
        }
        return {
            "schema_version": "1.0.0",
            "provider": "fixture_visual_provider",
            "provider_version": "1.0",
            "frame": {
                "frame_id": frame_id,
                "image_sha256": hashlib.sha256(frame_id.encode()).hexdigest(),
            },
            "calibration": {
                "profile_id": "pokerstars-1280x720-v1",
                "regions": {
                    "table": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "hero_cards": {"x": 0.4, "y": 0.72, "width": 0.2, "height": 0.12},
                    "board": {"x": 0.3, "y": 0.38, "width": 0.4, "height": 0.14},
                    "pot": {"x": 0.42, "y": 0.28, "width": 0.16, "height": 0.08},
                    "stack": {"x": 0.4, "y": 0.86, "width": 0.2, "height": 0.08},
                    "actions": {"x": 0.25, "y": 0.9, "width": 0.5, "height": 0.1},
                },
            },
            "fields": {
                name: {
                    "value": value,
                    "confidence": confidence,
                    "region": region_for.get(name, "table"),
                }
                for name, value in values.items()
            },
        }

    def test_two_distinct_high_confidence_frames_stabilize_once(self) -> None:
        adapter = VisualObservationAdapter()
        first = adapter.submit(self.manifest("frame-1"))
        duplicate = adapter.submit(self.manifest("frame-1"))
        ready = adapter.submit(self.manifest("frame-2"))
        unchanged = adapter.submit(self.manifest("frame-3"))
        self.assertEqual(first["status"], "pending_stability")
        self.assertEqual(duplicate["observed_stable_frames"], 1)
        self.assertEqual((ready["status"], ready["changed"], ready["revision"]), ("state_ready", True, 0))
        self.assertEqual((unchanged["changed"], unchanged["revision"]), (False, 0))
        self.assertEqual(ready["payload"]["source"], "visual_provider:fixture_visual_provider:1.0:pokerstars-1280x720-v1")

    def test_low_confidence_frame_is_gated_without_advancing_stability(self) -> None:
        adapter = VisualObservationAdapter()
        low = self.manifest("frame-low")
        low["fields"]["hero_cards"]["confidence"] = "0.62"
        rejected = adapter.submit(low)
        first_good = adapter.submit(self.manifest("frame-good-1"))
        self.assertEqual(rejected["status"], "low_confidence")
        self.assertEqual(rejected["low_confidence_fields"], ["hero_cards"])
        self.assertEqual(first_good["observed_stable_frames"], 1)

    def test_stable_visual_payload_matches_live_solver_state(self) -> None:
        adapter = VisualObservationAdapter()
        adapter.submit(self.manifest("frame-1"))
        visual = adapter.submit(self.manifest("frame-2"))
        live = LiveTableService()
        session = live.create_session("Visual Play Table")
        result = live.update_state(
            session["session_id"], visual["payload"], (self.spot,)
        )
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["match"]["confidence"], "exact")
        self.assertEqual(result["match"]["node_id"], self.spot.node_id)

    def test_changed_semantics_require_restabilization_and_advance_revision(self) -> None:
        adapter = VisualObservationAdapter()
        adapter.submit(self.manifest("base-1"))
        adapter.submit(self.manifest("base-2"))
        changed_one = self.manifest("changed-1")
        changed_two = self.manifest("changed-2")
        changed_one["fields"]["hand_id"]["value"] = "visual-hand-2"
        changed_two["fields"]["hand_id"]["value"] = "visual-hand-2"
        pending = adapter.submit(changed_one)
        ready = adapter.submit(changed_two)
        self.assertEqual(pending["status"], "pending_stability")
        self.assertEqual((ready["status"], ready["revision"]), ("state_ready", 1))

    def test_image_path_is_hashed_and_mismatch_fails(self) -> None:
        with TemporaryDirectory() as directory:
            image = Path(directory) / "frame.bin"
            image.write_bytes(b"auditable visual frame")
            manifest = self.manifest("file-frame")
            manifest["frame"] = {"frame_id": "file-frame", "image_path": str(image)}
            result = VisualObservationAdapter(stable_frames=1).submit(manifest)
            self.assertEqual(result["evidence"]["image_bytes"], image.stat().st_size)
            self.assertEqual(result["evidence"]["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
            manifest["frame"]["image_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                VisualObservationAdapter(stable_frames=1).submit(manifest)

    def test_invalid_region_and_missing_field_fail_explicitly(self) -> None:
        invalid = self.manifest()
        invalid["calibration"]["regions"]["pot"]["width"] = 0.9
        with self.assertRaisesRegex(ValueError, "normalized image bounds"):
            VisualObservationAdapter().submit(invalid)
        missing = self.manifest()
        del missing["fields"]["board"]
        with self.assertRaisesRegex(ValueError, "missing fields: board"):
            VisualObservationAdapter().submit(missing)


if __name__ == "__main__":
    unittest.main()
