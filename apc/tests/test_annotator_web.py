from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from apc.annotator import AnnotationProject
from apc.annotator.web import AnnotatorApplication, create_server
from apc.tests.test_annotation_project import complete_annotation, png_bytes


class AnnotatorWebTests(unittest.TestCase):
    def test_workbench_exposes_graphical_normalized_box_editor(self) -> None:
        assets = Path(__file__).parents[1] / "annotator" / "web_assets"
        index = (assets / "index.html").read_text(encoding="utf-8")
        script = (assets / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="boxCanvas"', index)
        self.assertIn('id="applyBoxButton"', index)
        self.assertIn('id="applySuggestionButton"', index)
        self.assertIn("normalizedPointer", script)
        self.assertIn("syncCanonicalState", script)
        self.assertIn("applySuggestionToDraft", script)

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.project = AnnotationProject.create(
            root / "project",
            project_id="web-fixture",
            source_kind="controlled_training_table",
            provider_id="fixture",
            layout_id="heads-up",
            theme_id="dark",
            locale="en-US",
            max_seats=2,
        )
        self.frame = root / "frame.png"
        self.frame.write_bytes(png_bytes())
        self.record, _ = self.project.import_frame(
            self.frame, capture_session_id="session", timestamp_ms=0
        )
        self.server = create_server(AnnotatorApplication(self.project))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.directory.cleanup()

    def request(self, path: str, *, payload: object | None = None) -> tuple[int, bytes]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method="GET" if payload is None else "POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            try:
                return error.code, error.read()
            finally:
                error.close()

    def test_static_project_annotation_and_frame_endpoints(self) -> None:
        status, html = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"APC Annotation Workbench", html)
        status, raw_project = self.request("/api/project")
        project = json.loads(raw_project)
        self.assertEqual((status, project["status"]["frames"]), (200, 1))
        status, frame = self.request(f"/api/frames/{self.record.sample_id}")
        self.assertEqual((status, frame), (200, self.frame.read_bytes()))
        status, raw_annotation = self.request(f"/api/annotations/{self.record.sample_id}")
        annotation = json.loads(raw_annotation)
        self.assertEqual(status, 200)
        self.assertFalse(annotation["saved"])
        self.assertIsNone(annotation["suggestion"])

    def test_fingerprinted_suggestion_is_exposed_without_modifying_annotation(self) -> None:
        payload = {
            "schema_version": "1.0.0",
            "kind": "apc_perception_suggestion",
            "sample_id": self.record.sample_id,
            "capture_session_id": self.record.capture_session_id,
            "image": {"sha256": self.record.sha256},
            "review_required": True,
            "auto_applied": False,
            "model_status": "abstain_incomplete_state",
            "minimum_supported_confidence": 0.4,
            "suggested_visible_state": {"street": "flop", "pot_bb": "4.5"},
        }
        self.project.save_suggestion(self.record.sample_id, payload)
        status, raw = self.request(f"/api/annotations/{self.record.sample_id}")
        result = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertFalse(result["saved"])
        self.assertTrue(result["suggestion"]["review_required"])
        self.assertFalse(result["suggestion"]["auto_applied"])
        self.assertRegex(result["suggestion"]["suggestion_sha256"], r"^[0-9a-f]{64}$")
        status, raw_project = self.request("/api/project")
        self.assertTrue(json.loads(raw_project)["records"][0]["suggested"])
        self.assertIsNone(self.project.load_annotation(self.record.sample_id))

    def test_invalid_then_valid_annotation_round_trip(self) -> None:
        status, raw = self.request(f"/api/annotations/{self.record.sample_id}")
        template = json.loads(raw)["annotation"]
        status, raw_error = self.request(
            f"/api/annotations/{self.record.sample_id}", payload=template
        )
        self.assertEqual(status, 422)
        self.assertIn("missing fields", json.loads(raw_error)["error"])

        completed = complete_annotation(template)
        status, raw_saved = self.request(
            f"/api/annotations/{self.record.sample_id}", payload=completed
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw_saved)["project"]["status"]["verified_annotations"], 1)


if __name__ == "__main__":
    unittest.main()
