import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from apc.capture.browser_frame import capture_browser_frame, save_browser_frame


class BrowserFrameTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.snapshot = {"viewport": [800, 900], "scroll": [0, 0],
                         "cards": [["hero", "As", 200, 300, 62, 78]]}
        stream = io.BytesIO()
        Image.new("RGB", (800, 875), "green").save(stream, format="PNG")
        self.png = stream.getvalue()

    async def observe(self):
        self.events.append("observe")
        return self.snapshot

    async def screenshot(self):
        self.events.append("screenshot")
        return self.png

    async def test_callbacks_bracket_screenshot_and_bind_original_bytes(self):
        result = await capture_browser_frame(self.observe, self.screenshot)
        self.assertEqual(self.events, ["observe", "screenshot", "observe"])
        self.assertEqual(result["png"], self.png)
        self.assertEqual(result["sha256"], hashlib.sha256(self.png).hexdigest())
        self.assertEqual(result["image_size"], [800, 875])
        self.assertEqual(result["cards"], [["hero", "As", 200, 300]])
        self.assertTrue(result["eligible_for_draft_scoring"])
        self.assertFalse(result["pixel_geometry_reviewed"])
        self.assertFalse(result["readiness_credit"])

    async def test_scroll_race_retains_frame_without_retry(self):
        async def moved_screenshot():
            self.snapshot["scroll"][1] = 80
            return await self.screenshot()
        result = await capture_browser_frame(self.observe, moved_screenshot)
        self.assertFalse(result["eligible_for_draft_scoring"])
        self.assertIn("moved", result["rejection_reason"])
        self.assertEqual(result["png"], self.png)
        self.assertEqual(result["capture_stability"]["before"]["scroll"], [0, 0])
        self.assertEqual(len(self.events), 3)

    async def test_malformed_snapshot_rejected_not_silently_repaired(self):
        self.snapshot = {"cards": None}
        result = await capture_browser_frame(self.observe, self.screenshot)
        self.assertFalse(result["eligible_for_draft_scoring"])

    async def test_acquisition_failure_propagates(self):
        async def failed():
            raise RuntimeError("capture unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await capture_browser_frame(self.observe, failed)
        self.assertEqual(self.events, ["observe"])

    async def test_non_png_rejected(self):
        stream = io.BytesIO()
        Image.new("RGB", (2, 2)).save(stream, format="BMP")
        self.png = stream.getvalue()
        with self.assertRaisesRegex(ValueError, "PNG"):
            await capture_browser_frame(self.observe, self.screenshot)

    async def test_save_retains_rejected_evidence_and_refuses_overwrite(self):
        async def moved():
            self.snapshot["scroll"][1] = 12
            return self.png
        result = await capture_browser_frame(self.observe, moved)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "sample"
            manifest = save_browser_frame(result, folder)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertFalse(saved["eligible_for_draft_scoring"])
            self.assertEqual((folder / saved["image"]).read_bytes(), self.png)
            with self.assertRaises(FileExistsError):
                save_browser_frame(result, folder)
            self.assertEqual(json.loads(manifest.read_text()), saved)

    async def test_changed_bytes_rejected_before_creating_folder(self):
        result = await capture_browser_frame(self.observe, self.screenshot)
        result["png"] += b"changed"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "sample"
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                save_browser_frame(result, folder)
            self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
