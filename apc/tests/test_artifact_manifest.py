from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.tools.artifact_manifest import build_manifest, verify_manifest


class ArtifactManifestTests(unittest.TestCase):
    def test_build_and_verify_portable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "model.bin"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"apc-checkpoint")
            report = root / "runs" / "metrics.json"
            report.parent.mkdir()
            report.write_text('{"accuracy": 1.0}', encoding="utf-8")

            manifest = build_manifest(
                root,
                [checkpoint, report],
                producer="test-run-v1",
                artifact_class="checkpoint_bundle",
                source_fingerprints={"dataset": "a" * 64},
            )
            result = verify_manifest(root, manifest)

            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["checked_artifacts"], 2)
            self.assertEqual(
                [row["path"] for row in manifest["artifacts"]],
                ["checkpoints/model.bin", "runs/metrics.json"],
            )

    def test_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "checkpoint.json"
            artifact.write_text("before", encoding="utf-8")
            manifest = build_manifest(
                root,
                [artifact],
                producer="test",
                artifact_class="checkpoint",
            )
            artifact.write_text("after", encoding="utf-8")

            result = verify_manifest(root, manifest)

            self.assertFalse(result["valid"])
            self.assertTrue(any("mismatch" in row for row in result["errors"]))

    def test_manifest_content_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "checkpoint.json"
            artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
            manifest = build_manifest(
                root,
                [artifact],
                producer="test",
                artifact_class="checkpoint",
            )
            manifest["producer"] = "changed"

            result = verify_manifest(root, manifest)

            self.assertFalse(result["valid"])
            self.assertIn("manifest_sha256 does not match manifest contents", result["errors"])

    def test_artifacts_outside_root_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "root"
            root.mkdir()
            outside = parent / "outside.bin"
            outside.write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "outside the declared root"):
                build_manifest(
                    root,
                    [outside],
                    producer="test",
                    artifact_class="checkpoint",
                )


if __name__ == "__main__":
    unittest.main()
