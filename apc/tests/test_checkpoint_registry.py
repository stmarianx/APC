from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.checkpoint_registry import CheckpointRegistry, sign_promotion_evidence
from apc.self_learning.evaluate_candidate_smoke import evaluate_candidate_smoke


ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "coach" / "examples" / "sample_solver_bundle.json"
HAND = ROOT / "coach" / "examples" / "sample_play_money_hand.txt"


def candidate_variant(source: Path, destination: Path) -> str:
    checkpoint = json.loads(source.read_text(encoding="utf-8"))
    checkpoint["configuration"]["seed"] += 1
    checkpoint.pop("checkpoint_fingerprint")
    import hashlib

    material = json.dumps(checkpoint, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    checkpoint["checkpoint_fingerprint"] = hashlib.sha256(material.encode("utf-8")).hexdigest()
    destination.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    return checkpoint["checkpoint_fingerprint"]


class CheckpointRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_directory = tempfile.TemporaryDirectory()
        smoke = Path(cls.fixture_directory.name) / "smoke"
        evaluate_candidate_smoke(SOLVER, HAND, smoke, replicas=30)
        cls.base_checkpoint = smoke / "candidate.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture_directory.cleanup()

    def evidence(self, candidate: str, incumbent: str, *, passed: bool = True) -> dict[str, object]:
        return sign_promotion_evidence(
            {
                "schema_version": "1.0.0",
                "candidate_checkpoint_fingerprint": candidate,
                "incumbent_checkpoint_fingerprint": incumbent,
                "paired_evaluation": {
                    "passed": passed,
                    "qualifies_as_incumbent": True,
                    "independent_nodes": 40,
                    "coverage": "0.95",
                    "lower_improvement_bb": "0.01",
                },
                "safety_non_regression": {"passed": passed},
                "promotion_gate_passed": passed,
                "activation_authorized": passed,
            }
        )

    def test_registration_does_not_activate_and_promotion_rolls_back_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.json"
            candidate = candidate_variant(self.base_checkpoint, candidate_path)
            registry = CheckpointRegistry.create(
                root / "registry",
                self.base_checkpoint,
                bootstrap_reason="test incumbent bootstrap",
            )
            initial = registry.status()
            incumbent = initial["active_checkpoint_fingerprint"]
            registered = registry.register_candidate(candidate_path, expected_revision=0)
            self.assertEqual(registered["active_checkpoint_fingerprint"], incumbent)
            promoted = registry.promote(
                candidate,
                self.evidence(candidate, incumbent),
                expected_revision=1,
            )
            self.assertEqual(promoted["active_checkpoint_fingerprint"], candidate)
            rolled_back = registry.rollback(expected_revision=2, reason="paired regression alarm")
            self.assertEqual(rolled_back["active_checkpoint_fingerprint"], incumbent)
            self.assertTrue(rolled_back["active_artifact_verified"])
            self.assertEqual(rolled_back["revision"], 3)

    def test_failed_or_tampered_evidence_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.json"
            candidate = candidate_variant(self.base_checkpoint, candidate_path)
            registry = CheckpointRegistry.create(root / "registry", self.base_checkpoint, bootstrap_reason="test")
            incumbent = registry.status()["active_checkpoint_fingerprint"]
            registry.register_candidate(candidate_path, expected_revision=0)
            with self.assertRaisesRegex(ValueError, "promotion evidence rejected"):
                registry.promote(candidate, self.evidence(candidate, incumbent, passed=False), expected_revision=1)
            tampered = self.evidence(candidate, incumbent)
            tampered["activation_authorized"] = False
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                registry.promote(candidate, tampered, expected_revision=1)
            self.assertEqual(registry.status()["active_checkpoint_fingerprint"], incumbent)

    def test_stale_revision_and_registry_or_artifact_tampering_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.json"
            candidate_variant(self.base_checkpoint, candidate_path)
            registry = CheckpointRegistry.create(root / "registry", self.base_checkpoint, bootstrap_reason="test")
            registry.register_candidate(candidate_path, expected_revision=0)
            with self.assertRaisesRegex(ValueError, "stale registry revision"):
                registry.register_candidate(candidate_path, expected_revision=0)
            state = json.loads(registry.registry_path.read_text(encoding="utf-8"))
            active = state["active_checkpoint_fingerprint"]
            artifact = registry.root / state["registered"][active]["artifact"]
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "active checkpoint artifact fingerprint mismatch"):
                registry.status()
            second = CheckpointRegistry.create(root / "registry-two", self.base_checkpoint, bootstrap_reason="test")
            state = json.loads(second.registry_path.read_text(encoding="utf-8"))
            state["revision"] = 999
            second.registry_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registry fingerprint mismatch"):
                second.status()


if __name__ == "__main__":
    unittest.main()
