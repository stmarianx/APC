from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from apc.self_learning.train_candidate import validate_candidate_checkpoint


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign_promotion_evidence(payload: dict[str, object]) -> dict[str, object]:
    if "evidence_fingerprint" in payload:
        raise ValueError("unsigned promotion evidence must not already contain a fingerprint")
    result = json.loads(json.dumps(payload))
    result["evidence_fingerprint"] = _fingerprint(result)
    return result


def validate_promotion_evidence(
    evidence: dict[str, object],
    *,
    candidate_fingerprint: str,
    incumbent_fingerprint: str,
) -> list[str]:
    issues: list[str] = []
    material = dict(evidence)
    observed = material.pop("evidence_fingerprint", None)
    if observed != _fingerprint(material):
        issues.append("promotion evidence fingerprint mismatch")
    if evidence.get("schema_version") != "1.0.0":
        issues.append("promotion evidence schema is invalid")
    if evidence.get("candidate_checkpoint_fingerprint") != candidate_fingerprint:
        issues.append("promotion evidence references a different candidate")
    if evidence.get("incumbent_checkpoint_fingerprint") != incumbent_fingerprint:
        issues.append("promotion evidence references a different incumbent")
    paired = evidence.get("paired_evaluation")
    if not isinstance(paired, dict) or paired.get("passed") is not True:
        issues.append("paired incumbent evaluation did not pass")
    elif paired.get("qualifies_as_incumbent") is not True:
        issues.append("paired comparator is not the declared incumbent")
    safety = evidence.get("safety_non_regression")
    if not isinstance(safety, dict) or safety.get("passed") is not True:
        issues.append("safety non-regression did not pass")
    if evidence.get("promotion_gate_passed") is not True:
        issues.append("promotion gate did not pass")
    if evidence.get("activation_authorized") is not True:
        issues.append("promotion evidence does not authorize registry activation")
    return issues


class CheckpointRegistry:
    """Content-addressed local checkpoint lifecycle with optimistic revisions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.registry_path = self.root / "registry.json"
        self.artifacts = self.root / "artifacts"

    @classmethod
    def create(
        cls,
        root: str | Path,
        incumbent_checkpoint: str | Path,
        *,
        bootstrap_reason: str,
    ) -> "CheckpointRegistry":
        registry = cls(root)
        if registry.root.exists():
            raise ValueError(f"checkpoint registry already exists: {registry.root}")
        if not bootstrap_reason:
            raise ValueError("checkpoint registry bootstrap requires a reason")
        registry.artifacts.mkdir(parents=True)
        incumbent = registry._store_checkpoint(Path(incumbent_checkpoint))
        state = {
            "schema_version": "1.0.0",
            "revision": 0,
            "active_checkpoint_fingerprint": incumbent["checkpoint_fingerprint"],
            "registered": {incumbent["checkpoint_fingerprint"]: incumbent},
            "history": [
                {
                    "kind": "bootstrap",
                    "checkpoint_fingerprint": incumbent["checkpoint_fingerprint"],
                    "reason": bootstrap_reason,
                }
            ],
        }
        registry._write(state)
        return registry

    def _load(self) -> dict[str, object]:
        try:
            state = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"checkpoint registry is unreadable: {error}") from error
        material = dict(state)
        observed = material.pop("registry_fingerprint", None)
        if observed != _fingerprint(material):
            raise ValueError("checkpoint registry fingerprint mismatch")
        return state

    def _write(self, state: dict[str, object]) -> None:
        material = dict(state)
        material.pop("registry_fingerprint", None)
        material["registry_fingerprint"] = _fingerprint(material)
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".registry-", dir=self.root) as temporary:
            path = Path(temporary) / "registry.json"
            path.write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")
            path.replace(self.registry_path)

    def _store_checkpoint(self, checkpoint_path: Path) -> dict[str, object]:
        resolved = checkpoint_path.resolve()
        validation = validate_candidate_checkpoint(resolved)
        if not validation["valid"]:
            raise ValueError("checkpoint is invalid: " + "; ".join(validation["issues"]))
        fingerprint = str(validation["checkpoint_fingerprint"])
        artifact_path = self.artifacts / f"{fingerprint}.json"
        content = resolved.read_bytes()
        if artifact_path.exists():
            if artifact_path.read_bytes() != content:
                raise ValueError("checkpoint fingerprint collision with different artifact bytes")
        else:
            artifact_path.write_bytes(content)
        return {
            "checkpoint_fingerprint": fingerprint,
            "artifact": f"artifacts/{fingerprint}.json",
            "artifact_sha256": hashlib.sha256(content).hexdigest(),
            "status": "registered_immutable",
        }

    @staticmethod
    def _expect_revision(state: dict[str, object], expected_revision: int) -> None:
        if state.get("revision") != expected_revision:
            raise ValueError(
                f"stale registry revision: expected {expected_revision}, current {state.get('revision')}"
            )

    def register_candidate(
        self,
        checkpoint: str | Path,
        *,
        expected_revision: int,
    ) -> dict[str, object]:
        state = self._load()
        self._expect_revision(state, expected_revision)
        record = self._store_checkpoint(Path(checkpoint))
        fingerprint = record["checkpoint_fingerprint"]
        registered = dict(state["registered"])
        if fingerprint not in registered:
            registered[fingerprint] = record
            state["history"] = [
                *state["history"],
                {"kind": "candidate_registered", "checkpoint_fingerprint": fingerprint},
            ]
        state["registered"] = registered
        state["revision"] = int(state["revision"]) + 1
        self._write(state)
        return self.status()

    def promote(
        self,
        candidate_fingerprint: str,
        evidence: dict[str, object],
        *,
        expected_revision: int,
    ) -> dict[str, object]:
        state = self._load()
        self._expect_revision(state, expected_revision)
        active = str(state["active_checkpoint_fingerprint"])
        if candidate_fingerprint == active:
            raise ValueError("candidate is already the active checkpoint")
        if candidate_fingerprint not in state["registered"]:
            raise ValueError("candidate checkpoint is not registered")
        issues = validate_promotion_evidence(
            evidence,
            candidate_fingerprint=candidate_fingerprint,
            incumbent_fingerprint=active,
        )
        if issues:
            raise ValueError("promotion evidence rejected: " + "; ".join(issues))
        state["active_checkpoint_fingerprint"] = candidate_fingerprint
        state["history"] = [
            *state["history"],
            {
                "kind": "promotion",
                "from_checkpoint_fingerprint": active,
                "to_checkpoint_fingerprint": candidate_fingerprint,
                "evidence_fingerprint": evidence["evidence_fingerprint"],
            },
        ]
        state["revision"] = int(state["revision"]) + 1
        self._write(state)
        return self.status()

    def rollback(self, *, expected_revision: int, reason: str) -> dict[str, object]:
        if not reason:
            raise ValueError("rollback requires a reason")
        state = self._load()
        self._expect_revision(state, expected_revision)
        active = str(state["active_checkpoint_fingerprint"])
        promotion = next(
            (
                row
                for row in reversed(state["history"])
                if row.get("kind") == "promotion"
                and row.get("to_checkpoint_fingerprint") == active
            ),
            None,
        )
        if promotion is None:
            raise ValueError("active checkpoint has no promotive predecessor to roll back to")
        target = str(promotion["from_checkpoint_fingerprint"])
        if target not in state["registered"]:
            raise ValueError("rollback target artifact is not registered")
        state["active_checkpoint_fingerprint"] = target
        state["history"] = [
            *state["history"],
            {
                "kind": "rollback",
                "from_checkpoint_fingerprint": active,
                "to_checkpoint_fingerprint": target,
                "reason": reason,
            },
        ]
        state["revision"] = int(state["revision"]) + 1
        self._write(state)
        return self.status()

    def status(self) -> dict[str, object]:
        state = self._load()
        active = str(state["active_checkpoint_fingerprint"])
        record = state["registered"].get(active)
        if not isinstance(record, dict):
            raise ValueError("active checkpoint is absent from the registry")
        artifact = self.root / str(record["artifact"])
        if not artifact.is_file() or _file_sha256(artifact) != record.get("artifact_sha256"):
            raise ValueError("active checkpoint artifact fingerprint mismatch")
        validation = validate_candidate_checkpoint(artifact)
        if not validation["valid"] or validation["checkpoint_fingerprint"] != active:
            raise ValueError("active checkpoint artifact is invalid")
        return {
            "schema_version": "1.0.0",
            "revision": state["revision"],
            "active_checkpoint_fingerprint": active,
            "registered_checkpoints": len(state["registered"]),
            "history_events": len(state["history"]),
            "registry_fingerprint": state["registry_fingerprint"],
            "active_artifact_verified": True,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage APC's local content-addressed checkpoint registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("root", type=Path)
    create.add_argument("incumbent", type=Path)
    create.add_argument("--reason", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("root", type=Path)
    register.add_argument("checkpoint", type=Path)
    register.add_argument("--expected-revision", type=int, required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("root", type=Path)
    promote.add_argument("candidate_fingerprint")
    promote.add_argument("evidence", type=Path)
    promote.add_argument("--expected-revision", type=int, required=True)
    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("root", type=Path)
    rollback.add_argument("--expected-revision", type=int, required=True)
    rollback.add_argument("--reason", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = CheckpointRegistry.create(
                args.root,
                args.incumbent,
                bootstrap_reason=args.reason,
            ).status()
        else:
            registry = CheckpointRegistry(args.root)
            if args.command == "register":
                result = registry.register_candidate(
                    args.checkpoint,
                    expected_revision=args.expected_revision,
                )
            elif args.command == "promote":
                evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
                result = registry.promote(
                    args.candidate_fingerprint,
                    evidence,
                    expected_revision=args.expected_revision,
                )
            elif args.command == "rollback":
                result = registry.rollback(
                    expected_revision=args.expected_revision,
                    reason=args.reason,
                )
            else:
                result = registry.status()
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
