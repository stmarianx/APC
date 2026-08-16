from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from apc.full_hand_table import _coach_types
from apc.self_learning.postflop_paired_rollout_dataset import _sha256_file
from apc.self_learning.profile_conditioned_postflop import evaluate_profile_conditioned_dataset
from apc.self_learning.train_postflop_policy_value import validate_postflop_policy_value_checkpoint
from apc.self_learning.train_value import _sha256

_coach_types()
from poker_coach.opponent_model import (
    PersistentProfileStore,
    infer_opponent_policy_mixture,
    validate_opponent_policy_mixture,
)
from poker_coach.profiles import Tendency


ARCHETYPES = {
    "overfolder": {
        Tendency.FOLD_TO_FLOP_CBET: (45, 50),
        Tendency.AGGRESSIVE_ACTION: (5, 50),
        Tendency.WENT_TO_SHOWDOWN: (5, 50),
    },
    "sticky_passive": {
        Tendency.FOLD_TO_FLOP_CBET: (5, 50),
        Tendency.AGGRESSIVE_ACTION: (5, 50),
        Tendency.WENT_TO_SHOWDOWN: (40, 50),
    },
    "aggressive_selective": {
        Tendency.FOLD_TO_FLOP_CBET: (20, 50),
        Tendency.AGGRESSIVE_ACTION: (45, 50),
        Tendency.WENT_TO_SHOWDOWN: (20, 50),
    },
}


def build_profile_conditioned_audit(
    output: str | Path,
    dataset: str | Path,
    checkpoint: str | Path,
) -> dict[str, object]:
    destination = Path(output).resolve()
    dataset_path = Path(dataset).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    if destination.exists():
        raise ValueError(f"profile-conditioned audit destination already exists: {destination}")
    checkpoint_validation = validate_postflop_policy_value_checkpoint(checkpoint_path)
    if not checkpoint_validation["valid"]:
        raise ValueError("postflop checkpoint is invalid: " + "; ".join(checkpoint_validation["issues"]))
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        root = Path(temporary)
        store = PersistentProfileStore()
        for label, tendencies in ARCHETYPES.items():
            profile_key = f"controlled-audit:{label}"
            for tendency, (successes, trials) in tendencies.items():
                outcomes = [True] * successes + [False] * (trials - successes)
                for index, success in enumerate(outcomes):
                    store.observe(
                        profile_key,
                        f"{label}:{tendency.value}:{index}",
                        tendency,
                        success,
                    )
        store.save(root / "profile_store.json")
        mixtures = {
            label: infer_opponent_policy_mixture(store.profile(f"controlled-audit:{label}"))
            for label in ARCHETYPES
        }
        (root / "mixtures.json").write_text(json.dumps(mixtures, indent=2) + "\n", encoding="utf-8")
        audit = evaluate_profile_conditioned_dataset(dataset_path, checkpoint_payload, mixtures)
        (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "1.0.0",
            "artifact_kind": "profile_conditioned_postflop_offline_audit",
            "immutable": True,
            "units": "BB",
            "training_eligible": False,
            "promotion_eligible": False,
            "recommendation_allowed": False,
            "activation_authorized": False,
            "external_actuation": False,
            "dataset_fingerprint": audit["dataset_fingerprint"],
            "checkpoint_fingerprint": checkpoint_payload["checkpoint_fingerprint"],
            "files": {
                name: _sha256_file(root / name)
                for name in ("profile_store.json", "mixtures.json", "audit.json")
            },
            "profile_scenarios": list(ARCHETYPES),
            "profile_event_count": store.revision,
            "mixture_fingerprints": audit["mixture_fingerprints"],
            "audit_fingerprint": audit["audit_fingerprint"],
            "gate_passed": audit["gate"]["passed"],
            "limitations": audit["limitations"],
        }
        manifest["artifact_fingerprint"] = _sha256(manifest)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validation = validate_profile_conditioned_audit(root)
        if not validation["valid"]:
            raise ValueError("generated profile-conditioned audit is invalid: " + "; ".join(validation["issues"]))
        root.replace(destination)
    return manifest


def validate_profile_conditioned_audit(root: str | Path) -> dict[str, object]:
    artifact = Path(root).resolve()
    issues = []
    try:
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((artifact / "audit.json").read_text(encoding="utf-8"))
        mixtures = json.loads((artifact / "mixtures.json").read_text(encoding="utf-8"))
        store = PersistentProfileStore.load(artifact / "profile_store.json")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        return {"schema_version": "1.0.0", "valid": False, "issues": [f"audit artifact unreadable: {error}"]}
    if manifest.get("artifact_kind") != "profile_conditioned_postflop_offline_audit" or manifest.get("units") != "BB":
        issues.append("manifest kind/BB contract is invalid")
    if any(manifest.get(key) is not False for key in ("training_eligible", "promotion_eligible", "recommendation_allowed", "activation_authorized", "external_actuation")):
        issues.append("audit artifact cannot authorize training, promotion, recommendations, activation or actuation")
    expected_files = {name: _sha256_file(artifact / name) for name in ("profile_store.json", "mixtures.json", "audit.json")}
    if manifest.get("files") != expected_files:
        issues.append("audit file fingerprint mismatch")
    if set(mixtures) != set(ARCHETYPES) or any(not validate_opponent_policy_mixture(row)["valid"] for row in mixtures.values()):
        issues.append("audit mixtures are missing or invalid")
    if manifest.get("profile_event_count") != store.revision or store.revision != len(ARCHETYPES) * 3 * 50:
        issues.append("profile store event coverage is invalid")
    if audit.get("audit_fingerprint") != manifest.get("audit_fingerprint") or audit.get("gate", {}).get("passed") is not True:
        issues.append("profile-conditioned audit fingerprint/gate is invalid")
    material = dict(manifest)
    observed = material.pop("artifact_fingerprint", None)
    if observed != _sha256(material):
        issues.append("artifact fingerprint mismatch")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "artifact_fingerprint": manifest.get("artifact_fingerprint"),
        "profile_event_count": store.revision,
        "profile_scenarios": len(mixtures),
        "test_visible_states": audit.get("test_visible_states"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate APC's frozen profile-conditioned postflop audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("output", type=Path)
    build.add_argument("dataset", type=Path)
    build.add_argument("checkpoint", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_profile_conditioned_audit(args.artifact)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        manifest = build_profile_conditioned_audit(args.output, args.dataset, args.checkpoint)
        print(json.dumps(manifest, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
