from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from apc.evaluate_full_hand_table import POLICIES, _command
from apc.full_hand_table import HeadsUpVirtualHand
from apc.self_learning.replay_dataset import _canonical, _sha256_bytes, _split


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bb(value: object) -> str:
    normalized = Decimal(str(value)).normalize()
    return "0" if normalized == 0 else format(normalized, "f")


def _bb_value_is_serialized(value: object) -> bool:
    if isinstance(value, dict):
        return all(_bb_value_is_serialized(child) for child in value.values())
    if isinstance(value, list):
        return all(_bb_value_is_serialized(child) for child in value)
    if not isinstance(value, str):
        return False
    try:
        parsed = Decimal(value)
    except Exception:
        return False
    return parsed.is_finite()


def _bb_fields_are_strings(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).endswith("_bb"):
                if not _bb_value_is_serialized(child):
                    return False
            if not _bb_fields_are_strings(child):
                return False
    elif isinstance(value, list):
        return all(_bb_fields_are_strings(child) for child in value)
    return True


def _trajectory(seed: int, button: int, policy: str, split_seed: int) -> list[dict[str, object]]:
    hand = HeadsUpVirtualHand(seed=seed, button_player=button)
    pending: list[dict[str, object]] = []
    final: dict[str, object] | None = None
    hero_step = 0
    while not hand.terminal:
        state = hand.observation()
        command = _command(policy, hand)
        feedback = hand.step(command)
        final = feedback
        if state["next_actor"] != "Hero":
            continue
        if state["opponent_cards"] is not None:
            raise ValueError("non-terminal trajectory state leaked opponent cards")
        legal_actions = [str(row["action"]) for row in state["action_buttons"]]
        if command.action not in legal_actions:
            raise ValueError("behavior policy selected an action outside the recorded legal set")
        group_id = "virtual-hand-" + hand.hand_id
        row = {
            "schema_version": "1.0.0",
            "example_id": f"{group_id}-hero-{hero_step:03d}",
            "group_id": group_id,
            "split": _split(
                group_id,
                split_seed,
                Decimal("0.70"),
                Decimal("0.15"),
            ),
            "units": "BB",
            "trajectory_step": hero_step,
            "state": state,
            "behavior": {
                "policy": f"deterministic_coverage_probe:{policy}",
                "chosen_action": feedback["command"],
                "legal_actions": legal_actions,
                "gto_verified": False,
            },
            "learning_signal": None,
            "provenance": {
                "environment": "controlled_virtual_chips",
                "engine": "heads_up_virtual_hand_v1",
                "hand_seed": seed,
                "button_player": button,
                "pre_state_fingerprint": state["state_fingerprint"],
                "transition_fingerprint": feedback["transition_fingerprint"],
                "external_actuation": False,
            },
        }
        pending.append(row)
        hero_step += 1
    assert final is not None
    outcome = final["completed_hand_feedback"]
    for row in pending:
        row["learning_signal"] = {
            "kind": "sampled_monte_carlo_terminal_return",
            "hero_return_bb": _bb(outcome["rewards_bb"]["Hero"]),
            "terminal_reason": outcome["terminal_reason"],
            "final_pot_bb": _bb(outcome["final_pot_bb"]),
            "solver_target": False,
        }
        row["example_sha256"] = _sha256_bytes(_canonical(row))
    return pending


def build_full_hand_dataset(
    output: str | Path,
    *,
    dataset_id: str,
    hands: int = 100,
    hand_seed_start: int = 1000,
    split_seed: int = 20260816,
    minimum_examples: int = 100,
    minimum_groups: int = 30,
) -> dict[str, object]:
    if not dataset_id or any(character in dataset_id for character in "\\/:*?\"<>|"):
        raise ValueError("dataset_id must be a non-empty portable name")
    if hands < 10 or minimum_examples <= 0 or minimum_groups <= 0:
        raise ValueError("hands must be at least 10 and minimums must be positive")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (hand_seed_start, split_seed)):
        raise ValueError("seeds must be non-negative integers")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(f"full-hand dataset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        row
        for offset in range(hands)
        for row in _trajectory(
            hand_seed_start + offset,
            0 if POLICIES[offset % len(POLICIES)] == "fold" else offset % 2,
            POLICIES[offset % len(POLICIES)],
            split_seed,
        )
    ]
    examples.sort(key=lambda row: str(row["example_id"]))
    split_counts = Counter(str(row["split"]) for row in examples)
    groups = {str(row["group_id"]) for row in examples}
    eligibility_reasons: list[str] = []
    if len(examples) < minimum_examples:
        eligibility_reasons.append("minimum_examples_not_met")
    if len(groups) < minimum_groups:
        eligibility_reasons.append("minimum_groups_not_met")
    if not all(split_counts.get(split, 0) for split in ("train", "validation", "test")):
        eligibility_reasons.append("train_validation_test_splits_must_all_be_nonempty")
    eligible = not eligibility_reasons
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        root = Path(temporary)
        examples_bytes = b"".join(_canonical(row) + b"\n" for row in examples)
        (root / "examples.jsonl").write_bytes(examples_bytes)
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": dataset_id,
            "dataset_kind": "sampled_full_hand_monte_carlo_trajectory",
            "immutable": True,
            "units": "BB",
            "training_eligible": eligible,
            "policy_promotion_eligible": False,
            "training_eligibility": {
                "passed": eligible,
                "minimum_examples": minimum_examples,
                "minimum_groups": minimum_groups,
                "required_nonempty_splits": ["train", "validation", "test"],
                "reasons": eligibility_reasons,
            },
            "examples_file": "examples.jsonl",
            "examples_sha256": _sha256_bytes(examples_bytes),
            "example_count": len(examples),
            "group_count": len(groups),
            "hand_count": hands,
            "split_counts": {key: split_counts.get(key, 0) for key in ("train", "validation", "test")},
            "group_exclusive": True,
            "generation": {
                "hand_seed_start": hand_seed_start,
                "split_seed": split_seed,
                "behavior_policies": list(POLICIES),
                "hero_decisions_only": True,
            },
            "source_fingerprints": {
                "full_hand_engine_sha256": _sha256_file(Path(__file__).resolve().parents[1] / "full_hand_table.py")
            },
            "limitations": [
                "Returns are sampled outcomes from deterministic coverage probes, not solver or GTO labels.",
                "Only Hero-perspective decisions are recorded; opponent hole cards never enter decision states.",
                "The source environment is equal-stack heads-up no-limit Hold'em with no rake.",
            ],
        }
        material = dict(manifest)
        manifest["dataset_fingerprint"] = _sha256_bytes(_canonical(material))
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validation = validate_full_hand_dataset(root)
        if not validation["valid"]:
            raise ValueError("generated full-hand dataset failed validation: " + "; ".join(validation["issues"]))
        root.replace(destination)
    return manifest


def validate_full_hand_dataset(root: str | Path) -> dict[str, object]:
    dataset = Path(root).resolve()
    issues: list[str] = []
    try:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        examples_path = dataset / str(manifest["examples_file"])
        examples = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        return {"valid": False, "issues": [f"dataset unreadable: {error}"]}
    if manifest.get("immutable") is not True or manifest.get("units") != "BB":
        issues.append("manifest immutability/BB contract is invalid")
    if manifest.get("policy_promotion_eligible") is not False or manifest.get("group_exclusive") is not True:
        issues.append("manifest promotion/group contract is invalid")
    if manifest.get("example_count") != len(examples) or manifest.get("examples_sha256") != _sha256_file(examples_path):
        issues.append("manifest count or examples fingerprint mismatch")
    material = dict(manifest)
    observed_fingerprint = material.pop("dataset_fingerprint", None)
    if observed_fingerprint != _sha256_bytes(_canonical(material)):
        issues.append("dataset fingerprint mismatch")
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    group_steps: defaultdict[str, list[int]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(examples):
        label = f"example[{index}]"
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in ids:
            issues.append(f"{label} identity is missing or duplicated")
        ids.add(example_id)
        split = str(row.get("split", ""))
        group = str(row.get("group_id", ""))
        if split not in {"train", "validation", "test"} or not group:
            issues.append(f"{label} split/group is invalid")
        previous = group_splits.setdefault(group, split)
        if previous != split:
            issues.append(f"{label} leaks a hand group across splits")
        split_counts[split] += 1
        step = row.get("trajectory_step")
        if not isinstance(step, int) or step < 0:
            issues.append(f"{label} trajectory step is invalid")
        else:
            group_steps[group].append(step)
        state = row.get("state", {})
        behavior = row.get("behavior", {})
        signal = row.get("learning_signal", {})
        chosen = behavior.get("chosen_action", {}).get("action") if isinstance(behavior, dict) else None
        if state.get("opponent_cards") is not None or chosen not in behavior.get("legal_actions", []):
            issues.append(f"{label} private-card/legal-action contract is invalid")
        if behavior.get("gto_verified") is not False or signal.get("solver_target") is not False:
            issues.append(f"{label} makes an unsupported solver/GTO claim")
        if row.get("units") != "BB" or not _bb_fields_are_strings(row):
            issues.append(f"{label} BB serialization contract is invalid")
        try:
            Decimal(str(signal["hero_return_bb"]))
            Decimal(str(signal["final_pot_bb"]))
        except Exception:
            issues.append(f"{label} terminal BB signal is invalid")
        provenance = row.get("provenance", {})
        if provenance.get("external_actuation") is not False:
            issues.append(f"{label} external actuation must be false")
        expected = row.get("example_sha256")
        row_material = dict(row)
        row_material.pop("example_sha256", None)
        if expected != _sha256_bytes(_canonical(row_material)):
            issues.append(f"{label} fingerprint mismatch")
    if any(sorted(steps) != list(range(len(steps))) for steps in group_steps.values()):
        issues.append("trajectory steps are not contiguous within a hand")
    expected_counts = {key: split_counts.get(key, 0) for key in ("train", "validation", "test")}
    if manifest.get("split_counts") != expected_counts or manifest.get("group_count") != len(group_splits):
        issues.append("manifest split/group counts do not match examples")
    eligibility = manifest.get("training_eligibility", {})
    expected_eligible = (
        len(examples) >= eligibility.get("minimum_examples", 10**18)
        and len(group_splits) >= eligibility.get("minimum_groups", 10**18)
        and all(split_counts.get(key, 0) for key in ("train", "validation", "test"))
    )
    if manifest.get("training_eligible") is not expected_eligible or eligibility.get("passed") is not expected_eligible:
        issues.append("training eligibility is inconsistent")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "example_count": len(examples),
        "group_count": len(group_splits),
        "split_counts": expected_counts,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate APC full-hand trajectory data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("output", type=Path)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--hands", type=int, default=100)
    build.add_argument("--hand-seed-start", type=int, default=1000)
    build.add_argument("--split-seed", type=int, default=20260816)
    build.add_argument("--minimum-examples", type=int, default=100)
    build.add_argument("--minimum-groups", type=int, default=30)
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_full_hand_dataset(args.dataset)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        manifest = build_full_hand_dataset(
            args.output,
            dataset_id=args.dataset_id,
            hands=args.hands,
            hand_seed_start=args.hand_seed_start,
            split_seed=args.split_seed,
            minimum_examples=args.minimum_examples,
            minimum_groups=args.minimum_groups,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
