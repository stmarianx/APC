from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from apc.deadline import ActionCommand
from apc.evaluate_full_hand_table import _command
from apc.full_hand_table import HeadsUpVirtualHand
from apc.self_learning.replay_dataset import _canonical, _sha256_bytes, _split
from apc.self_learning.train_action_value import action_issues


ACTION_ORDER = ("fold", "call", "raise", "all_in")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bb(value: object) -> str:
    normalized = Decimal(str(value)).normalize()
    return "0" if normalized == 0 else format(normalized, "f")


def _hand_class(cards: list[str]) -> str:
    rank_value = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}
    ordered = sorted(cards, key=lambda card: rank_value[card[0].upper()], reverse=True)
    ranks = ordered[0][0].upper() + ordered[1][0].upper()
    if ranks[0] == ranks[1]:
        return ranks
    return ranks + ("s" if ordered[0][-1] == ordered[1][-1] else "o")


def _counterfactual_commands(state: dict[str, object]) -> dict[str, ActionCommand]:
    buttons = {str(row["action"]): row for row in state["action_buttons"]}
    if set(buttons) != set(ACTION_ORDER):
        raise ValueError("paired rollout requires the four expected heads-up preflop actions")
    return {
        "fold": ActionCommand("fold"),
        "call": ActionCommand("call"),
        "raise": ActionCommand("raise", to_amount_bb=str(buttons["raise"]["minimum_to_bb"])),
        "all_in": ActionCommand("all_in"),
    }


def paired_rollout(seed: int, *, split_seed: int = 20260816) -> list[dict[str, object]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    base = HeadsUpVirtualHand(seed=seed, button_player=0)
    state = base.observation()
    commands = _counterfactual_commands(state)
    group_id = "paired-hand-" + base.hand_id
    split = _split(group_id, split_seed, Decimal("0.70"), Decimal("0.15"))
    rows: list[dict[str, object]] = []
    for action in ACTION_ORDER:
        hand = copy.deepcopy(base)
        first_feedback = hand.step(commands[action])
        feedback = first_feedback
        while not hand.terminal:
            feedback = hand.step(_command("check_call", hand))
        outcome = feedback["completed_hand_feedback"]
        row = {
            "schema_version": "1.0.0",
            "example_id": f"{group_id}-{action}",
            "group_id": group_id,
            "split": split,
            "units": "BB",
            "state": state,
            "counterfactual_action": first_feedback["command"],
            "learning_signal": {
                "kind": "paired_common_random_cards_terminal_return",
                "hero_return_bb": _bb(outcome["rewards_bb"]["Hero"]),
                "terminal_reason": outcome["terminal_reason"],
                "solver_target": False,
                "gto_verified": False,
            },
            "provenance": {
                "environment": "controlled_virtual_chips",
                "engine": "heads_up_virtual_hand_v1",
                "hand_seed": seed,
                "hero_hand_class": _hand_class(state["hero_cards"]),
                "common_random_cards_group": group_id,
                "continuation_policy": "deterministic_check_call_v1",
                "pre_state_fingerprint": state["state_fingerprint"],
                "external_actuation": False,
            },
        }
        row["example_sha256"] = _sha256_bytes(_canonical(row))
        rows.append(row)
    return rows


def _variance_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    by_group: defaultdict[str, dict[str, float]] = defaultdict(dict)
    by_action: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        action = str(row["counterfactual_action"]["action"])
        value = float(str(row["learning_signal"]["hero_return_bb"]))
        by_group[str(row["group_id"])][action] = value
        by_action[action].append(value)
    count = len(by_group)
    action_metrics = {}
    for action in ACTION_ORDER:
        values = by_action[action]
        variance = statistics.pvariance(values)
        action_metrics[action] = {
            "samples": len(values),
            "mean_return_bb": format(statistics.fmean(values), ".12g"),
            "population_variance_bb2": format(variance, ".12g"),
            "standard_error_bb": format(math.sqrt(variance / len(values)), ".12g"),
        }
    paired_differences = {}
    call_values = by_action["call"]
    call_variance = statistics.pvariance(call_values)
    for action in ("fold", "raise", "all_in"):
        values = by_action[action]
        differences = [by_group[group][action] - by_group[group]["call"] for group in sorted(by_group)]
        paired_variance = statistics.pvariance(differences)
        paired_se = math.sqrt(paired_variance / count)
        unpaired_se = math.sqrt((statistics.pvariance(values) + call_variance) / count)
        paired_differences[f"{action}_minus_call"] = {
            "samples": count,
            "mean_difference_bb": format(statistics.fmean(differences), ".12g"),
            "paired_standard_error_bb": format(paired_se, ".12g"),
            "unpaired_standard_error_bb": format(unpaired_se, ".12g"),
            "standard_error_reduction_fraction": format(
                0.0 if unpaired_se == 0 else 1.0 - paired_se / unpaired_se,
                ".12g",
            ),
        }
    return {"action_returns": action_metrics, "paired_differences": paired_differences}


def build_paired_rollout_dataset(
    output: str | Path,
    *,
    dataset_id: str,
    rollouts: int = 1000,
    hand_seed_start: int = 5000,
    split_seed: int = 20260816,
    minimum_rollouts: int = 500,
    minimum_hand_classes: int = 100,
) -> dict[str, object]:
    if not dataset_id or any(character in dataset_id for character in "\\/:*?\"<>|"):
        raise ValueError("dataset_id must be a non-empty portable name")
    if rollouts < 20 or minimum_rollouts <= 0 or minimum_hand_classes <= 0:
        raise ValueError("rollouts must be at least 20 and minimums must be positive")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(f"paired rollout destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        row
        for offset in range(rollouts)
        for row in paired_rollout(hand_seed_start + offset, split_seed=split_seed)
    ]
    examples.sort(key=lambda row: str(row["example_id"]))
    groups = {str(row["group_id"]) for row in examples}
    hand_classes = {str(row["provenance"]["hero_hand_class"]) for row in examples}
    split_counts = Counter(str(row["split"]) for row in examples)
    reasons = []
    if len(groups) < minimum_rollouts:
        reasons.append("minimum_paired_rollouts_not_met")
    if len(hand_classes) < minimum_hand_classes:
        reasons.append("minimum_hero_hand_classes_not_met")
    if not all(split_counts.get(split, 0) for split in ("train", "validation", "test")):
        reasons.append("train_validation_test_splits_must_all_be_nonempty")
    eligible = not reasons
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        root = Path(temporary)
        examples_bytes = b"".join(_canonical(row) + b"\n" for row in examples)
        (root / "examples.jsonl").write_bytes(examples_bytes)
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": dataset_id,
            "dataset_kind": "paired_common_random_cards_counterfactual_rollouts",
            "immutable": True,
            "units": "BB",
            "training_eligible": eligible,
            "policy_promotion_eligible": False,
            "eligibility": {
                "passed": eligible,
                "minimum_rollouts": minimum_rollouts,
                "minimum_hand_classes": minimum_hand_classes,
                "reasons": reasons,
            },
            "examples_file": "examples.jsonl",
            "examples_sha256": _sha256_bytes(examples_bytes),
            "example_count": len(examples),
            "paired_rollout_count": len(groups),
            "examples_per_rollout": len(ACTION_ORDER),
            "hero_hand_class_count": len(hand_classes),
            "action_vocabulary": list(ACTION_ORDER),
            "split_counts": {key: split_counts.get(key, 0) for key in ("train", "validation", "test")},
            "group_exclusive": True,
            "generation": {
                "hand_seed_start": hand_seed_start,
                "hand_seed_end": hand_seed_start + rollouts - 1,
                "split_seed": split_seed,
                "continuation_policy": "deterministic_check_call_v1",
                "common_random_numbers": "same_hole_cards_and_board_runout_across_actions",
            },
            "variance_audit": _variance_metrics(examples),
            "source_fingerprints": {
                "full_hand_engine_sha256": _sha256_file(Path(__file__).resolve().parents[1] / "full_hand_table.py")
            },
            "limitations": [
                "Counterfactuals cover the four legal Hero button preflop actions only.",
                "The opponent and continuation policy is deterministic check/call, not a GTO or learned policy.",
                "Same-deal pairing reduces comparison variance but does not remove sampled-showdown variance or create solver labels.",
            ],
        }
        material = dict(manifest)
        manifest["dataset_fingerprint"] = _sha256_bytes(_canonical(material))
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validation = validate_paired_rollout_dataset(root)
        if not validation["valid"]:
            raise ValueError("generated paired rollout dataset failed validation: " + "; ".join(validation["issues"]))
        root.replace(destination)
    return manifest


def validate_paired_rollout_dataset(root: str | Path) -> dict[str, object]:
    dataset = Path(root).resolve()
    issues: list[str] = []
    try:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        examples_path = dataset / str(manifest["examples_file"])
        examples = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        return {"valid": False, "issues": [f"dataset unreadable: {error}"]}
    if manifest.get("immutable") is not True or manifest.get("units") != "BB" or manifest.get("group_exclusive") is not True:
        issues.append("manifest immutable BB/group contract is invalid")
    if manifest.get("policy_promotion_eligible") is not False:
        issues.append("paired dataset cannot authorize policy promotion")
    if manifest.get("examples_per_rollout") != len(ACTION_ORDER) or manifest.get("action_vocabulary") != list(ACTION_ORDER):
        issues.append("manifest paired action coverage is invalid")
    if manifest.get("example_count") != len(examples) or manifest.get("examples_sha256") != _sha256_file(examples_path):
        issues.append("manifest count or examples fingerprint mismatch")
    material = dict(manifest)
    observed = material.pop("dataset_fingerprint", None)
    if observed != _sha256_bytes(_canonical(material)):
        issues.append("dataset fingerprint mismatch")
    ids: set[str] = set()
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(examples):
        label = f"example[{index}]"
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in ids:
            issues.append(f"{label} identity is missing or duplicated")
        ids.add(example_id)
        if row.get("units") != "BB" or row.get("state", {}).get("opponent_cards") is not None:
            issues.append(f"{label} BB/private-state contract is invalid")
        command = row.get("counterfactual_action")
        if not isinstance(command, dict) or action_issues(row.get("state", {}), command):
            issues.append(f"{label} counterfactual action is not exactly legal")
        signal = row.get("learning_signal", {})
        if signal.get("solver_target") is not False or signal.get("gto_verified") is not False:
            issues.append(f"{label} makes an unsupported solver/GTO claim")
        try:
            value = Decimal(str(signal["hero_return_bb"]))
        except (InvalidOperation, KeyError):
            issues.append(f"{label} terminal BB return is invalid")
        else:
            if not value.is_finite() or value < Decimal("-100") or value > Decimal("100"):
                issues.append(f"{label} terminal BB return is invalid")
        provenance = row.get("provenance", {})
        if provenance.get("external_actuation") is not False:
            issues.append(f"{label} external actuation must be false")
        expected_hash = row.get("example_sha256")
        row_material = dict(row)
        row_material.pop("example_sha256", None)
        if expected_hash != _sha256_bytes(_canonical(row_material)):
            issues.append(f"{label} fingerprint mismatch")
        group = str(row.get("group_id", ""))
        groups[group].append(row)
        split_counts[str(row.get("split", ""))] += 1
    for group, rows in groups.items():
        actions = {str(row["counterfactual_action"]["action"]) for row in rows}
        splits = {str(row["split"]) for row in rows}
        fingerprints = {str(row["provenance"]["pre_state_fingerprint"]) for row in rows}
        if len(rows) != len(ACTION_ORDER) or actions != set(ACTION_ORDER) or len(splits) != 1 or len(fingerprints) != 1:
            issues.append(f"paired group {group} is incomplete, leaked or not common-state")
    expected_counts = {key: split_counts.get(key, 0) for key in ("train", "validation", "test")}
    if manifest.get("split_counts") != expected_counts or manifest.get("paired_rollout_count") != len(groups):
        issues.append("manifest split/group counts do not match examples")
    eligibility = manifest.get("eligibility", {})
    expected_eligible = (
        len(groups) >= eligibility.get("minimum_rollouts", 10**18)
        and len({str(row.get("provenance", {}).get("hero_hand_class", "")) for row in examples})
        >= eligibility.get("minimum_hand_classes", 10**18)
        and all(split_counts.get(key, 0) for key in ("train", "validation", "test"))
    )
    if manifest.get("training_eligible") is not expected_eligible or eligibility.get("passed") is not expected_eligible:
        issues.append("training eligibility is inconsistent")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "example_count": len(examples),
        "paired_rollout_count": len(groups),
        "split_counts": expected_counts,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate APC paired counterfactual rollout data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("output", type=Path)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--rollouts", type=int, default=1000)
    build.add_argument("--hand-seed-start", type=int, default=5000)
    build.add_argument("--split-seed", type=int, default=20260816)
    build.add_argument("--minimum-rollouts", type=int, default=500)
    build.add_argument("--minimum-hand-classes", type=int, default=100)
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_paired_rollout_dataset(args.dataset)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        manifest = build_paired_rollout_dataset(
            args.output,
            dataset_id=args.dataset_id,
            rollouts=args.rollouts,
            hand_seed_start=args.hand_seed_start,
            split_seed=args.split_seed,
            minimum_rollouts=args.minimum_rollouts,
            minimum_hand_classes=args.minimum_hand_classes,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
