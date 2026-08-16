from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from apc.deadline import ActionCommand
from apc.full_hand_table import HeadsUpVirtualHand, _coach_types
from apc.self_learning.paired_rollout_dataset import _bb, _hand_class
from apc.self_learning.postflop_paired_rollout_dataset import (
    ACTIONS,
    STREETS,
    _commands,
    _sha256_file,
    _texture,
)
from apc.self_learning.replay_dataset import _canonical, _sha256_bytes, _split
from apc.self_learning.train_action_value import action_issues


OPPONENT_POLICIES = ("check_call", "fold_to_pressure", "made_hand_selective")


def _villain_command(hand: HeadsUpVirtualHand, policy: str) -> ActionCommand:
    buttons = {str(row["action"]): row for row in hand.legal_action_buttons()}
    if policy == "check_call":
        if "check" in buttons:
            return ActionCommand("check")
        if "call" in buttons:
            return ActionCommand("call")
    if policy == "fold_to_pressure":
        if "fold" in buttons:
            return ActionCommand("fold")
        if "check" in buttons:
            return ActionCommand("check")
    if policy == "made_hand_selective":
        best_hand_rank, _ = _coach_types()
        category = best_hand_rank((*hand.hole_cards[1], *hand.board)).category
        if "fold" in buttons:
            if category >= 2 and "raise" in buttons:
                return ActionCommand("raise", to_amount_bb=str(buttons["raise"]["minimum_to_bb"]))
            if category >= 1:
                return ActionCommand("call")
            return ActionCommand("fold")
        if category >= 2 and "bet" in buttons:
            return ActionCommand("bet", to_amount_bb=str(buttons["bet"]["minimum_to_bb"]))
        if "check" in buttons:
            return ActionCommand("check")
    raise RuntimeError(f"opponent policy {policy} found no legal action")


def _hero_command(hand: HeadsUpVirtualHand) -> ActionCommand:
    buttons = {str(row["action"]): row for row in hand.legal_action_buttons()}
    if "check" in buttons:
        return ActionCommand("check")
    if "call" in buttons:
        return ActionCommand("call")
    raise RuntimeError("Hero continuation found neither check nor call")


def _finish(
    hand: HeadsUpVirtualHand,
    first: ActionCommand,
    opponent_policy: str,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    prior_history = len(hand.history)
    first_feedback = hand.step(first)
    feedback = first_feedback
    while not hand.terminal:
        command = _hero_command(hand) if hand.next_actor == 0 else _villain_command(hand, opponent_policy)
        feedback = hand.step(command)
    continuation = list(hand.history[prior_history + 1 :])
    return first_feedback, feedback, continuation


def policy_postflop_rollout(seed: int, *, split_seed: int = 20260816) -> list[dict[str, object]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    base = HeadsUpVirtualHand(seed=seed, button_player=0)
    base.step(ActionCommand("call"))
    base.step(ActionCommand("check"))
    group_id = "postflop-policy-" + base.hand_id
    split = _split(group_id, split_seed, Decimal("0.70"), Decimal("0.15"))
    rows: list[dict[str, object]] = []
    for street in STREETS:
        base.step(ActionCommand("check"))
        state = base.observation()
        if state["street"] != street or state["next_actor"] != "Hero":
            raise ValueError("policy rollout trunk did not reach expected Hero state")
        commands = _commands(state)
        for policy in OPPONENT_POLICIES:
            for action in ACTIONS:
                hand = copy.deepcopy(base)
                first, terminal, continuation = _finish(hand, commands[action], policy)
                outcome = terminal["completed_hand_feedback"]
                row = {
                    "schema_version": "1.0.0",
                    "example_id": f"{group_id}-{street}-{policy}-{action}",
                    "group_id": group_id,
                    "state_id": f"{group_id}-{street}",
                    "policy_state_id": f"{group_id}-{street}-{policy}",
                    "split": split,
                    "units": "BB",
                    "street": street,
                    "state": state,
                    "opponent_policy": policy,
                    "counterfactual_action": first["command"],
                    "continuation_history": continuation,
                    "learning_signal": {
                        "kind": "postflop_multi_policy_paired_terminal_return",
                        "hero_return_bb": _bb(outcome["rewards_bb"]["Hero"]),
                        "solver_target": False,
                        "gto_verified": False,
                    },
                    "provenance": {
                        "environment": "controlled_virtual_chips",
                        "engine": "heads_up_virtual_hand_v1",
                        "hand_seed": seed,
                        "hero_hand_class": _hand_class(state["hero_cards"]),
                        "public_texture_class": _texture(state),
                        "common_random_cards_group": group_id,
                        "pre_state_fingerprint": state["state_fingerprint"],
                        "external_actuation": False,
                    },
                }
                row["example_sha256"] = _sha256_bytes(_canonical(row))
                rows.append(row)
        base.step(ActionCommand("check"))
    return rows


def _audits(rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, int]]:
    by_policy_state: defaultdict[str, dict[str, float]] = defaultdict(dict)
    opponent_actions: Counter[str] = Counter()
    for row in rows:
        by_policy_state[str(row["policy_state_id"])][str(row["counterfactual_action"]["action"])] = float(
            str(row["learning_signal"]["hero_return_bb"])
        )
        for history in row["continuation_history"]:
            if str(history).startswith("BB "):
                token = str(history).split(" ", 1)[1].split(":", 1)[0]
                if token == "raise_to":
                    token = "raise"
                opponent_actions[f"{row['opponent_policy']}:{token}"] += 1
    variance = {}
    means: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for policy in OPPONENT_POLICIES:
        variance[policy] = {}
        for street in STREETS:
            marker = f"-{street}-{policy}"
            states = [values for key, values in by_policy_state.items() if marker in key]
            street_rows = {}
            for action in ("bet", "all_in"):
                differences = [values[action] - values["check"] for values in states]
                action_values = [values[action] for values in states]
                check_values = [values["check"] for values in states]
                paired_se = math.sqrt(statistics.pvariance(differences) / len(states))
                unpaired_se = math.sqrt(
                    (statistics.pvariance(action_values) + statistics.pvariance(check_values)) / len(states)
                )
                mean = statistics.fmean(differences)
                means[f"{policy}:{action}"][street] = mean
                street_rows[f"{action}_minus_check"] = {
                    "samples": len(states),
                    "mean_difference_bb": format(mean, ".12g"),
                    "paired_standard_error_bb": format(paired_se, ".12g"),
                    "unpaired_standard_error_bb": format(unpaired_se, ".12g"),
                    "standard_error_reduction_fraction": format(
                        0.0 if unpaired_se == 0 else 1.0 - paired_se / unpaired_se,
                        ".12g",
                    ),
                }
            variance[policy][street] = street_rows
    differentiation = {
        key: {
            "street_means_bb": {street: format(values[street], ".12g") for street in STREETS},
            "max_minus_min_street_mean_bb": format(max(values.values()) - min(values.values()), ".12g"),
        }
        for key, values in means.items()
    }
    return {"paired_variance": variance, "street_differentiation": differentiation}, dict(sorted(opponent_actions.items()))


def build_postflop_policy_dataset(
    output: str | Path,
    *,
    dataset_id: str,
    rollouts: int = 1000,
    hand_seed_start: int = 40000,
    split_seed: int = 20260816,
    minimum_rollouts: int = 500,
    minimum_texture_classes: int = 20,
) -> dict[str, object]:
    if not dataset_id or any(character in dataset_id for character in "\\/:*?\"<>|"):
        raise ValueError("dataset_id must be a non-empty portable name")
    if rollouts < 20 or minimum_rollouts <= 0 or minimum_texture_classes <= 0:
        raise ValueError("rollouts must be at least 20 and minimums must be positive")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(f"postflop policy dataset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        row
        for offset in range(rollouts)
        for row in policy_postflop_rollout(hand_seed_start + offset, split_seed=split_seed)
    ]
    examples.sort(key=lambda row: str(row["example_id"]))
    groups = {str(row["group_id"]) for row in examples}
    textures = {str(row["provenance"]["public_texture_class"]) for row in examples}
    split_counts = Counter(str(row["split"]) for row in examples)
    audits, opponent_actions = _audits(examples)
    reasons = []
    if len(groups) < minimum_rollouts:
        reasons.append("minimum_rollouts_not_met")
    if len(textures) < minimum_texture_classes:
        reasons.append("minimum_texture_classes_not_met")
    if not all(split_counts.get(split, 0) for split in ("train", "validation", "test")):
        reasons.append("train_validation_test_splits_must_all_be_nonempty")
    required_selective = {"made_hand_selective:fold", "made_hand_selective:call", "made_hand_selective:raise", "made_hand_selective:bet"}
    if not required_selective.issubset(opponent_actions):
        reasons.append("selective_policy_did_not_cover_fold_call_bet_raise")
    eligible = not reasons
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        root = Path(temporary)
        examples_bytes = b"".join(_canonical(row) + b"\n" for row in examples)
        (root / "examples.jsonl").write_bytes(examples_bytes)
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": dataset_id,
            "dataset_kind": "postflop_multi_opponent_policy_paired_rollouts",
            "immutable": True,
            "units": "BB",
            "training_eligible": eligible,
            "policy_promotion_eligible": False,
            "eligibility": {
                "passed": eligible,
                "minimum_rollouts": minimum_rollouts,
                "minimum_texture_classes": minimum_texture_classes,
                "required_selective_actions": sorted(required_selective),
                "reasons": reasons,
            },
            "examples_file": "examples.jsonl",
            "examples_sha256": _sha256_bytes(examples_bytes),
            "example_count": len(examples),
            "paired_rollout_count": len(groups),
            "states_per_rollout": len(STREETS),
            "policies_per_state": len(OPPONENT_POLICIES),
            "actions_per_policy_state": len(ACTIONS),
            "public_texture_class_count": len(textures),
            "streets": list(STREETS),
            "hero_actions": list(ACTIONS),
            "opponent_policies": list(OPPONENT_POLICIES),
            "opponent_action_counts": opponent_actions,
            "split_counts": {key: split_counts.get(key, 0) for key in ("train", "validation", "test")},
            "group_exclusive": True,
            "generation": {
                "hand_seed_start": hand_seed_start,
                "hand_seed_end": hand_seed_start + rollouts - 1,
                "split_seed": split_seed,
                "base_line": "preflop_call_then_checkdown",
                "hero_continuation": "check_call",
                "common_random_numbers": "same_cards_within_street_policy_action_triplet",
            },
            **audits,
            "source_fingerprints": {
                "full_hand_engine_sha256": _sha256_file(Path(__file__).resolve().parents[1] / "full_hand_table.py")
            },
            "limitations": [
                "Opponent policies are declared deterministic probes, not learned population models or GTO strategies.",
                "States follow a preflop-call/checkdown trunk and cover Hero in position only.",
                "All-in targets remain high variance and are retained for audit rather than promotion.",
            ],
        }
        material = dict(manifest)
        manifest["dataset_fingerprint"] = _sha256_bytes(_canonical(material))
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validation = validate_postflop_policy_dataset(root)
        if not validation["valid"]:
            raise ValueError("generated postflop policy dataset failed validation: " + "; ".join(validation["issues"]))
        root.replace(destination)
    return manifest


def validate_postflop_policy_dataset(root: str | Path) -> dict[str, object]:
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
        issues.append("dataset cannot authorize promotion")
    if manifest.get("streets") != list(STREETS) or manifest.get("hero_actions") != list(ACTIONS) or manifest.get("opponent_policies") != list(OPPONENT_POLICIES):
        issues.append("manifest street/action/policy coverage is invalid")
    if manifest.get("example_count") != len(examples) or manifest.get("examples_sha256") != _sha256_file(examples_path):
        issues.append("manifest count or examples fingerprint mismatch")
    material = dict(manifest)
    observed = material.pop("dataset_fingerprint", None)
    if observed != _sha256_bytes(_canonical(material)):
        issues.append("dataset fingerprint mismatch")
    ids: set[str] = set()
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    policy_states: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(examples):
        label = f"example[{index}]"
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in ids:
            issues.append(f"{label} identity is missing or duplicated")
        ids.add(example_id)
        state = row.get("state", {})
        if row.get("units") != "BB" or state.get("opponent_cards") is not None or state.get("next_actor") != "Hero":
            issues.append(f"{label} private/Hero decision contract is invalid")
        command = row.get("counterfactual_action")
        if not isinstance(command, dict) or action_issues(state, command):
            issues.append(f"{label} counterfactual action is not legal")
        if row.get("opponent_policy") not in OPPONENT_POLICIES or not isinstance(row.get("continuation_history"), list):
            issues.append(f"{label} policy/continuation evidence is invalid")
        signal = row.get("learning_signal", {})
        if signal.get("solver_target") is not False or signal.get("gto_verified") is not False:
            issues.append(f"{label} makes unsupported solver/GTO claims")
        try:
            value = Decimal(str(signal["hero_return_bb"]))
        except (InvalidOperation, KeyError):
            issues.append(f"{label} return is invalid")
        else:
            if not value.is_finite() or value < -100 or value > 100:
                issues.append(f"{label} return is invalid")
        if row.get("provenance", {}).get("external_actuation") is not False:
            issues.append(f"{label} external actuation must be false")
        expected_hash = row.get("example_sha256")
        row_material = dict(row)
        row_material.pop("example_sha256", None)
        if expected_hash != _sha256_bytes(_canonical(row_material)):
            issues.append(f"{label} fingerprint mismatch")
        groups[str(row.get("group_id", ""))].append(row)
        policy_states[str(row.get("policy_state_id", ""))].append(row)
        split_counts[str(row.get("split", ""))] += 1
    for group, rows in groups.items():
        if len(rows) != len(STREETS) * len(OPPONENT_POLICIES) * len(ACTIONS) or len({str(row["split"]) for row in rows}) != 1:
            issues.append(f"hand group {group} is incomplete or split-leaked")
    for state_id, rows in policy_states.items():
        if len(rows) != len(ACTIONS) or {str(row["counterfactual_action"]["action"]) for row in rows} != set(ACTIONS) or len({str(row["provenance"]["pre_state_fingerprint"]) for row in rows}) != 1:
            issues.append(f"policy state {state_id} is incomplete or not common-state")
    expected_counts = {key: split_counts.get(key, 0) for key in ("train", "validation", "test")}
    if manifest.get("split_counts") != expected_counts or manifest.get("paired_rollout_count") != len(groups):
        issues.append("manifest split/group counts do not match examples")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "example_count": len(examples),
        "paired_rollout_count": len(groups),
        "policy_state_count": len(policy_states),
        "split_counts": expected_counts,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate APC multi-policy postflop paired data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("output", type=Path)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--rollouts", type=int, default=1000)
    build.add_argument("--hand-seed-start", type=int, default=40000)
    build.add_argument("--split-seed", type=int, default=20260816)
    build.add_argument("--minimum-rollouts", type=int, default=500)
    build.add_argument("--minimum-texture-classes", type=int, default=20)
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_postflop_policy_dataset(args.dataset)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        manifest = build_postflop_policy_dataset(
            args.output,
            dataset_id=args.dataset_id,
            rollouts=args.rollouts,
            hand_seed_start=args.hand_seed_start,
            split_seed=args.split_seed,
            minimum_rollouts=args.minimum_rollouts,
            minimum_texture_classes=args.minimum_texture_classes,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
