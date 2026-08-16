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
from apc.full_hand_table import HeadsUpVirtualHand, _coach_types
from apc.self_learning.paired_rollout_dataset import _bb, _hand_class
from apc.self_learning.replay_dataset import _canonical, _sha256_bytes, _split
from apc.self_learning.train_action_value import action_issues


STREETS = ("flop", "turn", "river")
ACTIONS = ("check", "bet", "all_in")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _texture(state: dict[str, object]) -> str:
    board = [str(card) for card in state["board"]]
    hero = [str(card) for card in state["hero_cards"]]
    best_hand_rank, _ = _coach_types()
    from poker_coach.models import Card

    cards = [Card(card[0].upper(), card[-1]) for card in [*hero, *board]]
    rank = best_hand_rank(cards)
    ranks = [card[0].upper() for card in board]
    suits = Counter(card[-1] for card in board)
    return ":".join([
        str(state["street"]),
        rank.name,
        "paired" if len(set(ranks)) < len(ranks) else "unpaired",
        f"maxsuit{max(suits.values())}",
    ])


def _commands(state: dict[str, object]) -> dict[str, ActionCommand]:
    buttons = {str(row["action"]): row for row in state["action_buttons"]}
    if set(buttons) != set(ACTIONS):
        raise ValueError("postflop paired rollout requires check, bet and all-in")
    return {
        "check": ActionCommand("check"),
        "bet": ActionCommand("bet", to_amount_bb=str(buttons["bet"]["minimum_to_bb"])),
        "all_in": ActionCommand("all_in"),
    }


def _finish(hand: HeadsUpVirtualHand, first: ActionCommand) -> tuple[dict[str, object], dict[str, object]]:
    first_feedback = hand.step(first)
    feedback = first_feedback
    while not hand.terminal:
        feedback = hand.step(_command("check_call", hand))
    return first_feedback, feedback


def postflop_paired_rollout(seed: int, *, split_seed: int = 20260816) -> list[dict[str, object]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    base = HeadsUpVirtualHand(seed=seed, button_player=0)
    base.step(ActionCommand("call"))
    base.step(ActionCommand("check"))
    group_id = "postflop-paired-" + base.hand_id
    split = _split(group_id, split_seed, Decimal("0.70"), Decimal("0.15"))
    rows: list[dict[str, object]] = []
    for street in STREETS:
        if base.street != street or base.next_actor != 1:
            raise ValueError("base checkdown path did not reach expected out-of-position actor")
        base.step(ActionCommand("check"))
        state = base.observation()
        if state["next_actor"] != "Hero":
            raise ValueError("postflop paired state is not a Hero decision")
        commands = _commands(state)
        for action in ACTIONS:
            hand = copy.deepcopy(base)
            first, terminal = _finish(hand, commands[action])
            outcome = terminal["completed_hand_feedback"]
            row = {
                "schema_version": "1.0.0",
                "example_id": f"{group_id}-{street}-{action}",
                "group_id": group_id,
                "state_id": f"{group_id}-{street}",
                "split": split,
                "units": "BB",
                "street": street,
                "state": state,
                "counterfactual_action": first["command"],
                "learning_signal": {
                    "kind": "postflop_paired_common_random_cards_terminal_return",
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
                    "continuation_policy": "deterministic_check_call_v1",
                    "pre_state_fingerprint": state["state_fingerprint"],
                    "external_actuation": False,
                },
            }
            row["example_sha256"] = _sha256_bytes(_canonical(row))
            rows.append(row)
        base.step(ActionCommand("check"))
    if not base.terminal:
        raise ValueError("base checkdown path did not settle at river")
    return rows


def _variance_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    by_state: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_state[str(row["state_id"])][str(row["counterfactual_action"]["action"])] = float(
            str(row["learning_signal"]["hero_return_bb"])
        )
    output = {}
    for street in STREETS:
        states = [values for state_id, values in by_state.items() if f"-{street}" in state_id]
        street_rows = {}
        for action in ("bet", "all_in"):
            differences = [values[action] - values["check"] for values in states]
            action_values = [values[action] for values in states]
            check_values = [values["check"] for values in states]
            paired_variance = statistics.pvariance(differences)
            paired_se = math.sqrt(paired_variance / len(states))
            unpaired_se = math.sqrt(
                (statistics.pvariance(action_values) + statistics.pvariance(check_values)) / len(states)
            )
            street_rows[f"{action}_minus_check"] = {
                "samples": len(states),
                "mean_difference_bb": format(statistics.fmean(differences), ".12g"),
                "paired_standard_error_bb": format(paired_se, ".12g"),
                "unpaired_standard_error_bb": format(unpaired_se, ".12g"),
                "standard_error_reduction_fraction": format(
                    0.0 if unpaired_se == 0 else 1.0 - paired_se / unpaired_se,
                    ".12g",
                ),
            }
        output[street] = street_rows
    return output


def build_postflop_paired_dataset(
    output: str | Path,
    *,
    dataset_id: str,
    rollouts: int = 1000,
    hand_seed_start: int = 30000,
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
        raise ValueError(f"postflop paired destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        row
        for offset in range(rollouts)
        for row in postflop_paired_rollout(hand_seed_start + offset, split_seed=split_seed)
    ]
    examples.sort(key=lambda row: str(row["example_id"]))
    groups = {str(row["group_id"]) for row in examples}
    textures = {str(row["provenance"]["public_texture_class"]) for row in examples}
    split_counts = Counter(str(row["split"]) for row in examples)
    reasons = []
    if len(groups) < minimum_rollouts:
        reasons.append("minimum_paired_rollouts_not_met")
    if len(textures) < minimum_texture_classes:
        reasons.append("minimum_texture_classes_not_met")
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
            "dataset_kind": "postflop_paired_common_random_cards_counterfactual_rollouts",
            "immutable": True,
            "units": "BB",
            "training_eligible": eligible,
            "policy_promotion_eligible": False,
            "eligibility": {
                "passed": eligible,
                "minimum_rollouts": minimum_rollouts,
                "minimum_texture_classes": minimum_texture_classes,
                "reasons": reasons,
            },
            "examples_file": "examples.jsonl",
            "examples_sha256": _sha256_bytes(examples_bytes),
            "example_count": len(examples),
            "paired_rollout_count": len(groups),
            "states_per_rollout": len(STREETS),
            "examples_per_state": len(ACTIONS),
            "public_texture_class_count": len(textures),
            "streets": list(STREETS),
            "action_vocabulary": list(ACTIONS),
            "split_counts": {key: split_counts.get(key, 0) for key in ("train", "validation", "test")},
            "group_exclusive": True,
            "generation": {
                "hand_seed_start": hand_seed_start,
                "hand_seed_end": hand_seed_start + rollouts - 1,
                "split_seed": split_seed,
                "base_line": "preflop_call_then_checkdown",
                "continuation_policy": "deterministic_check_call_v1",
                "common_random_numbers": "same_hole_cards_and_runout_within_each_street_action_triplet",
            },
            "variance_audit": _variance_audit(examples),
            "source_fingerprints": {
                "full_hand_engine_sha256": _sha256_file(Path(__file__).resolve().parents[1] / "full_hand_table.py")
            },
            "limitations": [
                "States follow one preflop-call/checkdown trunk and cover Hero in-position decisions only.",
                "Counterfactual actions are check, minimum bet and all-in against deterministic check/call continuation.",
                "Targets are sampled outcomes, not solver values, GTO labels or population-opponent estimates.",
            ],
        }
        material = dict(manifest)
        manifest["dataset_fingerprint"] = _sha256_bytes(_canonical(material))
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validation = validate_postflop_paired_dataset(root)
        if not validation["valid"]:
            raise ValueError("generated postflop paired dataset failed validation: " + "; ".join(validation["issues"]))
        root.replace(destination)
    return manifest


def validate_postflop_paired_dataset(root: str | Path) -> dict[str, object]:
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
        issues.append("postflop dataset cannot authorize promotion")
    if manifest.get("streets") != list(STREETS) or manifest.get("action_vocabulary") != list(ACTIONS):
        issues.append("manifest street/action coverage is invalid")
    if manifest.get("example_count") != len(examples) or manifest.get("examples_sha256") != _sha256_file(examples_path):
        issues.append("manifest count or examples fingerprint mismatch")
    material = dict(manifest)
    observed = material.pop("dataset_fingerprint", None)
    if observed != _sha256_bytes(_canonical(material)):
        issues.append("dataset fingerprint mismatch")
    ids: set[str] = set()
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    states: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(examples):
        label = f"example[{index}]"
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in ids:
            issues.append(f"{label} identity is missing or duplicated")
        ids.add(example_id)
        state = row.get("state", {})
        if row.get("units") != "BB" or state.get("opponent_cards") is not None or state.get("next_actor") != "Hero":
            issues.append(f"{label} BB/private/Hero-decision contract is invalid")
        command = row.get("counterfactual_action")
        if not isinstance(command, dict) or action_issues(state, command):
            issues.append(f"{label} counterfactual action is not exactly legal")
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
        states[str(row.get("state_id", ""))].append(row)
        split_counts[str(row.get("split", ""))] += 1
    for group, rows in groups.items():
        if len(rows) != len(STREETS) * len(ACTIONS) or {str(row["street"]) for row in rows} != set(STREETS) or len({str(row["split"]) for row in rows}) != 1:
            issues.append(f"hand group {group} is incomplete or split-leaked")
    for state_id, rows in states.items():
        if len(rows) != len(ACTIONS) or {str(row["counterfactual_action"]["action"]) for row in rows} != set(ACTIONS) or len({str(row["provenance"]["pre_state_fingerprint"]) for row in rows}) != 1:
            issues.append(f"state group {state_id} is incomplete or not common-state")
    expected_counts = {key: split_counts.get(key, 0) for key in ("train", "validation", "test")}
    if manifest.get("split_counts") != expected_counts or manifest.get("paired_rollout_count") != len(groups):
        issues.append("manifest split/group counts do not match examples")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "example_count": len(examples),
        "paired_rollout_count": len(groups),
        "state_count": len(states),
        "split_counts": expected_counts,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or validate APC postflop paired rollout data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("output", type=Path)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--rollouts", type=int, default=1000)
    build.add_argument("--hand-seed-start", type=int, default=30000)
    build.add_argument("--split-seed", type=int, default=20260816)
    build.add_argument("--minimum-rollouts", type=int, default=500)
    build.add_argument("--minimum-texture-classes", type=int, default=20)
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_postflop_paired_dataset(args.dataset)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        manifest = build_postflop_paired_dataset(
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
