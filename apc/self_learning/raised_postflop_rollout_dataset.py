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
from apc.full_hand_table import HeadsUpVirtualHand
from apc.self_learning.paired_rollout_dataset import _bb, _hand_class
from apc.self_learning.postflop_paired_rollout_dataset import STREETS, _sha256_file, _texture
from apc.self_learning.postflop_policy_rollout_dataset import OPPONENT_POLICIES, _finish
from apc.self_learning.postflop_position_rollout_dataset import HERO_POSITIONS
from apc.self_learning.replay_dataset import _canonical, _sha256_bytes, _split
from apc.self_learning.train_action_value import action_issues


PREFLOP_OPEN_TO_BB = Decimal("2.5")
NODE_FAMILIES = ("lead", "facing_33", "facing_75")
LEAD_BET_FRACTIONS = {
    "bet_33": Decimal("0.33"),
    "bet_67": Decimal("0.67"),
    "bet_100": Decimal("1"),
}
FACING_BET_FRACTIONS = {
    "facing_33": Decimal("0.33"),
    "facing_75": Decimal("0.75"),
}
ACTION_KEYS_BY_NODE = {
    "lead": ("check", *LEAD_BET_FRACTIONS),
    "facing_33": ("fold", "call", "raise_min", "raise_3x"),
    "facing_75": ("fold", "call", "raise_min", "raise_3x"),
}


def _buttons(state: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(row.get("action")): row
        for row in state.get("action_buttons", [])
        if isinstance(row, dict)
    }


def _raised_trunk(seed: int, hero_position: str) -> HeadsUpVirtualHand:
    if hero_position not in HERO_POSITIONS:
        raise ValueError("hero_position must be BTN or BB")
    hand = HeadsUpVirtualHand(
        seed=seed, button_player=0 if hero_position == "BTN" else 1
    )
    hand.step(ActionCommand("raise", to_amount_bb=_bb(PREFLOP_OPEN_TO_BB)))
    hand.step(ActionCommand("call"))
    if hand.street != "flop" or hand.pot != Decimal("5"):
        raise ValueError("raised rollout trunk did not reach a 5 BB flop pot")
    return hand


def _pot_bet_target(
    state: dict[str, object], fraction: Decimal, action: str = "bet"
) -> Decimal:
    buttons = _buttons(state)
    if action not in buttons:
        raise ValueError(f"visible state does not expose {action}")
    minimum = Decimal(str(buttons[action]["minimum_to_bb"]))
    maximum = Decimal(str(buttons[action]["maximum_to_bb"]))
    desired = Decimal(str(state["pot_bb"])) * fraction
    return min(max(desired, minimum), maximum)


def _node(
    base: HeadsUpVirtualHand, hero_position: str, node_family: str
) -> tuple[HeadsUpVirtualHand, dict[str, object], dict[str, object]]:
    if node_family not in NODE_FAMILIES:
        raise ValueError("unsupported raised-postflop node family")
    hand = copy.deepcopy(base)
    metadata: dict[str, object] = {
        "node_family": node_family,
        "forced_opponent_bet_pot_fraction": None,
        "forced_opponent_bet_to_bb": None,
    }
    if node_family == "lead":
        if hero_position == "BTN":
            hand.step(ActionCommand("check"))
    else:
        if hero_position == "BB":
            hand.step(ActionCommand("check"))
        if hand.next_actor == 0:
            raise ValueError("facing-bet trunk expected the opponent to act")
        before = hand.observation()
        fraction = FACING_BET_FRACTIONS[node_family]
        target = _pot_bet_target(before, fraction)
        hand.step(ActionCommand("bet", to_amount_bb=_bb(target)))
        metadata["forced_opponent_bet_pot_fraction"] = _bb(fraction)
        metadata["forced_opponent_bet_to_bb"] = _bb(target)
    state = hand.observation()
    if (
        state["hero_position"] != hero_position
        or state["next_actor"] != "Hero"
        or state["street"] not in STREETS
    ):
        raise ValueError("raised-postflop node did not reach the expected Hero state")
    if node_family == "lead" and Decimal(str(state["to_call_bb"])) != 0:
        raise ValueError("lead node cannot face a bet")
    if node_family != "lead" and Decimal(str(state["to_call_bb"])) <= 0:
        raise ValueError("facing node must expose a positive call price")
    return hand, state, metadata


def _commands(
    state: dict[str, object], node_family: str
) -> dict[str, ActionCommand]:
    buttons = _buttons(state)
    if node_family == "lead":
        if "check" not in buttons or "bet" not in buttons:
            raise ValueError("lead node requires check and bet buttons")
        commands = {"check": ActionCommand("check")}
        for key, fraction in LEAD_BET_FRACTIONS.items():
            commands[key] = ActionCommand(
                "bet", to_amount_bb=_bb(_pot_bet_target(state, fraction))
            )
    else:
        if not {"fold", "call", "raise"}.issubset(buttons):
            raise ValueError("facing node requires fold, call and raise buttons")
        minimum = Decimal(str(buttons["raise"]["minimum_to_bb"]))
        maximum = Decimal(str(buttons["raise"]["maximum_to_bb"]))
        opponent_to = max(
            Decimal(str(value))
            for value in state["street_contributions_bb"].values()
        )
        three_x = min(max(opponent_to * 3, minimum), maximum)
        if three_x == minimum:
            raise ValueError("facing node cannot distinguish minimum and 3x raises")
        commands = {
            "fold": ActionCommand("fold"),
            "call": ActionCommand(
                "call", amount_bb=str(buttons["call"]["amount_bb"])
            ),
            "raise_min": ActionCommand("raise", to_amount_bb=_bb(minimum)),
            "raise_3x": ActionCommand("raise", to_amount_bb=_bb(three_x)),
        }
    if tuple(commands) != ACTION_KEYS_BY_NODE[node_family]:
        raise ValueError("raised-postflop command order/coverage is invalid")
    return commands


def _advance_checkdown(hand: HeadsUpVirtualHand) -> None:
    hand.step(ActionCommand("check"))
    hand.step(ActionCommand("check"))


def raised_postflop_rollout(
    seed: int, *, split_seed: int = 20260822
) -> list[dict[str, object]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    group_id = f"raised-postflop-{seed:012d}"
    split = _split(group_id, split_seed, Decimal("0.70"), Decimal("0.15"))
    rows = []
    for hero_position in HERO_POSITIONS:
        base = _raised_trunk(seed, hero_position)
        for street in STREETS:
            if base.street != street:
                raise ValueError("raised rollout base street drifted")
            for node_family in NODE_FAMILIES:
                node_hand, state, node_metadata = _node(
                    base, hero_position, node_family
                )
                commands = _commands(state, node_family)
                for policy in OPPONENT_POLICIES:
                    for action_key, command in commands.items():
                        hand = copy.deepcopy(node_hand)
                        first, terminal, continuation = _finish(
                            hand, command, policy
                        )
                        outcome = terminal["completed_hand_feedback"]
                        row = {
                            "schema_version": "1.0.0",
                            "example_id": (
                                f"{group_id}-{hero_position}-{street}-"
                                f"{node_family}-{policy}-{action_key}"
                            ),
                            "group_id": group_id,
                            "state_id": (
                                f"{group_id}-{hero_position}-{street}-{node_family}"
                            ),
                            "policy_state_id": (
                                f"{group_id}-{hero_position}-{street}-"
                                f"{node_family}-{policy}"
                            ),
                            "split": split,
                            "units": "BB",
                            "street": street,
                            "hero_position": hero_position,
                            "node_family": node_family,
                            "counterfactual_action_key": action_key,
                            "state": state,
                            "opponent_policy": policy,
                            "counterfactual_action": first["command"],
                            "continuation_history": continuation,
                            "learning_signal": {
                                "kind": (
                                    "raised_postflop_position_policy_matched_"
                                    "terminal_return"
                                ),
                                "hero_return_bb": _bb(
                                    outcome["rewards_bb"]["Hero"]
                                ),
                                "solver_target": False,
                                "gto_verified": False,
                            },
                            "provenance": {
                                "environment": "controlled_virtual_chips",
                                "engine": "heads_up_virtual_hand_v1",
                                "hand_seed": seed,
                                "hero_hand_class": _hand_class(
                                    state["hero_cards"]
                                ),
                                "public_texture_class": _texture(state),
                                "common_random_cards_group": group_id,
                                "matched_position_group": group_id,
                                "raised_preflop_pot": True,
                                "preflop_open_to_bb": _bb(PREFLOP_OPEN_TO_BB),
                                "pre_state_fingerprint": state[
                                    "state_fingerprint"
                                ],
                                **node_metadata,
                                "external_actuation": False,
                            },
                        }
                        row["example_sha256"] = _sha256_bytes(_canonical(row))
                        rows.append(row)
            _advance_checkdown(base)
    return rows


def _paired_stat(
    states: list[dict[str, float]], action: str, baseline: str
) -> dict[str, object]:
    differences = [row[action] - row[baseline] for row in states]
    action_values = [row[action] for row in states]
    baseline_values = [row[baseline] for row in states]
    paired_se = math.sqrt(statistics.pvariance(differences) / len(states))
    unpaired_se = math.sqrt(
        (
            statistics.pvariance(action_values)
            + statistics.pvariance(baseline_values)
        )
        / len(states)
    )
    return {
        "samples": len(states),
        "mean_difference_bb": format(statistics.fmean(differences), ".12g"),
        "paired_standard_error_bb": format(paired_se, ".12g"),
        "unpaired_standard_error_bb": format(unpaired_se, ".12g"),
        "standard_error_reduction_fraction": format(
            0 if unpaired_se == 0 else 1 - paired_se / unpaired_se, ".12g"
        ),
    }


def _audits(
    rows: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, int]]:
    values: defaultdict[
        tuple[str, str, str, str], dict[str, dict[str, float]]
    ] = defaultdict(dict)
    opponent_actions: Counter[str] = Counter()
    for row in rows:
        key = (
            str(row["hero_position"]),
            str(row["street"]),
            str(row["node_family"]),
            str(row["opponent_policy"]),
        )
        state_values = values[key].setdefault(str(row["state_id"]), {})
        state_values[str(row["counterfactual_action_key"])] = float(
            str(row["learning_signal"]["hero_return_bb"])
        )
        opponent_position = "BB" if row["hero_position"] == "BTN" else "BTN"
        for history in row["continuation_history"]:
            if str(history).startswith(opponent_position + " "):
                token = str(history).split(" ", 1)[1].split(":", 1)[0]
                if token == "raise_to":
                    token = "raise"
                opponent_actions[
                    f"{row['hero_position']}:{row['opponent_policy']}:{token}"
                ] += 1
    comparisons: dict[str, object] = {}
    for position in HERO_POSITIONS:
        comparisons[position] = {}
        for policy in OPPONENT_POLICIES:
            comparisons[position][policy] = {}
            for street in STREETS:
                comparisons[position][policy][street] = {}
                for node_family in NODE_FAMILIES:
                    state_rows = list(
                        values[(position, street, node_family, policy)].values()
                    )
                    baseline = "check" if node_family == "lead" else "call"
                    comparisons[position][policy][street][node_family] = {
                        f"{action}_minus_{baseline}": _paired_stat(
                            state_rows, action, baseline
                        )
                        for action in ACTION_KEYS_BY_NODE[node_family]
                        if action != baseline
                    }
    return {"paired_comparisons": comparisons}, dict(
        sorted(opponent_actions.items())
    )


def build_raised_postflop_dataset(
    output: str | Path,
    *,
    dataset_id: str,
    rollouts: int = 600,
    hand_seed_start: int = 60000,
    split_seed: int = 20260822,
    minimum_rollouts: int = 300,
    minimum_texture_classes: int = 20,
) -> dict[str, object]:
    if not dataset_id or any(character in dataset_id for character in "\\/:*?\"<>|"):
        raise ValueError("dataset_id must be a non-empty portable name")
    if rollouts < 20 or minimum_rollouts <= 0 or minimum_texture_classes <= 0:
        raise ValueError("rollouts must be at least 20 and minimums must be positive")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(
            f"raised postflop dataset destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        row
        for offset in range(rollouts)
        for row in raised_postflop_rollout(
            hand_seed_start + offset, split_seed=split_seed
        )
    ]
    examples.sort(key=lambda row: str(row["example_id"]))
    groups = {str(row["group_id"]) for row in examples}
    textures = {str(row["provenance"]["public_texture_class"]) for row in examples}
    split_counts = Counter(str(row["split"]) for row in examples)
    audits, opponent_actions = _audits(examples)
    required_selective = {
        f"{position}:made_hand_selective:{action}"
        for position in HERO_POSITIONS
        for action in ("fold", "call", "bet", "raise")
    }
    reasons = []
    if len(groups) < minimum_rollouts:
        reasons.append("minimum_rollouts_not_met")
    if len(textures) < minimum_texture_classes:
        reasons.append("minimum_texture_classes_not_met")
    if not all(split_counts.get(split, 0) for split in ("train", "validation", "test")):
        reasons.append("train_validation_test_splits_must_all_be_nonempty")
    if not required_selective.issubset(opponent_actions):
        reasons.append(
            "selective_policy_did_not_cover_fold_call_bet_raise_in_both_positions"
        )
    eligible = not reasons
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        root = Path(temporary)
        examples_bytes = b"".join(_canonical(row) + b"\n" for row in examples)
        (root / "examples.jsonl").write_bytes(examples_bytes)
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": dataset_id,
            "dataset_kind": "raised_postflop_position_multi_policy_paired_rollouts",
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
            "states_per_rollout": (
                len(HERO_POSITIONS) * len(STREETS) * len(NODE_FAMILIES)
            ),
            "policies_per_state": len(OPPONENT_POLICIES),
            "actions_per_policy_state": 4,
            "public_texture_class_count": len(textures),
            "hero_positions": list(HERO_POSITIONS),
            "streets": list(STREETS),
            "node_families": list(NODE_FAMILIES),
            "hero_action_keys_by_node": {
                node: list(actions) for node, actions in ACTION_KEYS_BY_NODE.items()
            },
            "opponent_policies": list(OPPONENT_POLICIES),
            "opponent_action_counts": opponent_actions,
            "split_counts": {
                key: split_counts.get(key, 0)
                for key in ("train", "validation", "test")
            },
            "group_exclusive": True,
            "position_card_matched": True,
            "raised_preflop_pot": True,
            "generation": {
                "hand_seed_start": hand_seed_start,
                "hand_seed_end": hand_seed_start + rollouts - 1,
                "split_seed": split_seed,
                "preflop_open_to_bb": _bb(PREFLOP_OPEN_TO_BB),
                "preflop_response": "call",
                "lead_bet_pot_fractions": {
                    key: _bb(value) for key, value in LEAD_BET_FRACTIONS.items()
                },
                "facing_bet_pot_fractions": {
                    key: _bb(value) for key, value in FACING_BET_FRACTIONS.items()
                },
                "facing_raise_sizes": ["minimum_legal", "three_times_bet_to"],
                "future_hero_continuation": "check_call",
                "common_random_numbers": (
                    "same_cards_across_positions_nodes_policies_and_actions"
                ),
            },
            **audits,
            "source_fingerprints": {
                "full_hand_engine_sha256": _sha256_file(
                    Path(__file__).resolve().parents[1] / "full_hand_table.py"
                ),
                "opponent_policy_sha256": _sha256_file(
                    Path(__file__).resolve().parent
                    / "postflop_policy_rollout_dataset.py"
                ),
            },
            "limitations": [
                "Opponent policies are deterministic probes, not learned population models or GTO strategies.",
                "The corpus covers one 2.5 BB heads-up preflop raise/call trunk with no rake or side pots.",
                "Bet fractions are fixed experimental sizes and terminal returns are sampled virtual-chip outcomes, not solver counterfactual values.",
            ],
        }
        material = dict(manifest)
        manifest["dataset_fingerprint"] = _sha256_bytes(_canonical(material))
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        validation = validate_raised_postflop_dataset(root)
        if not validation["valid"]:
            raise ValueError(
                "generated raised postflop dataset failed validation: "
                + "; ".join(validation["issues"])
            )
        root.replace(destination)
    return manifest


def validate_raised_postflop_dataset(root: str | Path) -> dict[str, object]:
    dataset = Path(root).resolve()
    issues = []
    try:
        manifest = json.loads(
            (dataset / "manifest.json").read_text(encoding="utf-8")
        )
        examples_path = dataset / str(manifest["examples_file"])
        examples = [
            json.loads(line)
            for line in examples_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        return {
            "schema_version": "1.0.0",
            "valid": False,
            "issues": [f"dataset unreadable: {error}"],
        }
    if (
        manifest.get("dataset_kind")
        != "raised_postflop_position_multi_policy_paired_rollouts"
        or manifest.get("immutable") is not True
        or manifest.get("units") != "BB"
        or manifest.get("group_exclusive") is not True
        or manifest.get("position_card_matched") is not True
        or manifest.get("raised_preflop_pot") is not True
    ):
        issues.append("manifest raised-pot immutable BB/group contract is invalid")
    if manifest.get("policy_promotion_eligible") is not False:
        issues.append("dataset cannot authorize promotion")
    if (
        manifest.get("hero_positions") != list(HERO_POSITIONS)
        or manifest.get("streets") != list(STREETS)
        or manifest.get("node_families") != list(NODE_FAMILIES)
        or manifest.get("opponent_policies") != list(OPPONENT_POLICIES)
        or manifest.get("hero_action_keys_by_node")
        != {node: list(actions) for node, actions in ACTION_KEYS_BY_NODE.items()}
    ):
        issues.append("manifest position/street/node/action/policy coverage is invalid")
    if (
        manifest.get("example_count") != len(examples)
        or manifest.get("examples_sha256") != _sha256_file(examples_path)
    ):
        issues.append("manifest count or examples fingerprint mismatch")
    material = dict(manifest)
    observed_fingerprint = material.pop("dataset_fingerprint", None)
    if observed_fingerprint != _sha256_bytes(_canonical(material)):
        issues.append("dataset fingerprint mismatch")

    ids = set()
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    states: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    policy_states: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(examples):
        label = f"example[{index}]"
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in ids:
            issues.append(f"{label} identity is missing or duplicated")
        ids.add(example_id)
        state = row.get("state", {})
        position = row.get("hero_position")
        node_family = row.get("node_family")
        action_key = row.get("counterfactual_action_key")
        if (
            row.get("units") != "BB"
            or position not in HERO_POSITIONS
            or node_family not in NODE_FAMILIES
            or action_key not in ACTION_KEYS_BY_NODE.get(str(node_family), ())
            or state.get("hero_position") != position
            or state.get("next_actor") != "Hero"
            or state.get("opponent_cards") is not None
        ):
            issues.append(f"{label} position/node/private/Hero contract is invalid")
        try:
            to_call = Decimal(str(state["to_call_bb"]))
            pot = Decimal(str(state["pot_bb"]))
        except (InvalidOperation, KeyError):
            issues.append(f"{label} pot/call state is invalid")
        else:
            if pot <= 0 or (node_family == "lead") != (to_call == 0):
                issues.append(f"{label} node family does not match visible call price")
        history = state.get("action_history", [])
        if (
            not isinstance(history, list)
            or not any("raise_to:2.5" in str(action) for action in history)
            or not any(str(action).endswith(" call") for action in history)
        ):
            issues.append(f"{label} raised preflop trunk evidence is missing")
        command = row.get("counterfactual_action")
        if not isinstance(command, dict) or action_issues(state, command):
            issues.append(f"{label} counterfactual action is not legal")
        elif node_family == "lead":
            expected_action = "check" if action_key == "check" else "bet"
            if command.get("action") != expected_action:
                issues.append(f"{label} lead action key/command mismatch")
        else:
            expected_action = (
                "raise" if str(action_key).startswith("raise_") else action_key
            )
            if command.get("action") != expected_action:
                issues.append(f"{label} facing action key/command mismatch")
        if (
            row.get("opponent_policy") not in OPPONENT_POLICIES
            or not isinstance(row.get("continuation_history"), list)
        ):
            issues.append(f"{label} policy/continuation evidence is invalid")
        signal = row.get("learning_signal", {})
        if (
            signal.get("kind")
            != "raised_postflop_position_policy_matched_terminal_return"
            or signal.get("solver_target") is not False
            or signal.get("gto_verified") is not False
        ):
            issues.append(f"{label} learning-signal scope is invalid")
        try:
            value = Decimal(str(signal["hero_return_bb"]))
        except (InvalidOperation, KeyError):
            issues.append(f"{label} return is invalid")
        else:
            if not value.is_finite() or value < -100 or value > 100:
                issues.append(f"{label} return is invalid")
        provenance = row.get("provenance", {})
        if (
            provenance.get("external_actuation") is not False
            or provenance.get("raised_preflop_pot") is not True
            or provenance.get("preflop_open_to_bb") != "2.5"
            or provenance.get("node_family") != node_family
        ):
            issues.append(f"{label} provenance is invalid")
        expected_hash = row.get("example_sha256")
        row_material = dict(row)
        row_material.pop("example_sha256", None)
        if expected_hash != _sha256_bytes(_canonical(row_material)):
            issues.append(f"{label} fingerprint mismatch")
        groups[str(row.get("group_id", ""))].append(row)
        states[str(row.get("state_id", ""))].append(row)
        policy_states[str(row.get("policy_state_id", ""))].append(row)
        split_counts[str(row.get("split", ""))] += 1

    expected_group_rows = (
        len(HERO_POSITIONS)
        * len(STREETS)
        * len(NODE_FAMILIES)
        * len(OPPONENT_POLICIES)
        * 4
    )
    for group_id, group_rows in groups.items():
        if (
            len(group_rows) != expected_group_rows
            or len({str(row["split"]) for row in group_rows}) != 1
        ):
            issues.append(f"hand group {group_id} is incomplete or split-leaked")
        for street in STREETS:
            for node_family in NODE_FAMILIES:
                position_rows = {
                    str(row["hero_position"]): row
                    for row in group_rows
                    if row["street"] == street
                    and row["node_family"] == node_family
                    and row["opponent_policy"] == OPPONENT_POLICIES[0]
                    and row["counterfactual_action_key"]
                    == ACTION_KEYS_BY_NODE[node_family][0]
                }
                known_cards = {
                    tuple([*row["state"]["hero_cards"], *row["state"]["board"]])
                    for row in position_rows.values()
                }
                if set(position_rows) != set(HERO_POSITIONS) or len(known_cards) != 1:
                    issues.append(
                        f"hand group {group_id} {street}/{node_family} is not card-matched"
                    )
    for state_id, state_rows in states.items():
        if (
            len(state_rows) != len(OPPONENT_POLICIES) * 4
            or len(
                {
                    str(row["provenance"]["pre_state_fingerprint"])
                    for row in state_rows
                }
            )
            != 1
        ):
            issues.append(f"raised state {state_id} is incomplete or not common-state")
    for policy_state_id, state_rows in policy_states.items():
        node_family = str(state_rows[0].get("node_family", "")) if state_rows else ""
        if (
            len(state_rows) != 4
            or {str(row["counterfactual_action_key"]) for row in state_rows}
            != set(ACTION_KEYS_BY_NODE.get(node_family, ()))
        ):
            issues.append(f"policy state {policy_state_id} is incomplete")
    expected_counts = {
        key: split_counts.get(key, 0)
        for key in ("train", "validation", "test")
    }
    if (
        manifest.get("split_counts") != expected_counts
        or manifest.get("paired_rollout_count") != len(groups)
    ):
        issues.append("manifest split/group counts do not match examples")
    try:
        recomputed_audits, recomputed_actions = (
            _audits(examples) if examples else ({}, {})
        )
    except (KeyError, ValueError, ZeroDivisionError, statistics.StatisticsError):
        issues.append("paired/action audit cannot be recomputed from malformed examples")
    else:
        if manifest.get("paired_comparisons") != recomputed_audits.get(
            "paired_comparisons"
        ) or manifest.get("opponent_action_counts") != recomputed_actions:
            issues.append("manifest paired/action audit does not match examples")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "example_count": len(examples),
        "paired_rollout_count": len(groups),
        "state_count": len(states),
        "policy_state_count": len(policy_states),
        "split_counts": expected_counts,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate APC raised-pot multi-sizing postflop data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("output", type=Path)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--rollouts", type=int, default=600)
    build.add_argument("--hand-seed-start", type=int, default=60000)
    build.add_argument("--split-seed", type=int, default=20260822)
    build.add_argument("--minimum-rollouts", type=int, default=300)
    build.add_argument("--minimum-texture-classes", type=int, default=20)
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_raised_postflop_dataset(args.dataset)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        manifest = build_raised_postflop_dataset(
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
