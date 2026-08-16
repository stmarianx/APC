from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from apc.full_hand_table import _coach_types
from apc.self_learning.train_postflop_policy_value import (
    OPPONENT_POLICIES,
    SUPPORTED_ACTIONS,
    _prediction,
    _load,
    validate_postflop_policy_value_checkpoint,
)
from apc.self_learning.train_value import _sha256, value_state_issues


def _mixture_validation(payload: dict[str, object]) -> dict[str, object]:
    _coach_types()
    from poker_coach.opponent_model import validate_opponent_policy_mixture

    return validate_opponent_policy_mixture(payload)


def _commands(state: dict[str, object]) -> dict[str, dict[str, object]]:
    buttons = {
        str(row.get("action")): row
        for row in state.get("action_buttons", [])
        if isinstance(row, dict)
    }
    if "check" not in buttons or "bet" not in buttons:
        raise ValueError("profile-conditioned postflop inference requires visible check and bet buttons")
    try:
        minimum = str(buttons["bet"]["minimum_to_bb"])
    except KeyError as error:
        raise ValueError("visible bet button has no minimum BB target") from error
    return {"check": {"action": "check"}, "bet": {"action": "bet", "to_amount_bb": minimum}}


def _bounded_value(
    policy_values: dict[str, float],
    intervals: dict[str, list[object]],
    *,
    maximize: bool,
) -> float:
    bounds = {
        policy: (float(str(intervals[policy][0])), float(str(intervals[policy][1])))
        for policy in OPPONENT_POLICIES
    }
    weights = {policy: bounds[policy][0] for policy in OPPONENT_POLICIES}
    remaining = 1.0 - sum(weights.values())
    if remaining < -1e-8 or sum(bounds[policy][1] for policy in OPPONENT_POLICIES) < 1 - 1e-8:
        raise ValueError("mixture uncertainty bounds cannot form a probability simplex")
    order = sorted(OPPONENT_POLICIES, key=lambda policy: policy_values[policy], reverse=maximize)
    for policy in order:
        increment = min(max(0.0, bounds[policy][1] - weights[policy]), max(0.0, remaining))
        weights[policy] += increment
        remaining -= increment
    if remaining > 1e-7:
        raise ValueError("mixture uncertainty bounds do not cover total probability one")
    return sum(weights[policy] * policy_values[policy] for policy in OPPONENT_POLICIES)


def predict_profile_conditioned_postflop(
    checkpoint_or_path: dict[str, object] | str | Path,
    state: dict[str, object],
    mixture: dict[str, object],
) -> dict[str, object]:
    checkpoint = checkpoint_or_path if isinstance(checkpoint_or_path, dict) else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    checkpoint_validation = validate_postflop_policy_value_checkpoint(checkpoint)
    if not checkpoint_validation["valid"]:
        raise ValueError("postflop checkpoint is invalid: " + "; ".join(checkpoint_validation["issues"]))
    mixture_validation = _mixture_validation(mixture)
    if not mixture_validation["valid"]:
        raise ValueError("opponent policy mixture is invalid: " + "; ".join(mixture_validation["issues"]))
    state_issues = value_state_issues(state)
    if state_issues or state.get("street") not in checkpoint["configuration"]["supported_streets"]:
        return {
            "schema_version": "1.0.0",
            "status": "abstain_unsupported_or_invalid_state",
            "reasons": sorted(set([*state_issues, "street_not_supported"] if state.get("street") not in checkpoint["configuration"]["supported_streets"] else state_issues)),
            "profile_conditioned_action": None,
            "recommendation_allowed": False,
            "activation_authorized": False,
        }
    try:
        commands = _commands(state)
    except ValueError as error:
        return {
            "schema_version": "1.0.0",
            "status": "abstain_unsupported_or_invalid_state",
            "reasons": [str(error)],
            "profile_conditioned_action": None,
            "recommendation_allowed": False,
            "activation_authorized": False,
        }
    weights = {policy: float(str(mixture["opponent_policy_weights"][policy])) for policy in OPPONENT_POLICIES}
    intervals = mixture["weight_uncertainty_approximate_95"]
    policy_values: dict[str, dict[str, float]] = {}
    central_values: dict[str, float] = {}
    value_bounds: dict[str, tuple[float, float]] = {}
    coverage: dict[str, dict[str, bool]] = {}
    for action in SUPPORTED_ACTIONS:
        policy_values[action] = {}
        coverage[action] = {}
        for policy in OPPONENT_POLICIES:
            row = {"street": state["street"], "opponent_policy": policy, "counterfactual_action": commands[action], "state": state}
            value, exact, _ = _prediction(
                checkpoint["street_policy_action_train_means_bb"],
                checkpoint["abstraction_action_table"],
                row,
                checkpoint["configuration"]["selected_feature_family_by_validation"],
                float(checkpoint["configuration"]["selected_shrinkage_by_validation"]),
            )
            policy_values[action][policy] = value
            coverage[action][policy] = exact
        central_values[action] = sum(weights[policy] * policy_values[action][policy] for policy in OPPONENT_POLICIES)
        value_bounds[action] = (
            _bounded_value(policy_values[action], intervals, maximize=False),
            _bounded_value(policy_values[action], intervals, maximize=True),
        )
    selected = max(SUPPORTED_ACTIONS, key=lambda action: (central_values[action], action == "check"))
    alternative = next(action for action in SUPPORTED_ACTIONS if action != selected)
    advantage_by_policy = {
        policy: policy_values[selected][policy] - policy_values[alternative][policy]
        for policy in OPPONENT_POLICIES
    }
    robust_advantage_lower = _bounded_value(advantage_by_policy, intervals, maximize=False)
    evidence_passed = mixture["evidence_gate"]["passed"] is True
    abstraction_complete = all(value for actions in coverage.values() for value in actions.values())
    robust = robust_advantage_lower >= 0
    if not evidence_passed:
        status = "profile_evidence_observe_only"
        conditioned_action = None
    elif not abstraction_complete:
        status = "abstain_incomplete_value_abstraction"
        conditioned_action = None
    elif not robust:
        status = "abstain_profile_mixture_uncertainty_changes_action"
        conditioned_action = None
    else:
        status = "offline_profile_conditioned_action_stable_not_recommendation"
        conditioned_action = commands[selected]
    result = {
        "schema_version": "1.0.0",
        "status": status,
        "units": "BB",
        "street": state["street"],
        "profile_key": mixture["profile_key"],
        "profile_conditioned_action": conditioned_action,
        "central_action_values_bb": {action: format(central_values[action], ".12g") for action in SUPPORTED_ACTIONS},
        "uncertainty_action_value_bounds_bb": {
            action: [format(value_bounds[action][0], ".12g"), format(value_bounds[action][1], ".12g")]
            for action in SUPPORTED_ACTIONS
        },
        "selected_minus_alternative_central_bb": format(central_values[selected] - central_values[alternative], ".12g"),
        "selected_minus_alternative_uncertainty_lower_bb": format(robust_advantage_lower, ".12g"),
        "policy_action_values_bb": {
            action: {policy: format(policy_values[action][policy], ".12g") for policy in OPPONENT_POLICIES}
            for action in SUPPORTED_ACTIONS
        },
        "exact_abstraction_coverage": coverage,
        "evidence_gate_passed": evidence_passed,
        "uncertainty_stable_action": robust,
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "mixture_fingerprint": mixture["mixture_fingerprint"],
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
        "external_actuation": False,
        "limitations": [
            "The output combines synthetic probe-policy values and is not a solver/GTO recommendation.",
            "A profile-conditioned action is exposed only for offline evaluation when evidence, abstraction coverage and uncertainty stability all pass.",
            "Confidence, recommendation, activation and external actuation remain prohibited.",
        ],
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def evaluate_profile_conditioned_latency(
    checkpoint: dict[str, object] | str | Path,
    state: dict[str, object],
    mixture: dict[str, object],
    *,
    repetitions: int = 200,
) -> dict[str, object]:
    if repetitions < 20:
        raise ValueError("latency evaluation needs at least 20 repetitions")
    loaded = checkpoint if isinstance(checkpoint, dict) else json.loads(Path(checkpoint).read_text(encoding="utf-8"))
    durations = []
    prediction_fingerprints = set()
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = predict_profile_conditioned_postflop(loaded, state, mixture)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
        prediction_fingerprints.add(result.get("prediction_fingerprint"))
    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "schema_version": "1.0.0",
        "repetitions": repetitions,
        "p50_ms": format(statistics.median(durations), ".12g"),
        "p95_ms": format(p95, ".12g"),
        "maximum_ms": format(max(durations), ".12g"),
        "deterministic_prediction": len(prediction_fingerprints) == 1,
        "latency_gate": {"passed": p95 <= 5.0, "threshold_p95_ms": "5"},
        "checkpoint_fingerprint": loaded["checkpoint_fingerprint"],
        "mixture_fingerprint": mixture["mixture_fingerprint"],
    }


def evaluate_profile_conditioned_dataset(
    dataset: str | Path,
    checkpoint_or_path: dict[str, object] | str | Path,
    mixtures: dict[str, dict[str, object]],
) -> dict[str, object]:
    if len(mixtures) < 3:
        raise ValueError("profile-conditioned audit requires at least three profile mixtures")
    checkpoint = checkpoint_or_path if isinstance(checkpoint_or_path, dict) else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    manifest, rows = _load(Path(dataset).resolve())
    if checkpoint.get("dataset", {}).get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
        raise ValueError("profile-conditioned audit dataset does not match checkpoint")
    unique_states = {
        str(row["state_id"]): row["state"]
        for row in rows
        if row["split"] == "test" and row["counterfactual_action"]["action"] == "check"
    }
    status_counts: dict[str, dict[str, int]] = {}
    action_counts: dict[str, dict[str, int]] = {}
    stable_coverage: dict[str, str] = {}
    durations = []
    all_non_authorizing = True
    fingerprints: dict[str, list[str]] = {}
    for label, mixture in sorted(mixtures.items()):
        if not _mixture_validation(mixture)["valid"] or mixture["evidence_gate"]["passed"] is not True:
            raise ValueError(f"audit mixture {label} is invalid or lacks evidence")
        statuses: dict[str, int] = {}
        actions: dict[str, int] = {}
        observed_fingerprints = []
        for state in unique_states.values():
            started = time.perf_counter_ns()
            result = predict_profile_conditioned_postflop(checkpoint, state, mixture)
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
            status = str(result["status"])
            statuses[status] = statuses.get(status, 0) + 1
            command = result.get("profile_conditioned_action")
            if isinstance(command, dict):
                action = str(command["action"])
                actions[action] = actions.get(action, 0) + 1
            all_non_authorizing = all_non_authorizing and all(
                result.get(key) is False
                for key in ("confidence_calibrated", "recommendation_allowed", "activation_authorized", "external_actuation")
            )
            observed_fingerprints.append(str(result["prediction_fingerprint"]))
        stable = statuses.get("offline_profile_conditioned_action_stable_not_recommendation", 0)
        status_counts[label] = dict(sorted(statuses.items()))
        action_counts[label] = dict(sorted(actions.items()))
        stable_coverage[label] = format(stable / len(unique_states), ".12g")
        fingerprints[label] = observed_fingerprints
    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    gate = (
        all(float(value) >= 0.50 for value in stable_coverage.values())
        and all_non_authorizing
        and p95 <= 5.0
        and len({str(mixture["mixture_fingerprint"]) for mixture in mixtures.values()}) == len(mixtures)
    )
    result = {
        "schema_version": "1.0.0",
        "status": "offline_profile_conditioned_audit_passed" if gate else "offline_profile_conditioned_audit_failed",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "test_complete_hands": len({str(row["group_id"]) for row in rows if row["split"] == "test"}),
        "test_visible_states": len(unique_states),
        "profile_scenarios": len(mixtures),
        "profile_state_evaluations": len(durations),
        "stable_action_coverage": stable_coverage,
        "status_counts": status_counts,
        "stable_action_counts": action_counts,
        "all_outputs_non_authorizing": all_non_authorizing,
        "latency_ms": {
            "p50": format(statistics.median(durations), ".12g"),
            "p95": format(p95, ".12g"),
            "maximum": format(max(durations), ".12g"),
            "threshold_p95": "5",
        },
        "gate": {
            "passed": gate,
            "minimum_stable_action_coverage_per_evidenced_profile": "0.50",
            "maximum_p95_ms": "5",
            "requires_non_authorizing_outputs": True,
        },
        "mixture_fingerprints": {
            label: mixture["mixture_fingerprint"] for label, mixture in sorted(mixtures.items())
        },
        "limitations": [
            "Archetypes are constructed Beta-posterior profiles and do not prove calibration on real player populations.",
            "The audit uses the same three synthetic probe-policy families represented in value training.",
            "Stable actions remain offline-only and are not solver/GTO recommendations.",
        ],
    }
    result["audit_fingerprint"] = _sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate APC profile-conditioned postflop values offline.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("mixture", type=Path)
    parser.add_argument("--latency-repetitions", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        state = json.loads(args.state.read_text(encoding="utf-8"))
        mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
        if args.latency_repetitions:
            result = evaluate_profile_conditioned_latency(args.checkpoint, state, mixture, repetitions=args.latency_repetitions)
        else:
            result = predict_profile_conditioned_postflop(args.checkpoint, state, mixture)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
