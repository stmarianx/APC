from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from apc.self_learning.postflop_position_rollout_dataset import HERO_POSITIONS
from apc.self_learning.postflop_policy_rollout_dataset import OPPONENT_POLICIES
from apc.self_learning.profile_conditioned_postflop import (
    _bounded_value,
    _commands,
    _mixture_validation,
)
from apc.self_learning.train_position_postflop_value import (
    SUPPORTED_ACTIONS,
    _load,
    _prediction,
    validate_position_postflop_value_checkpoint,
)
from apc.self_learning.train_value import _sha256, value_state_issues


def _abstention(
    checkpoint: dict[str, object],
    mixture: dict[str, object],
    state: dict[str, object],
    status: str,
    reasons: list[str],
) -> dict[str, object]:
    result = {
        "schema_version": "1.0.0",
        "status": status,
        "units": "BB",
        "hero_position": state.get("hero_position"),
        "street": state.get("street"),
        "profile_key": mixture.get("profile_key"),
        "reasons": sorted(set(reasons)),
        "profile_conditioned_action": None,
        "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint"),
        "mixture_fingerprint": mixture.get("mixture_fingerprint"),
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
        "external_actuation": False,
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def predict_position_profile_conditioned_postflop(
    checkpoint_or_path: dict[str, object] | str | Path,
    state: dict[str, object],
    mixture: dict[str, object],
) -> dict[str, object]:
    checkpoint = (
        checkpoint_or_path
        if isinstance(checkpoint_or_path, dict)
        else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    )
    checkpoint_validation = validate_position_postflop_value_checkpoint(checkpoint)
    if not checkpoint_validation["valid"]:
        raise ValueError(
            "position postflop checkpoint is invalid: "
            + "; ".join(checkpoint_validation["issues"])
        )
    mixture_validation = _mixture_validation(mixture)
    if not mixture_validation["valid"]:
        raise ValueError(
            "opponent policy mixture is invalid: "
            + "; ".join(mixture_validation["issues"])
        )

    issues = value_state_issues(state)
    if state.get("hero_position") not in HERO_POSITIONS:
        issues.append("hero_position_not_supported")
    if state.get("street") not in checkpoint["configuration"]["supported_streets"]:
        issues.append("street_not_supported")
    if issues:
        return _abstention(
            checkpoint,
            mixture,
            state,
            "abstain_unsupported_or_invalid_state",
            issues,
        )
    try:
        commands = _commands(state)
    except ValueError as error:
        return _abstention(
            checkpoint,
            mixture,
            state,
            "abstain_unsupported_or_invalid_state",
            [str(error)],
        )

    weights = {
        policy: float(str(mixture["opponent_policy_weights"][policy]))
        for policy in OPPONENT_POLICIES
    }
    intervals = mixture["weight_uncertainty_approximate_95"]
    policy_values: dict[str, dict[str, float]] = {}
    central_values: dict[str, float] = {}
    value_bounds: dict[str, tuple[float, float]] = {}
    coverage: dict[str, dict[str, bool]] = {}
    training_samples: dict[str, dict[str, int]] = {}
    for action in SUPPORTED_ACTIONS:
        policy_values[action] = {}
        coverage[action] = {}
        training_samples[action] = {}
        for policy in OPPONENT_POLICIES:
            row = {
                "hero_position": state["hero_position"],
                "street": state["street"],
                "opponent_policy": policy,
                "counterfactual_action": commands[action],
                "state": state,
            }
            value, exact, samples = _prediction(
                checkpoint["position_street_policy_action_train_means_bb"],
                checkpoint["abstraction_action_table"],
                row,
                checkpoint["configuration"]["selected_feature_family_by_validation"],
                float(checkpoint["configuration"]["selected_shrinkage_by_validation"]),
            )
            policy_values[action][policy] = value
            coverage[action][policy] = exact
            training_samples[action][policy] = samples
        central_values[action] = sum(
            weights[policy] * policy_values[action][policy]
            for policy in OPPONENT_POLICIES
        )
        value_bounds[action] = (
            _bounded_value(policy_values[action], intervals, maximize=False),
            _bounded_value(policy_values[action], intervals, maximize=True),
        )

    selected = max(
        SUPPORTED_ACTIONS,
        key=lambda action: (central_values[action], action == "check"),
    )
    alternative = next(action for action in SUPPORTED_ACTIONS if action != selected)
    advantage_by_policy = {
        policy: policy_values[selected][policy] - policy_values[alternative][policy]
        for policy in OPPONENT_POLICIES
    }
    robust_advantage_lower = _bounded_value(
        advantage_by_policy, intervals, maximize=False
    )
    evidence_passed = mixture["evidence_gate"]["passed"] is True
    generalization_passed = checkpoint["generalization_gate"]["passed"] is True
    abstraction_complete = all(
        exact for action_rows in coverage.values() for exact in action_rows.values()
    )
    robust = robust_advantage_lower >= 0
    if not generalization_passed:
        status = "abstain_position_value_generalization_gate"
        conditioned_action = None
    elif not evidence_passed:
        status = "profile_evidence_observe_only"
        conditioned_action = None
    elif not abstraction_complete:
        status = "abstain_incomplete_value_abstraction"
        conditioned_action = None
    elif not robust:
        status = "abstain_profile_mixture_uncertainty_changes_action"
        conditioned_action = None
    else:
        status = "offline_position_profile_action_stable_not_recommendation"
        conditioned_action = commands[selected]

    result = {
        "schema_version": "1.0.0",
        "status": status,
        "units": "BB",
        "hero_position": state["hero_position"],
        "street": state["street"],
        "profile_key": mixture["profile_key"],
        "profile_conditioned_action": conditioned_action,
        "central_action_values_bb": {
            action: format(central_values[action], ".12g")
            for action in SUPPORTED_ACTIONS
        },
        "uncertainty_action_value_bounds_bb": {
            action: [
                format(value_bounds[action][0], ".12g"),
                format(value_bounds[action][1], ".12g"),
            ]
            for action in SUPPORTED_ACTIONS
        },
        "selected_minus_alternative_central_bb": format(
            central_values[selected] - central_values[alternative], ".12g"
        ),
        "selected_minus_alternative_uncertainty_lower_bb": format(
            robust_advantage_lower, ".12g"
        ),
        "policy_action_values_bb": {
            action: {
                policy: format(policy_values[action][policy], ".12g")
                for policy in OPPONENT_POLICIES
            }
            for action in SUPPORTED_ACTIONS
        },
        "exact_abstraction_coverage": coverage,
        "training_samples": training_samples,
        "evidence_gate_passed": evidence_passed,
        "position_value_generalization_gate_passed": generalization_passed,
        "uncertainty_stable_action": robust,
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "mixture_fingerprint": mixture["mixture_fingerprint"],
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
        "external_actuation": False,
        "limitations": [
            "The output combines synthetic probe-policy values and is not a solver/GTO recommendation.",
            "An offline action is exposed only when profile evidence, the position-value gate, abstraction coverage and mixture-interval stability all pass.",
            "Only limped BTN/BB check and minimum-bet nodes are represented; confidence and external actuation remain prohibited.",
        ],
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def evaluate_position_profile_conditioned_latency(
    checkpoint: dict[str, object] | str | Path,
    state: dict[str, object],
    mixture: dict[str, object],
    *,
    repetitions: int = 200,
) -> dict[str, object]:
    if repetitions < 20:
        raise ValueError("latency evaluation needs at least 20 repetitions")
    loaded = (
        checkpoint
        if isinstance(checkpoint, dict)
        else json.loads(Path(checkpoint).read_text(encoding="utf-8"))
    )
    durations = []
    prediction_fingerprints = set()
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = predict_position_profile_conditioned_postflop(
            loaded, state, mixture
        )
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


def evaluate_position_profile_conditioned_dataset(
    dataset: str | Path,
    checkpoint_or_path: dict[str, object] | str | Path,
    mixtures: dict[str, dict[str, object]],
) -> dict[str, object]:
    if len(mixtures) < 3:
        raise ValueError("position-profile audit requires at least three mixtures")
    checkpoint = (
        checkpoint_or_path
        if isinstance(checkpoint_or_path, dict)
        else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    )
    validation = validate_position_postflop_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError(
            "position postflop checkpoint is invalid: "
            + "; ".join(validation["issues"])
        )
    manifest, rows = _load(Path(dataset).resolve())
    if checkpoint.get("dataset", {}).get("dataset_fingerprint") != manifest.get(
        "dataset_fingerprint"
    ):
        raise ValueError("position-profile audit dataset does not match checkpoint")
    unique_states = {
        str(row["state_id"]): row["state"]
        for row in rows
        if row["split"] == "test"
        and row["counterfactual_action"]["action"] == "check"
    }
    position_counts = {
        position: sum(
            state.get("hero_position") == position for state in unique_states.values()
        )
        for position in HERO_POSITIONS
    }
    if any(count <= 0 for count in position_counts.values()):
        raise ValueError("position-profile audit requires every supported position")

    status_counts: dict[str, dict[str, int]] = {}
    action_counts: dict[str, dict[str, int]] = {}
    stable_coverage: dict[str, str] = {}
    by_position_coverage: dict[str, dict[str, str]] = {}
    durations = []
    all_non_authorizing = True
    for label, mixture in sorted(mixtures.items()):
        mixture_validation = _mixture_validation(mixture)
        if not mixture_validation["valid"] or mixture["evidence_gate"]["passed"] is not True:
            raise ValueError(f"audit mixture {label} is invalid or lacks evidence")
        statuses: dict[str, int] = {}
        actions: dict[str, int] = {}
        stable_by_position = {position: 0 for position in HERO_POSITIONS}
        for state in unique_states.values():
            started = time.perf_counter_ns()
            result = predict_position_profile_conditioned_postflop(
                checkpoint, state, mixture
            )
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
            status = str(result["status"])
            statuses[status] = statuses.get(status, 0) + 1
            command = result.get("profile_conditioned_action")
            if isinstance(command, dict):
                action = str(command["action"])
                actions[action] = actions.get(action, 0) + 1
                stable_by_position[str(state["hero_position"])] += 1
            all_non_authorizing = all_non_authorizing and all(
                result.get(key) is False
                for key in (
                    "confidence_calibrated",
                    "recommendation_allowed",
                    "activation_authorized",
                    "external_actuation",
                )
            )
        stable = sum(stable_by_position.values())
        status_counts[label] = dict(sorted(statuses.items()))
        action_counts[label] = dict(sorted(actions.items()))
        stable_coverage[label] = format(stable / len(unique_states), ".12g")
        by_position_coverage[label] = {
            position: format(
                stable_by_position[position] / position_counts[position], ".12g"
            )
            for position in HERO_POSITIONS
        }

    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    unique_mixtures = len(
        {str(mixture["mixture_fingerprint"]) for mixture in mixtures.values()}
    ) == len(mixtures)
    gate = (
        checkpoint["generalization_gate"]["passed"] is True
        and all(
            float(by_position_coverage[label][position]) >= 0.50
            for label in by_position_coverage
            for position in HERO_POSITIONS
        )
        and all_non_authorizing
        and p95 <= 5.0
        and unique_mixtures
    )
    result = {
        "schema_version": "1.0.0",
        "status": (
            "offline_position_profile_conditioned_audit_passed"
            if gate
            else "offline_position_profile_conditioned_audit_failed"
        ),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "test_complete_hands": len(
            {str(row["group_id"]) for row in rows if row["split"] == "test"}
        ),
        "test_visible_states": len(unique_states),
        "test_visible_states_by_position": position_counts,
        "profile_scenarios": len(mixtures),
        "profile_state_evaluations": len(durations),
        "stable_action_coverage": stable_coverage,
        "stable_action_coverage_by_position": by_position_coverage,
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
            "requires_position_value_generalization_gate": True,
            "minimum_stable_action_coverage_per_profile_position": "0.50",
            "maximum_p95_ms": "5",
            "requires_non_authorizing_outputs": True,
        },
        "mixture_fingerprints": {
            label: mixture["mixture_fingerprint"]
            for label, mixture in sorted(mixtures.items())
        },
        "limitations": [
            "Archetypes are constructed Beta-posterior profiles and do not prove calibration on real player populations.",
            "The audit uses the same three synthetic probe-policy families represented in value training.",
            "Coverage is limited to limped BTN/BB check and minimum-bet states; stable actions remain offline and non-GTO.",
        ],
    }
    result["audit_fingerprint"] = _sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate APC's position-aware profile-conditioned postflop values."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("mixture", type=Path)
    parser.add_argument("--latency-repetitions", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        state = json.loads(args.state.read_text(encoding="utf-8"))
        mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
        if args.latency_repetitions:
            result = evaluate_position_profile_conditioned_latency(
                args.checkpoint,
                state,
                mixture,
                repetitions=args.latency_repetitions,
            )
        else:
            result = predict_position_profile_conditioned_postflop(
                args.checkpoint, state, mixture
            )
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
