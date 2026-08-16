from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from apc.self_learning.postflop_position_rollout_dataset import HERO_POSITIONS, validate_postflop_position_dataset
from apc.self_learning.postflop_policy_rollout_dataset import OPPONENT_POLICIES
from apc.self_learning.postflop_paired_rollout_dataset import STREETS
from apc.self_learning.train_action_value import action_issues
from apc.self_learning.train_postflop_policy_value import (
    FEATURE_FAMILIES,
    SHRINKAGE_GRID,
    SUPPORTED_ACTIONS,
    postflop_feature_key,
)
from apc.self_learning.train_value import _sha256, value_state_issues


def _load(dataset: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    validation = validate_postflop_position_dataset(dataset)
    if not validation["valid"]:
        raise ValueError("postflop position dataset is invalid: " + "; ".join(validation["issues"]))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_eligible") is not True:
        raise ValueError("postflop position dataset is not training eligible")
    rows = [
        json.loads(line)
        for line in (dataset / str(manifest["examples_file"])).read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [row for row in rows if row["counterfactual_action"]["action"] in SUPPORTED_ACTIONS]
    if any(not any(row["split"] == split for row in rows) for split in ("train", "validation", "test")):
        raise ValueError("position-aware value training requires every complete-hand split")
    return manifest, rows


def _target(row: dict[str, object]) -> float:
    return float(str(row["learning_signal"]["hero_return_bb"]))


def _base_key(row: dict[str, object]) -> str:
    return "|".join(
        (
            str(row["hero_position"]),
            str(row["street"]),
            str(row["opponent_policy"]),
            str(row["counterfactual_action"]["action"]),
        )
    )


def _fine_key(row: dict[str, object], family: str) -> str:
    return _base_key(row) + "|" + postflop_feature_key(row["state"], family)


def _statistics(
    rows: list[dict[str, object]], family: str
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    base_values: defaultdict[str, list[float]] = defaultdict(list)
    fine_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _target(row)
        base_values[_base_key(row)].append(value)
        fine_values[_fine_key(row, family)].append(value)
    base = {key: format(statistics.fmean(values), ".12g") for key, values in sorted(base_values.items())}
    table = {
        key: {
            "samples": len(values),
            "sum_return_bb": format(sum(values), ".12g"),
            "mean_return_bb": format(statistics.fmean(values), ".12g"),
        }
        for key, values in sorted(fine_values.items())
    }
    return base, table


def _prediction(
    base: dict[str, str],
    table: dict[str, dict[str, object]],
    row: dict[str, object],
    family: str,
    shrinkage: float,
) -> tuple[float, bool, int]:
    baseline = float(base[_base_key(row)])
    detail = table.get(_fine_key(row, family))
    if detail is None:
        return baseline, False, 0
    count = int(detail["samples"])
    return (float(str(detail["sum_return_bb"])) + shrinkage * baseline) / (count + shrinkage), True, count


def _value_metrics(predictions: list[float], actual: list[float], baselines: list[float]) -> dict[str, object]:
    errors = [prediction - observed for prediction, observed in zip(predictions, actual)]
    baseline_errors = [prediction - observed for prediction, observed in zip(baselines, actual)]
    mae = statistics.fmean(abs(value) for value in errors)
    baseline_mae = statistics.fmean(abs(value) for value in baseline_errors)
    return {
        "examples": len(actual),
        "mae_bb": format(mae, ".12g"),
        "rmse_bb": format(math.sqrt(statistics.fmean(value * value for value in errors)), ".12g"),
        "bias_bb": format(statistics.fmean(errors), ".12g"),
        "position_street_policy_action_mean_baseline_mae_bb": format(baseline_mae, ".12g"),
        "mae_improvement_bb": format(baseline_mae - mae, ".12g"),
    }


def _calibration(predictions: list[float], actual: list[float]) -> dict[str, object]:
    ordered = sorted(range(len(predictions)), key=lambda index: (predictions[index], index))
    bins = []
    weighted_gap = 0.0
    for bin_index in range(5):
        indices = ordered[bin_index * len(ordered) // 5 : (bin_index + 1) * len(ordered) // 5]
        mean_prediction = statistics.fmean(predictions[index] for index in indices)
        mean_actual = statistics.fmean(actual[index] for index in indices)
        gap = abs(mean_prediction - mean_actual)
        weighted_gap += len(indices) * gap
        bins.append(
            {
                "examples": len(indices),
                "mean_prediction_bb": format(mean_prediction, ".12g"),
                "mean_actual_bb": format(mean_actual, ".12g"),
                "absolute_gap_bb": format(gap, ".12g"),
            }
        )
    return {
        "bins": bins,
        "expected_absolute_calibration_error_bb": format(weighted_gap / len(ordered), ".12g"),
        "maximum_bin_gap_bb": format(max(float(row["absolute_gap_bb"]) for row in bins), ".12g"),
        "confidence_calibrated": False,
    }


def _decision_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    candidate = [float(row["candidate_return"]) for row in records]
    baseline = [float(row["baseline_return"]) for row in records]
    oracle = [float(row["oracle_return"]) for row in records]
    return {
        "policy_states": len(records),
        "decision_accuracy": format(statistics.fmean(float(row["correct"]) for row in records), ".12g"),
        "realized_policy_value_bb": format(statistics.fmean(candidate), ".12g"),
        "baseline_policy_value_bb": format(statistics.fmean(baseline), ".12g"),
        "policy_value_improvement_bb": format(statistics.fmean(candidate) - statistics.fmean(baseline), ".12g"),
        "oracle_policy_value_bb": format(statistics.fmean(oracle), ".12g"),
        "mean_regret_bb": format(statistics.fmean(oracle) - statistics.fmean(candidate), ".12g"),
    }


def _evaluate(
    rows: list[dict[str, object]],
    base: dict[str, str],
    table: dict[str, dict[str, object]],
    family: str,
    shrinkage: float,
    *,
    bootstrap_samples: int = 0,
) -> dict[str, object]:
    predictions = []
    actual = []
    baselines = []
    covered = []
    positions = []
    paired: defaultdict[str, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    context: dict[str, tuple[str, str, str, str]] = {}
    for row in rows:
        predicted, exact, _ = _prediction(base, table, row, family, shrinkage)
        observed = _target(row)
        baseline = float(base[_base_key(row)])
        predictions.append(predicted)
        actual.append(observed)
        baselines.append(baseline)
        covered.append(exact)
        positions.append(str(row["hero_position"]))
        policy_state_id = str(row["policy_state_id"])
        action = str(row["counterfactual_action"]["action"])
        paired[policy_state_id][action] = (predicted, observed, baseline)
        context[policy_state_id] = (
            str(row["group_id"]),
            str(row["hero_position"]),
            str(row["street"]),
            str(row["opponent_policy"]),
        )
    decision_records = []
    for policy_state_id, state in paired.items():
        if set(state) != set(SUPPORTED_ACTIONS):
            raise ValueError("evaluation found incomplete position policy state")
        candidate_action = max(SUPPORTED_ACTIONS, key=lambda action: (state[action][0], action == "check"))
        baseline_action = max(SUPPORTED_ACTIONS, key=lambda action: (state[action][2], action == "check"))
        oracle_return = max(value[1] for value in state.values())
        group_id, position, street, policy = context[policy_state_id]
        decision_records.append(
            {
                "group_id": group_id,
                "position": position,
                "street": street,
                "policy": policy,
                "candidate_return": state[candidate_action][1],
                "baseline_return": state[baseline_action][1],
                "oracle_return": oracle_return,
                "correct": state[candidate_action][1] == oracle_return,
            }
        )
    aggregate = _value_metrics(predictions, actual, baselines)
    aggregate["exact_abstraction_coverage"] = format(sum(covered) / len(covered), ".12g")
    aggregate["value_calibration"] = _calibration(predictions, actual)
    aggregate["decision"] = _decision_metrics(decision_records)
    aggregate["by_position"] = {}
    for position in HERO_POSITIONS:
        indices = [index for index, observed in enumerate(positions) if observed == position]
        metrics = _value_metrics(
            [predictions[index] for index in indices],
            [actual[index] for index in indices],
            [baselines[index] for index in indices],
        )
        metrics["exact_abstraction_coverage"] = format(sum(covered[index] for index in indices) / len(indices), ".12g")
        metrics["decision"] = _decision_metrics([row for row in decision_records if row["position"] == position])
        aggregate["by_position"][position] = metrics
    if bootstrap_samples:
        if bootstrap_samples < 200:
            raise ValueError("paired bootstrap requires at least 200 samples")
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for row in decision_records:
            grouped[str(row["group_id"])].append(float(row["candidate_return"]) - float(row["baseline_return"]))
        group_means = [statistics.fmean(values) for _, values in sorted(grouped.items())]
        generator = random.Random(20260816)
        sampled = sorted(
            statistics.fmean(group_means[generator.randrange(len(group_means))] for _ in group_means)
            for _ in range(bootstrap_samples)
        )
        aggregate["paired_policy_value_bootstrap"] = {
            "unit": "matched_complete_hand_group",
            "independent_groups": len(group_means),
            "samples": bootstrap_samples,
            "seed": 20260816,
            "mean_improvement_bb": format(statistics.fmean(group_means), ".12g"),
            "lower_95_bb": format(sampled[math.floor(0.025 * bootstrap_samples)], ".12g"),
            "upper_95_bb": format(sampled[math.ceil(0.975 * bootstrap_samples) - 1], ".12g"),
        }
    return aggregate


def train_position_postflop_value_model(
    dataset: str | Path,
    output: str | Path,
    *,
    feature_families: tuple[str, ...] = FEATURE_FAMILIES,
    shrinkage_grid: tuple[str, ...] = SHRINKAGE_GRID,
) -> dict[str, object]:
    if not feature_families or any(family not in FEATURE_FAMILIES for family in feature_families):
        raise ValueError("feature families are empty or unsupported")
    try:
        shrinkages = tuple(float(value) for value in shrinkage_grid)
    except ValueError as error:
        raise ValueError("shrinkage grid values must be numbers") from error
    if not shrinkages or any(not math.isfinite(value) or value < 0 for value in shrinkages):
        raise ValueError("shrinkage grid must contain finite non-negative values")
    dataset_path = Path(dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"position postflop checkpoint already exists: {output_path}")
    manifest, rows = _load(dataset_path)
    splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    candidates = []
    selection_trace = []
    for family in feature_families:
        base, table = _statistics(splits["train"], family)
        for raw, shrinkage in zip(shrinkage_grid, shrinkages):
            metrics = _evaluate(splits["validation"], base, table, family, shrinkage)
            selection_trace.append(
                {
                    "feature_family": family,
                    "shrinkage": raw,
                    "validation_mae_bb": metrics["mae_bb"],
                    "validation_policy_value_improvement_bb": metrics["decision"]["policy_value_improvement_bb"],
                }
            )
            candidates.append((float(metrics["mae_bb"]), family, raw, base, table))
    _, family, raw_shrinkage, base, table = min(candidates, key=lambda item: (item[0], item[1], float(item[2])))
    shrinkage = float(raw_shrinkage)
    metrics = {
        split: _evaluate(split_rows, base, table, family, shrinkage, bootstrap_samples=2000 if split == "test" else 0)
        for split, split_rows in splits.items()
    }
    test = metrics["test"]
    gate = (
        float(test["mae_improvement_bb"]) > 0
        and float(test["paired_policy_value_bootstrap"]["lower_95_bb"]) >= 0
        and float(test["exact_abstraction_coverage"]) >= 0.90
        and all(
            float(test["by_position"][position]["mae_improvement_bb"]) > 0
            and float(test["by_position"][position]["decision"]["policy_value_improvement_bb"]) >= 0
            and float(test["by_position"][position]["decision"]["decision_accuracy"]) >= 0.55
            for position in HERO_POSITIONS
        )
    )
    checkpoint = {
        "schema_version": "1.0.0",
        "model_kind": "matched_position_postflop_multi_policy_action_value_candidate",
        "status": "offline_position_postflop_value_candidate_not_promoted",
        "units": "BB",
        "activation_authorized": False,
        "recommendation_allowed": False,
        "confidence_calibrated": False,
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "examples_sha256": manifest["examples_sha256"],
            "complete_hand_group_exclusive": manifest["group_exclusive"],
            "position_card_matched": manifest["position_card_matched"],
        },
        "configuration": {
            "supported_positions": list(HERO_POSITIONS),
            "supported_streets": list(STREETS),
            "supported_actions": list(SUPPORTED_ACTIONS),
            "unsupported_actions": ["all_in"],
            "supported_opponent_policies": list(OPPONENT_POLICIES),
            "selected_feature_family_by_validation": family,
            "selected_shrinkage_by_validation": raw_shrinkage,
            "selection_trace": selection_trace,
            "target": "matched_position_same_state_terminal_return_bb",
        },
        "position_street_policy_action_train_means_bb": base,
        "abstraction_action_table": table,
        "metrics": metrics,
        "generalization_gate": {
            "passed": gate,
            "criterion": "untouched_matched_hand_test_improves_both_positions_with_nonnegative_paired_95pct_lower_bound_90pct_coverage_and_55pct_decision_accuracy",
            "activation_authorized": False,
        },
        "limitations": [
            "Values apply only to two limped-pot positions and three deterministic opponent probes, not GTO or learned population play.",
            "The candidate supports check and minimum bet only; all-in is excluded from training because of high variance.",
            "Passing this offline gate cannot authorize coaching, confidence calibration, activation or external actuation.",
        ],
    }
    checkpoint["checkpoint_fingerprint"] = _sha256(checkpoint)
    validation = validate_position_postflop_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("position postflop checkpoint failed validation: " + "; ".join(validation["issues"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temporary:
        temporary_file = Path(temporary) / output_path.name
        temporary_file.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        temporary_file.replace(output_path)
    return checkpoint


def validate_position_postflop_value_checkpoint(checkpoint_or_path: dict[str, object] | str | Path) -> dict[str, object]:
    issues = []
    if isinstance(checkpoint_or_path, dict):
        checkpoint = checkpoint_or_path
    else:
        try:
            checkpoint = json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {"schema_version": "1.0.0", "valid": False, "issues": [f"checkpoint unreadable: {error}"]}
    if checkpoint.get("schema_version") != "1.0.0" or checkpoint.get("units") != "BB":
        issues.append("checkpoint schema/BB contract is invalid")
    if checkpoint.get("status") != "offline_position_postflop_value_candidate_not_promoted":
        issues.append("checkpoint status is invalid")
    if any(checkpoint.get(key) is not False for key in ("activation_authorized", "recommendation_allowed", "confidence_calibrated")):
        issues.append("checkpoint cannot authorize activation, recommendations or confidence")
    configuration = checkpoint.get("configuration", {})
    if not isinstance(configuration, dict) or configuration.get("supported_positions") != list(HERO_POSITIONS) or configuration.get("supported_actions") != list(SUPPORTED_ACTIONS) or configuration.get("unsupported_actions") != ["all_in"]:
        issues.append("checkpoint position/action contract is invalid")
    if configuration.get("supported_streets") != list(STREETS) or configuration.get("supported_opponent_policies") != list(OPPONENT_POLICIES) or configuration.get("selected_feature_family_by_validation") not in FEATURE_FAMILIES:
        issues.append("checkpoint street/policy/feature contract is invalid")
    base = checkpoint.get("position_street_policy_action_train_means_bb")
    expected_base = {
        "|".join((position, street, policy, action))
        for position in HERO_POSITIONS
        for street in STREETS
        for policy in OPPONENT_POLICIES
        for action in SUPPORTED_ACTIONS
    }
    try:
        base_valid = isinstance(base, dict) and set(base) == expected_base and all(math.isfinite(float(str(value))) for value in base.values())
    except (TypeError, ValueError):
        base_valid = False
    if not base_valid:
        issues.append("checkpoint position baseline table is invalid")
    table = checkpoint.get("abstraction_action_table")
    try:
        table_valid = isinstance(table, dict) and bool(table) and all(int(row["samples"]) > 0 and math.isfinite(float(str(row["sum_return_bb"]))) for row in table.values())
    except (KeyError, TypeError, ValueError):
        table_valid = False
    if not table_valid:
        issues.append("checkpoint abstraction table is invalid")
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict) or any(split not in metrics or metrics[split].get("examples", 0) <= 0 for split in ("train", "validation", "test")):
        issues.append("checkpoint requires non-empty complete-hand split metrics")
    material = dict(checkpoint)
    observed = material.pop("checkpoint_fingerprint", None)
    if observed != _sha256(material):
        issues.append("checkpoint fingerprint mismatch")
    return {"schema_version": "1.0.0", "valid": not issues, "issues": issues, "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint")}


def predict_position_postflop_value(
    checkpoint_or_path: dict[str, object] | str | Path,
    state: dict[str, object],
    command: dict[str, object],
    opponent_policy: str,
) -> dict[str, object]:
    checkpoint = checkpoint_or_path if isinstance(checkpoint_or_path, dict) else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    validation = validate_position_postflop_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("position postflop checkpoint is invalid: " + "; ".join(validation["issues"]))
    issues = [*value_state_issues(state), *action_issues(state, command)]
    action = str(command.get("action", "")) if isinstance(command, dict) else ""
    if state.get("hero_position") not in HERO_POSITIONS:
        issues.append("hero_position_not_supported")
    if state.get("street") not in STREETS:
        issues.append("street_not_supported")
    if action not in SUPPORTED_ACTIONS:
        issues.append("action_not_supported")
    if opponent_policy not in OPPONENT_POLICIES:
        issues.append("opponent_policy_not_supported")
    if issues:
        return {
            "schema_version": "1.0.0",
            "status": "abstain_unsupported_or_invalid",
            "predicted_terminal_return_bb": None,
            "reasons": sorted(set(issues)),
            "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
            "recommendation_allowed": False,
            "activation_authorized": False,
        }
    row = {
        "hero_position": state["hero_position"],
        "street": state["street"],
        "opponent_policy": opponent_policy,
        "counterfactual_action": command,
        "state": state,
    }
    predicted, exact, samples = _prediction(
        checkpoint["position_street_policy_action_train_means_bb"],
        checkpoint["abstraction_action_table"],
        row,
        checkpoint["configuration"]["selected_feature_family_by_validation"],
        float(checkpoint["configuration"]["selected_shrinkage_by_validation"]),
    )
    result = {
        "schema_version": "1.0.0",
        "status": "offline_position_postflop_value_prediction_uncalibrated",
        "units": "BB",
        "hero_position": state["hero_position"],
        "street": state["street"],
        "opponent_policy": opponent_policy,
        "action": command,
        "predicted_terminal_return_bb": format(predicted, ".12g"),
        "exact_abstraction_coverage": exact,
        "training_samples": samples,
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def evaluate_position_postflop_latency(
    dataset: str | Path,
    checkpoint_or_path: dict[str, object] | str | Path,
    *,
    repetitions: int = 200,
) -> dict[str, object]:
    if repetitions < 20:
        raise ValueError("latency evaluation needs at least 20 repetitions")
    checkpoint = checkpoint_or_path if isinstance(checkpoint_or_path, dict) else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    manifest, rows = _load(Path(dataset).resolve())
    if checkpoint.get("dataset", {}).get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
        raise ValueError("latency dataset does not match checkpoint")
    test_rows = [row for row in rows if row["split"] == "test"]
    durations = []
    for index in range(repetitions):
        row = test_rows[index % len(test_rows)]
        started = time.perf_counter_ns()
        predict_position_postflop_value(checkpoint, row["state"], row["counterfactual_action"], str(row["opponent_policy"]))
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "schema_version": "1.0.0",
        "repetitions": repetitions,
        "p50_ms": format(statistics.median(durations), ".12g"),
        "p95_ms": format(p95, ".12g"),
        "maximum_ms": format(max(durations), ".12g"),
        "latency_gate": {"passed": p95 <= 5.0, "threshold_p95_ms": "5"},
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train, validate or latency-audit APC's position-aware postflop value candidate.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("checkpoint", type=Path)
    latency = subparsers.add_parser("latency")
    latency.add_argument("dataset", type=Path)
    latency.add_argument("checkpoint", type=Path)
    latency.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_position_postflop_value_checkpoint(args.checkpoint)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        if args.command == "latency":
            report = evaluate_position_postflop_latency(args.dataset, args.checkpoint, repetitions=args.repetitions)
            print(json.dumps(report, indent=2))
            return 0 if report["latency_gate"]["passed"] else 3
        checkpoint = train_position_postflop_value_model(args.dataset, args.output)
        print(json.dumps(checkpoint, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
