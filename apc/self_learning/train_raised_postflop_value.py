from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from apc.self_learning.postflop_paired_rollout_dataset import STREETS
from apc.self_learning.postflop_policy_rollout_dataset import OPPONENT_POLICIES
from apc.self_learning.postflop_position_rollout_dataset import HERO_POSITIONS
from apc.self_learning.raised_postflop_rollout_dataset import (
    ACTION_KEYS_BY_NODE,
    NODE_FAMILIES,
    validate_raised_postflop_dataset,
)
from apc.self_learning.train_action_value import action_issues
from apc.self_learning.train_position_postflop_value import (
    _calibration,
    _decision_metrics,
    _value_metrics,
)
from apc.self_learning.train_postflop_policy_value import (
    FEATURE_FAMILIES,
    SHRINKAGE_GRID,
    postflop_feature_key,
)
from apc.self_learning.train_value import _sha256, value_state_issues


TIE_BREAK_BY_NODE = {
    "lead": ("check", "bet_33", "bet_67", "bet_100"),
    "facing_33": ("call", "fold", "raise_min", "raise_3x"),
    "facing_75": ("call", "fold", "raise_min", "raise_3x"),
}
GENERALIZATION_CRITERION = (
    "untouched_complete_hand_test_improves_aggregate_and_each_position_node_"
    "slice_with_nonnegative_paired_95pct_lower_bound_90pct_coverage_and_"
    "35pct_decision_accuracy"
)


@dataclass(frozen=True)
class PreparedRaisedPostflopValue:
    """Validated, detached checkpoint handle for repeated offline lookups."""

    _checkpoint: dict[str, object]


def prepare_raised_postflop_value(
    checkpoint_or_path: dict[str, object] | str | Path | PreparedRaisedPostflopValue,
) -> PreparedRaisedPostflopValue:
    if isinstance(checkpoint_or_path, PreparedRaisedPostflopValue):
        return checkpoint_or_path
    loaded = (
        checkpoint_or_path
        if isinstance(checkpoint_or_path, dict)
        else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    )
    validation = validate_raised_postflop_value_checkpoint(loaded)
    if not validation["valid"]:
        raise ValueError(
            "raised postflop checkpoint is invalid: "
            + "; ".join(validation["issues"])
        )
    detached = json.loads(json.dumps(loaded, separators=(",", ":")))
    return PreparedRaisedPostflopValue(detached)


def _load(dataset: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    validation = validate_raised_postflop_dataset(dataset)
    if not validation["valid"]:
        raise ValueError(
            "raised postflop dataset is invalid: "
            + "; ".join(validation["issues"])
        )
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_eligible") is not True:
        raise ValueError("raised postflop dataset is not training eligible")
    rows = [
        json.loads(line)
        for line in (dataset / str(manifest["examples_file"]))
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if any(
        not any(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    ):
        raise ValueError("raised value training requires every complete-hand split")
    return manifest, rows


def _target(row: dict[str, object]) -> float:
    return float(str(row["learning_signal"]["hero_return_bb"]))


def _base_key(row: dict[str, object]) -> str:
    return "|".join(
        (
            str(row["hero_position"]),
            str(row["street"]),
            str(row["node_family"]),
            str(row["opponent_policy"]),
            str(row["counterfactual_action_key"]),
        )
    )


def _feature_key(
    row: dict[str, object],
    family: str,
    cache: dict[tuple[str, str], str],
) -> str:
    identity = (str(row["provenance"]["pre_state_fingerprint"]), family)
    if identity not in cache:
        cache[identity] = postflop_feature_key(row["state"], family)
    return cache[identity]


def _fine_key(
    row: dict[str, object],
    family: str,
    cache: dict[tuple[str, str], str],
) -> str:
    return _base_key(row) + "|" + _feature_key(row, family, cache)


def _statistics(
    rows: list[dict[str, object]],
    family: str,
    cache: dict[tuple[str, str], str],
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    base_values: defaultdict[str, list[float]] = defaultdict(list)
    fine_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _target(row)
        base_values[_base_key(row)].append(value)
        fine_values[_fine_key(row, family, cache)].append(value)
    base = {
        key: format(statistics.fmean(values), ".12g")
        for key, values in sorted(base_values.items())
    }
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
    cache: dict[tuple[str, str], str],
) -> tuple[float, bool, int]:
    baseline = float(base[_base_key(row)])
    detail = table.get(_fine_key(row, family, cache))
    if detail is None:
        return baseline, False, 0
    count = int(detail["samples"])
    predicted = (
        float(str(detail["sum_return_bb"])) + shrinkage * baseline
    ) / (count + shrinkage)
    return predicted, True, count


def _select_action(
    state: dict[str, tuple[float, float, float]],
    node_family: str,
    value_index: int,
) -> str:
    priority = TIE_BREAK_BY_NODE[node_family]
    return max(
        priority,
        key=lambda action: (state[action][value_index], -priority.index(action)),
    )


def _slice_metrics(
    indices: list[int],
    predictions: list[float],
    actual: list[float],
    baselines: list[float],
    covered: list[bool],
    decision_rows: list[dict[str, object]],
) -> dict[str, object]:
    metrics = _value_metrics(
        [predictions[index] for index in indices],
        [actual[index] for index in indices],
        [baselines[index] for index in indices],
    )
    metrics["exact_abstraction_coverage"] = format(
        sum(covered[index] for index in indices) / len(indices), ".12g"
    )
    metrics["decision"] = _decision_metrics(decision_rows)
    return metrics


def _evaluate(
    rows: list[dict[str, object]],
    base: dict[str, str],
    table: dict[str, dict[str, object]],
    family: str,
    shrinkage: float,
    cache: dict[tuple[str, str], str],
    *,
    bootstrap_samples: int = 0,
) -> dict[str, object]:
    predictions: list[float] = []
    actual: list[float] = []
    baselines: list[float] = []
    covered: list[bool] = []
    positions: list[str] = []
    nodes: list[str] = []
    paired: defaultdict[
        str, dict[str, tuple[float, float, float]]
    ] = defaultdict(dict)
    context: dict[str, tuple[str, str, str, str, str]] = {}
    for row in rows:
        predicted, exact, _ = _prediction(
            base, table, row, family, shrinkage, cache
        )
        observed = _target(row)
        baseline = float(base[_base_key(row)])
        predictions.append(predicted)
        actual.append(observed)
        baselines.append(baseline)
        covered.append(exact)
        positions.append(str(row["hero_position"]))
        nodes.append(str(row["node_family"]))
        policy_state_id = str(row["policy_state_id"])
        action_key = str(row["counterfactual_action_key"])
        paired[policy_state_id][action_key] = (predicted, observed, baseline)
        context[policy_state_id] = (
            str(row["group_id"]),
            str(row["hero_position"]),
            str(row["street"]),
            str(row["node_family"]),
            str(row["opponent_policy"]),
        )
    decision_records = []
    for policy_state_id, state in paired.items():
        group_id, position, street, node_family, policy = context[policy_state_id]
        if set(state) != set(ACTION_KEYS_BY_NODE[node_family]):
            raise ValueError("evaluation found incomplete raised policy state")
        candidate_action = _select_action(state, node_family, 0)
        baseline_action = _select_action(state, node_family, 2)
        oracle_return = max(value[1] for value in state.values())
        decision_records.append(
            {
                "group_id": group_id,
                "position": position,
                "street": street,
                "node": node_family,
                "policy": policy,
                "candidate_return": state[candidate_action][1],
                "baseline_return": state[baseline_action][1],
                "oracle_return": oracle_return,
                "correct": state[candidate_action][1] == oracle_return,
            }
        )
    aggregate = _value_metrics(predictions, actual, baselines)
    aggregate["exact_abstraction_coverage"] = format(
        sum(covered) / len(covered), ".12g"
    )
    aggregate["value_calibration"] = _calibration(predictions, actual)
    aggregate["decision"] = _decision_metrics(decision_records)
    aggregate["by_position"] = {}
    for position in HERO_POSITIONS:
        indices = [
            index for index, observed in enumerate(positions) if observed == position
        ]
        aggregate["by_position"][position] = _slice_metrics(
            indices,
            predictions,
            actual,
            baselines,
            covered,
            [row for row in decision_records if row["position"] == position],
        )
    aggregate["by_node"] = {}
    for node in NODE_FAMILIES:
        indices = [index for index, observed in enumerate(nodes) if observed == node]
        aggregate["by_node"][node] = _slice_metrics(
            indices,
            predictions,
            actual,
            baselines,
            covered,
            [row for row in decision_records if row["node"] == node],
        )
    aggregate["by_position_node"] = {}
    for position in HERO_POSITIONS:
        aggregate["by_position_node"][position] = {}
        for node in NODE_FAMILIES:
            indices = [
                index
                for index, (observed_position, observed_node) in enumerate(
                    zip(positions, nodes)
                )
                if observed_position == position and observed_node == node
            ]
            aggregate["by_position_node"][position][node] = _slice_metrics(
                indices,
                predictions,
                actual,
                baselines,
                covered,
                [
                    row
                    for row in decision_records
                    if row["position"] == position and row["node"] == node
                ],
            )
    if bootstrap_samples:
        if bootstrap_samples < 200:
            raise ValueError("paired bootstrap requires at least 200 samples")
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for row in decision_records:
            grouped[str(row["group_id"])].append(
                float(row["candidate_return"]) - float(row["baseline_return"])
            )
        group_means = [
            statistics.fmean(values) for _, values in sorted(grouped.items())
        ]
        generator = random.Random(20260823)
        sampled = sorted(
            statistics.fmean(
                group_means[generator.randrange(len(group_means))]
                for _ in group_means
            )
            for _ in range(bootstrap_samples)
        )
        aggregate["paired_policy_value_bootstrap"] = {
            "unit": "matched_complete_hand_group",
            "independent_groups": len(group_means),
            "samples": bootstrap_samples,
            "seed": 20260823,
            "mean_improvement_bb": format(statistics.fmean(group_means), ".12g"),
            "lower_95_bb": format(
                sampled[math.floor(0.025 * bootstrap_samples)], ".12g"
            ),
            "upper_95_bb": format(
                sampled[math.ceil(0.975 * bootstrap_samples) - 1], ".12g"
            ),
        }
    return aggregate


def _slice_passed(row: dict[str, object]) -> bool:
    return (
        float(row["mae_improvement_bb"]) > 0
        and float(row["exact_abstraction_coverage"]) >= 0.90
        and float(row["decision"]["policy_value_improvement_bb"]) >= 0
        and float(row["decision"]["decision_accuracy"]) >= 0.35
    )


def _generalization_gate_passed(test: dict[str, object]) -> bool:
    return (
        float(test["mae_improvement_bb"]) > 0
        and float(test["paired_policy_value_bootstrap"]["lower_95_bb"]) >= 0
        and float(test["exact_abstraction_coverage"]) >= 0.90
        and all(
            _slice_passed(test["by_position"][position])
            for position in HERO_POSITIONS
        )
        and all(_slice_passed(test["by_node"][node]) for node in NODE_FAMILIES)
    )


def train_raised_postflop_value_model(
    dataset: str | Path,
    output: str | Path,
    *,
    feature_families: tuple[str, ...] = FEATURE_FAMILIES,
    shrinkage_grid: tuple[str, ...] = SHRINKAGE_GRID,
) -> dict[str, object]:
    if not feature_families or any(
        family not in FEATURE_FAMILIES for family in feature_families
    ):
        raise ValueError("feature families are empty or unsupported")
    try:
        shrinkages = tuple(float(value) for value in shrinkage_grid)
    except ValueError as error:
        raise ValueError("shrinkage grid values must be numbers") from error
    if not shrinkages or any(
        not math.isfinite(value) or value < 0 for value in shrinkages
    ):
        raise ValueError("shrinkage grid must contain finite non-negative values")
    dataset_path = Path(dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"raised postflop checkpoint already exists: {output_path}")
    manifest, rows = _load(dataset_path)
    splits = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    cache: dict[tuple[str, str], str] = {}
    candidates = []
    selection_trace = []
    for family in feature_families:
        base, table = _statistics(splits["train"], family, cache)
        for raw, shrinkage in zip(shrinkage_grid, shrinkages):
            metrics = _evaluate(
                splits["validation"],
                base,
                table,
                family,
                shrinkage,
                cache,
            )
            selection_trace.append(
                {
                    "feature_family": family,
                    "shrinkage": raw,
                    "validation_mae_bb": metrics["mae_bb"],
                    "validation_policy_value_improvement_bb": metrics["decision"][
                        "policy_value_improvement_bb"
                    ],
                }
            )
            candidates.append(
                (float(metrics["mae_bb"]), family, raw, base, table)
            )
    _, family, raw_shrinkage, base, table = min(
        candidates, key=lambda item: (item[0], item[1], float(item[2]))
    )
    shrinkage = float(raw_shrinkage)
    metrics = {
        split: _evaluate(
            split_rows,
            base,
            table,
            family,
            shrinkage,
            cache,
            bootstrap_samples=2000 if split == "test" else 0,
        )
        for split, split_rows in splits.items()
    }
    test = metrics["test"]

    gate = _generalization_gate_passed(test)
    checkpoint = {
        "schema_version": "1.0.0",
        "model_kind": "raised_postflop_position_node_policy_action_value_candidate",
        "status": "offline_raised_postflop_value_candidate_not_promoted",
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
            "raised_preflop_pot": manifest["raised_preflop_pot"],
        },
        "configuration": {
            "supported_positions": list(HERO_POSITIONS),
            "supported_streets": list(STREETS),
            "supported_nodes": list(NODE_FAMILIES),
            "supported_action_keys_by_node": {
                node: list(actions) for node, actions in ACTION_KEYS_BY_NODE.items()
            },
            "supported_opponent_policies": list(OPPONENT_POLICIES),
            "selected_feature_family_by_validation": family,
            "selected_shrinkage_by_validation": raw_shrinkage,
            "selection_trace": selection_trace,
            "target": "matched_raised_position_node_terminal_return_bb",
        },
        "position_street_node_policy_action_train_means_bb": base,
        "abstraction_action_table": table,
        "metrics": metrics,
        "generalization_gate": {
            "passed": gate,
            "criterion": GENERALIZATION_CRITERION,
            "activation_authorized": False,
        },
        "limitations": [
            "Values apply only to one 2.5 BB raised heads-up trunk and three deterministic opponent probes, not GTO or learned population play.",
            "Supported sizes are the discrete action keys represented by each node; all-in, 3-bet, 4-bet, multiway and rake contexts are absent.",
            "Passing an offline gate cannot authorize confidence, coaching, promotion, activation or external actuation.",
        ],
    }
    checkpoint["checkpoint_fingerprint"] = _sha256(checkpoint)
    validation = validate_raised_postflop_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError(
            "raised postflop checkpoint failed validation: "
            + "; ".join(validation["issues"])
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}-", dir=output_path.parent
    ) as temporary:
        temporary_file = Path(temporary) / output_path.name
        temporary_file.write_text(
            json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
        )
        temporary_file.replace(output_path)
    return checkpoint


def validate_raised_postflop_value_checkpoint(
    checkpoint_or_path: dict[str, object] | str | Path,
) -> dict[str, object]:
    issues = []
    if isinstance(checkpoint_or_path, dict):
        checkpoint = checkpoint_or_path
    else:
        try:
            checkpoint = json.loads(
                Path(checkpoint_or_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            return {
                "schema_version": "1.0.0",
                "valid": False,
                "issues": [f"checkpoint unreadable: {error}"],
            }
    if checkpoint.get("schema_version") != "1.0.0" or checkpoint.get("units") != "BB":
        issues.append("checkpoint schema/BB contract is invalid")
    if checkpoint.get("status") != "offline_raised_postflop_value_candidate_not_promoted":
        issues.append("checkpoint status is invalid")
    if any(
        checkpoint.get(key) is not False
        for key in ("activation_authorized", "recommendation_allowed", "confidence_calibrated")
    ):
        issues.append("checkpoint cannot authorize activation, recommendations or confidence")
    configuration = checkpoint.get("configuration", {})
    if (
        not isinstance(configuration, dict)
        or configuration.get("supported_positions") != list(HERO_POSITIONS)
        or configuration.get("supported_streets") != list(STREETS)
        or configuration.get("supported_nodes") != list(NODE_FAMILIES)
        or configuration.get("supported_action_keys_by_node")
        != {node: list(actions) for node, actions in ACTION_KEYS_BY_NODE.items()}
        or configuration.get("supported_opponent_policies") != list(OPPONENT_POLICIES)
        or configuration.get("selected_feature_family_by_validation") not in FEATURE_FAMILIES
    ):
        issues.append("checkpoint configuration contract is invalid")
    base = checkpoint.get("position_street_node_policy_action_train_means_bb")
    expected_base = {
        "|".join((position, street, node, policy, action))
        for position in HERO_POSITIONS
        for street in STREETS
        for node in NODE_FAMILIES
        for policy in OPPONENT_POLICIES
        for action in ACTION_KEYS_BY_NODE[node]
    }
    try:
        base_valid = (
            isinstance(base, dict)
            and set(base) == expected_base
            and all(math.isfinite(float(str(value))) for value in base.values())
        )
    except (TypeError, ValueError):
        base_valid = False
    if not base_valid:
        issues.append("checkpoint baseline table is invalid")
    table = checkpoint.get("abstraction_action_table")
    try:
        table_valid = (
            isinstance(table, dict)
            and bool(table)
            and all(
                int(row["samples"]) > 0
                and math.isfinite(float(str(row["sum_return_bb"])))
                for row in table.values()
            )
        )
    except (KeyError, TypeError, ValueError):
        table_valid = False
    if not table_valid:
        issues.append("checkpoint abstraction table is invalid")
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict) or any(
        split not in metrics or metrics[split].get("examples", 0) <= 0
        for split in ("train", "validation", "test")
    ):
        issues.append("checkpoint requires non-empty complete-hand split metrics")
    gate = checkpoint.get("generalization_gate")
    try:
        gate_valid = (
            isinstance(gate, dict)
            and gate.get("criterion") == GENERALIZATION_CRITERION
            and gate.get("activation_authorized") is False
            and gate.get("passed") is _generalization_gate_passed(metrics["test"])
        )
    except (KeyError, TypeError, ValueError):
        gate_valid = False
    if not gate_valid:
        issues.append("checkpoint generalization gate does not match test metrics")
    material = dict(checkpoint)
    observed = material.pop("checkpoint_fingerprint", None)
    if observed != _sha256(material):
        issues.append("checkpoint fingerprint mismatch")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint"),
    }


def _command_key_issues(
    state: dict[str, object], node_family: str, action_key: str, command: dict[str, object]
) -> list[str]:
    issues = action_issues(state, command)
    if node_family not in NODE_FAMILIES:
        issues.append("node_family_not_supported")
        return issues
    if action_key not in ACTION_KEYS_BY_NODE[node_family]:
        issues.append("action_key_not_supported")
        return issues
    expected_action = (
        "check"
        if action_key == "check"
        else "bet"
        if action_key.startswith("bet_")
        else "raise"
        if action_key.startswith("raise_")
        else action_key
    )
    if command.get("action") != expected_action:
        issues.append("action_key_command_mismatch")
    try:
        to_call = float(str(state["to_call_bb"]))
    except (KeyError, TypeError, ValueError):
        issues.append("visible_to_call_invalid")
    else:
        if (node_family == "lead") != (to_call == 0):
            issues.append("node_family_call_price_mismatch")
    return sorted(set(issues))


def predict_raised_postflop_value(
    checkpoint_or_path: (
        dict[str, object] | str | Path | PreparedRaisedPostflopValue
    ),
    state: dict[str, object],
    node_family: str,
    action_key: str,
    command: dict[str, object],
    opponent_policy: str,
) -> dict[str, object]:
    prepared = prepare_raised_postflop_value(checkpoint_or_path)
    checkpoint = prepared._checkpoint
    issues = [
        *value_state_issues(state),
        *_command_key_issues(state, node_family, action_key, command),
    ]
    if state.get("hero_position") not in HERO_POSITIONS:
        issues.append("hero_position_not_supported")
    if state.get("street") not in STREETS:
        issues.append("street_not_supported")
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
        "node_family": node_family,
        "opponent_policy": opponent_policy,
        "counterfactual_action_key": action_key,
        "counterfactual_action": command,
        "state": state,
        "provenance": {"pre_state_fingerprint": state["state_fingerprint"]},
    }
    predicted, exact, samples = _prediction(
        checkpoint["position_street_node_policy_action_train_means_bb"],
        checkpoint["abstraction_action_table"],
        row,
        checkpoint["configuration"]["selected_feature_family_by_validation"],
        float(checkpoint["configuration"]["selected_shrinkage_by_validation"]),
        {},
    )
    result = {
        "schema_version": "1.0.0",
        "status": "offline_raised_postflop_value_prediction_uncalibrated",
        "units": "BB",
        "hero_position": state["hero_position"],
        "street": state["street"],
        "node_family": node_family,
        "opponent_policy": opponent_policy,
        "action_key": action_key,
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


def evaluate_raised_postflop_latency(
    dataset: str | Path,
    checkpoint_or_path: (
        dict[str, object] | str | Path | PreparedRaisedPostflopValue
    ),
    *,
    repetitions: int = 200,
) -> dict[str, object]:
    if repetitions < 20:
        raise ValueError("latency evaluation needs at least 20 repetitions")
    prepared = prepare_raised_postflop_value(checkpoint_or_path)
    checkpoint = prepared._checkpoint
    manifest, rows = _load(Path(dataset).resolve())
    if checkpoint.get("dataset", {}).get("dataset_fingerprint") != manifest.get(
        "dataset_fingerprint"
    ):
        raise ValueError("latency dataset does not match checkpoint")
    test_rows = [row for row in rows if row["split"] == "test"]
    durations = []
    fingerprints = set()
    for index in range(repetitions):
        row = test_rows[index % len(test_rows)]
        started = time.perf_counter_ns()
        result = predict_raised_postflop_value(
            prepared,
            row["state"],
            str(row["node_family"]),
            str(row["counterfactual_action_key"]),
            row["counterfactual_action"],
            str(row["opponent_policy"]),
        )
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
        fingerprints.add(str(result["prediction_fingerprint"]))
    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "schema_version": "1.0.0",
        "repetitions": repetitions,
        "p50_ms": format(statistics.median(durations), ".12g"),
        "p95_ms": format(p95, ".12g"),
        "maximum_ms": format(max(durations), ".12g"),
        "unique_prediction_fingerprints": len(fingerprints),
        "latency_gate": {"passed": p95 <= 5.0, "threshold_p95_ms": "5"},
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train, validate or latency-audit APC raised-postflop values."
    )
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
            report = validate_raised_postflop_value_checkpoint(args.checkpoint)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        if args.command == "latency":
            report = evaluate_raised_postflop_latency(
                args.dataset, args.checkpoint, repetitions=args.repetitions
            )
            print(json.dumps(report, indent=2))
            return 0 if report["latency_gate"]["passed"] else 3
        checkpoint = train_raised_postflop_value_model(args.dataset, args.output)
        print(json.dumps(checkpoint, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
