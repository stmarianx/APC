from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from apc.self_learning.paired_rollout_dataset import _hand_class, validate_paired_rollout_dataset
from apc.self_learning.train_action_value import action_issues
from apc.self_learning.train_value import _sha256, value_state_issues


SUPPORTED_ACTIONS = ("fold", "call", "raise")
SHRINKAGE_GRID = ("0", "2", "5", "10", "20", "50", "100")


def _load(dataset: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    report = validate_paired_rollout_dataset(dataset)
    if not report["valid"]:
        raise ValueError("paired rollout dataset is invalid: " + "; ".join(report["issues"]))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_eligible") is not True:
        raise ValueError("paired rollout dataset is not training eligible")
    rows = [
        json.loads(line)
        for line in (dataset / str(manifest["examples_file"])).read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [row for row in rows if row["counterfactual_action"]["action"] in SUPPORTED_ACTIONS]
    if any(not any(row["split"] == split for row in rows) for split in ("train", "validation", "test")):
        raise ValueError("paired value training requires every split")
    return manifest, rows


def _target(row: dict[str, object]) -> float:
    return float(str(row["learning_signal"]["hero_return_bb"]))


def _statistics(rows: list[dict[str, object]]) -> tuple[dict[str, float], dict[str, dict[str, dict[str, object]]]]:
    by_action: defaultdict[str, list[float]] = defaultdict(list)
    by_class_action: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        action = str(row["counterfactual_action"]["action"])
        hand_class = str(row["provenance"]["hero_hand_class"])
        value = _target(row)
        by_action[action].append(value)
        by_class_action[(hand_class, action)].append(value)
    action_means = {action: statistics.fmean(by_action[action]) for action in SUPPORTED_ACTIONS}
    table: defaultdict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for (hand_class, action), values in sorted(by_class_action.items()):
        table[hand_class][action] = {
            "samples": len(values),
            "sum_return_bb": format(sum(values), ".12g"),
            "mean_return_bb": format(statistics.fmean(values), ".12g"),
            "population_variance_bb2": format(statistics.pvariance(values), ".12g"),
        }
    return action_means, dict(table)


def _predict(
    table: dict[str, dict[str, dict[str, object]]],
    action_means: dict[str, float],
    hand_class: str,
    action: str,
    shrinkage: float,
) -> tuple[float, bool, int]:
    row = table.get(hand_class, {}).get(action)
    if row is None:
        return action_means[action], False, 0
    count = int(row["samples"])
    total = float(str(row["sum_return_bb"]))
    return (total + shrinkage * action_means[action]) / (count + shrinkage), True, count


def _evaluate(
    rows: list[dict[str, object]],
    table: dict[str, dict[str, dict[str, object]]],
    action_means: dict[str, float],
    shrinkage: float,
) -> dict[str, object]:
    predictions: list[float] = []
    actual: list[float] = []
    baselines: list[float] = []
    covered: list[bool] = []
    actions: list[str] = []
    for row in rows:
        action = str(row["counterfactual_action"]["action"])
        hand_class = str(row["provenance"]["hero_hand_class"])
        prediction, exact_class, _ = _predict(table, action_means, hand_class, action, shrinkage)
        predictions.append(prediction)
        actual.append(_target(row))
        baselines.append(action_means[action])
        covered.append(exact_class)
        actions.append(action)

    def slice_metrics(indices: list[int]) -> dict[str, object]:
        errors = [predictions[index] - actual[index] for index in indices]
        baseline_errors = [baselines[index] - actual[index] for index in indices]
        mae = sum(abs(value) for value in errors) / len(indices)
        baseline_mae = sum(abs(value) for value in baseline_errors) / len(indices)
        return {
            "examples": len(indices),
            "mae_bb": format(mae, ".12g"),
            "rmse_bb": format(math.sqrt(sum(value * value for value in errors) / len(indices)), ".12g"),
            "bias_bb": format(sum(errors) / len(indices), ".12g"),
            "action_mean_baseline_mae_bb": format(baseline_mae, ".12g"),
            "mae_improvement_bb": format(baseline_mae - mae, ".12g"),
        }

    aggregate = slice_metrics(list(range(len(rows))))
    aggregate["exact_hand_class_coverage"] = format(sum(covered) / len(covered), ".12g")
    aggregate["by_action"] = {
        action: slice_metrics([index for index, observed in enumerate(actions) if observed == action])
        for action in SUPPORTED_ACTIONS
    }
    return aggregate


def train_paired_value_model(
    dataset: str | Path,
    output: str | Path,
    *,
    shrinkage_grid: tuple[str, ...] = SHRINKAGE_GRID,
) -> dict[str, object]:
    if not shrinkage_grid:
        raise ValueError("shrinkage grid cannot be empty")
    try:
        candidates = tuple(float(value) for value in shrinkage_grid)
    except ValueError as error:
        raise ValueError("shrinkage grid values must be numbers") from error
    if any(not math.isfinite(value) or value < 0 for value in candidates):
        raise ValueError("shrinkage grid values must be finite and non-negative")
    dataset_path = Path(dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"paired value checkpoint already exists: {output_path}")
    manifest, rows = _load(dataset_path)
    splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    action_means, table = _statistics(splits["train"])
    validation_rows = splits["validation"]
    selection = []
    for raw, candidate in zip(shrinkage_grid, candidates):
        metrics = _evaluate(validation_rows, table, action_means, candidate)
        selection.append({"shrinkage": raw, "validation_mae_bb": metrics["mae_bb"]})
    selected = min(
        range(len(candidates)),
        key=lambda index: (float(selection[index]["validation_mae_bb"]), candidates[index]),
    )
    shrinkage = candidates[selected]
    metrics = {split: _evaluate(split_rows, table, action_means, shrinkage) for split, split_rows in splits.items()}
    test = metrics["test"]
    gate = (
        float(test["exact_hand_class_coverage"]) >= 0.95
        and float(test["mae_improvement_bb"]) > 0
        and float(test["by_action"]["call"]["mae_improvement_bb"]) > 0
        and float(test["by_action"]["raise"]["mae_improvement_bb"]) > 0
    )
    checkpoint = {
        "schema_version": "1.0.0",
        "model_kind": "preflop_hand_class_shrinkage_counterfactual_value_candidate",
        "status": "offline_paired_value_candidate_not_promoted",
        "units": "BB",
        "activation_authorized": False,
        "recommendation_allowed": False,
        "confidence_calibrated": False,
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "examples_sha256": manifest["examples_sha256"],
        },
        "configuration": {
            "supported_street": "preflop",
            "supported_actions": list(SUPPORTED_ACTIONS),
            "unsupported_actions": ["all_in"],
            "shrinkage_grid": list(shrinkage_grid),
            "selected_shrinkage_by_validation": shrinkage_grid[selected],
            "selection_trace": selection,
            "target": "paired_common_random_cards_terminal_return_bb",
            "continuation_policy": manifest["generation"]["continuation_policy"],
        },
        "action_train_means_bb": {action: format(action_means[action], ".12g") for action in SUPPORTED_ACTIONS},
        "hand_class_action_table": table,
        "metrics": metrics,
        "generalization_gate": {
            "passed": gate,
            "criterion": "fresh_test_overall_call_and_raise_mae_below_action_means_with_95pct_class_coverage",
            "activation_authorized": False,
        },
        "limitations": [
            "This model supports Hero-button preflop fold/call/minimum-raise values only and abstains on all-in.",
            "Values are against the declared deterministic check/call continuation policy, not GTO or a population opponent model.",
            "Even a passing offline error gate cannot authorize recommendations, confidence calibration or policy activation.",
        ],
    }
    checkpoint["checkpoint_fingerprint"] = _sha256(checkpoint)
    validation = validate_paired_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("paired value checkpoint failed validation: " + "; ".join(validation["issues"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temporary:
        temporary_file = Path(temporary) / output_path.name
        temporary_file.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        temporary_file.replace(output_path)
    return checkpoint


def validate_paired_value_checkpoint(checkpoint_or_path: dict[str, object] | str | Path) -> dict[str, object]:
    issues: list[str] = []
    if isinstance(checkpoint_or_path, dict):
        checkpoint = checkpoint_or_path
    else:
        try:
            checkpoint = json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {"valid": False, "issues": [f"checkpoint unreadable: {error}"]}
    if checkpoint.get("schema_version") != "1.0.0" or checkpoint.get("units") != "BB":
        issues.append("checkpoint schema/BB contract is invalid")
    if checkpoint.get("status") != "offline_paired_value_candidate_not_promoted":
        issues.append("checkpoint status is invalid")
    if any(checkpoint.get(key) is not False for key in ("activation_authorized", "recommendation_allowed", "confidence_calibrated")):
        issues.append("checkpoint cannot authorize activation, recommendations or confidence")
    configuration = checkpoint.get("configuration", {})
    if not isinstance(configuration, dict) or configuration.get("supported_actions") != list(SUPPORTED_ACTIONS) or configuration.get("unsupported_actions") != ["all_in"]:
        issues.append("checkpoint supported-action contract is invalid")
    table = checkpoint.get("hand_class_action_table")
    if not isinstance(table, dict) or not table:
        issues.append("checkpoint hand-class table is missing")
    else:
        try:
            valid_rows = all(
                isinstance(action_rows, dict)
                and set(action_rows) == set(SUPPORTED_ACTIONS)
                and all(int(row["samples"]) > 0 and math.isfinite(float(row["mean_return_bb"])) for row in action_rows.values())
                for action_rows in table.values()
            )
        except (KeyError, TypeError, ValueError):
            valid_rows = False
        if not valid_rows:
            issues.append("checkpoint hand-class rows are invalid or incomplete")
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict) or any(split not in metrics or metrics[split].get("examples", 0) <= 0 for split in ("train", "validation", "test")):
        issues.append("checkpoint requires non-empty split metrics")
    material = dict(checkpoint)
    observed = material.pop("checkpoint_fingerprint", None)
    if observed != _sha256(material):
        issues.append("checkpoint fingerprint mismatch")
    return {"schema_version": "1.0.0", "valid": not issues, "issues": issues, "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint")}


def predict_paired_value(
    checkpoint_or_path: dict[str, object] | str | Path,
    state: dict[str, object],
    command: dict[str, object],
) -> dict[str, object]:
    checkpoint = checkpoint_or_path if isinstance(checkpoint_or_path, dict) else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    validation = validate_paired_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("paired value checkpoint is invalid: " + "; ".join(validation["issues"]))
    issues = sorted(set([*value_state_issues(state), *action_issues(state, command)]))
    action = str(command.get("action", "")) if isinstance(command, dict) else ""
    if state.get("street") != "preflop":
        issues.append("street_not_supported")
    if state.get("hero_position") != "BTN":
        issues.append("hero_position_not_supported")
    if action not in SUPPORTED_ACTIONS:
        issues.append("action_not_supported")
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
    hand_class = _hand_class([str(card) for card in state["hero_cards"]])
    action_means = {key: float(value) for key, value in checkpoint["action_train_means_bb"].items()}
    prediction, exact, samples = _predict(
        checkpoint["hand_class_action_table"],
        action_means,
        hand_class,
        action,
        float(checkpoint["configuration"]["selected_shrinkage_by_validation"]),
    )
    result = {
        "schema_version": "1.0.0",
        "status": "offline_paired_value_prediction_uncalibrated",
        "hero_hand_class": hand_class,
        "action": command,
        "predicted_terminal_return_bb": format(prediction, ".12g"),
        "exact_hand_class_coverage": exact,
        "training_samples": samples,
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or validate APC's paired preflop value candidate.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_paired_value_checkpoint(args.checkpoint)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        checkpoint = train_paired_value_model(args.dataset, args.output)
        print(json.dumps(checkpoint, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
