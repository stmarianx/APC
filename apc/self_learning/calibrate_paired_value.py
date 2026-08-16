from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path

from apc.self_learning.paired_rollout_dataset import _hand_class, validate_paired_rollout_dataset
from apc.self_learning.train_paired_value import (
    _predict,
    predict_paired_value,
    validate_paired_value_checkpoint,
)
from apc.self_learning.train_value import _sha256


ACTIONS = ("call", "raise")
CALIBRATION_THRESHOLDS = {
    "call_eace_bb_max": "0.12",
    "raise_eace_bb_max": "0.24",
    "call_max_bin_gap_bb_max": "0.45",
    "raise_max_bin_gap_bb_max": "0.90",
    "mae_regression_bb_max": "0.01",
    "exact_hand_class_coverage_min": "0.95",
}


def _read_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _base_predictions(
    rows: list[dict[str, object]], checkpoint: dict[str, object]
) -> list[dict[str, object]]:
    configuration = checkpoint["configuration"]
    table = checkpoint["hand_class_action_table"]
    action_means = {key: float(value) for key, value in checkpoint["action_train_means_bb"].items()}
    shrinkage = float(configuration["selected_shrinkage_by_validation"])
    output = []
    for row in rows:
        action = str(row["counterfactual_action"]["action"])
        hand_class = _hand_class([str(card) for card in row["state"]["hero_cards"]])
        prediction, covered, samples = _predict(table, action_means, hand_class, action, shrinkage)
        output.append({
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "action": action,
            "raw_prediction_bb": prediction,
            "actual_bb": float(str(row["learning_signal"]["hero_return_bb"])),
            "exact_hand_class_coverage": covered,
            "training_samples": samples,
        })
    return output


def _fit_affine(rows: list[dict[str, object]]) -> dict[str, str]:
    x = [float(row["raw_prediction_bb"]) for row in rows]
    y = [float(row["actual_bb"]) for row in rows]
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    variance = sum((value - mean_x) ** 2 for value in x)
    slope = 0.0 if variance == 0 else sum(
        (left - mean_x) * (right - mean_y) for left, right in zip(x, y)
    ) / variance
    slope = min(1.0, max(0.0, slope))
    intercept = mean_y - slope * mean_x
    return {"intercept_bb": format(intercept, ".12g"), "slope": format(slope, ".12g")}


def _apply(value: float, parameters: dict[str, str], action: str) -> float:
    calibrated = float(parameters["intercept_bb"]) + float(parameters["slope"]) * value
    bound = 1.0 if action == "call" else 2.0
    return max(-bound, min(bound, calibrated))


def _metrics(rows: list[dict[str, object]], parameters: dict[str, str], bins: int) -> dict[str, object]:
    material = []
    for row in rows:
        raw = float(row["raw_prediction_bb"])
        actual = float(row["actual_bb"])
        calibrated = _apply(raw, parameters, str(row["action"]))
        material.append((raw, calibrated, actual, str(row["example_id"])))
    ordered = sorted(material, key=lambda row: (row[1], row[3]))
    calibration = []
    weighted_gap = 0.0
    for index in range(bins):
        members = ordered[index * len(ordered) // bins : (index + 1) * len(ordered) // bins]
        if not members:
            continue
        predicted = statistics.fmean(row[1] for row in members)
        actual = statistics.fmean(row[2] for row in members)
        gap = abs(predicted - actual)
        weighted_gap += gap * len(members)
        calibration.append({
            "examples": len(members),
            "mean_prediction_bb": format(predicted, ".12g"),
            "mean_actual_bb": format(actual, ".12g"),
            "absolute_gap_bb": format(gap, ".12g"),
        })
    raw_mae = statistics.fmean(abs(row[0] - row[2]) for row in material)
    calibrated_mae = statistics.fmean(abs(row[1] - row[2]) for row in material)
    return {
        "examples": len(material),
        "raw_mae_bb": format(raw_mae, ".12g"),
        "calibrated_mae_bb": format(calibrated_mae, ".12g"),
        "mae_change_bb": format(calibrated_mae - raw_mae, ".12g"),
        "calibrated_bias_bb": format(statistics.fmean(row[1] - row[2] for row in material), ".12g"),
        "expected_absolute_calibration_error_bb": format(weighted_gap / len(material), ".12g"),
        "maximum_bin_gap_bb": format(max(float(row["absolute_gap_bb"]) for row in calibration), ".12g"),
        "bins": calibration,
    }


def calibrate_paired_value(
    calibration_dataset: str | Path,
    base_checkpoint: str | Path,
    output: str | Path,
    *,
    bins: int = 10,
) -> dict[str, object]:
    if bins < 5:
        raise ValueError("calibration requires at least five bins")
    dataset_path = Path(calibration_dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"calibration artifact already exists: {output_path}")
    dataset_validation = validate_paired_rollout_dataset(dataset_path)
    if not dataset_validation["valid"]:
        raise ValueError("calibration dataset is invalid: " + "; ".join(dataset_validation["issues"]))
    manifest = _read_json(dataset_path / "manifest.json")
    checkpoint = _read_json(base_checkpoint)
    checkpoint_validation = validate_paired_value_checkpoint(checkpoint)
    if not checkpoint_validation["valid"]:
        raise ValueError("base checkpoint is invalid: " + "; ".join(checkpoint_validation["issues"]))
    if manifest["dataset_fingerprint"] == checkpoint["dataset"]["dataset_fingerprint"]:
        raise ValueError("calibration corpus must be independent from base training/evaluation data")
    examples = [
        json.loads(line)
        for line in (dataset_path / str(manifest["examples_file"])).read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [
        row
        for row in examples
        if row["counterfactual_action"]["action"] in ACTIONS
    ]
    splits = {
        split: _base_predictions([row for row in selected if row["split"] == split], checkpoint)
        for split in ("validation", "test")
    }
    parameters = {
        action: _fit_affine([row for row in splits["validation"] if row["action"] == action])
        for action in ACTIONS
    }
    test_metrics = {
        action: _metrics(
            [row for row in splits["test"] if row["action"] == action],
            parameters[action],
            bins,
        )
        for action in ACTIONS
    }
    coverage = statistics.fmean(
        1.0 if row["exact_hand_class_coverage"] else 0.0 for row in splits["test"]
    )
    thresholds = {key: float(value) for key, value in CALIBRATION_THRESHOLDS.items()}
    gate = (
        coverage >= thresholds["exact_hand_class_coverage_min"]
        and float(test_metrics["call"]["expected_absolute_calibration_error_bb"]) <= thresholds["call_eace_bb_max"]
        and float(test_metrics["raise"]["expected_absolute_calibration_error_bb"]) <= thresholds["raise_eace_bb_max"]
        and float(test_metrics["call"]["maximum_bin_gap_bb"]) <= thresholds["call_max_bin_gap_bb_max"]
        and float(test_metrics["raise"]["maximum_bin_gap_bb"]) <= thresholds["raise_max_bin_gap_bb_max"]
        and all(float(test_metrics[action]["mae_change_bb"]) <= thresholds["mae_regression_bb_max"] for action in ACTIONS)
    )
    artifact = {
        "schema_version": "1.0.0",
        "artifact_kind": "paired_preflop_value_affine_calibration",
        "status": "calibration_gate_passed" if gate else "calibration_gate_failed",
        "units": "BB",
        "base_checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "base_dataset_fingerprint": checkpoint["dataset"]["dataset_fingerprint"],
        "calibration_dataset": {
            "dataset_id": manifest["dataset_id"],
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "examples_sha256": manifest["examples_sha256"],
            "validation_examples": len(splits["validation"]),
            "test_examples": len(splits["test"]),
        },
        "selection": {
            "fit_split": "validation",
            "evaluation_split": "test",
            "method": "per_action_nonnegative_unit_slope_affine",
            "test_used_for_fit": False,
            "bins": bins,
        },
        "parameters": parameters,
        "test_metrics": test_metrics,
        "exact_hand_class_coverage": format(coverage, ".12g"),
        "thresholds_declared_before_test": dict(CALIBRATION_THRESHOLDS),
        "calibration_gate": {
            "passed": gate,
            "confidence_calibrated": gate,
            "activation_authorized": False,
            "recommendation_allowed": False,
        },
        "limitations": [
            "Calibration applies only to Hero-button preflop call/minimum-raise values against the declared check/call policy.",
            "Calibrated terminal-return values are not action-optimality probabilities or GTO frequencies.",
            "A passing calibration gate still cannot authorize coaching recommendations or activation.",
        ],
    }
    artifact["calibration_fingerprint"] = _sha256(artifact)
    validation = validate_paired_value_calibration(artifact)
    if not validation["valid"]:
        raise ValueError("calibration artifact failed validation: " + "; ".join(validation["issues"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temporary:
        temporary_file = Path(temporary) / output_path.name
        temporary_file.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        temporary_file.replace(output_path)
    return artifact


def validate_paired_value_calibration(artifact_or_path: dict[str, object] | str | Path) -> dict[str, object]:
    issues: list[str] = []
    if isinstance(artifact_or_path, dict):
        artifact = artifact_or_path
    else:
        try:
            artifact = _read_json(artifact_or_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return {"valid": False, "issues": [f"calibration unreadable: {error}"]}
    if artifact.get("schema_version") != "1.0.0" or artifact.get("units") != "BB":
        issues.append("calibration schema/BB contract is invalid")
    gate = artifact.get("calibration_gate", {})
    if not isinstance(gate, dict) or gate.get("activation_authorized") is not False or gate.get("recommendation_allowed") is not False:
        issues.append("calibration cannot authorize recommendations or activation")
    parameters = artifact.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(ACTIONS):
        issues.append("calibration parameters are invalid")
    else:
        try:
            finite = all(
                math.isfinite(float(row["intercept_bb"])) and 0 <= float(row["slope"]) <= 1
                for row in parameters.values()
            )
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            issues.append("calibration parameters must be finite and constrained")
    if artifact.get("thresholds_declared_before_test") != CALIBRATION_THRESHOLDS:
        issues.append("calibration thresholds do not match the declared contract")
    material = dict(artifact)
    observed = material.pop("calibration_fingerprint", None)
    if observed != _sha256(material):
        issues.append("calibration fingerprint mismatch")
    return {"schema_version": "1.0.0", "valid": not issues, "issues": issues, "calibration_fingerprint": artifact.get("calibration_fingerprint")}


def predict_calibrated_paired_value(
    base_checkpoint: dict[str, object] | str | Path,
    calibration: dict[str, object] | str | Path,
    state: dict[str, object],
    command: dict[str, object],
) -> dict[str, object]:
    checkpoint = base_checkpoint if isinstance(base_checkpoint, dict) else _read_json(base_checkpoint)
    artifact = calibration if isinstance(calibration, dict) else _read_json(calibration)
    validation = validate_paired_value_calibration(artifact)
    if not validation["valid"]:
        raise ValueError("calibration artifact is invalid: " + "; ".join(validation["issues"]))
    if artifact["base_checkpoint_fingerprint"] != checkpoint.get("checkpoint_fingerprint"):
        raise ValueError("calibration does not match the base checkpoint")
    base = predict_paired_value(checkpoint, state, command)
    if base["predicted_terminal_return_bb"] is None:
        return base
    action = str(command["action"])
    calibrated = _apply(float(base["predicted_terminal_return_bb"]), artifact["parameters"][action], action)
    result = {
        "schema_version": "1.0.0",
        "status": "offline_calibrated_paired_value" if artifact["calibration_gate"]["passed"] else "offline_paired_value_calibration_failed",
        "action": command,
        "raw_terminal_return_bb": base["predicted_terminal_return_bb"],
        "calibrated_terminal_return_bb": format(calibrated, ".12g"),
        "base_checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "calibration_fingerprint": artifact["calibration_fingerprint"],
        "confidence_calibrated": artifact["calibration_gate"]["passed"],
        "recommendation_allowed": False,
        "activation_authorized": False,
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate or validate APC paired preflop values.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("calibration_dataset", type=Path)
    fit.add_argument("base_checkpoint", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--bins", type=int, default=10)
    validate = subparsers.add_parser("validate")
    validate.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_paired_value_calibration(args.artifact)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        artifact = calibrate_paired_value(args.calibration_dataset, args.base_checkpoint, args.output, bins=args.bins)
        print(json.dumps(artifact, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
