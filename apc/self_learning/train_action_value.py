from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from apc.self_learning.train_value import (
    _canonical,
    _load_dataset,
    _sha256,
    _target,
    value_feature_tokens,
    value_state_issues,
)


def action_issues(state: dict[str, object], command: dict[str, object]) -> list[str]:
    if not isinstance(command, dict):
        return ["command_must_be_an_object"]
    issues: list[str] = []
    action = command.get("action")
    buttons = state.get("action_buttons") if isinstance(state, dict) else None
    legal = {
        str(row.get("action")): row
        for row in buttons
        if isinstance(buttons, list) and isinstance(row, dict) and isinstance(row.get("action"), str)
    } if isinstance(buttons, list) else {}
    if not isinstance(action, str) or action not in legal:
        issues.append("action_not_in_visible_legal_set")
        return issues
    allowed_fields = {"action", "amount_bb", "to_amount_bb"}
    if any(key not in allowed_fields for key in command):
        issues.append("command_field_not_allowed")
    decimals: dict[str, Decimal] = {}
    for field in ("amount_bb", "to_amount_bb"):
        if field not in command:
            continue
        if not isinstance(command[field], str):
            issues.append(f"{field}_must_be_a_BB_string")
            continue
        try:
            value = Decimal(command[field])
        except InvalidOperation:
            issues.append(f"{field}_invalid")
            continue
        if not value.is_finite() or value < 0:
            issues.append(f"{field}_invalid")
        else:
            decimals[field] = value
    if action in {"fold", "check"} and decimals:
        issues.append("passive_action_must_not_have_size")
    if action == "call":
        expected = legal[action].get("amount_bb")
        if expected is None or decimals.get("amount_bb") != Decimal(str(expected)):
            issues.append("call_amount_does_not_match_visible_price")
    if action in {"bet", "raise"}:
        target = decimals.get("to_amount_bb")
        try:
            minimum = Decimal(str(legal[action]["minimum_to_bb"]))
            maximum = Decimal(str(legal[action]["maximum_to_bb"]))
        except (InvalidOperation, KeyError):
            issues.append("visible_size_range_invalid")
        else:
            if target is None or target < minimum or target > maximum:
                issues.append("aggressive_target_outside_visible_range")
    if action == "all_in":
        expected = legal[action].get("to_amount_bb")
        if expected is None or decimals.get("to_amount_bb") != Decimal(str(expected)):
            issues.append("all_in_target_does_not_match_visible_button")
    return sorted(set(issues))


def _poker_shape_tokens(state: dict[str, object]) -> tuple[str, ...]:
    rank_value = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}
    hero = [str(card) for card in state["hero_cards"]]
    board = [str(card) for card in state["board"]]
    hero_ranks = sorted((rank_value[card[0].upper()] for card in hero), reverse=True)
    gap = hero_ranks[0] - hero_ranks[1]
    tokens = [
        f"hero_high_rank={hero_ranks[0]}",
        f"hero_low_rank={hero_ranks[1]}",
        f"hero_pair={hero_ranks[0] == hero_ranks[1]}",
        f"hero_suited={hero[0][-1] == hero[1][-1]}",
        f"hero_gap={min(gap, 5)}",
        f"hero_broadway_count={sum(value >= 10 for value in hero_ranks)}",
    ]
    known = [*hero, *board]
    rank_counts: defaultdict[int, int] = defaultdict(int)
    suit_counts: defaultdict[str, int] = defaultdict(int)
    for card in known:
        rank_counts[rank_value[card[0].upper()]] += 1
        suit_counts[card[-1]] += 1
    if board:
        tokens.extend([
            f"known_max_rank_count={max(rank_counts.values())}",
            f"known_max_suit_count={max(suit_counts.values())}",
            f"board_paired={len({card[0] for card in board}) < len(board)}",
            f"hero_rank_hits_board={sum(card[0] in {row[0] for row in board} for card in hero)}",
        ])
    return tuple(tokens)


def action_value_feature_tokens(state: dict[str, object], command: dict[str, object]) -> tuple[str, ...]:
    state_issues = value_state_issues(state)
    command_issues = action_issues(state, command)
    if state_issues or command_issues:
        raise ValueError("invalid action-value input: " + ",".join([*state_issues, *command_issues]))
    action = str(command["action"])
    base = (*value_feature_tokens(state), *_poker_shape_tokens(state))
    tokens = [*base, f"chosen_action={action}"]
    tokens.extend(f"chosen_{key}={value}" for key, value in sorted(command.items()) if key != "action")
    tokens.extend(f"action_cross={action}|{token}" for token in base)
    return tuple(tokens)


def action_value_features(state: dict[str, object], command: dict[str, object], dimension: int) -> dict[int, float]:
    if dimension < 64:
        raise ValueError("action-value feature dimension must be at least 64")
    vector: dict[int, float] = defaultdict(float)
    import hashlib

    for token in action_value_feature_tokens(state, command):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        vector[index] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        raise ValueError("action-value feature vector cannot be zero")
    return {index: value / norm for index, value in vector.items()}


def _action(row: dict[str, object]) -> str:
    return str(row["behavior"]["chosen_action"]["action"])


def _action_means(rows: list[dict[str, object]]) -> tuple[float, dict[str, float]]:
    global_mean = sum(_target(row) for row in rows) / len(rows)
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_action(row)].append(_target(row))
    return global_mean, {action: sum(values) / len(values) for action, values in grouped.items()}


def _prediction(
    weights: list[float],
    features: dict[int, float],
    action: str,
    global_mean: float,
    action_means: dict[str, float],
) -> float:
    baseline = action_means.get(action, global_mean)
    return max(-100.0, min(100.0, baseline + sum(weights[index] * value for index, value in features.items())))


def _evaluate(
    rows: list[dict[str, object]],
    weights: list[float],
    dimension: int,
    global_mean: float,
    action_means: dict[str, float],
) -> dict[str, object]:
    actual: list[float] = []
    predicted: list[float] = []
    action_baseline: list[float] = []
    for row in rows:
        action = _action(row)
        actual.append(_target(row))
        action_baseline.append(action_means.get(action, global_mean))
        predicted.append(_prediction(
            weights,
            action_value_features(row["state"], row["behavior"]["chosen_action"], dimension),
            action,
            global_mean,
            action_means,
        ))
    errors = [prediction - target for prediction, target in zip(predicted, actual)]
    action_errors = [prediction - target for prediction, target in zip(action_baseline, actual)]
    global_errors = [global_mean - target for target in actual]
    order = sorted(range(len(rows)), key=lambda index: (predicted[index], str(rows[index]["example_id"])))
    bins = []
    for bin_index in range(5):
        indices = order[bin_index * len(order) // 5 : (bin_index + 1) * len(order) // 5]
        if indices:
            bins.append({
                "examples": len(indices),
                "mean_prediction_bb": format(sum(predicted[index] for index in indices) / len(indices), ".12g"),
                "mean_actual_bb": format(sum(actual[index] for index in indices) / len(indices), ".12g"),
            })
    count = len(rows)
    mae = sum(abs(value) for value in errors) / count
    action_mae = sum(abs(value) for value in action_errors) / count
    return {
        "examples": count,
        "mae_bb": format(mae, ".12g"),
        "rmse_bb": format(math.sqrt(sum(value * value for value in errors) / count), ".12g"),
        "bias_bb": format(sum(errors) / count, ".12g"),
        "action_mean_baseline_mae_bb": format(action_mae, ".12g"),
        "global_train_mean_baseline_mae_bb": format(sum(abs(value) for value in global_errors) / count, ".12g"),
        "mae_improvement_vs_action_baseline_bb": format(action_mae - mae, ".12g"),
        "sign_accuracy": format(sum((prediction >= 0) == (target >= 0) for prediction, target in zip(predicted, actual)) / count, ".12g"),
        "calibration_bins": bins,
    }


def train_action_value_model(
    dataset: str | Path,
    output: str | Path,
    *,
    seed: int = 20260816,
    feature_dimension: int = 1024,
    epochs: int = 100,
    learning_rate: float = 0.015,
    l2: float = 0.001,
) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if feature_dimension < 64 or epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("action-value training configuration is invalid")
    dataset_path = Path(dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"action-value checkpoint already exists: {output_path}")
    manifest, examples = _load_dataset(dataset_path)
    splits = {split: [row for row in examples if row["split"] == split] for split in ("train", "validation", "test")}
    global_mean, action_means = _action_means(splits["train"])
    if any(_action(row) not in action_means for split in ("validation", "test") for row in splits[split]):
        raise ValueError("fresh evaluation contains an action absent from training")
    cache = {
        str(row["example_id"]): action_value_features(row["state"], row["behavior"]["chosen_action"], feature_dimension)
        for row in examples
    }
    weights = [0.0] * feature_dimension
    best_weights = list(weights)
    best_epoch = 0
    best_validation_mae = float(_evaluate(splits["validation"], weights, feature_dimension, global_mean, action_means)["mae_bb"])
    for epoch in range(1, epochs + 1):
        ordered = list(splits["train"])
        random.Random(seed + epoch).shuffle(ordered)
        rate = learning_rate / math.sqrt(1.0 + epoch / 20.0)
        for row in ordered:
            features = cache[str(row["example_id"])]
            action = _action(row)
            prediction = _prediction(weights, features, action, global_mean, action_means)
            error = max(-200.0, min(200.0, prediction - _target(row)))
            for index, value in features.items():
                weights[index] -= rate * (error * value + l2 * weights[index])
        validation_mae = float(_evaluate(splits["validation"], weights, feature_dimension, global_mean, action_means)["mae_bb"])
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_weights = list(weights)
    metrics = {
        split: _evaluate(rows, best_weights, feature_dimension, global_mean, action_means)
        for split, rows in splits.items()
    }
    test_improvement = Decimal(metrics["test"]["mae_improvement_vs_action_baseline_bb"])
    checkpoint = {
        "schema_version": "1.0.0",
        "model_kind": "hashed_linear_action_conditioned_terminal_return_candidate",
        "status": "offline_action_value_candidate_not_promoted",
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
            "seed": seed,
            "feature_dimension": feature_dimension,
            "maximum_epochs": epochs,
            "selected_epoch_by_validation": best_epoch,
            "learning_rate": format(learning_rate, ".12g"),
            "l2": format(l2, ".12g"),
            "feature_schema": "hashed_virtual_hero_state_action_cross_v1",
            "target": "sampled_monte_carlo_terminal_return_bb",
            "global_train_mean_bb": format(global_mean, ".12g"),
            "action_train_means_bb": {key: format(value, ".12g") for key, value in sorted(action_means.items())},
            "prediction_bounds_bb": ["-100", "100"],
        },
        "weights": [format(value, ".12g") for value in best_weights],
        "metrics": metrics,
        "generalization_gate": {
            "passed": test_improvement > 0,
            "criterion": "fresh_test_mae_below_train_action_mean_baseline",
            "test_mae_improvement_bb": str(test_improvement),
            "activation_authorized": False,
        },
        "limitations": [
            "This estimates returns of observed coverage-probe actions, not counterfactual values for every legal action.",
            "The target is a sampled terminal return rather than a solver or GTO label.",
            "Passing an error baseline does not authorize coaching, confidence calibration or policy activation.",
        ],
    }
    checkpoint["checkpoint_fingerprint"] = _sha256(checkpoint)
    validation = validate_action_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("action-value checkpoint failed validation: " + "; ".join(validation["issues"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temporary:
        temporary_file = Path(temporary) / output_path.name
        temporary_file.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        temporary_file.replace(output_path)
    return checkpoint


def validate_action_value_checkpoint(checkpoint_or_path: dict[str, object] | str | Path) -> dict[str, object]:
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
    if checkpoint.get("status") != "offline_action_value_candidate_not_promoted":
        issues.append("checkpoint status is invalid")
    if any(checkpoint.get(key) is not False for key in ("activation_authorized", "recommendation_allowed", "confidence_calibrated")):
        issues.append("checkpoint cannot authorize activation, recommendations or confidence")
    configuration = checkpoint.get("configuration", {})
    dimension = configuration.get("feature_dimension") if isinstance(configuration, dict) else None
    weights = checkpoint.get("weights")
    if not isinstance(dimension, int) or dimension < 64 or not isinstance(weights, list) or len(weights) != dimension:
        issues.append("checkpoint feature/weight dimensions are invalid")
    else:
        try:
            finite = all(math.isfinite(float(value)) for value in weights)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            issues.append("checkpoint weights must be finite")
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict) or any(
        split not in metrics or not isinstance(metrics[split], dict) or metrics[split].get("examples", 0) <= 0
        for split in ("train", "validation", "test")
    ):
        issues.append("checkpoint requires non-empty split metrics")
    material = dict(checkpoint)
    observed = material.pop("checkpoint_fingerprint", None)
    if observed != _sha256(material):
        issues.append("checkpoint fingerprint mismatch")
    return {"schema_version": "1.0.0", "valid": not issues, "issues": issues, "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint")}


def predict_action_value(
    checkpoint_or_path: dict[str, object] | str | Path,
    state: dict[str, object],
    command: dict[str, object],
) -> dict[str, object]:
    checkpoint = checkpoint_or_path if isinstance(checkpoint_or_path, dict) else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    validation = validate_action_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("action-value checkpoint is invalid: " + "; ".join(validation["issues"]))
    issues = sorted(set([*value_state_issues(state), *action_issues(state, command)]))
    if issues:
        return {
            "schema_version": "1.0.0",
            "status": "abstain_invalid_state_or_action",
            "predicted_terminal_return_bb": None,
            "reasons": issues,
            "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
            "recommendation_allowed": False,
            "activation_authorized": False,
        }
    configuration = checkpoint["configuration"]
    action = str(command["action"])
    action_means = {key: float(value) for key, value in configuration["action_train_means_bb"].items()}
    prediction = _prediction(
        [float(value) for value in checkpoint["weights"]],
        action_value_features(state, command, configuration["feature_dimension"]),
        action,
        float(configuration["global_train_mean_bb"]),
        action_means,
    )
    result = {
        "schema_version": "1.0.0",
        "status": "offline_action_value_prediction_uncalibrated",
        "action": command,
        "predicted_terminal_return_bb": format(prediction, ".12g"),
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or validate APC's offline action-value candidate.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--seed", type=int, default=20260816)
    train.add_argument("--feature-dimension", type=int, default=1024)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=0.015)
    train.add_argument("--l2", type=float, default=0.001)
    validate = subparsers.add_parser("validate")
    validate.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_action_value_checkpoint(args.checkpoint)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        checkpoint = train_action_value_model(
            args.dataset,
            args.output,
            seed=args.seed,
            feature_dimension=args.feature_dimension,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        print(json.dumps(checkpoint, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
