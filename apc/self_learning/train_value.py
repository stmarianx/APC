from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from apc.self_learning.full_hand_dataset import validate_full_hand_dataset
from apc.self_learning.train_candidate import _canonical_cards


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def value_state_issues(state: dict[str, object]) -> list[str]:
    if not isinstance(state, dict):
        return ["state_must_be_an_object"]
    issues: list[str] = []
    if state.get("environment") != "controlled_virtual_chips" or state.get("units") != "BB":
        issues.append("environment_or_units_invalid")
    if state.get("game") != "holdem_no_limit" or state.get("scope") != "complete_heads_up_hand":
        issues.append("game_or_scope_invalid")
    if state.get("next_actor") != "Hero" or state.get("terminal") is not False:
        issues.append("state_is_not_a_live_hero_decision")
    if state.get("opponent_cards") is not None:
        issues.append("opponent_cards_must_be_hidden")
    board = state.get("board")
    hero_cards = state.get("hero_cards")
    if not isinstance(board, list) or len(board) not in (0, 3, 4, 5):
        issues.append("board_invalid")
    if not isinstance(hero_cards, list) or len(hero_cards) != 2:
        issues.append("hero_cards_invalid")
    if isinstance(board, list) and isinstance(hero_cards, list):
        try:
            _canonical_cards(board, hero_cards)
        except ValueError:
            issues.append("card_token_invalid")
        if len({str(card) for card in [*board, *hero_cards]}) != len(board) + len(hero_cards):
            issues.append("duplicate_known_cards")
    for field in ("pot_bb", "to_call_bb"):
        try:
            value = Decimal(str(state.get(field)))
        except (InvalidOperation, ValueError):
            issues.append(f"{field}_invalid")
            continue
        if not value.is_finite() or value < 0 or (field == "pot_bb" and value <= 0):
            issues.append(f"{field}_invalid")
    stacks = state.get("stacks_bb")
    if not isinstance(stacks, dict) or set(stacks) != {"Hero", "Villain"}:
        issues.append("stacks_invalid")
    else:
        try:
            parsed_stacks = [Decimal(str(stacks[player])) for player in ("Hero", "Villain")]
        except (InvalidOperation, ValueError):
            issues.append("stacks_invalid")
        else:
            if any(not value.is_finite() or value < 0 for value in parsed_stacks):
                issues.append("stacks_invalid")
    history = state.get("action_history")
    buttons = state.get("action_buttons")
    if not isinstance(history, list) or any(not isinstance(value, str) for value in history):
        issues.append("action_history_invalid")
    if not isinstance(buttons, list) or not buttons or any(
        not isinstance(row, dict) or not isinstance(row.get("action"), str) for row in buttons
    ):
        issues.append("legal_actions_invalid")
    provider = state.get("provider")
    if not isinstance(provider, dict) or provider.get("external_actuation") is not False:
        issues.append("provider_contract_invalid")
    fingerprint = state.get("state_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        issues.append("state_fingerprint_invalid")
    return sorted(set(issues))


def _bucket(value: object, width: Decimal) -> str:
    number = Decimal(str(value))
    return format((number // width) * width, "f")


def value_feature_tokens(state: dict[str, object]) -> tuple[str, ...]:
    issues = value_state_issues(state)
    if issues:
        raise ValueError("invalid value-model state: " + ",".join(issues))
    board, hero_cards = _canonical_cards(state["board"], state["hero_cards"])
    stacks = state["stacks_bb"]
    tokens = [
        "bias",
        f"street={state['street']}",
        f"position={state['hero_position']}",
        f"pot_2bb={_bucket(state['pot_bb'], Decimal('2'))}",
        f"to_call_1bb={_bucket(state['to_call_bb'], Decimal('1'))}",
        f"hero_stack_5bb={_bucket(stacks['Hero'], Decimal('5'))}",
        f"effective_stack_5bb={_bucket(min(Decimal(stacks['Hero']), Decimal(stacks['Villain'])), Decimal('5'))}",
        f"history_length={len(state['action_history'])}",
    ]
    tokens.extend(f"board[{index}]={card}" for index, card in enumerate(board))
    tokens.extend(f"hero[{index}]={card}" for index, card in enumerate(hero_cards))
    tokens.extend(f"history[{index}]={action}" for index, action in enumerate(state["action_history"]))
    tokens.extend(f"legal={row['action']}" for row in state["action_buttons"])
    return tuple(tokens)


def value_features(state: dict[str, object], dimension: int) -> dict[int, float]:
    if dimension < 32:
        raise ValueError("value feature dimension must be at least 32")
    vector: dict[int, float] = defaultdict(float)
    for token in value_feature_tokens(state):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        vector[index] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        raise ValueError("value feature vector cannot be zero")
    return {index: value / norm for index, value in vector.items()}


def _load_dataset(dataset: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    validation = validate_full_hand_dataset(dataset)
    if not validation["valid"]:
        raise ValueError("full-hand dataset is invalid: " + "; ".join(validation["issues"]))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_eligible") is not True:
        raise ValueError("full-hand dataset is not training eligible")
    rows = [
        json.loads(line)
        for line in (dataset / str(manifest["examples_file"])).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not any(row["split"] == split for row in rows) for split in ("train", "validation", "test")):
        raise ValueError("value training requires non-empty train, validation and test splits")
    return manifest, rows


def _target(row: dict[str, object]) -> float:
    return float(str(row["learning_signal"]["hero_return_bb"]))


def _predict(weights: list[float], features: dict[int, float], mean: float) -> float:
    return max(-100.0, min(100.0, mean + sum(weights[index] * value for index, value in features.items())))


def _metrics(rows: list[dict[str, object]], weights: list[float], dimension: int, mean: float) -> dict[str, object]:
    actual = [_target(row) for row in rows]
    predicted = [_predict(weights, value_features(row["state"], dimension), mean) for row in rows]
    errors = [prediction - target for prediction, target in zip(predicted, actual)]
    baseline_errors = [mean - target for target in actual]
    order = sorted(range(len(rows)), key=lambda index: (predicted[index], str(rows[index]["example_id"])))
    bins: list[dict[str, object]] = []
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
    baseline_mae = sum(abs(value) for value in baseline_errors) / count
    return {
        "examples": count,
        "mae_bb": format(mae, ".12g"),
        "rmse_bb": format(math.sqrt(sum(value * value for value in errors) / count), ".12g"),
        "bias_bb": format(sum(errors) / count, ".12g"),
        "baseline_train_mean_mae_bb": format(baseline_mae, ".12g"),
        "mae_improvement_vs_baseline_bb": format(baseline_mae - mae, ".12g"),
        "sign_accuracy": format(sum((prediction >= 0) == (target >= 0) for prediction, target in zip(predicted, actual)) / count, ".12g"),
        "calibration_bins": bins,
    }


def train_value_model(
    dataset: str | Path,
    output: str | Path,
    *,
    seed: int = 20260816,
    feature_dimension: int = 256,
    epochs: int = 120,
    learning_rate: float = 0.02,
    l2: float = 0.0005,
) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if feature_dimension < 32 or epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("value training configuration is invalid")
    dataset_path = Path(dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"value checkpoint already exists: {output_path}")
    manifest, examples = _load_dataset(dataset_path)
    splits = {split: [row for row in examples if row["split"] == split] for split in ("train", "validation", "test")}
    train_mean = sum(_target(row) for row in splits["train"]) / len(splits["train"])
    weights = [0.0] * feature_dimension
    for epoch in range(epochs):
        ordered = list(splits["train"])
        random.Random(seed + epoch).shuffle(ordered)
        rate = learning_rate / math.sqrt(1.0 + epoch / 20.0)
        for row in ordered:
            features = value_features(row["state"], feature_dimension)
            prediction = _predict(weights, features, train_mean)
            error = max(-200.0, min(200.0, prediction - _target(row)))
            for index, value in features.items():
                weights[index] -= rate * (error * value + l2 * weights[index])
    serialized_weights = [format(value, ".12g") for value in weights]
    metrics = {split: _metrics(rows, weights, feature_dimension, train_mean) for split, rows in splits.items()}
    test_improvement = Decimal(metrics["test"]["mae_improvement_vs_baseline_bb"])
    checkpoint = {
        "schema_version": "1.0.0",
        "model_kind": "hashed_linear_terminal_return_value_candidate",
        "status": "offline_value_candidate_not_promoted",
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
            "epochs": epochs,
            "learning_rate": format(learning_rate, ".12g"),
            "l2": format(l2, ".12g"),
            "feature_schema": "hashed_virtual_hero_state_v1",
            "target": "sampled_monte_carlo_terminal_return_bb",
            "train_target_mean_bb": format(train_mean, ".12g"),
            "prediction_bounds_bb": ["-100", "100"],
        },
        "weights": serialized_weights,
        "metrics": metrics,
        "generalization_gate": {
            "passed": test_improvement > 0,
            "criterion": "test_mae_below_train_mean_baseline",
            "test_mae_improvement_bb": str(test_improvement),
            "activation_authorized": False,
        },
        "limitations": [
            "Targets are sampled returns under deterministic coverage behavior, not counterfactual action values or GTO labels.",
            "Repeated decisions in a hand share one return; split assignment is hand-exclusive.",
            "The structured-state model is not a visible-table perception checkpoint and cannot produce coaching recommendations.",
        ],
    }
    checkpoint["checkpoint_fingerprint"] = _sha256(checkpoint)
    validation = validate_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("value checkpoint failed validation: " + "; ".join(validation["issues"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temporary:
        temporary_file = Path(temporary) / output_path.name
        temporary_file.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        temporary_file.replace(output_path)
    return checkpoint


def validate_value_checkpoint(checkpoint_or_path: dict[str, object] | str | Path) -> dict[str, object]:
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
    if checkpoint.get("status") != "offline_value_candidate_not_promoted":
        issues.append("checkpoint status is invalid")
    if any(checkpoint.get(key) is not False for key in ("activation_authorized", "recommendation_allowed", "confidence_calibrated")):
        issues.append("value checkpoint cannot authorize activation, confidence or recommendations")
    configuration = checkpoint.get("configuration", {})
    dimension = configuration.get("feature_dimension") if isinstance(configuration, dict) else None
    weights = checkpoint.get("weights")
    if not isinstance(dimension, int) or dimension < 32 or not isinstance(weights, list) or len(weights) != dimension:
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


def predict_value(checkpoint_or_path: dict[str, object] | str | Path, state: dict[str, object]) -> dict[str, object]:
    checkpoint = checkpoint_or_path if isinstance(checkpoint_or_path, dict) else json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    validation = validate_value_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("value checkpoint is invalid: " + "; ".join(validation["issues"]))
    issues = value_state_issues(state)
    if issues:
        return {
            "schema_version": "1.0.0",
            "status": "abstain_invalid_state",
            "predicted_terminal_return_bb": None,
            "reasons": issues,
            "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
            "recommendation_allowed": False,
            "activation_authorized": False,
        }
    dimension = checkpoint["configuration"]["feature_dimension"]
    prediction = _predict(
        [float(value) for value in checkpoint["weights"]],
        value_features(state, dimension),
        float(checkpoint["configuration"]["train_target_mean_bb"]),
    )
    result = {
        "schema_version": "1.0.0",
        "status": "offline_value_prediction_uncalibrated",
        "predicted_terminal_return_bb": format(prediction, ".12g"),
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train, validate or inspect APC's offline value model.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--seed", type=int, default=20260816)
    train.add_argument("--feature-dimension", type=int, default=256)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--learning-rate", type=float, default=0.02)
    train.add_argument("--l2", type=float, default=0.0005)
    validate = subparsers.add_parser("validate")
    validate.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_value_checkpoint(args.checkpoint)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        checkpoint = train_value_model(
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
