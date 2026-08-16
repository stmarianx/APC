from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from apc.self_learning.replay_dataset import validate_replay_dataset


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _bucket(value: object, width: Decimal) -> str:
    number = Decimal(str(value))
    lower = (number // width) * width
    return format(lower, "f")


def _canonical_cards(board: list[object], hero_cards: list[object]) -> tuple[list[str], list[str]]:
    suit_map: dict[str, str] = {}
    canonical_suits = ("a", "b", "c", "d")
    rows: list[str] = []
    for raw in [*board, *hero_cards]:
        card = str(raw)
        if len(card) != 2 or card[0].upper() not in "23456789TJQKA" or card[-1] not in "cdhs":
            raise ValueError(f"invalid candidate card token: {card!r}")
        suit = card[-1]
        if suit not in suit_map:
            suit_map[suit] = canonical_suits[len(suit_map)]
        rows.append(card[:-1].upper() + suit_map[suit])
    return rows[: len(board)], rows[len(board) :]


_HISTORY_ACTION = re.compile(
    r"^[A-Z][A-Z0-9_]* (?:fold|check|call|all_in(?::[0-9]+(?:\.[0-9]+)?)?|bet:[0-9]+(?:\.[0-9]+)?|raise_to:[0-9]+(?:\.[0-9]+)?)$"
)
_LEGAL_ACTION = re.compile(
    r"^(?:fold|check|call|all_in|bet:[0-9]+(?:\.[0-9]+)?|raise_to:[0-9]+(?:\.[0-9]+)?)$"
)


def candidate_state_issues(state: dict[str, object], legal_actions: list[str]) -> list[str]:
    issues: list[str] = []
    if not isinstance(state, dict):
        return ["state_must_be_an_object"]
    if not isinstance(state.get("game"), str) or not state.get("game"):
        issues.append("game_missing")
    players = state.get("players")
    if isinstance(players, bool) or not isinstance(players, int) or players < 2:
        issues.append("players_invalid")
    if not isinstance(state.get("hero_position"), str) or not state.get("hero_position"):
        issues.append("hero_position_missing")
    decimals: dict[str, Decimal] = {}
    for field in ("effective_stack_bb", "pot_bb", "to_call_bb"):
        try:
            value = Decimal(str(state.get(field)))
        except (InvalidOperation, ValueError):
            issues.append(f"{field}_invalid")
            continue
        if not value.is_finite() or value < 0 or (field == "pot_bb" and value <= 0):
            issues.append(f"{field}_invalid")
        else:
            decimals[field] = value
    if (
        "to_call_bb" in decimals
        and "effective_stack_bb" in decimals
        and decimals["to_call_bb"] > decimals["effective_stack_bb"]
    ):
        issues.append("to_call_exceeds_effective_stack")
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
    history = state.get("action_history")
    if not isinstance(history, list) or any(
        not isinstance(action, str) or not _HISTORY_ACTION.fullmatch(action) for action in history
    ):
        issues.append("action_history_not_canonical")
    if not legal_actions or len(legal_actions) != len(set(legal_actions)) or any(
        not isinstance(action, str) or not _LEGAL_ACTION.fullmatch(action) for action in legal_actions
    ):
        issues.append("legal_actions_invalid")
    observed_legal = state.get("legal_actions")
    if not isinstance(observed_legal, list) or observed_legal != legal_actions:
        issues.append("legal_actions_do_not_match_state")
    if not isinstance(state.get("rake_model"), str) or not state.get("rake_model"):
        issues.append("rake_model_missing")
    if not isinstance(state.get("utility_model"), str) or not state.get("utility_model"):
        issues.append("utility_model_missing")
    return sorted(set(issues))


def feature_tokens(state: dict[str, object]) -> tuple[str, ...]:
    board = state.get("board")
    hero_cards = state.get("hero_cards")
    history = state.get("action_history")
    if not isinstance(board, list) or not isinstance(hero_cards, list) or not isinstance(history, list):
        raise ValueError("candidate state requires board, hero_cards and action_history arrays")
    canonical_board, canonical_hero = _canonical_cards(board, hero_cards)
    tokens = [
        "bias",
        f"game={state.get('game')}",
        f"players={state.get('players')}",
        f"position={state.get('hero_position')}",
        f"street_cards={len(board)}",
        f"stack_5bb={_bucket(state.get('effective_stack_bb'), Decimal('5'))}",
        f"pot_1bb={_bucket(state.get('pot_bb'), Decimal('1'))}",
    ]
    tokens.extend(f"board[{index}]={card}" for index, card in enumerate(canonical_board))
    tokens.extend(f"hero[{index}]={card}" for index, card in enumerate(canonical_hero))
    tokens.extend(f"history[{index}]={action}" for index, action in enumerate(history))
    return tuple(tokens)


def hashed_features(state: dict[str, object], dimension: int) -> dict[int, float]:
    if dimension < 16:
        raise ValueError("feature dimension must be at least 16")
    vector: dict[int, float] = defaultdict(float)
    for token in feature_tokens(state):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        raise ValueError("candidate feature vector cannot be zero")
    return {index: value / norm for index, value in vector.items()}


def _examples(dataset: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    report = validate_replay_dataset(dataset)
    if not report["valid"]:
        raise ValueError("replay dataset is invalid: " + "; ".join(report["issues"]))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_eligible") is not True:
        raise ValueError("replay dataset is not training eligible")
    eligibility = manifest.get("training_eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("passed") is not True:
        raise ValueError("replay dataset lacks passed training eligibility evidence")
    rows = [
        json.loads(line)
        for line in (dataset / str(manifest["examples_file"])).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not any(row.get("split") == split for row in rows) for split in ("train", "validation", "test")):
        raise ValueError("candidate training requires non-empty train, validation and test splits")
    return manifest, rows


def _target(example: dict[str, object]) -> tuple[list[str], dict[str, float], dict[str, float]]:
    target = example["target"]
    actions = target["actions"]
    action_ids = [str(row["action_id"]) for row in actions]
    frequencies = {str(row["action_id"]): float(str(row["frequency"])) for row in actions}
    evs = {str(row["action_id"]): float(str(row["ev_bb"])) for row in actions}
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("candidate target actions must be unique")
    return action_ids, frequencies, evs


def _probabilities(
    weights: dict[str, list[float]],
    features: dict[int, float],
    legal_actions: list[str],
) -> dict[str, float]:
    logits = {
        action: sum(weights[action][index] * value for index, value in features.items())
        for action in legal_actions
    }
    offset = max(logits.values())
    exponentials = {action: math.exp(value - offset) for action, value in logits.items()}
    total = sum(exponentials.values())
    return {action: value / total for action, value in exponentials.items()}


def _evaluate(
    rows: list[dict[str, object]],
    weights: dict[str, list[float]],
    dimension: int,
) -> dict[str, object]:
    cross_entropy = l1 = regret = 0.0
    top_agreement = 0
    for example in rows:
        legal, target, evs = _target(example)
        probabilities = _probabilities(weights, hashed_features(example["state"], dimension), legal)
        cross_entropy -= sum(target[action] * math.log(max(probabilities[action], 1e-15)) for action in legal)
        l1 += sum(abs(probabilities[action] - target[action]) for action in legal)
        predicted_top = max(legal, key=lambda action: (probabilities[action], action))
        target_top = max(legal, key=lambda action: (target[action], action))
        top_agreement += predicted_top == target_top
        regret += max(evs.values()) - evs[predicted_top]
    count = len(rows)
    return {
        "examples": count,
        "cross_entropy": format(cross_entropy / count, ".12g"),
        "mean_l1": format(l1 / count, ".12g"),
        "top_action_agreement": format(top_agreement / count, ".12g"),
        "mean_top_action_regret_bb": format(regret / count, ".12g"),
    }


def train_candidate(
    dataset: str | Path,
    output: str | Path,
    *,
    seed: int = 20260816,
    feature_dimension: int = 128,
    epochs: int = 40,
    learning_rate: float = 0.08,
    l2: float = 0.0001,
) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if feature_dimension < 16 or epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("candidate training configuration is invalid")
    dataset_path = Path(dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"candidate checkpoint already exists: {output_path}")
    manifest, examples = _examples(dataset_path)
    splits = {
        split: [row for row in examples if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    vocabulary = sorted(
        {
            str(action["action_id"])
            for row in examples
            for action in row["target"]["actions"]
        }
    )
    weights = {action: [0.0] * feature_dimension for action in vocabulary}
    train_rows = list(splits["train"])
    for epoch in range(epochs):
        ordered = list(train_rows)
        random.Random(seed + epoch).shuffle(ordered)
        for example in ordered:
            legal, target, _ = _target(example)
            features = hashed_features(example["state"], feature_dimension)
            predicted = _probabilities(weights, features, legal)
            for action in legal:
                error = predicted[action] - target[action]
                action_weights = weights[action]
                for index, value in features.items():
                    action_weights[index] -= learning_rate * (
                        error * value + l2 * action_weights[index]
                    )
    serialized_weights = {
        action: [format(value, ".12g") for value in values]
        for action, values in sorted(weights.items())
    }
    checkpoint = {
        "schema_version": "1.0.0",
        "model_kind": "hashed_linear_mixed_strategy_candidate",
        "status": "candidate_not_promoted",
        "activation_authorized": False,
        "incumbent_replaced": False,
        "units": "BB",
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
            "feature_schema": "hashed_poker_state_v1",
        },
        "action_vocabulary": vocabulary,
        "weights": serialized_weights,
        "metrics": {
            split: _evaluate(rows, weights, feature_dimension)
            for split, rows in splits.items()
        },
        "limitations": [
            "This is a candidate policy distillation model; it cannot replace the incumbent without paired promotion and safety evaluation.",
            "Imported solver supervision is not automatically GTO-verified.",
            "The initial feature schema is structured-state based and does not train visible-table perception.",
        ],
    }
    checkpoint["checkpoint_fingerprint"] = _sha256(checkpoint)
    validation = validate_candidate_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("candidate checkpoint failed validation: " + "; ".join(validation["issues"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temporary:
        temporary_file = Path(temporary) / output_path.name
        temporary_file.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        temporary_file.replace(output_path)
    return checkpoint


def validate_candidate_checkpoint(checkpoint_or_path: dict[str, object] | str | Path) -> dict[str, object]:
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
    if checkpoint.get("status") != "candidate_not_promoted":
        issues.append("checkpoint must remain an unpromoted candidate")
    if checkpoint.get("activation_authorized") is not False or checkpoint.get("incumbent_replaced") is not False:
        issues.append("candidate checkpoint cannot authorize activation or replace the incumbent")
    configuration = checkpoint.get("configuration")
    vocabulary = checkpoint.get("action_vocabulary")
    weights = checkpoint.get("weights")
    if not isinstance(configuration, dict) or not isinstance(configuration.get("feature_dimension"), int):
        issues.append("checkpoint configuration is invalid")
        dimension = 0
    else:
        dimension = configuration["feature_dimension"]
    if not isinstance(vocabulary, list) or not vocabulary or len(vocabulary) != len(set(vocabulary)):
        issues.append("checkpoint action vocabulary is invalid")
        vocabulary = []
    if not isinstance(weights, dict) or set(weights) != set(vocabulary):
        issues.append("checkpoint weights do not match the action vocabulary")
    elif any(not isinstance(row, list) or len(row) != dimension for row in weights.values()):
        issues.append("checkpoint weight dimensions are invalid")
    else:
        try:
            finite_weights = all(math.isfinite(float(value)) for row in weights.values() for value in row)
        except (TypeError, ValueError):
            finite_weights = False
        if not finite_weights:
            issues.append("checkpoint weights must be finite numbers")
    material = dict(checkpoint)
    observed = material.pop("checkpoint_fingerprint", None)
    if observed != _sha256(material):
        issues.append("checkpoint fingerprint mismatch")
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict) or any(
        split not in metrics or not isinstance(metrics[split], dict) or metrics[split].get("examples", 0) <= 0
        for split in ("train", "validation", "test")
    ):
        issues.append("checkpoint requires non-empty train, validation and test metrics")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint"),
    }


def predict_candidate(
    checkpoint_or_path: dict[str, object] | str | Path,
    state: dict[str, object],
    legal_actions: list[str],
) -> dict[str, object]:
    if isinstance(checkpoint_or_path, dict):
        checkpoint = checkpoint_or_path
    else:
        checkpoint = json.loads(Path(checkpoint_or_path).read_text(encoding="utf-8"))
    validation = validate_candidate_checkpoint(checkpoint)
    if not validation["valid"]:
        raise ValueError("candidate checkpoint is invalid: " + "; ".join(validation["issues"]))
    state_issues = candidate_state_issues(state, legal_actions)
    if state_issues:
        return {
            "schema_version": "1.0.0",
            "status": "abstain_invalid_state",
            "probabilities": None,
            "reasons": state_issues,
            "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
            "confidence_calibrated": False,
            "offline_evaluation_allowed": False,
            "recommendation_allowed": False,
            "activation_authorized": False,
        }
    vocabulary = set(checkpoint["action_vocabulary"])
    unsupported = sorted(action for action in legal_actions if action not in vocabulary)
    if unsupported:
        return {
            "schema_version": "1.0.0",
            "status": "abstain_unsupported_actions",
            "probabilities": None,
            "unsupported_actions": unsupported,
            "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
            "confidence_calibrated": False,
            "offline_evaluation_allowed": False,
            "recommendation_allowed": False,
            "activation_authorized": False,
        }
    dimension = checkpoint["configuration"]["feature_dimension"]
    weights = {
        action: [float(value) for value in checkpoint["weights"][action]]
        for action in legal_actions
    }
    probabilities = _probabilities(weights, hashed_features(state, dimension), legal_actions)
    result = {
        "schema_version": "1.0.0",
        "status": "prediction_ready_uncalibrated",
        "probabilities": {
            action: format(probabilities[action], ".12g") for action in legal_actions
        },
        "unsupported_actions": [],
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "confidence_calibrated": False,
        "offline_evaluation_allowed": True,
        "recommendation_allowed": False,
        "activation_authorized": False,
    }
    result["prediction_fingerprint"] = _sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or validate an APC candidate policy checkpoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--seed", type=int, default=20260816)
    train.add_argument("--feature-dimension", type=int, default=128)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--l2", type=float, default=0.0001)
    validate = subparsers.add_parser("validate")
    validate.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_candidate_checkpoint(args.checkpoint)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        checkpoint = train_candidate(
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
