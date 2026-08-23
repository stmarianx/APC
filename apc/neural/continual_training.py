from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from apc.neural.model import APCNetwork, load_apc_weights, save_apc_weights
from apc.neural.replay_adapter import ReplayTemporalCorpus, load_replay_temporal_corpus


SCHEMA_VERSION = "1.0.0"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _model_fingerprint(model: APCNetwork) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _batch(corpus: ReplayTemporalCorpus, indices: np.ndarray) -> dict[str, Tensor]:
    return {
        "state_tokens": torch.from_numpy(corpus.state_tokens[indices]),
        "state_padding_mask": torch.from_numpy(corpus.state_padding_mask[indices]),
        "profile_features": torch.from_numpy(corpus.profile_features[indices]),
        "modality_available": torch.from_numpy(corpus.modality_available[indices]),
        "legal_action_mask": torch.from_numpy(corpus.legal_action_mask[indices]),
        "candidate_action_index": torch.from_numpy(corpus.chosen_action_index[indices]),
        "candidate_size_features": torch.from_numpy(corpus.chosen_size_features[indices]),
    }


def _predict(model: APCNetwork, corpus: ReplayTemporalCorpus, indices: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    if len(indices) == 0:
        raise ValueError("APC replay evaluation split is empty")
    collected: dict[str, list[np.ndarray]] = {"value": [], "policy": [], "temporal": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            output = model(**_batch(corpus, indices[start:start + batch_size]))
            collected["value"].append(output["candidate_action_value_bb"].cpu().numpy())
            collected["policy"].append(output["policy_logits"].cpu().numpy())
            collected["temporal"].append(output["temporal_consistency"].cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def _metrics(corpus: ReplayTemporalCorpus, indices: np.ndarray, prediction: dict[str, np.ndarray]) -> dict[str, object]:
    actual = corpus.target_return_bb[indices].astype(np.float64)
    predicted = prediction["value"].astype(np.float64)
    legal_policy = prediction["policy"].argmax(axis=1)
    chosen = corpus.chosen_action_index[indices]
    errors = predicted - actual
    return {
        "decisions": len(indices),
        "complete_hands": len({corpus.replay_fingerprints[int(index)] for index in indices}),
        "mae_bb": format(float(np.abs(errors).mean()), ".12g"),
        "rmse_bb": format(float(np.sqrt(np.mean(errors ** 2))), ".12g"),
        "bias_bb": format(float(errors.mean()), ".12g"),
        "observed_action_accuracy": format(float(np.mean(legal_policy == chosen)), ".12g"),
        "temporal_consistency_mean": format(float(prediction["temporal"].mean()), ".12g"),
    }


def evaluate_temporal_latency(
    model: APCNetwork,
    corpus: ReplayTemporalCorpus,
    *,
    repetitions: int = 100,
) -> dict[str, object]:
    if repetitions < 20:
        raise ValueError("APC temporal latency audit requires at least 20 repetitions")
    test = corpus.indices("test")
    if len(test) == 0:
        raise ValueError("APC temporal latency audit requires test decisions")
    # Audit the longest held-out history, which is the cost-driving fast path.
    visible = (~corpus.state_padding_mask[test]).sum(axis=(1, 2))
    index = test[int(np.argmax(visible)) : int(np.argmax(visible)) + 1]
    inputs = _batch(corpus, index)
    model.eval()
    durations = []
    with torch.inference_mode():
        for _ in range(10):
            model(**inputs)
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            model(**inputs)
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * repetitions) - 1]
    return {
        "repetitions": repetitions,
        "audited_visible_tokens": int(visible.max()),
        "p50_ms": format(float(np.median(durations)), ".12g"),
        "p95_ms": format(p95, ".12g"),
        "maximum_ms": format(max(durations), ".12g"),
        "threshold_p95_ms": "50",
        "passed": p95 <= 50.0,
    }


def train_completed_replay_candidate(
    corpus: ReplayTemporalCorpus,
    incumbent: APCNetwork,
    *,
    seed: int = 20260825,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    incumbent_retention_weight: float = 0.20,
) -> tuple[APCNetwork, dict[str, object]]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or incumbent_retention_weight < 0:
        raise ValueError("APC continual-training parameters are invalid")
    split_indices = {split: corpus.indices(split) for split in ("train", "validation", "test")}
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise ValueError("APC continual training requires non-empty hand-exclusive train/validation/test splits")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    incumbent.eval()
    incumbent_before = _model_fingerprint(incumbent)
    candidate = copy.deepcopy(incumbent)
    mean = float(corpus.target_return_bb[split_indices["train"]].mean())
    scale = max(float(corpus.target_return_bb[split_indices["train"]].std()), 1e-3)
    candidate.value_mean_bb.fill_(mean)
    candidate.value_scale_bb.fill_(scale)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_validation = (math.inf, math.inf)
    history = []
    for epoch in range(epochs):
        candidate.train()
        shuffled = split_indices["train"][torch.randperm(len(split_indices["train"]), generator=generator).numpy()]
        losses = []
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            inputs = _batch(corpus, indices)
            output = candidate(**inputs)
            with torch.inference_mode():
                prior = incumbent(**inputs)
            target = torch.from_numpy(corpus.target_return_bb[indices])
            normalized_error = (output["candidate_action_value_bb"] - target) / scale
            value_loss = torch.nn.functional.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error))
            imitation_loss = torch.nn.functional.cross_entropy(output["policy_logits"], inputs["candidate_action_index"])
            temporal_loss = torch.nn.functional.binary_cross_entropy(output["temporal_consistency"], torch.ones_like(output["temporal_consistency"]))
            finite_legal = inputs["legal_action_mask"]
            retention_value = torch.nn.functional.smooth_l1_loss(output["action_value_bb"][finite_legal], prior["action_value_bb"][finite_legal])
            prior_probability = torch.softmax(prior["policy_logits"], dim=-1)
            retention_policy = torch.nn.functional.kl_div(
                torch.log_softmax(output["policy_logits"], dim=-1), prior_probability, reduction="batchmean"
            )
            loss = value_loss + 0.05 * imitation_loss + 0.02 * temporal_loss + incumbent_retention_weight * (retention_value + retention_policy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_prediction = _predict(candidate, corpus, split_indices["validation"], batch_size)
        validation = _metrics(corpus, split_indices["validation"], validation_prediction)
        selection = (float(validation["mae_bb"]), -float(validation["observed_action_accuracy"]))
        history.append({"epoch": epoch + 1, "training_loss": format(float(np.mean(losses)), ".12g"), **validation})
        if selection < best_validation:
            best_validation = selection
            best_state = {name: tensor.detach().clone() for name, tensor in candidate.state_dict().items()}
    if best_state is None:
        raise RuntimeError("APC continual training did not produce a candidate")
    candidate.load_state_dict(best_state)
    metrics: dict[str, object] = {"history": history}
    for split, indices in split_indices.items():
        incumbent_prediction = _predict(incumbent, corpus, indices, batch_size)
        candidate_prediction = _predict(candidate, corpus, indices, batch_size)
        metrics[split] = {
            "incumbent": _metrics(corpus, indices, incumbent_prediction),
            "candidate": _metrics(corpus, indices, candidate_prediction),
            "mean_absolute_value_drift_bb": format(float(np.mean(np.abs(candidate_prediction["value"] - incumbent_prediction["value"]))), ".12g"),
        }
    if _model_fingerprint(incumbent) != incumbent_before:
        raise RuntimeError("APC continual training mutated the incumbent")
    return candidate, metrics


def audit_completed_replay_candidate(
    corpus: ReplayTemporalCorpus,
    incumbent: APCNetwork,
    candidate: APCNetwork,
    metrics: dict[str, object],
) -> dict[str, object]:
    incumbent_fingerprint = _model_fingerprint(incumbent)
    candidate_fingerprint = _model_fingerprint(candidate)
    test = metrics["test"]
    improves_value = float(test["candidate"]["mae_bb"]) < float(test["incumbent"]["mae_bb"])
    action_non_regression = float(test["candidate"]["observed_action_accuracy"]) >= float(test["incumbent"]["observed_action_accuracy"])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "incumbent.apc"
        saved = save_apc_weights(incumbent, path)
        restored = load_apc_weights(path, str(saved["weights_sha256"]))
        rollback_verified = _model_fingerprint(restored) == incumbent_fingerprint
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_name": "APC",
        "audit_kind": "completed_hand_continual_retraining",
        "replay_adapter_fingerprint": corpus.manifest["adapter_fingerprint"],
        "incumbent_fingerprint": incumbent_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "weights_updated_during_hand": False,
        "selection_uses_test": False,
        "automatic_promotion": False,
        "metrics": metrics,
        "gates": {
            "held_out_value_improves": improves_value,
            "held_out_observed_action_non_regression": action_non_regression,
            "rollback_verified": rollback_verified,
            "promotion_authorized": False,
        },
    }
    return report


def build_completed_replay_checkpoint(
    replay_buffer: str | Path,
    incumbent_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = 20260825,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    incumbent_retention_weight: float = 0.20,
    strategy_regression_dataset: str | Path | None = None,
) -> dict[str, object]:
    from apc.neural.train_candidate import validate_checkpoint

    incumbent_path = Path(incumbent_checkpoint).resolve()
    validation = validate_checkpoint(incumbent_path)
    if not validation["valid"]:
        raise ValueError("APC incumbent checkpoint is invalid: " + "; ".join(validation["issues"]))
    incumbent_record = json.loads(incumbent_path.read_text(encoding="utf-8"))
    incumbent = load_apc_weights(
        incumbent_path.parent / str(incumbent_record["weights"]["file"]),
        str(incumbent_record["weights"]["weights_sha256"]),
    )
    corpus = load_replay_temporal_corpus(replay_buffer)
    candidate, metrics = train_completed_replay_candidate(
        corpus,
        incumbent,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        incumbent_retention_weight=incumbent_retention_weight,
    )
    audit = audit_completed_replay_candidate(corpus, incumbent, candidate, metrics)
    latency = evaluate_temporal_latency(candidate, corpus)
    strategy_regression = None
    if strategy_regression_dataset is not None:
        from apc.neural.train_candidate import _predict as predict_strategy
        from apc.neural.train_candidate import _sliced_metrics, load_raised_corpus

        regression_manifest, regression_corpus = load_raised_corpus(strategy_regression_dataset)
        regression_indices = regression_corpus.indices("test")
        incumbent_value, _, incumbent_uncertainty = predict_strategy(incumbent, regression_corpus, regression_indices, batch_size)
        candidate_value, _, candidate_uncertainty = predict_strategy(candidate, regression_corpus, regression_indices, batch_size)
        incumbent_metrics = _sliced_metrics(regression_corpus, regression_indices, incumbent_value, incumbent_uncertainty)
        candidate_metrics = _sliced_metrics(regression_corpus, regression_indices, candidate_value, candidate_uncertainty)
        strategy_regression = {
            "dataset_id": regression_manifest["dataset_id"],
            "dataset_fingerprint": regression_manifest["dataset_fingerprint"],
            "evaluated_split": "test",
            "used_for_training_or_selection": False,
            "incumbent": incumbent_metrics,
            "candidate": candidate_metrics,
            "mae_non_regression": float(candidate_metrics["mae_bb"]) <= float(incumbent_metrics["mae_bb"]),
            "chosen_action_regret_non_regression": float(candidate_metrics["chosen_action_regret_bb"]) <= float(incumbent_metrics["chosen_action_regret_bb"]),
            "decision_accuracy_non_regression": float(candidate_metrics["decision_accuracy"]) >= float(incumbent_metrics["decision_accuracy"]),
        }
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite APC continual checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent))
    try:
        weights = save_apc_weights(candidate, temporary / "weights.apc")
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "model_name": "APC",
            "model_family": "multimodal_temporal_neural_network",
            "status": "offline_completed_replay_candidate_unpromoted",
            "incumbent": {
                "checkpoint_fingerprint": incumbent_record["checkpoint_fingerprint"],
                "weights_sha256": incumbent_record["weights"]["weights_sha256"],
            },
            "replay": corpus.manifest,
            "training": {
                "pipeline_version": "1.0.0",
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": format(learning_rate, ".12g"),
                "incumbent_retention_weight": format(incumbent_retention_weight, ".12g"),
                "selection": "validation_mae_then_observed_action_accuracy",
                "complete_hand_group_exclusive": True,
                "policy_weight_updates_during_hand": False,
            },
            "weights": {"file": "weights.apc", **weights},
            "audit": audit,
            "strategy_regression": strategy_regression,
            "latency": latency,
            "gates": {
                **audit["gates"],
                "temporal_strategy_p95_under_50_ms": latency["passed"],
                "calibration_passed": False,
                "paired_incumbent_confidence_interval_passed": False,
                "strategy_regression_passed": bool(
                    strategy_regression is not None
                    and strategy_regression["mae_non_regression"]
                    and strategy_regression["chosen_action_regret_non_regression"]
                    and strategy_regression["decision_accuracy_non_regression"]
                ),
                "promotion_authorized": False,
            },
            "activation_authorized": False,
            "recommendation_allowed": False,
        }
        checkpoint["checkpoint_fingerprint"] = hashlib.sha256(_canonical(checkpoint)).hexdigest()
        (temporary / "checkpoint.json").write_bytes(_canonical(checkpoint) + b"\n")
        os.replace(temporary, target)
        return checkpoint
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def validate_completed_replay_checkpoint(path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(path)
    issues = []
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"schema_version": SCHEMA_VERSION, "valid": False, "issues": [f"checkpoint unreadable: {error}"]}
    material = dict(checkpoint)
    observed = material.pop("checkpoint_fingerprint", None)
    if observed != hashlib.sha256(_canonical(material)).hexdigest():
        issues.append("checkpoint fingerprint mismatch")
    try:
        if checkpoint["model_name"] != "APC" or checkpoint["model_family"] != "multimodal_temporal_neural_network":
            issues.append("checkpoint model identity is invalid")
        if checkpoint["status"] != "offline_completed_replay_candidate_unpromoted":
            issues.append("checkpoint status is invalid")
        if checkpoint["activation_authorized"] is not False or checkpoint["recommendation_allowed"] is not False:
            issues.append("checkpoint activation isolation is invalid")
        if checkpoint["training"]["policy_weight_updates_during_hand"] is not False:
            issues.append("checkpoint permits mid-hand weight updates")
        if checkpoint["audit"]["selection_uses_test"] is not False or checkpoint["audit"]["automatic_promotion"] is not False:
            issues.append("checkpoint selection/promotion isolation is invalid")
        if checkpoint["gates"]["rollback_verified"] is not True or checkpoint["gates"]["promotion_authorized"] is not False:
            issues.append("checkpoint rollback/promotion gate is invalid")
        strategy = checkpoint.get("strategy_regression")
        expected_strategy_gate = bool(
            isinstance(strategy, dict)
            and strategy.get("used_for_training_or_selection") is False
            and strategy.get("evaluated_split") == "test"
            and strategy.get("mae_non_regression") is True
            and strategy.get("chosen_action_regret_non_regression") is True
            and strategy.get("decision_accuracy_non_regression") is True
        )
        if checkpoint["gates"].get("strategy_regression_passed") is not expected_strategy_gate:
            issues.append("checkpoint strategy regression gate is invalid")
        latency = checkpoint["latency"]
        expected_latency = float(latency["p95_ms"]) <= float(latency["threshold_p95_ms"])
        if latency["passed"] is not expected_latency or checkpoint["gates"]["temporal_strategy_p95_under_50_ms"] is not expected_latency:
            issues.append("checkpoint temporal latency gate is invalid")
        weights = checkpoint["weights"]
        loaded = load_apc_weights(checkpoint_path.parent / str(weights["file"]), str(weights["weights_sha256"]))
        if len(loaded.state_dict()) != int(weights["tensor_count"]):
            issues.append("checkpoint tensor count mismatch")
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        issues.append(f"checkpoint evidence is invalid: {error}")
    return {"schema_version": SCHEMA_VERSION, "valid": not issues, "issues": issues, "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or validate an APC completed-replay candidate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("replay_buffer", type=Path)
    train.add_argument("incumbent_checkpoint", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--seed", type=int, default=20260825)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--incumbent-retention-weight", type=float, default=0.20)
    train.add_argument("--strategy-regression-dataset", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        report = validate_completed_replay_checkpoint(args.checkpoint)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["valid"] else 1)
    result = build_completed_replay_checkpoint(
        args.replay_buffer,
        args.incumbent_checkpoint,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        incumbent_retention_weight=args.incumbent_retention_weight,
        strategy_regression_dataset=args.strategy_regression_dataset,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
