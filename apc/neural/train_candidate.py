from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from apc.neural.contract import ACTION_VOCABULARY, load_apc_neural_config
from apc.neural.features import (
    PROFILE_FEATURE_DIMENSION,
    STATE_TOKEN_COUNT,
    STATE_TOKEN_DIMENSION,
    EncodedDecision,
    encode_raised_row,
)
from apc.neural.model import APCArchitecture, APCNetwork, load_apc_weights, save_apc_weights
from apc.self_learning.raised_postflop_rollout_dataset import validate_raised_postflop_dataset


SCHEMA_VERSION = "1.0.0"
SPLIT_INDEX = {"train": 0, "validation": 1, "test": 2}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass
class EncodedCorpus:
    tokens: np.ndarray
    padding: np.ndarray
    profile: np.ndarray
    modality_available: np.ndarray
    legal: np.ndarray
    action: np.ndarray
    sizes: np.ndarray
    target: np.ndarray
    policy_probability: np.ndarray
    split: np.ndarray
    group_ids: list[str]
    policy_state_ids: list[str]
    action_keys: list[str]
    positions: list[str]
    nodes: list[str]

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == SPLIT_INDEX[split])


def encode_rows(rows: list[dict[str, object]]) -> EncodedCorpus:
    encoded: list[EncodedDecision] = [encode_raised_row(row) for row in rows]
    if not encoded:
        raise ValueError("APC neural training corpus cannot be empty")
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(encoded):
        if item.split not in SPLIT_INDEX:
            raise ValueError("APC neural row has invalid split")
        groups[item.policy_state_id].append(index)
    probabilities = np.zeros(len(encoded), dtype=np.float32)
    for policy_state_id, indices in groups.items():
        if len(indices) != 4:
            raise ValueError(f"APC policy state {policy_state_id} is not four-action paired")
        values = np.asarray([encoded[index].target_return_bb for index in indices], dtype=np.float64)
        centered = np.clip((values - values.max()) / 2.0, -40.0, 0.0)
        weights = np.exp(centered)
        weights /= weights.sum()
        probabilities[indices] = weights.astype(np.float32)
    return EncodedCorpus(
        tokens=np.stack([item.state_tokens for item in encoded]),
        padding=np.stack([item.state_padding_mask for item in encoded]),
        profile=np.stack([item.profile_features for item in encoded]),
        modality_available=np.stack([item.modality_available for item in encoded]),
        legal=np.stack([item.legal_action_mask for item in encoded]),
        action=np.asarray([item.candidate_action_index for item in encoded], dtype=np.int64),
        sizes=np.stack([item.candidate_size_features for item in encoded]),
        target=np.asarray([item.target_return_bb for item in encoded], dtype=np.float32),
        policy_probability=probabilities,
        split=np.asarray([SPLIT_INDEX[item.split] for item in encoded], dtype=np.int8),
        group_ids=[item.group_id for item in encoded],
        policy_state_ids=[item.policy_state_id for item in encoded],
        action_keys=[item.action_key for item in encoded],
        positions=[str(row["hero_position"]) for row in rows],
        nodes=[str(row["node_family"]) for row in rows],
    )


def load_raised_corpus(dataset: str | Path) -> tuple[dict[str, object], EncodedCorpus]:
    root = Path(dataset).resolve()
    validation = validate_raised_postflop_dataset(root)
    if not validation["valid"]:
        raise ValueError("APC neural source dataset is invalid: " + "; ".join(validation["issues"]))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_eligible") is not True:
        raise ValueError("APC neural source dataset is not training eligible")
    print(f"APC dataset verified: {manifest['example_count']} examples", flush=True)
    rows = []
    with (root / str(manifest["examples_file"])).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    corpus = encode_rows(rows)
    print("APC feature encoding complete", flush=True)
    return manifest, corpus


def _batch(corpus: EncodedCorpus, indices: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    return {
        "state_tokens": torch.from_numpy(corpus.tokens[indices]).to(device),
        "state_padding_mask": torch.from_numpy(corpus.padding[indices]).to(device),
        "profile_features": torch.from_numpy(corpus.profile[indices]).to(device),
        "modality_available": torch.from_numpy(corpus.modality_available[indices]).to(device),
        "legal_action_mask": torch.from_numpy(corpus.legal[indices]).to(device),
        "candidate_action_index": torch.from_numpy(corpus.action[indices]).to(device),
        "candidate_size_features": torch.from_numpy(corpus.sizes[indices]).to(device),
    }


def _policy_groups(corpus: EncodedCorpus, split: str) -> np.ndarray:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index in corpus.indices(split):
        grouped[corpus.policy_state_ids[int(index)]].append(int(index))
    ordered = []
    for policy_state_id, indices in sorted(grouped.items()):
        if len(indices) != 4:
            raise ValueError(f"APC {split} policy state {policy_state_id} is incomplete")
        ordered.append(sorted(indices, key=lambda index: corpus.action_keys[index]))
    return np.asarray(ordered, dtype=np.int64)


def _predict(model: APCNetwork, corpus: EncodedCorpus, indices: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    values, policies, uncertainty = [], [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            output = model(**_batch(corpus, chosen, torch.device("cpu")))
            values.append(output["candidate_action_value_bb"].cpu().numpy())
            policies.append(output["policy_logits"].cpu().numpy())
            uncertainty.append(output["uncertainty"].cpu().numpy())
    return np.concatenate(values), np.concatenate(policies), np.concatenate(uncertainty)


def _metrics(corpus: EncodedCorpus, indices: np.ndarray, predicted: np.ndarray, uncertainty: np.ndarray) -> dict[str, object]:
    actual = corpus.target[indices].astype(np.float64)
    predicted64 = predicted.astype(np.float64)
    errors = predicted64 - actual
    baseline = float(corpus.target[corpus.indices("train")].mean())
    by_state: defaultdict[str, list[int]] = defaultdict(list)
    for local, global_index in enumerate(indices):
        by_state[corpus.policy_state_ids[int(global_index)]].append(local)
    correct = 0
    gain = []
    for local_indices in by_state.values():
        actual_best = max(local_indices, key=lambda i: (actual[i], corpus.action_keys[int(indices[i])]))
        predicted_best = max(local_indices, key=lambda i: (predicted64[i], corpus.action_keys[int(indices[i])]))
        correct += corpus.action_keys[int(indices[actual_best])] == corpus.action_keys[int(indices[predicted_best])]
        gain.append(actual[predicted_best] - actual[actual_best])
    absolute = np.abs(errors)
    confidence_error = np.abs(np.clip(uncertainty.astype(np.float64), 0, 1) - np.clip(absolute / 25.0, 0, 1))
    return {
        "examples": len(indices),
        "policy_states": len(by_state),
        "mae_bb": format(float(absolute.mean()), ".12g"),
        "rmse_bb": format(float(np.sqrt(np.mean(errors ** 2))), ".12g"),
        "bias_bb": format(float(errors.mean()), ".12g"),
        "train_global_mean_baseline_mae_bb": format(float(np.mean(np.abs(actual - baseline))), ".12g"),
        "decision_accuracy": format(correct / len(by_state), ".12g"),
        "chosen_action_regret_bb": format(float(-np.mean(gain)), ".12g"),
        "uncertainty_eace": format(float(confidence_error.mean()), ".12g"),
    }


def _sliced_metrics(corpus: EncodedCorpus, indices: np.ndarray, predicted: np.ndarray, uncertainty: np.ndarray) -> dict[str, object]:
    result = _metrics(corpus, indices, predicted, uncertainty)
    result["by_position"] = {}
    result["by_node"] = {}
    for field, values, key in (("by_position", corpus.positions, "position"), ("by_node", corpus.nodes, "node")):
        for value in sorted(set(values[int(index)] for index in indices)):
            local = np.asarray([offset for offset, index in enumerate(indices) if values[int(index)] == value], dtype=np.int64)
            subset = indices[local]
            result[field][value] = _metrics(corpus, subset, predicted[local], uncertainty[local])
    return result


def train_apc_candidate(
    corpus: EncodedCorpus,
    *,
    architecture: APCArchitecture | None = None,
    seed: int = 20260823,
    epochs: int = 3,
    batch_size: int = 384,
    learning_rate: float = 3e-4,
) -> tuple[APCNetwork, dict[str, object]]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("APC neural training parameters must be positive")
    for split in SPLIT_INDEX:
        if len(corpus.indices(split)) == 0:
            raise ValueError("APC neural training requires every group-exclusive split")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    try:
        model = APCNetwork(architecture)
        train_indices = corpus.indices("train")
        mean = float(corpus.target[train_indices].mean())
        scale = max(float(corpus.target[train_indices].std()), 1e-3)
        model.value_mean_bb.fill_(mean)
        model.value_scale_bb.fill_(scale)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        best_state = None
        best_validation = (math.inf, math.inf)
        history = []
        train_groups = _policy_groups(corpus, "train")
        groups_per_batch = max(1, batch_size // 4)
        for epoch in range(epochs):
            model.train()
            permutation = train_groups[
                torch.randperm(len(train_groups), generator=generator).numpy()
            ]
            losses = []
            for start in range(0, len(permutation), groups_per_batch):
                grouped_indices = permutation[start:start + groups_per_batch]
                chosen = grouped_indices.reshape(-1)
                inputs = _batch(corpus, chosen, torch.device("cpu"))
                output = model(**inputs)
                target = torch.from_numpy(corpus.target[chosen])
                normalized_error = (output["candidate_action_value_bb"] - target) / scale
                value_loss = torch.nn.functional.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error))
                log_policy = torch.log_softmax(output["policy_logits"], dim=-1)
                row_log_probability = log_policy.gather(1, inputs["candidate_action_index"][:, None]).squeeze(1)
                policy_weights = torch.from_numpy(corpus.policy_probability[chosen])
                policy_loss = -(policy_weights * row_log_probability).mean() * 4.0
                grouped_values = output["candidate_action_value_bb"].reshape(-1, 4)
                grouped_targets = policy_weights.reshape(-1, 4)
                ranking_loss = -(
                    grouped_targets * torch.log_softmax(grouped_values / 2.0, dim=1)
                ).sum(1).mean()
                uncertainty_target = normalized_error.detach().abs().clamp(0, 1)
                uncertainty_loss = torch.nn.functional.mse_loss(output["uncertainty"], uncertainty_target)
                loss = value_loss + 0.30 * ranking_loss + 0.05 * policy_loss + 0.02 * uncertainty_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            validation_indices = corpus.indices("validation")
            validation_prediction, _, validation_uncertainty = _predict(model, corpus, validation_indices, batch_size)
            validation = _metrics(corpus, validation_indices, validation_prediction, validation_uncertainty)
            validation_mae = float(validation["mae_bb"])
            validation_regret = float(validation["chosen_action_regret_bb"])
            history.append({"epoch": epoch + 1, "training_loss": format(statistics.fmean(losses), ".12g"), "validation_mae_bb": validation["mae_bb"], "validation_chosen_action_regret_bb": validation["chosen_action_regret_bb"]})
            print(
                f"APC epoch {epoch + 1}/{epochs}: loss={history[-1]['training_loss']} "
                f"validation_mae_bb={history[-1]['validation_mae_bb']} "
                f"validation_regret_bb={history[-1]['validation_chosen_action_regret_bb']}",
                flush=True,
            )
            selection = (validation_regret, validation_mae)
            if selection < best_validation:
                best_validation = selection
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("APC neural training did not produce a candidate")
        model.load_state_dict(best_state)
        metrics = {"history": history}
        for split in SPLIT_INDEX:
            indices = corpus.indices(split)
            predicted, _, uncertainty = _predict(model, corpus, indices, batch_size)
            metrics[split] = _sliced_metrics(corpus, indices, predicted, uncertainty)
        return model, metrics
    finally:
        torch.set_num_threads(previous_threads)


def evaluate_latency(model: APCNetwork, corpus: EncodedCorpus, repetitions: int = 200) -> dict[str, object]:
    if repetitions < 20:
        raise ValueError("APC latency audit needs at least 20 repetitions")
    index = corpus.indices("test")[:1]
    inputs = _batch(corpus, index, torch.device("cpu"))
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(**inputs)
        durations = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            model(**inputs)
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "repetitions": repetitions,
        "p50_ms": format(statistics.median(durations), ".12g"),
        "p95_ms": format(p95, ".12g"),
        "maximum_ms": format(max(durations), ".12g"),
        "threshold_p95_ms": "50",
        "passed": p95 <= 50.0,
    }


def build_checkpoint(
    dataset: str | Path,
    output: str | Path,
    *,
    seed: int = 20260823,
    epochs: int = 3,
    batch_size: int = 384,
    learning_rate: float = 3e-4,
) -> dict[str, object]:
    root = Path(output).resolve()
    checkpoint_path = root / "checkpoint.json"
    weights_path = root / "weights.apc"
    if root.exists() or checkpoint_path.exists() or weights_path.exists():
        raise FileExistsError(f"refusing to overwrite APC neural run: {root}")
    manifest, corpus = load_raised_corpus(dataset)
    model, metrics = train_apc_candidate(corpus, seed=seed, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    root.mkdir(parents=True)
    weights = save_apc_weights(model, weights_path)
    latency = evaluate_latency(model, corpus)
    config = load_apc_neural_config()
    test = metrics["test"]
    value_gate = float(test["mae_bb"]) < float(test["train_global_mean_baseline_mae_bb"])
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "model_name": "APC",
        "model_family": "multimodal_temporal_neural_network",
        "framework": {"name": "pytorch", "version": torch.__version__, "device": "cpu"},
        "architecture_contract_fingerprint": config["config_fingerprint"],
        "architecture": asdict(model.architecture),
        "dataset": {"dataset_id": manifest["dataset_id"], "dataset_fingerprint": manifest["dataset_fingerprint"], "examples_sha256": manifest["examples_sha256"]},
        "training": {"pipeline_version": "3.0.0", "feature_schema_version": "2.0.0", "seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": format(learning_rate, ".12g"), "validation_selection": "minimum_chosen_action_regret_then_mae_bb", "group_exclusive_split": True, "counterfactual_groups_kept_complete_per_batch": True, "trained_modalities": ["canonical_state_sequence", "player_profile"], "untrained_modalities": ["visible_frame_sequence"]},
        "weights": {"file": weights_path.name, **weights},
        "metrics": metrics,
        "latency": latency,
        "gates": {"finite_outputs": True, "test_value_improves_global_mean": value_gate, "strategy_p95_under_50_ms": latency["passed"], "visible_table_training_ready": False, "evaluated_coaching_ready": False},
        "status": "offline_neural_candidate_unpromoted",
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
        "limitations": [
            "The first neural checkpoint trains the canonical state/history and controlled-policy profile branches on virtual-chip rollouts.",
            "Rollout returns are policy-matched Monte Carlo targets, not verified GTO solver labels.",
            "The visual encoder is initialized but untrained and the profile branch has only controlled-policy labels, so this checkpoint is not visible-table ready.",
            "Policy weights never update during a hand and this candidate cannot activate automatically."
        ],
    }
    checkpoint["checkpoint_fingerprint"] = _fingerprint(checkpoint)
    checkpoint_path.write_bytes(_canonical(checkpoint) + b"\n")
    return checkpoint


def validate_checkpoint(path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(path).resolve()
    issues = []
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "issues": [f"checkpoint unreadable: {error}"]}
    material = dict(checkpoint)
    observed = material.pop("checkpoint_fingerprint", None)
    if observed != _fingerprint(material):
        issues.append("checkpoint fingerprint mismatch")
    if checkpoint.get("model_name") != "APC" or checkpoint.get("model_family") != "multimodal_temporal_neural_network":
        issues.append("checkpoint neural identity is invalid")
    if checkpoint.get("architecture_contract_fingerprint") != load_apc_neural_config()["config_fingerprint"]:
        issues.append("checkpoint architecture contract fingerprint mismatch")
    framework = checkpoint.get("framework", {})
    if not isinstance(framework, dict) or framework.get("name") != "pytorch" or framework.get("device") != "cpu":
        issues.append("checkpoint framework evidence is invalid")
    if checkpoint.get("status") != "offline_neural_candidate_unpromoted" or checkpoint.get("activation_authorized") is not False or checkpoint.get("recommendation_allowed") is not False:
        issues.append("checkpoint activation isolation is invalid")
    try:
        weights = checkpoint["weights"]
        weights_path = checkpoint_path.parent / weights["file"]
        model = load_apc_weights(weights_path, weights["weights_sha256"])
        if weights_path.stat().st_size != int(weights["weights_bytes"]):
            issues.append("weights byte count mismatch")
        if len(model.state_dict()) != int(weights["tensor_count"]):
            issues.append("weights tensor count mismatch")
        if asdict(model.architecture) != checkpoint.get("architecture"):
            # JSON turns the channel tuple into a list.
            expected = asdict(model.architecture)
            expected["visual_channels"] = list(expected["visual_channels"])
            if expected != checkpoint.get("architecture"):
                issues.append("weights architecture mismatch")
    except (KeyError, OSError, ValueError, RuntimeError) as error:
        issues.append(f"weights invalid: {error}")
    try:
        trained = checkpoint["training"]["trained_modalities"]
        untrained = checkpoint["training"]["untrained_modalities"]
        valid_modality_sets = (
            (trained == ["canonical_state_sequence"] and set(untrained) == {"visible_frame_sequence", "player_profile"})
            or (trained == ["canonical_state_sequence", "player_profile"] and untrained == ["visible_frame_sequence"])
        )
        if not valid_modality_sets:
            issues.append("checkpoint modality evidence is invalid")
        if checkpoint["training"]["group_exclusive_split"] is not True:
            issues.append("checkpoint group split evidence is invalid")
        if checkpoint["training"].get("pipeline_version") == "3.0.0" and (
            checkpoint["training"].get("counterfactual_groups_kept_complete_per_batch") is not True
            or checkpoint["training"].get("validation_selection") != "minimum_chosen_action_regret_then_mae_bb"
        ):
            issues.append("checkpoint grouped-ranking evidence is invalid")
        if checkpoint["gates"]["visible_table_training_ready"] is not False or checkpoint["gates"]["evaluated_coaching_ready"] is not False:
            issues.append("checkpoint readiness gates must remain closed")
    except (KeyError, TypeError):
        issues.append("checkpoint training/readiness evidence is missing")
    try:
        metrics = checkpoint["metrics"]
        for split in ("train", "validation", "test"):
            row = metrics[split]
            if int(row["examples"]) <= 0 or int(row["policy_states"]) <= 0:
                raise ValueError("empty split")
            for field in (
                "mae_bb", "rmse_bb", "bias_bb", "train_global_mean_baseline_mae_bb",
                "decision_accuracy", "chosen_action_regret_bb", "uncertainty_eace",
            ):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"non-finite {split} {field}")
        test = metrics["test"]
        expected_value_gate = float(test["mae_bb"]) < float(test["train_global_mean_baseline_mae_bb"])
        latency = checkpoint["latency"]
        expected_latency_gate = float(latency["p95_ms"]) <= float(latency["threshold_p95_ms"])
        if checkpoint["gates"]["finite_outputs"] is not True:
            issues.append("finite-output gate is invalid")
        if checkpoint["gates"]["test_value_improves_global_mean"] is not expected_value_gate:
            issues.append("value gate does not match held-out metrics")
        if checkpoint["gates"]["strategy_p95_under_50_ms"] is not expected_latency_gate or latency["passed"] is not expected_latency_gate:
            issues.append("latency gate does not match measured latency")
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"checkpoint metric evidence is invalid: {error}")
    return {"schema_version": SCHEMA_VERSION, "valid": not issues, "issues": issues, "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first genuine APC PyTorch candidate")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        report = validate_checkpoint(args.dataset)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["valid"] else 1)
    if args.output is None:
        parser.error("output is required for training")
    checkpoint = build_checkpoint(args.dataset, args.output, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    print(json.dumps(checkpoint, indent=2))


if __name__ == "__main__":
    main()
