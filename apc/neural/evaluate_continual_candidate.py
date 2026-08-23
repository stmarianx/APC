from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from apc.neural.continual_training import (
    _metrics,
    _predict,
    evaluate_temporal_latency,
    validate_completed_replay_checkpoint,
)
from apc.neural.model import load_apc_weights
from apc.neural.replay_adapter import ReplayTemporalCorpus, load_replay_temporal_corpus
from apc.neural.train_candidate import validate_checkpoint


SCHEMA_VERSION = "1.0.0"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_hand_bootstrap(
    corpus: ReplayTemporalCorpus,
    indices: np.ndarray,
    incumbent_prediction: dict[str, np.ndarray],
    candidate_prediction: dict[str, np.ndarray],
    *,
    samples: int = 5000,
    seed: int = 20260826,
) -> dict[str, object]:
    if samples < 100 or len(indices) == 0:
        raise ValueError("APC paired replay bootstrap parameters are invalid")
    by_hand: defaultdict[str, list[int]] = defaultdict(list)
    for local, global_index in enumerate(indices):
        by_hand[corpus.replay_fingerprints[int(global_index)]].append(local)
    hands = sorted(by_hand)
    if len(hands) < 2:
        raise ValueError("APC paired replay bootstrap needs at least two complete hands")
    actual = corpus.target_return_bb[indices].astype(np.float64)
    incumbent_error = incumbent_prediction["value"].astype(np.float64) - actual
    candidate_error = candidate_prediction["value"].astype(np.float64) - actual
    chosen = corpus.chosen_action_index[indices]
    incumbent_correct = incumbent_prediction["policy"].argmax(axis=1) == chosen
    candidate_correct = candidate_prediction["policy"].argmax(axis=1) == chosen
    rng = random.Random(seed)
    distributions = {"mae_improvement_bb": [], "rmse_improvement_bb": [], "action_accuracy_improvement": []}
    for _ in range(samples):
        selected_hands = [hands[rng.randrange(len(hands))] for _ in hands]
        selected = np.asarray([local for hand in selected_hands for local in by_hand[hand]], dtype=np.int64)
        distributions["mae_improvement_bb"].append(float(np.abs(incumbent_error[selected]).mean() - np.abs(candidate_error[selected]).mean()))
        distributions["rmse_improvement_bb"].append(float(np.sqrt(np.mean(incumbent_error[selected] ** 2)) - np.sqrt(np.mean(candidate_error[selected] ** 2))))
        distributions["action_accuracy_improvement"].append(float(candidate_correct[selected].mean() - incumbent_correct[selected].mean()))
    result: dict[str, object] = {
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "complete_hands": len(hands),
        "resampling_unit": "complete_hand",
    }
    for key, values in distributions.items():
        result[key] = {
            "mean": format(float(np.mean(values)), ".12g"),
            "lower_95": format(_percentile(values, 0.025), ".12g"),
            "upper_95": format(_percentile(values, 0.975), ".12g"),
        }
    return result


def evaluate_fresh_completed_replay(
    replay_buffer: str | Path,
    incumbent_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260826,
) -> dict[str, object]:
    incumbent_path = Path(incumbent_checkpoint).resolve()
    candidate_path = Path(candidate_checkpoint).resolve()
    incumbent_validation = validate_checkpoint(incumbent_path)
    candidate_validation = validate_completed_replay_checkpoint(candidate_path)
    if not incumbent_validation["valid"]:
        raise ValueError("APC incumbent is invalid: " + "; ".join(incumbent_validation["issues"]))
    if not candidate_validation["valid"]:
        raise ValueError("APC continual candidate is invalid: " + "; ".join(candidate_validation["issues"]))
    incumbent_record = json.loads(incumbent_path.read_text(encoding="utf-8"))
    candidate_record = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate_record["incumbent"]["checkpoint_fingerprint"] != incumbent_record["checkpoint_fingerprint"]:
        raise ValueError("APC fresh audit incumbent does not match the candidate declaration")
    corpus = load_replay_temporal_corpus(replay_buffer)
    training_fingerprints = {str(row["replay_fingerprint"]) for row in candidate_record["replay"]["sources"]}
    audit_fingerprints = set(corpus.replay_fingerprints)
    if training_fingerprints & audit_fingerprints:
        raise ValueError("APC fresh replay audit overlaps candidate training hands")
    if corpus.manifest.get("replay_buffer_content_fingerprint") == candidate_record["replay"].get("replay_buffer_content_fingerprint"):
        raise ValueError("APC fresh replay audit reuses the candidate training buffer")
    incumbent = load_apc_weights(incumbent_path.parent / str(incumbent_record["weights"]["file"]), str(incumbent_record["weights"]["weights_sha256"]))
    candidate = load_apc_weights(candidate_path.parent / str(candidate_record["weights"]["file"]), str(candidate_record["weights"]["weights_sha256"]))
    indices = corpus.indices("test")
    if len({corpus.replay_fingerprints[int(index)] for index in indices}) < 20:
        raise ValueError("APC fresh replay audit requires at least 20 complete test hands")
    incumbent_prediction = _predict(incumbent, corpus, indices, 64)
    candidate_prediction = _predict(candidate, corpus, indices, 64)
    incumbent_metrics = _metrics(corpus, indices, incumbent_prediction)
    candidate_metrics = _metrics(corpus, indices, candidate_prediction)
    paired = paired_hand_bootstrap(
        corpus,
        indices,
        incumbent_prediction,
        candidate_prediction,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    mae_ci = paired["mae_improvement_bb"]
    rmse_ci = paired["rmse_improvement_bb"]
    action_ci = paired["action_accuracy_improvement"]
    gates = {
        "fresh_buffer_disjoint": True,
        "minimum_20_complete_test_hands": True,
        "mae_improvement_lower_95_above_zero": float(mae_ci["lower_95"]) > 0,
        "rmse_improvement_lower_95_above_zero": float(rmse_ci["lower_95"]) > 0,
        "action_accuracy_lower_95_nonnegative": float(action_ci["lower_95"]) >= 0,
        "strategy_regression_passed": candidate_record["gates"].get("strategy_regression_passed") is True,
        "calibration_passed": False,
        "promotion_authorized": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_name": "APC",
        "audit_kind": "fresh_completed_replay_paired_incumbent",
        "status": "evaluated_not_promoted",
        "audit_replay": {
            "content_fingerprint": corpus.manifest["replay_buffer_content_fingerprint"],
            "adapter_fingerprint": corpus.manifest["adapter_fingerprint"],
            "complete_hands": corpus.manifest["completed_hands"],
            "decisions": corpus.manifest["decisions"],
            "evaluated_split": "test",
            "used_for_training_or_selection": False,
        },
        "incumbent_checkpoint_fingerprint": incumbent_record["checkpoint_fingerprint"],
        "candidate_checkpoint_fingerprint": candidate_record["checkpoint_fingerprint"],
        "incumbent": incumbent_metrics,
        "candidate": candidate_metrics,
        "paired_bootstrap": paired,
        "latency": evaluate_temporal_latency(candidate, corpus),
        "gates": gates,
        "recommendation_allowed": False,
        "activation_authorized": False,
    }
    report["report_fingerprint"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def validate_fresh_replay_report(path: str | Path) -> dict[str, object]:
    issues = []
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"schema_version": SCHEMA_VERSION, "valid": False, "issues": [f"report unreadable: {error}"]}
    material = dict(report)
    observed = material.pop("report_fingerprint", None)
    if observed != hashlib.sha256(_canonical(material)).hexdigest():
        issues.append("report fingerprint mismatch")
    try:
        if report["model_name"] != "APC" or report["audit_kind"] != "fresh_completed_replay_paired_incumbent":
            issues.append("report identity is invalid")
        if report["status"] != "evaluated_not_promoted":
            issues.append("report status is invalid")
        if report["recommendation_allowed"] is not False or report["activation_authorized"] is not False:
            issues.append("report activation isolation is invalid")
        if report["audit_replay"]["used_for_training_or_selection"] is not False or report["audit_replay"]["evaluated_split"] != "test":
            issues.append("report audit provenance is invalid")
        if report["incumbent_checkpoint_fingerprint"] == report["candidate_checkpoint_fingerprint"]:
            issues.append("report candidate and incumbent are identical")
        paired = report["paired_bootstrap"]
        gates = report["gates"]
        expected = {
            "minimum_20_complete_test_hands": int(paired["complete_hands"]) >= 20,
            "mae_improvement_lower_95_above_zero": float(paired["mae_improvement_bb"]["lower_95"]) > 0,
            "rmse_improvement_lower_95_above_zero": float(paired["rmse_improvement_bb"]["lower_95"]) > 0,
            "action_accuracy_lower_95_nonnegative": float(paired["action_accuracy_improvement"]["lower_95"]) >= 0,
        }
        if any(gates.get(key) is not value for key, value in expected.items()):
            issues.append("report paired evidence gates are invalid")
        latency = report["latency"]
        if latency["passed"] is not (float(latency["p95_ms"]) <= float(latency["threshold_p95_ms"])):
            issues.append("report latency gate is invalid")
        if gates["promotion_authorized"] is not False or gates["calibration_passed"] is not False:
            issues.append("report promotion/calibration gates are invalid")
        for side in ("incumbent", "candidate"):
            for field in ("mae_bb", "rmse_bb", "bias_bb", "observed_action_accuracy", "temporal_consistency_mean"):
                if not math.isfinite(float(report[side][field])):
                    raise ValueError(f"non-finite {side} {field}")
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"report evidence is invalid: {error}")
    return {"schema_version": SCHEMA_VERSION, "valid": not issues, "issues": issues, "report_fingerprint": report.get("report_fingerprint")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fresh paired APC continual-candidate replay audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("replay_buffer", type=Path)
    evaluate.add_argument("incumbent_checkpoint", type=Path)
    evaluate.add_argument("candidate_checkpoint", type=Path)
    evaluate.add_argument("--bootstrap-samples", type=int, default=5000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=20260826)
    evaluate.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        validation = validate_fresh_replay_report(args.report)
        print(json.dumps(validation, indent=2))
        raise SystemExit(0 if validation["valid"] else 1)
    report = evaluate_fresh_completed_replay(
        args.replay_buffer,
        args.incumbent_checkpoint,
        args.candidate_checkpoint,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = _canonical(report) + b"\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite APC audit report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
