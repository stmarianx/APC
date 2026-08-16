from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from apc.perception.baseline import _percentile
from apc.self_learning.paired_rollout_dataset import _hand_class, validate_paired_rollout_dataset
from apc.self_learning.train_paired_value import (
    _predict,
    validate_paired_value_checkpoint,
)
from apc.self_learning.train_value import _sha256


EVALUATED_ACTIONS = ("call", "raise")


def _load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _calibration(rows: list[dict[str, object]], bins: int) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: (float(row["prediction_bb"]), str(row["example_id"])))
    output = []
    weighted_gap = 0.0
    for index in range(bins):
        members = ordered[index * len(ordered) // bins : (index + 1) * len(ordered) // bins]
        if not members:
            continue
        predicted = sum(float(row["prediction_bb"]) for row in members) / len(members)
        actual = sum(float(row["actual_bb"]) for row in members) / len(members)
        gap = abs(predicted - actual)
        weighted_gap += gap * len(members)
        output.append({
            "examples": len(members),
            "mean_prediction_bb": format(predicted, ".12g"),
            "mean_actual_bb": format(actual, ".12g"),
            "absolute_gap_bb": format(gap, ".12g"),
        })
    return {
        "bins": output,
        "expected_absolute_calibration_error_bb": format(weighted_gap / len(ordered), ".12g"),
        "maximum_bin_gap_bb": format(max(float(row["absolute_gap_bb"]) for row in output), ".12g"),
    }


def evaluate_paired_value_confidence(
    dataset: str | Path,
    checkpoint: str | Path,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260816,
    calibration_bins: int = 10,
) -> dict[str, object]:
    if bootstrap_samples < 200 or calibration_bins < 2:
        raise ValueError("confidence evaluation requires >=200 bootstraps and >=2 calibration bins")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    dataset_path = Path(dataset).resolve()
    dataset_validation = validate_paired_rollout_dataset(dataset_path)
    if not dataset_validation["valid"]:
        raise ValueError("paired rollout dataset is invalid: " + "; ".join(dataset_validation["issues"]))
    manifest = _load_json(dataset_path / "manifest.json")
    checkpoint_payload = _load_json(checkpoint)
    checkpoint_validation = validate_paired_value_checkpoint(checkpoint_payload)
    if not checkpoint_validation["valid"]:
        raise ValueError("paired value checkpoint is invalid: " + "; ".join(checkpoint_validation["issues"]))
    if (
        checkpoint_payload["dataset"]["dataset_fingerprint"] != manifest["dataset_fingerprint"]
        or checkpoint_payload["dataset"]["examples_sha256"] != manifest["examples_sha256"]
    ):
        raise ValueError("checkpoint and paired dataset fingerprints do not match")
    examples = [
        json.loads(line)
        for line in (dataset_path / str(manifest["examples_file"])).read_text(encoding="utf-8").splitlines()
        if line
    ]
    test_rows = [
        row
        for row in examples
        if row["split"] == "test" and row["counterfactual_action"]["action"] in EVALUATED_ACTIONS
    ]
    configuration = checkpoint_payload["configuration"]
    table = checkpoint_payload["hand_class_action_table"]
    action_means = {key: float(value) for key, value in checkpoint_payload["action_train_means_bb"].items()}
    shrinkage = float(configuration["selected_shrinkage_by_validation"])
    evaluated: list[dict[str, object]] = []
    latencies: list[float] = []
    for row in test_rows:
        action = str(row["counterfactual_action"]["action"])
        hand_class = _hand_class([str(card) for card in row["state"]["hero_cards"]])
        started = time.perf_counter()
        prediction, covered, samples = _predict(table, action_means, hand_class, action, shrinkage)
        latencies.append((time.perf_counter() - started) * 1000.0)
        actual = float(str(row["learning_signal"]["hero_return_bb"]))
        baseline = action_means[action]
        evaluated.append({
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "action": action,
            "hand_class": hand_class,
            "prediction_bb": format(prediction, ".12g"),
            "actual_bb": format(actual, ".12g"),
            "baseline_bb": format(baseline, ".12g"),
            "absolute_error_improvement_bb": format(
                abs(baseline - actual) - abs(prediction - actual), ".12g"
            ),
            "exact_hand_class_coverage": covered,
            "training_samples": samples,
        })
    by_group: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for row in evaluated:
        by_group[str(row["group_id"])][str(row["action"])] = float(str(row["absolute_error_improvement_bb"]))
    if not by_group or any(set(values) != set(EVALUATED_ACTIONS) for values in by_group.values()):
        raise ValueError("test groups do not contain complete paired call/raise evidence")
    group_ids = sorted(by_group)
    point = {
        action: sum(by_group[group][action] for group in group_ids) / len(group_ids)
        for action in EVALUATED_ACTIONS
    }
    point["aggregate"] = sum(point[action] for action in EVALUATED_ACTIONS) / len(EVALUATED_ACTIONS)
    bootstrap: dict[str, list[float]] = {action: [] for action in (*EVALUATED_ACTIONS, "aggregate")}
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        sampled = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
        action_means_sample = {
            action: sum(by_group[group][action] for group in sampled) / len(sampled)
            for action in EVALUATED_ACTIONS
        }
        for action in EVALUATED_ACTIONS:
            bootstrap[action].append(action_means_sample[action])
        bootstrap["aggregate"].append(
            sum(action_means_sample.values()) / len(EVALUATED_ACTIONS)
        )
    intervals = {
        key: {
            "point_improvement_bb": format(point[key], ".12g"),
            "lower_95_bb": format(_percentile(values, 0.025), ".12g"),
            "upper_95_bb": format(_percentile(values, 0.975), ".12g"),
        }
        for key, values in bootstrap.items()
    }
    calibration = {
        action: _calibration([row for row in evaluated if row["action"] == action], calibration_bins)
        for action in EVALUATED_ACTIONS
    }
    coverage = sum(row["exact_hand_class_coverage"] is True for row in evaluated) / len(evaluated)
    confidence_passed = coverage >= 0.95 and all(
        float(intervals[key]["lower_95_bb"]) > 0 for key in (*EVALUATED_ACTIONS, "aggregate")
    )
    semantic = {
        "schema_version": "1.0.0",
        "evaluation_kind": "paired_preflop_value_confidence_and_calibration",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "checkpoint_fingerprint": checkpoint_payload["checkpoint_fingerprint"],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "test_groups": len(group_ids),
        "test_examples": len(evaluated),
        "exact_hand_class_coverage": format(coverage, ".12g"),
        "improvement_intervals": intervals,
        "calibration": calibration,
        "confidence_gate": {
            "passed": confidence_passed,
            "criterion": "paired_95pct_lower_bound_above_zero_for_call_raise_and_aggregate_with_95pct_coverage",
            "activation_authorized": False,
            "recommendation_allowed": False,
        },
    }
    report = dict(semantic)
    report["passed"] = True
    report["promotion_eligible"] = False
    report["validated_lookup_latency_ms"] = {
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "max": max(latencies),
    }
    report["evaluation_fingerprint"] = _sha256(semantic)
    report["limitations"] = [
        "Bootstrap units are complete same-deal test groups and preserve call/raise pairing.",
        "Calibration describes terminal returns against one deterministic continuation policy, not confidence in action optimality.",
        "This evaluation does not support all-in, postflop, non-button, GTO or population-opponent claims.",
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate paired APC value confidence and calibration.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_paired_value_confidence(
            args.dataset,
            args.checkpoint,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            calibration_bins=args.calibration_bins,
        )
        if args.output:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 3
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
