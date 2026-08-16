from __future__ import annotations

import argparse
import json
import random
import sys
from decimal import Decimal
from pathlib import Path

from apc.self_learning.train_candidate import predict_candidate, validate_candidate_checkpoint
from apc.virtual_table import VirtualDecisionTable


def _registry() -> object:
    try:
        from poker_coach import SolverExportRegistry
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach import SolverExportRegistry
    return SolverExportRegistry


def bootstrap_mean_ci(
    differences_bb: list[Decimal],
    *,
    seed: int = 20260816,
    samples: int = 5000,
    confidence: Decimal = Decimal("0.95"),
) -> dict[str, str]:
    if not differences_bb:
        raise ValueError("paired confidence interval requires differences")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if not Decimal("0") < confidence < Decimal("1"):
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(seed)
    count = len(differences_bb)
    means = sorted(
        sum((differences_bb[rng.randrange(count)] for _ in range(count)), Decimal("0"))
        / Decimal(count)
        for _ in range(samples)
    )
    alpha = (Decimal("1") - confidence) / Decimal("2")
    lower_index = int(alpha * Decimal(samples - 1))
    upper_index = int((Decimal("1") - alpha) * Decimal(samples - 1))
    mean = sum(differences_bb, Decimal("0")) / Decimal(count)
    return {
        "method": "paired_node_bootstrap_percentile",
        "confidence": format(confidence, "f"),
        "samples": str(samples),
        "seed": str(seed),
        "mean_improvement_bb": format(mean, "f"),
        "lower_bb": format(means[lower_index], "f"),
        "upper_bb": format(means[upper_index], "f"),
    }


def evaluate_candidate_against_uniform(
    checkpoint: str | Path,
    solver_export: str | Path,
    *,
    minimum_nodes: int = 30,
    minimum_coverage: Decimal = Decimal("0.90"),
    bootstrap_samples: int = 5000,
    seed: int = 20260816,
) -> dict[str, object]:
    if minimum_nodes <= 0:
        raise ValueError("minimum_nodes must be positive")
    if not Decimal("0") <= minimum_coverage <= Decimal("1"):
        raise ValueError("minimum_coverage must be between zero and one")
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    validation = validate_candidate_checkpoint(checkpoint_payload)
    if not validation["valid"]:
        raise ValueError("candidate checkpoint is invalid: " + "; ".join(validation["issues"]))
    bundle = _registry()().parse_file(Path(solver_export).resolve()).bundle
    rows: list[dict[str, object]] = []
    unsupported: list[dict[str, object]] = []
    differences: list[Decimal] = []
    candidate_total = reference_total = Decimal("0")
    for spot in bundle.spots:
        observation = VirtualDecisionTable(spot).observation()
        legal_actions = list(observation["state"]["legal_actions"])
        prediction = predict_candidate(checkpoint_payload, observation["state"], legal_actions)
        if prediction["status"] != "prediction_ready":
            unsupported.append(
                {
                    "node_id": spot.node_id,
                    "unsupported_actions": prediction["unsupported_actions"],
                }
            )
            continue
        probabilities = {
            action: Decimal(value) for action, value in prediction["probabilities"].items()
        }
        candidate_ev = sum(
            (probabilities[action.action] * action.ev for action in spot.actions),
            Decimal("0"),
        )
        reference_ev = sum((action.ev for action in spot.actions), Decimal("0")) / Decimal(len(spot.actions))
        difference = candidate_ev - reference_ev
        differences.append(difference)
        candidate_total += candidate_ev
        reference_total += reference_ev
        rows.append(
            {
                "node_id": spot.node_id,
                "candidate_expected_ev_bb": format(candidate_ev, "f"),
                "uniform_reference_expected_ev_bb": format(reference_ev, "f"),
                "paired_improvement_bb": format(difference, "f"),
                "prediction_fingerprint": prediction["prediction_fingerprint"],
            }
        )
    covered = len(rows)
    total = len(bundle.spots)
    coverage = Decimal(covered) / Decimal(total) if total else Decimal("0")
    interval = None if not differences else bootstrap_mean_ci(
        differences,
        seed=seed,
        samples=bootstrap_samples,
    )
    statistical_gate = (
        interval is not None
        and covered >= minimum_nodes
        and coverage >= minimum_coverage
        and Decimal(interval["lower_bb"]) > 0
    )
    reasons: list[str] = []
    if covered < minimum_nodes:
        reasons.append("minimum_independent_nodes_not_met")
    if coverage < minimum_coverage:
        reasons.append("candidate_action_vocabulary_coverage_below_threshold")
    if interval is None or Decimal(interval["lower_bb"]) <= 0:
        reasons.append("paired_improvement_confidence_interval_does_not_exclude_zero")
    reasons.append("uniform_reference_is_not_a_declared_incumbent")
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "paired_virtual_decision_policy_reference",
        "passed": covered > 0,
        "promotion_eligible": False,
        "candidate_checkpoint_fingerprint": checkpoint_payload["checkpoint_fingerprint"],
        "comparison": {
            "kind": "uniform_legal_action_reference",
            "qualifies_as_incumbent": False,
        },
        "coverage": {
            "solver_nodes": total,
            "evaluated_nodes": covered,
            "unsupported_nodes": len(unsupported),
            "fraction": format(coverage, "f"),
        },
        "objective": {
            "units": "BB",
            "candidate_mean_ev_bb": None if not covered else format(candidate_total / Decimal(covered), "f"),
            "reference_mean_ev_bb": None if not covered else format(reference_total / Decimal(covered), "f"),
            "paired_confidence_interval": interval,
        },
        "promotion_gate": {
            "passed": False,
            "statistical_subgate_passed": statistical_gate,
            "minimum_nodes": minimum_nodes,
            "minimum_coverage": format(minimum_coverage, "f"),
            "reasons": reasons,
            "activation_authorized": False,
        },
        "rows": rows,
        "unsupported": unsupported,
        "limitations": [
            "The comparator is a uniform legal-action reference, not the deployed incumbent policy.",
            "Episodes use imported solver EVs and are not full-hand sampled virtual-chip outcomes.",
            "A promotion claim requires an incumbent checkpoint, representative independent nodes, safety non-regression and rollback evidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paired APC candidate/reference virtual-decision evaluation.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("solver_export", type=Path)
    parser.add_argument("--minimum-nodes", type=int, default=30)
    parser.add_argument("--minimum-coverage", type=Decimal, default=Decimal("0.90"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_candidate_against_uniform(
            args.checkpoint,
            args.solver_export,
            minimum_nodes=args.minimum_nodes,
            minimum_coverage=args.minimum_coverage,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
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
