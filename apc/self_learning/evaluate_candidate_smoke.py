from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from apc.self_learning.replay_dataset import build_replay_dataset
from apc.self_learning.train_candidate import train_candidate, validate_candidate_checkpoint


def _coach_types() -> tuple[object, object]:
    try:
        from poker_coach import PokerStarsParser, SolverExportRegistry
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach import PokerStarsParser, SolverExportRegistry
    return PokerStarsParser, SolverExportRegistry


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_candidate_smoke(
    solver_export: str | Path,
    hand_history: str | Path,
    output: str | Path,
    *,
    replicas: int = 60,
    seed: int = 20260816,
) -> dict[str, object]:
    """Exercise replay build -> candidate train -> checkpoint validation.

    Replicated hand groups prove mechanics only. They are deliberately unsuitable for
    generalization or promotion claims because their poker state is identical.
    """
    if replicas < 20:
        raise ValueError("candidate smoke requires at least 20 replicated hand groups")
    root = Path(output).resolve()
    if root.exists():
        raise ValueError(f"candidate smoke destination already exists: {root}")
    solver_path = Path(solver_export).resolve()
    hand_path = Path(hand_history).resolve()
    PokerStarsParser, SolverExportRegistry = _coach_types()
    parsed_hands = PokerStarsParser().parse_file(hand_path)
    if len(parsed_hands) != 1:
        raise ValueError("candidate smoke requires exactly one completed fixture hand")
    bundle = SolverExportRegistry().parse_file(solver_path).bundle
    source_hand = parsed_hands[0]
    hands = tuple(
        replace(source_hand, hand_id=f"candidate-smoke-{index:04d}")
        for index in range(replicas)
    )
    dataset_path = root / "dataset"
    checkpoint_path = root / "candidate.json"
    manifest = build_replay_dataset(
        dataset_path,
        hands,
        bundle.spots,
        dataset_id="candidate-smoke-v1",
        source_fingerprints={
            "solver_export_sha256": _file_sha256(solver_path),
            "hand_history_fixture_sha256": _file_sha256(hand_path),
        },
        seed=seed,
        split_ratios=(Decimal("0.60"), Decimal("0.20"), Decimal("0.20")),
        minimum_examples=replicas,
        minimum_groups=replicas,
    )
    checkpoint = train_candidate(
        dataset_path,
        checkpoint_path,
        seed=seed,
        feature_dimension=64,
        epochs=20,
    )
    validation = validate_candidate_checkpoint(checkpoint_path)
    report = {
        "schema_version": "1.0.0",
        "evaluation_kind": "candidate_training_pipeline_smoke",
        "passed": manifest["training_eligible"] is True and validation["valid"] is True,
        "promotion_eligible": False,
        "replicated_groups": replicas,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "split_counts": manifest["split_counts"],
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "metrics": checkpoint["metrics"],
        "activation_authorized": checkpoint["activation_authorized"],
        "incumbent_replaced": checkpoint["incumbent_replaced"],
        "limitations": [
            "All groups replicate one identical poker state; split metrics prove deterministic plumbing, not generalization.",
            "The imported solver target is not GTO-verified.",
            "No paired virtual-chip promotion, safety non-regression or rollback claim is made.",
        ],
    }
    (root / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run APC's end-to-end candidate training smoke evaluation.")
    parser.add_argument("solver_export", type=Path)
    parser.add_argument("hand_history", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicas", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)
    try:
        report = evaluate_candidate_smoke(
            args.solver_export,
            args.hand_history,
            args.output,
            replicas=args.replicas,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 3
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
