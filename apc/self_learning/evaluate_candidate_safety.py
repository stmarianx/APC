from __future__ import annotations

import argparse
import copy
import json
import sys
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


def _mutations(state: dict[str, object]) -> list[tuple[str, dict[str, object], list[str]]]:
    legal = list(state["legal_actions"])
    rows: list[tuple[str, dict[str, object], list[str]]] = []

    def add(name: str, field: str, value: object, actions: list[str] | None = None) -> None:
        mutated = copy.deepcopy(state)
        mutated[field] = value
        rows.append((name, mutated, list(actions or legal)))

    add("negative_stack", "effective_stack_bb", "-1")
    add("zero_pot", "pot_bb", "0")
    add("call_exceeds_stack", "to_call_bb", "9999")
    add("invalid_players", "players", 1)
    add("invalid_board_length", "board", list(state["board"])[:2])
    duplicate_cards = list(state["board"])
    if duplicate_cards:
        duplicate_cards[0] = state["hero_cards"][0]
    add("duplicate_cards", "board", duplicate_cards)
    invalid_cards = list(state["hero_cards"])
    invalid_cards[0] = "1x"
    add("invalid_card_token", "hero_cards", invalid_cards)
    add("noncanonical_history", "action_history", ["raises a little"])
    add("missing_rake", "rake_model", "")
    mismatched_legal = list(legal)
    mismatched_legal.append("fold")
    add("legal_state_mismatch", "legal_actions", mismatched_legal, legal)
    return rows


def evaluate_candidate_safety(
    checkpoint: str | Path,
    solver_export: str | Path,
) -> dict[str, object]:
    checkpoint_payload = json.loads(Path(checkpoint).resolve().read_text(encoding="utf-8"))
    validation = validate_candidate_checkpoint(checkpoint_payload)
    if not validation["valid"]:
        raise ValueError("candidate checkpoint is invalid: " + "; ".join(validation["issues"]))
    bundle = _registry()().parse_file(Path(solver_export).resolve()).bundle
    valid_rows: list[dict[str, object]] = []
    supported_state: dict[str, object] | None = None
    for spot in bundle.spots:
        state = VirtualDecisionTable(spot).observation()["state"]
        legal = list(state["legal_actions"])
        result = predict_candidate(checkpoint_payload, state, legal)
        if result["status"] == "prediction_ready_uncalibrated" and supported_state is None:
            supported_state = state
        valid_rows.append(
            {
                "node_id": spot.node_id,
                "status": result["status"],
                "recommendation_allowed": result["recommendation_allowed"],
                "activation_authorized": result["activation_authorized"],
                "confidence_calibrated": result["confidence_calibrated"],
            }
        )
    if supported_state is None:
        raise ValueError("candidate supports no solver node for adversarial safety evaluation")
    adversarial_rows: list[dict[str, object]] = []
    for name, state, legal in _mutations(supported_state):
        result = predict_candidate(checkpoint_payload, state, legal)
        adversarial_rows.append(
            {
                "mutation": name,
                "status": result["status"],
                "reasons": result.get("reasons", []),
                "probabilities_present": result["probabilities"] is not None,
                "recommendation_allowed": result["recommendation_allowed"],
                "activation_authorized": result["activation_authorized"],
            }
        )
    invalid_acceptances = sum(row["status"].startswith("prediction_ready") for row in adversarial_rows)
    recommendation_violations = sum(
        row["recommendation_allowed"] is not False or row["activation_authorized"] is not False
        for row in [*valid_rows, *adversarial_rows]
    )
    uncalibrated_ready = sum(row["status"] == "prediction_ready_uncalibrated" for row in valid_rows)
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "candidate_invalid_state_and_confidence_safety",
        "passed": invalid_acceptances == 0 and recommendation_violations == 0,
        "promotion_eligible": False,
        "paired_incumbent_non_regression": False,
        "candidate_checkpoint_fingerprint": checkpoint_payload["checkpoint_fingerprint"],
        "metrics": {
            "valid_solver_nodes": len(valid_rows),
            "uncalibrated_supported_nodes": uncalibrated_ready,
            "unsupported_valid_nodes": sum(row["status"] == "abstain_unsupported_actions" for row in valid_rows),
            "adversarial_invalid_states": len(adversarial_rows),
            "invalid_state_acceptances": invalid_acceptances,
            "recommendation_or_activation_violations": recommendation_violations,
        },
        "valid_rows": valid_rows,
        "adversarial_rows": adversarial_rows,
        "limitations": [
            "This is standalone candidate safety evidence; no declared incumbent checkpoint is evaluated side by side.",
            "The candidate remains uncalibrated and therefore cannot emit coaching recommendations.",
            "The mutation corpus is deterministic and bounded; broader fuzzing and controlled-visible safety evaluation remain open."
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit APC candidate invalid-state and confidence safety.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("solver_export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_candidate_safety(args.checkpoint, args.solver_export)
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
