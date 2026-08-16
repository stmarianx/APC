from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from apc.perception.baseline import _percentile
from apc.recommendation import build_auditable_recommendation


def _coach_types() -> tuple[object, object]:
    try:
        from poker_coach import LiveTableService, SolverExportRegistry
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach import LiveTableService, SolverExportRegistry
    return LiveTableService, SolverExportRegistry


def evaluate_recommendations(solver_export: str | Path) -> dict[str, object]:
    LiveTableService, SolverExportRegistry = _coach_types()
    bundle = SolverExportRegistry().parse_file(Path(solver_export).resolve()).bundle
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for revision, spot in enumerate(bundle.spots):
        service = LiveTableService()
        session = service.create_session(f"recommendation-eval-{revision}")
        key = spot.key
        payload = {
            "schema_version": "1.0.0",
            "table_id": f"recommendation-eval-{revision}",
            "hand_id": f"hand-{revision}",
            "revision": revision,
            "game": key.game,
            "players": key.players,
            "hero_position": key.hero_position,
            "effective_stack_bb": format(key.effective_stack_bb, "f"),
            "pot_bb": format(key.pot_bb, "f"),
            "to_call_bb": "0",
            "board": [str(card) for card in key.board],
            "hero_cards": [str(card) for card in key.hero_cards],
            "action_history": list(key.action_history),
            "legal_actions": [action.action for action in spot.actions],
            "rake_model": key.rake_model,
            "utility_model": key.utility_model,
            "source": "apc_recommendation_regression",
        }
        started = time.perf_counter()
        backend = service.update_state(session["session_id"], payload, bundle.spots)
        plan = {
            "status": "compute",
            "strategy_tier": "cached_exact_solver",
            "remaining_ms": 10_000,
            "compute_budget_ms": 9_000,
        }
        first = build_auditable_recommendation(
            backend,
            recommendation_allowed=True,
            perception_calibrated=True,
            virtual_chip_environment=True,
            decision_plan=plan,
            sampling_key=key.fingerprint,
        )
        replay = build_auditable_recommendation(
            backend,
            recommendation_allowed=True,
            perception_calibrated=True,
            virtual_chip_environment=True,
            decision_plan=plan,
            sampling_key=key.fingerprint,
        )
        closed = build_auditable_recommendation(
            backend,
            recommendation_allowed=False,
            perception_calibrated=True,
            virtual_chip_environment=True,
            decision_plan=plan,
            sampling_key=key.fingerprint,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)
        recommendation = first.get("recommendation")
        audit = first.get("audit")
        rows.append(
            {
                "node_id": spot.node_id,
                "backend_status": backend.get("status"),
                "recommendation_status": first.get("status"),
                "recommendation_sha256": first.get("recommendation_sha256"),
                "deterministic_replay": first.get("recommendation_sha256") == replay.get("recommendation_sha256"),
                "gate_closed_abstained": closed.get("status") == "abstain_recommendation_gate" and closed.get("recommendation") is None,
                "bb_only_command": isinstance(recommendation, dict) and isinstance(recommendation.get("command"), dict),
                "provenance_complete": isinstance(audit, dict) and bool(audit.get("solver_fingerprint")) and bool(audit.get("strategy_source_version")),
                "actuation_authorized": first.get("actuation_authorized"),
                "latency_ms": elapsed,
            }
        )
    ready = sum(row["recommendation_status"] == "recommendation_ready" for row in rows)
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "solver_to_auditable_bb_recommendation_regression",
        "promotion_eligible": False,
        "solver_source": bundle.source,
        "solver_source_version": bundle.source_version,
        "nodes": len(rows),
        "metrics": {
            "exact_backend_matches": sum(row["backend_status"] == "matched" for row in rows),
            "recommendations_ready": ready,
            "deterministic_replays": sum(bool(row["deterministic_replay"]) for row in rows),
            "closed_gate_abstentions": sum(bool(row["gate_closed_abstained"]) for row in rows),
            "bb_only_commands": sum(bool(row["bb_only_command"]) for row in rows),
            "provenance_complete": sum(bool(row["provenance_complete"]) for row in rows),
            "actuation_authorization_violations": sum(row["actuation_authorized"] is not False for row in rows),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "passed": bool(rows) and ready == len(rows) and all(
            row["deterministic_replay"]
            and row["gate_closed_abstained"]
            and row["bb_only_command"]
            and row["provenance_complete"]
            and row["actuation_authorized"] is False
            for row in rows
        ),
        "rows": rows,
        "limitations": [
            "Strategy-only regression using an imported fixture; it excludes visible-table perception latency and cannot open Gate C.",
            "The evaluator verifies recommendation generation, gating, BB conversion, provenance and deterministic replay; it performs no actuation.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate APC's auditable BB recommendation layer.")
    parser.add_argument("solver_export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_recommendations(args.solver_export)
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
