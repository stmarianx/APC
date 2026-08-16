from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from apc.perception.baseline import _percentile
from apc.virtual_table import VirtualDecisionTable


def _registry() -> object:
    try:
        from poker_coach import SolverExportRegistry
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach import SolverExportRegistry
    return SolverExportRegistry


def evaluate_virtual_table(solver_export: str | Path) -> dict[str, object]:
    bundle = _registry()().parse_file(Path(solver_export).resolve()).bundle
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    illegal_rejections = duplicate_rejections = 0
    for spot in bundle.spots:
        probe = VirtualDecisionTable(spot)
        try:
            probe.step("not_a_legal_action")
        except ValueError:
            illegal_rejections += 1
        for action in spot.actions:
            table = VirtualDecisionTable(spot)
            observation = table.observation()
            started = time.perf_counter()
            feedback = table.step(action.action)
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed)
            try:
                table.step(action.action)
            except ValueError:
                duplicate_rejections += 1
            command = feedback["command"]
            rows.append(
                {
                    "node_id": spot.node_id,
                    "action_id": action.action,
                    "units": feedback["units"],
                    "legal_in_observation": action.action in observation["state"]["legal_actions"],
                    "command_bb_only": isinstance(command, dict)
                    and all(not key.endswith("_bb") or isinstance(value, str) for key, value in command.items()),
                    "reward_exact": feedback["reward_bb"] == format(action.ev, "f"),
                    "regret_non_negative": float(feedback["regret_bb"]) >= 0,
                    "terminal": feedback["status"] == "terminal",
                    "external_actuation": feedback["external_actuation"],
                    "feedback_fingerprint_length": len(feedback["feedback_fingerprint"]),
                    "latency_ms": elapsed,
                }
            )
    expected_actions = sum(len(spot.actions) for spot in bundle.spots)
    passed = (
        len(rows) == expected_actions
        and illegal_rejections == len(bundle.spots)
        and duplicate_rejections == expected_actions
        and all(
            row["units"] == "BB"
            and row["legal_in_observation"]
            and row["command_bb_only"]
            and row["reward_exact"]
            and row["regret_non_negative"]
            and row["terminal"]
            and row["external_actuation"] is False
            and row["feedback_fingerprint_length"] == 64
            for row in rows
        )
    )
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "controlled_virtual_chip_decision_provider",
        "passed": passed,
        "promotion_eligible": False,
        "solver_source": bundle.source,
        "solver_source_version": bundle.source_version,
        "metrics": {
            "nodes": len(bundle.spots),
            "actions": len(rows),
            "expected_actions": expected_actions,
            "illegal_action_rejections": illegal_rejections,
            "duplicate_step_rejections": duplicate_rejections,
            "external_actuation_violations": sum(row["external_actuation"] is not False for row in rows),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "rows": rows,
        "limitations": [
            "Each episode is one solver-backed decision, not a complete dealt poker hand.",
            "Reward is imported action EV in BB rather than sampled showdown chips.",
            "This provider enables controlled policy evaluation but cannot by itself satisfy paired promotion or full self-play gates.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate APC's controlled virtual-chip decision provider.")
    parser.add_argument("solver_export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_virtual_table(args.solver_export)
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
