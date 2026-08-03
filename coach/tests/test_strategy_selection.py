from __future__ import annotations

import time
import unittest
from decimal import Decimal
from pathlib import Path

from poker_coach import (
    ActionSolution,
    RefinementSafetyCertificate,
    SAFE_CERTIFICATE_METHOD,
    SolverExportRegistry,
    StrategySelectionService,
    SubgameRefinement,
)


D = Decimal
ROOT = Path(__file__).resolve().parents[1]


class StaticRefiner:
    def __init__(self, factory):
        self.factory = factory
        self.requests = []

    def refine(self, request, blueprint):
        self.requests.append(request)
        return self.factory(request, blueprint)


def certified(request, blueprint, *, actions=None, **certificate_changes):
    certificate = {
        "method": SAFE_CERTIFICATE_METHOD,
        "parent_blueprint_fingerprint": blueprint.key.fingerprint,
        "max_parent_cfv_violation_bb": D("0"),
        "verified": True,
        "details": "Boundary counterfactual values checked against parent blueprint.",
    }
    certificate.update(certificate_changes)
    return SubgameRefinement(
        actions=actions or blueprint.actions,
        source="test_safe_subgame_solver",
        source_version="1.0",
        certificate=RefinementSafetyCertificate(**certificate),
    )


class StrategySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blueprint = SolverExportRegistry().parse_file(
            ROOT / "examples" / "sample_solver_export.csv"
        ).bundle.spots[-1]

    def select(self, service, **changes):
        arguments = {
            "state_id": "a" * 64,
            "revision": 7,
            "legal_actions": tuple(
                action.action for action in self.blueprint.actions
            ),
            "latency_budget_ms": 25,
            "public_belief_state_id": "pbs:test:7",
        }
        arguments.update(changes)
        return service.select(self.blueprint, **arguments)

    def test_missing_refiner_returns_versioned_blueprint_fallback(self) -> None:
        result = self.select(StrategySelectionService())
        self.assertEqual(result["selection_status"], "blueprint_fallback")
        self.assertEqual(result["fallback"]["reason"], "refiner_not_configured")
        self.assertEqual(
            result["blueprint"]["fingerprint"], self.blueprint.key.fingerprint
        )
        self.assertEqual(result["request"]["public_belief_state_id"], "pbs:test:7")
        self.assertEqual(len(result["actions"]), len(self.blueprint.actions))

    def test_verified_parent_cfv_certificate_selects_refinement(self) -> None:
        actions = (
            ActionSolution("check", D("0.25"), D("9.70")),
            ActionSolution("bet:0.75", D("0.30"), D("9.82")),
            ActionSolution("bet:1", D("0.45"), D("9.91")),
        )
        refiner = StaticRefiner(
            lambda request, blueprint: certified(
                request, blueprint, actions=actions
            )
        )
        result = self.select(StrategySelectionService(refiner))
        self.assertEqual(result["selection_status"], "certified_refinement")
        self.assertFalse(result["fallback"]["used"])
        self.assertEqual(result["selected_provenance"]["source"], "test_safe_subgame_solver")
        self.assertEqual(result["actions"][2]["frequency"], "0.45")
        self.assertTrue(result["safety"]["certificate"]["verified"])
        self.assertEqual(refiner.requests[0].latency_budget_ms, 25)

    def test_wrong_parent_or_excess_cfv_violation_falls_back(self) -> None:
        cases = (
            {"parent_blueprint_fingerprint": "f" * 64},
            {"max_parent_cfv_violation_bb": D("0.01")},
            {"verified": False},
            {"method": "unsafe_local_best_response"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                refiner = StaticRefiner(
                    lambda request, blueprint, row=changes: certified(
                        request, blueprint, **row
                    )
                )
                result = self.select(StrategySelectionService(refiner))
                self.assertEqual(result["selection_status"], "blueprint_fallback")
                self.assertEqual(result["fallback"]["reason"], "unsafe_refinement")
                self.assertTrue(result["fallback"]["detail"])
                self.assertIsNone(result["safety"]["certificate"])

    def test_refinement_with_action_outside_table_legality_falls_back(self) -> None:
        actions = (
            ActionSolution("check", D("0.5"), D("9.7")),
            ActionSolution("bet:0.1", D("0.5"), D("9.8")),
        )
        refiner = StaticRefiner(
            lambda request, blueprint: certified(
                request, blueprint, actions=actions
            )
        )
        result = self.select(StrategySelectionService(refiner))
        self.assertEqual(result["fallback"]["reason"], "unsafe_refinement")
        self.assertIn("bet:0.1", result["fallback"]["detail"])

    def test_deadline_or_provider_error_returns_blueprint(self) -> None:
        class SlowRefiner:
            def refine(self, request, blueprint):
                time.sleep(0.05)
                return certified(request, blueprint)

        started = time.perf_counter()
        timed_out = self.select(
            StrategySelectionService(SlowRefiner()), latency_budget_ms=5
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(timed_out["fallback"]["reason"], "latency_budget_exceeded")
        self.assertLess(elapsed, 0.04)

        class BrokenRefiner:
            def refine(self, request, blueprint):
                raise RuntimeError("solver process unavailable")

        failed = self.select(StrategySelectionService(BrokenRefiner()))
        self.assertEqual(failed["fallback"]["reason"], "refiner_error")
        self.assertIn("solver process unavailable", failed["fallback"]["detail"])


if __name__ == "__main__":
    unittest.main()
