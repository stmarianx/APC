from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Protocol

from .solutions import ActionSolution, SolvedSpot


D = Decimal
DEFAULT_LATENCY_BUDGET_MS = 75
DEFAULT_CFV_TOLERANCE_BB = D("0.000001")
SAFE_CERTIFICATE_METHOD = "parent_cfv_bounds_v1"


@dataclass(frozen=True)
class SubgameRefinementRequest:
    request_id: str
    state_id: str
    revision: int
    blueprint_fingerprint: str
    blueprint_node_id: str
    legal_actions: tuple[str, ...]
    latency_budget_ms: int
    public_belief_state_id: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.state_id:
            raise ValueError("Subgame request requires request and state ids")
        if self.revision < 0:
            raise ValueError("Subgame request revision cannot be negative")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        if not self.blueprint_fingerprint:
            raise ValueError("Subgame request requires a blueprint fingerprint")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("Subgame request legal actions must be unique")


@dataclass(frozen=True)
class RefinementSafetyCertificate:
    method: str
    parent_blueprint_fingerprint: str
    max_parent_cfv_violation_bb: Decimal
    verified: bool
    details: str = ""

    def __post_init__(self) -> None:
        if self.max_parent_cfv_violation_bb < 0:
            raise ValueError("CFV violation cannot be negative")


@dataclass(frozen=True)
class SubgameRefinement:
    actions: tuple[ActionSolution, ...]
    source: str
    source_version: str
    certificate: RefinementSafetyCertificate

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("Subgame refinement requires actions")
        if not self.source or not self.source_version:
            raise ValueError("Subgame refinement requires versioned provenance")
        if len({action.action for action in self.actions}) != len(self.actions):
            raise ValueError("Refined action ids must be unique")
        total = sum((action.frequency for action in self.actions), D("0"))
        if abs(total - D("1")) > D("0.000001"):
            raise ValueError("Refined action frequencies must sum to one")


class SubgameRefiner(Protocol):
    def refine(
        self, request: SubgameRefinementRequest, blueprint: SolvedSpot
    ) -> SubgameRefinement: ...


class StrategySelectionService:
    """Select a certified refinement or return the cached blueprint by deadline."""

    def __init__(
        self,
        refiner: SubgameRefiner | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        cfv_tolerance_bb: Decimal = DEFAULT_CFV_TOLERANCE_BB,
    ) -> None:
        if cfv_tolerance_bb < 0:
            raise ValueError("cfv_tolerance_bb cannot be negative")
        self.refiner = refiner
        self.clock = clock
        self.cfv_tolerance_bb = cfv_tolerance_bb

    def select(
        self,
        blueprint: SolvedSpot,
        *,
        state_id: str,
        revision: int,
        legal_actions: tuple[str, ...] = (),
        latency_budget_ms: int = DEFAULT_LATENCY_BUDGET_MS,
        public_belief_state_id: str | None = None,
    ) -> dict[str, object]:
        request = SubgameRefinementRequest(
            request_id=f"{state_id[:16]}:{revision}",
            state_id=state_id,
            revision=revision,
            blueprint_fingerprint=blueprint.key.fingerprint,
            blueprint_node_id=blueprint.node_id,
            legal_actions=legal_actions,
            latency_budget_ms=latency_budget_ms,
            public_belief_state_id=public_belief_state_id,
        )
        started = self.clock()
        if self.refiner is None:
            return self._blueprint_result(
                request, blueprint, "refiner_not_configured", started
            )

        outcomes: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def run_refiner() -> None:
            try:
                outcomes.put(("result", self.refiner.refine(request, blueprint)))
            except Exception as error:  # provider boundary; fallback remains deterministic
                outcomes.put(("error", error))

        worker = threading.Thread(
            target=run_refiner,
            name=f"subgame-refiner-{request.request_id}",
            daemon=True,
        )
        worker.start()
        worker.join(latency_budget_ms / 1000)
        if worker.is_alive():
            return self._blueprint_result(
                request, blueprint, "latency_budget_exceeded", started
            )
        try:
            kind, value = outcomes.get_nowait()
        except queue.Empty:
            return self._blueprint_result(
                request, blueprint, "refiner_returned_no_result", started
            )
        if kind == "error":
            return self._blueprint_result(
                request,
                blueprint,
                "refiner_error",
                started,
                detail=f"{type(value).__name__}: {value}",
            )
        assert isinstance(value, SubgameRefinement)
        violation = self._safety_violation(request, value)
        if violation is not None:
            return self._blueprint_result(
                request, blueprint, "unsafe_refinement", started, detail=violation
            )
        return self._refined_result(request, blueprint, value, started)

    def _safety_violation(
        self,
        request: SubgameRefinementRequest,
        refinement: SubgameRefinement,
    ) -> str | None:
        certificate = refinement.certificate
        if not certificate.verified:
            return "Safety certificate is not verified"
        if certificate.method != SAFE_CERTIFICATE_METHOD:
            return f"Unsupported safety method: {certificate.method}"
        if certificate.parent_blueprint_fingerprint != request.blueprint_fingerprint:
            return "Safety certificate references a different parent blueprint"
        if certificate.max_parent_cfv_violation_bb > self.cfv_tolerance_bb:
            return "Parent counterfactual-value violation exceeds tolerance"
        legal = set(request.legal_actions)
        if legal:
            unsupported = sorted(
                action.action for action in refinement.actions if action.action not in legal
            )
            if unsupported:
                return f"Refinement contains illegal actions: {', '.join(unsupported)}"
        return None

    def _blueprint_result(
        self,
        request: SubgameRefinementRequest,
        blueprint: SolvedSpot,
        reason: str,
        started: float,
        *,
        detail: str = "",
    ) -> dict[str, object]:
        legal = set(request.legal_actions)
        actions = tuple(
            action for action in blueprint.actions if not legal or action.action in legal
        )
        return self._result(
            request=request,
            blueprint=blueprint,
            actions=actions,
            selection_status="blueprint_fallback",
            selected_source=blueprint.source,
            selected_source_version=blueprint.source_version,
            started=started,
            fallback_reason=reason,
            fallback_detail=detail,
            certificate=None,
        )

    def _refined_result(
        self,
        request: SubgameRefinementRequest,
        blueprint: SolvedSpot,
        refinement: SubgameRefinement,
        started: float,
    ) -> dict[str, object]:
        return self._result(
            request=request,
            blueprint=blueprint,
            actions=refinement.actions,
            selection_status="certified_refinement",
            selected_source=refinement.source,
            selected_source_version=refinement.source_version,
            started=started,
            fallback_reason=None,
            fallback_detail="",
            certificate=refinement.certificate,
        )

    def _result(
        self,
        *,
        request: SubgameRefinementRequest,
        blueprint: SolvedSpot,
        actions: tuple[ActionSolution, ...],
        selection_status: str,
        selected_source: str,
        selected_source_version: str,
        started: float,
        fallback_reason: str | None,
        fallback_detail: str,
        certificate: RefinementSafetyCertificate | None,
    ) -> dict[str, object]:
        elapsed_ms = max(0, int(round((self.clock() - started) * 1000)))
        return {
            "selection_status": selection_status,
            "request": {
                "request_id": request.request_id,
                "state_id": request.state_id,
                "revision": request.revision,
                "public_belief_state_id": request.public_belief_state_id,
            },
            "latency": {
                "budget_ms": request.latency_budget_ms,
                "elapsed_ms": elapsed_ms,
                "completed_within_budget": elapsed_ms <= request.latency_budget_ms,
            },
            "fallback": {
                "used": fallback_reason is not None,
                "reason": fallback_reason,
                "detail": fallback_detail,
            },
            "blueprint": {
                "fingerprint": blueprint.key.fingerprint,
                "node_id": blueprint.node_id,
                "source": blueprint.source,
                "source_version": blueprint.source_version,
            },
            "selected_provenance": {
                "source": selected_source,
                "source_version": selected_source_version,
            },
            "safety": {
                "required_method": SAFE_CERTIFICATE_METHOD,
                "cfv_tolerance_bb": format(self.cfv_tolerance_bb, "f"),
                "certificate": None
                if certificate is None
                else {
                    "method": certificate.method,
                    "parent_blueprint_fingerprint": certificate.parent_blueprint_fingerprint,
                    "max_parent_cfv_violation_bb": format(
                        certificate.max_parent_cfv_violation_bb, "f"
                    ),
                    "verified": certificate.verified,
                    "details": certificate.details,
                },
            },
            "actions": [
                {
                    "action": action.action,
                    "frequency": format(action.frequency, "f"),
                    "ev_bb": format(action.ev, "f"),
                }
                for action in actions
            ],
        }
