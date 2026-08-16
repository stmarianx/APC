from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal, InvalidOperation
from typing import Iterable

from apc.deadline import ActionCommand


MATCH_CONFIDENCE_ORDER = {"approximate": 1, "close": 2, "exact": 3}


def _decimal(value: object, name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a decimal number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a decimal number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _bb(value: Decimal) -> str:
    return format(value, "f")


def _abstain(status: str, reasons: Iterable[str], started: float) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "status": status,
        "recommendation": None,
        "units": "BB",
        "reasons": sorted(set(reasons)),
        "actuation_authorized": False,
        "latency_ms": max(0.0, (time.perf_counter() - started) * 1000.0),
    }


def action_command_from_solver_id(
    action_id: str,
    *,
    pot_bb: object,
    to_call_bb: object,
    effective_stack_bb: object,
) -> ActionCommand:
    """Convert the solver interchange action grammar to APC's BB command contract."""
    if action_id in {"fold", "check"}:
        return ActionCommand(action_id)
    if action_id == "call":
        return ActionCommand("call", amount_bb=_bb(_decimal(to_call_bb, "to_call_bb")))
    if action_id == "all_in":
        return ActionCommand(
            "all_in",
            amount_bb=_bb(_decimal(effective_stack_bb, "effective_stack_bb")),
        )
    if ":" not in action_id:
        raise ValueError(f"solver action has no explicit sizing semantics: {action_id!r}")
    kind, raw_size = action_id.split(":", 1)
    size = _decimal(raw_size, f"{kind} size")
    if kind == "bet":
        # The solver interchange defines bet:FRACTION, so convert the fraction to BB.
        amount = _decimal(pot_bb, "pot_bb") * size
        return ActionCommand("bet", amount_bb=_bb(amount), to_amount_bb=_bb(amount))
    if kind == "raise_to":
        return ActionCommand("raise", to_amount_bb=_bb(size))
    raise ValueError(f"unsupported solver action id: {action_id!r}")


def _sample_index(rows: list[dict[str, object]], key: str) -> tuple[int, str]:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    draw = Decimal(int(digest[:16], 16)) / Decimal(2**64)
    cumulative = Decimal("0")
    for index, row in enumerate(rows):
        cumulative += Decimal(str(row["conditional_frequency"]))
        if draw < cumulative:
            return index, digest
    return len(rows) - 1, digest


def build_auditable_recommendation(
    backend_response: dict[str, object],
    *,
    recommendation_allowed: bool,
    perception_calibrated: bool,
    virtual_chip_environment: bool,
    decision_plan: dict[str, object],
    sampling_key: str,
    minimum_match_confidence: str = "exact",
) -> dict[str, object]:
    """Build a reproducible solver-backed recommendation or explicitly abstain.

    This function has no provider coordinates and performs no actuation. Its output is
    a BB-only coaching artifact that can be separately authorized by ``apc.deadline``.
    """
    started = time.perf_counter()
    gate_reasons: list[str] = []
    if not recommendation_allowed:
        gate_reasons.append("recommendation_gate_closed")
    if not perception_calibrated:
        gate_reasons.append("perception_not_calibrated")
    if not virtual_chip_environment:
        gate_reasons.append("virtual_chip_environment_not_verified")
    if not isinstance(sampling_key, str) or not sampling_key:
        gate_reasons.append("sampling_key_missing")
    if minimum_match_confidence not in MATCH_CONFIDENCE_ORDER:
        raise ValueError("minimum_match_confidence must be approximate, close or exact")
    if gate_reasons:
        return _abstain("abstain_recommendation_gate", gate_reasons, started)

    plan_status = decision_plan.get("status")
    if plan_status == "fallback_required":
        fallback = decision_plan.get("fallback")
        if not isinstance(fallback, dict) or not isinstance(fallback.get("action"), str):
            return _abstain("abstain_invalid_deadline_fallback", ["fallback_missing"], started)
        command = ActionCommand(str(fallback["action"]))
        return {
            "schema_version": "1.0.0",
            "status": "safe_fallback_only",
            "recommendation": {
                "command": command.payload(),
                "gto_claim": False,
                "explanation": "No strategy tier fit the decision window; use the declared safe fallback.",
            },
            "units": "BB",
            "reasons": ["deadline_compute_budget_exhausted"],
            "actuation_authorized": False,
            "latency_ms": max(0.0, (time.perf_counter() - started) * 1000.0),
        }
    if plan_status != "compute" or not decision_plan.get("strategy_tier"):
        return _abstain(
            "abstain_deadline",
            [f"deadline_plan_{plan_status or 'missing'}"],
            started,
        )

    if backend_response.get("status") != "matched":
        return _abstain("abstain_solver_uncovered", ["solver_state_unmatched"], started)
    match = backend_response.get("match")
    state = backend_response.get("state")
    route = backend_response.get("strategy_route")
    if not isinstance(match, dict) or not isinstance(state, dict) or not isinstance(route, dict):
        return _abstain("abstain_invalid_strategy_response", ["strategy_evidence_missing"], started)
    confidence = str(match.get("confidence"))
    if MATCH_CONFIDENCE_ORDER.get(confidence, 0) < MATCH_CONFIDENCE_ORDER[minimum_match_confidence]:
        return _abstain(
            "abstain_match_confidence",
            [f"match_confidence_{confidence}"],
            started,
        )
    latency = route.get("latency")
    provenance = route.get("selected_provenance")
    blueprint = route.get("blueprint")
    actions = route.get("actions")
    if (
        not isinstance(latency, dict)
        or latency.get("completed_within_budget") is not True
        or not isinstance(provenance, dict)
        or not provenance.get("source")
        or not provenance.get("source_version")
        or not isinstance(blueprint, dict)
        or not blueprint.get("fingerprint")
        or not blueprint.get("node_id")
        or not isinstance(actions, list)
        or not actions
    ):
        return _abstain(
            "abstain_invalid_strategy_response",
            ["strategy_route_incomplete_or_late"],
            started,
        )

    parsed: list[dict[str, object]] = []
    try:
        for row in actions:
            if not isinstance(row, dict):
                raise ValueError("strategy action must be structured")
            frequency = _decimal(row.get("frequency"), "action frequency")
            ev = Decimal(str(row.get("ev_bb")))
            if not ev.is_finite():
                raise ValueError("action EV must be finite")
            action_id = str(row.get("action"))
            parsed.append(
                {
                    "action_id": action_id,
                    "command": action_command_from_solver_id(
                        action_id,
                        pot_bb=state.get("pot_bb"),
                        to_call_bb=state.get("to_call_bb"),
                        effective_stack_bb=state.get("effective_stack_bb"),
                    ).payload(),
                    "raw_frequency": frequency,
                    "ev_bb": ev,
                }
            )
    except (InvalidOperation, ValueError) as error:
        return _abstain("abstain_invalid_strategy_response", [str(error)], started)

    total = sum((row["raw_frequency"] for row in parsed), Decimal("0"))
    if total <= 0:
        return _abstain(
            "abstain_invalid_strategy_response",
            ["legal_strategy_frequency_mass_is_zero"],
            started,
        )
    best_ev = max(row["ev_bb"] for row in parsed)
    mix: list[dict[str, object]] = []
    for row in parsed:
        conditional = row["raw_frequency"] / total
        mix.append(
            {
                "action_id": row["action_id"],
                "command": row["command"],
                "raw_frequency": _bb(row["raw_frequency"]),
                "conditional_frequency": _bb(conditional),
                "ev_bb": _bb(row["ev_bb"]),
                "ev_loss_from_best_bb": _bb(best_ev - row["ev_bb"]),
            }
        )
    sample_material = json.dumps(
        {
            "sampling_key": sampling_key,
            "state_id": backend_response.get("state_id"),
            "revision": backend_response.get("last_revision"),
            "blueprint_fingerprint": blueprint["fingerprint"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    selected_index, sample_digest = _sample_index(mix, sample_material)
    selected = mix[selected_index]
    gto_verified = provenance.get("gto_verified") is True
    result = {
        "schema_version": "1.0.0",
        "status": "recommendation_ready",
        "recommendation": {
            "command": selected["command"],
            "action_id": selected["action_id"],
            "selected_frequency": selected["conditional_frequency"],
            "ev_bb": selected["ev_bb"],
            "ev_loss_from_best_bb": selected["ev_loss_from_best_bb"],
            "mixed_strategy": mix,
            "gto_claim": gto_verified,
            "explanation": (
                f"Solver-backed {confidence} node {blueprint['node_id']}; "
                f"reproducibly sampled {selected['action_id']} from the normalized mixed strategy."
            ),
        },
        "units": "BB",
        "actuation_authorized": False,
        "strategy_tier": decision_plan["strategy_tier"],
        "audit": {
            "state_id": backend_response.get("state_id"),
            "revision": backend_response.get("last_revision"),
            "match_confidence": confidence,
            "card_match": match.get("card_match"),
            "solver_node_id": blueprint["node_id"],
            "solver_fingerprint": blueprint["fingerprint"],
            "strategy_selection_status": route.get("selection_status"),
            "strategy_source": provenance["source"],
            "strategy_source_version": provenance["source_version"],
            "gto_verified": gto_verified,
            "route_latency": latency,
            "legal_frequency_mass": _bb(total),
            "sampling_sha256": sample_digest,
        },
        "reasons": [],
        "latency_ms": max(0.0, (time.perf_counter() - started) * 1000.0),
    }
    result["recommendation_sha256"] = hashlib.sha256(
        json.dumps(result["recommendation"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result
