from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable


ACTIONS = {"fold", "check", "call", "bet", "raise", "all_in"}
TIER_REQUIREMENTS_MS = {
    "fast_policy": 10,
    "cached_exact_solver": 20,
    "cached_blueprint": 75,
    "bounded_refinement": 500,
}
TIER_QUALITY = {
    "fast_policy": 1,
    "cached_blueprint": 2,
    "cached_exact_solver": 3,
    "bounded_refinement": 4,
}


def _bb(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a BB decimal string") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return format(parsed, "f")


@dataclass(frozen=True)
class ActionCommand:
    action: str
    amount_bb: str | None = None
    to_amount_bb: str | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unsupported poker action: {self.action}")
        object.__setattr__(self, "amount_bb", _bb(self.amount_bb, "amount_bb"))
        object.__setattr__(self, "to_amount_bb", _bb(self.to_amount_bb, "to_amount_bb"))
        if self.action in {"fold", "check"} and (
            self.amount_bb is not None or self.to_amount_bb is not None
        ):
            raise ValueError(f"{self.action} cannot carry a BB size")
        if self.action in {"bet", "raise"} and self.to_amount_bb is None:
            raise ValueError(f"{self.action} requires an explicit to_amount_bb")

    def payload(self) -> dict[str, str]:
        row = {"action": self.action}
        if self.amount_bb is not None:
            row["amount_bb"] = self.amount_bb
        if self.to_amount_bb is not None:
            row["to_amount_bb"] = self.to_amount_bb
        return row


@dataclass(frozen=True)
class DecisionWindow:
    state_revision: int
    state_sha256: str
    observed_at_ms: int
    deadline_ms: int
    legal_actions: tuple[str, ...]
    safety_margin_ms: int = 750
    actuation_reserve_ms: int = 250
    max_state_age_ms: int = 1500

    def __post_init__(self) -> None:
        if self.state_revision < 0:
            raise ValueError("state_revision must be nonnegative")
        if len(self.state_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.state_sha256
        ):
            raise ValueError("state_sha256 must be a lowercase SHA-256 digest")
        if self.deadline_ms <= self.observed_at_ms:
            raise ValueError("deadline must be later than the observation")
        if not self.legal_actions or any(action not in ACTIONS for action in self.legal_actions):
            raise ValueError("legal_actions must contain supported actions")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal_actions cannot contain duplicates")
        if min(self.safety_margin_ms, self.actuation_reserve_ms, self.max_state_age_ms) < 0:
            raise ValueError("deadline margins must be nonnegative")

    def remaining_ms(self, now_ms: int) -> int:
        return self.deadline_ms - now_ms

    def compute_budget_ms(self, now_ms: int) -> int:
        return max(
            0,
            self.deadline_ms
            - now_ms
            - self.safety_margin_ms
            - self.actuation_reserve_ms,
        )

    def state_is_fresh(self, now_ms: int) -> bool:
        return 0 <= now_ms - self.observed_at_ms <= self.max_state_age_ms


def fallback_action(legal_actions: Iterable[str]) -> ActionCommand | None:
    legal = set(legal_actions)
    if "check" in legal:
        return ActionCommand("check")
    if "fold" in legal:
        return ActionCommand("fold")
    return None


def select_strategy_tier(
    available_tiers: Iterable[str], compute_budget_ms: int
) -> str | None:
    unknown = set(available_tiers) - set(TIER_REQUIREMENTS_MS)
    if unknown:
        raise ValueError(f"unsupported strategy tiers: {sorted(unknown)}")
    eligible = [
        tier
        for tier in set(available_tiers)
        if TIER_REQUIREMENTS_MS[tier] <= compute_budget_ms
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda tier: (TIER_QUALITY[tier], -TIER_REQUIREMENTS_MS[tier]))


def plan_deadline_decision(
    window: DecisionWindow,
    *,
    now_ms: int,
    available_tiers: Iterable[str],
) -> dict[str, object]:
    remaining = window.remaining_ms(now_ms)
    budget = window.compute_budget_ms(now_ms)
    if remaining <= window.safety_margin_ms:
        return {
            "status": "expired_or_unsafe",
            "remaining_ms": remaining,
            "compute_budget_ms": 0,
            "strategy_tier": None,
            "fallback": None,
        }
    tier = select_strategy_tier(available_tiers, budget)
    if tier is not None:
        return {
            "status": "compute",
            "remaining_ms": remaining,
            "compute_budget_ms": budget,
            "strategy_tier": tier,
            "fallback": fallback_action(window.legal_actions).payload()
            if fallback_action(window.legal_actions)
            else None,
        }
    fallback = fallback_action(window.legal_actions)
    return {
        "status": "fallback_required" if fallback else "abstain_no_safe_action",
        "remaining_ms": remaining,
        "compute_budget_ms": budget,
        "strategy_tier": None,
        "fallback": fallback.payload() if fallback else None,
    }


def decision_window_from_observation(
    state: dict[str, object],
    *,
    state_revision: int,
    state_sha256: str,
    observed_at_ms: int,
    safety_margin_ms: int = 750,
    actuation_reserve_ms: int = 250,
    max_state_age_ms: int = 1500,
) -> DecisionWindow:
    if state.get("hero_to_act") is not True:
        raise ValueError("a decision window requires explicit hero_to_act evidence")
    remaining = state.get("decision_time_remaining_ms")
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining <= 0:
        raise ValueError("decision_time_remaining_ms must be a positive integer")
    legal = state.get("legal_actions")
    if not isinstance(legal, list):
        raise ValueError("legal_actions must be an array")
    return DecisionWindow(
        state_revision=state_revision,
        state_sha256=state_sha256,
        observed_at_ms=observed_at_ms,
        deadline_ms=observed_at_ms + remaining,
        legal_actions=tuple(str(action) for action in legal),
        safety_margin_ms=safety_margin_ms,
        actuation_reserve_ms=actuation_reserve_ms,
        max_state_age_ms=max_state_age_ms,
    )


def authorize_action(
    window: DecisionWindow,
    command: ActionCommand,
    *,
    now_ms: int,
    current_state_revision: int,
    current_state_sha256: str,
    previously_authorized_tokens: Iterable[str] = (),
) -> dict[str, object]:
    reasons: list[str] = []
    if current_state_revision != window.state_revision:
        reasons.append("stale_state_revision")
    if current_state_sha256 != window.state_sha256:
        reasons.append("stale_state_fingerprint")
    if not window.state_is_fresh(now_ms):
        reasons.append("stale_observation")
    if command.action not in window.legal_actions:
        reasons.append("illegal_action")
    if now_ms >= window.deadline_ms - window.safety_margin_ms:
        reasons.append("deadline_safety_margin_reached")
    token_material = {
        "state_revision": window.state_revision,
        "state_sha256": window.state_sha256,
        "deadline_ms": window.deadline_ms,
        "command": command.payload(),
    }
    token = hashlib.sha256(
        json.dumps(token_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if token in set(previously_authorized_tokens):
        reasons.append("duplicate_action_token")
    return {
        "authorized": not reasons,
        "authorization_token": token,
        "command": command.payload(),
        "remaining_ms": window.remaining_ms(now_ms),
        "reasons": reasons,
    }
