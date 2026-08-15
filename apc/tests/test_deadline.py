from __future__ import annotations

import unittest

from apc.deadline import (
    ActionCommand,
    DecisionWindow,
    authorize_action,
    decision_window_from_observation,
    fallback_action,
    plan_deadline_decision,
    select_strategy_tier,
)


STATE_SHA = "a" * 64


def window(**overrides: object) -> DecisionWindow:
    values: dict[str, object] = {
        "state_revision": 7,
        "state_sha256": STATE_SHA,
        "observed_at_ms": 10_000,
        "deadline_ms": 40_000,
        "legal_actions": ("fold", "call", "raise"),
    }
    values.update(overrides)
    return DecisionWindow(**values)  # type: ignore[arg-type]


class DeadlineTests(unittest.TestCase):
    def test_long_budget_selects_bounded_refinement(self) -> None:
        plan = plan_deadline_decision(
            window(),
            now_ms=10_100,
            available_tiers={"fast_policy", "cached_exact_solver", "bounded_refinement"},
        )
        self.assertEqual(plan["status"], "compute")
        self.assertEqual(plan["strategy_tier"], "bounded_refinement")

    def test_short_budget_uses_fastest_available_tier(self) -> None:
        self.assertEqual(
            select_strategy_tier({"cached_blueprint", "fast_policy"}, 20),
            "fast_policy",
        )

    def test_no_compute_budget_uses_check_then_fold_fallback(self) -> None:
        check_window = window(
            deadline_ms=11_050,
            legal_actions=("check", "bet"),
        )
        check_plan = plan_deadline_decision(
            check_window, now_ms=10_100, available_tiers={"fast_policy"}
        )
        self.assertEqual(check_plan["status"], "fallback_required")
        self.assertEqual(check_plan["fallback"], {"action": "check"})
        self.assertEqual(fallback_action(("fold", "call")).action, "fold")

    def test_action_requires_current_fresh_legal_state(self) -> None:
        result = authorize_action(
            window(),
            ActionCommand("call", amount_bb="2.5"),
            now_ms=10_200,
            current_state_revision=7,
            current_state_sha256=STATE_SHA,
        )
        self.assertTrue(result["authorized"], result["reasons"])

    def test_stale_or_illegal_action_is_rejected(self) -> None:
        result = authorize_action(
            window(max_state_age_ms=100),
            ActionCommand("check"),
            now_ms=10_200,
            current_state_revision=8,
            current_state_sha256="b" * 64,
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(
            set(result["reasons"]),
            {
                "stale_state_revision",
                "stale_state_fingerprint",
                "stale_observation",
                "illegal_action",
            },
        )

    def test_deadline_margin_blocks_late_action(self) -> None:
        result = authorize_action(
            window(),
            ActionCommand("fold"),
            now_ms=39_250,
            current_state_revision=7,
            current_state_sha256=STATE_SHA,
        )
        self.assertFalse(result["authorized"])
        self.assertIn("deadline_safety_margin_reached", result["reasons"])

    def test_duplicate_authorization_token_is_rejected(self) -> None:
        first = authorize_action(
            window(),
            ActionCommand("fold"),
            now_ms=10_100,
            current_state_revision=7,
            current_state_sha256=STATE_SHA,
        )
        second = authorize_action(
            window(),
            ActionCommand("fold"),
            now_ms=10_100,
            current_state_revision=7,
            current_state_sha256=STATE_SHA,
            previously_authorized_tokens=[first["authorization_token"]],
        )
        self.assertFalse(second["authorized"])
        self.assertIn("duplicate_action_token", second["reasons"])

    def test_raise_requires_unambiguous_raise_to_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "to_amount_bb"):
            ActionCommand("raise", amount_bb="2.5")

    def test_visible_timer_observation_builds_actual_deadline(self) -> None:
        observed = decision_window_from_observation(
            {
                "hero_to_act": True,
                "decision_time_remaining_ms": 12_500,
                "legal_actions": ["check", "bet"],
            },
            state_revision=3,
            state_sha256=STATE_SHA,
            observed_at_ms=50_000,
        )
        self.assertEqual(observed.deadline_ms, 62_500)
        self.assertEqual(observed.legal_actions, ("check", "bet"))


if __name__ == "__main__":
    unittest.main()
