from __future__ import annotations

import argparse
import json
import time
from decimal import Decimal
from pathlib import Path

from apc.deadline import ActionCommand
from apc.full_hand_table import HeadsUpVirtualHand
from apc.perception.baseline import _percentile


POLICIES = ("fold", "check_call", "raise_once", "street_bet", "all_in")


def _command(policy: str, hand: HeadsUpVirtualHand) -> ActionCommand:
    buttons = {row["action"]: row for row in hand.legal_action_buttons()}
    if policy == "fold" and "fold" in buttons:
        return ActionCommand("fold")
    if policy == "all_in" and "all_in" in buttons:
        return ActionCommand("all_in")
    if policy == "raise_once" and not hand.history and "raise" in buttons:
        return ActionCommand("raise", to_amount_bb=buttons["raise"]["minimum_to_bb"])
    if policy == "street_bet" and "bet" in buttons:
        return ActionCommand("bet", to_amount_bb=buttons["bet"]["minimum_to_bb"])
    if "check" in buttons:
        return ActionCommand("check")
    if "call" in buttons:
        return ActionCommand("call")
    raise RuntimeError("policy found no legal action")


def _play(seed: int, button: int, policy: str) -> tuple[dict[str, object], list[str], list[float]]:
    hand = HeadsUpVirtualHand(seed=seed, button_player=button)
    fingerprints: list[str] = []
    latencies: list[float] = []
    action_kinds: set[str] = set()
    streets = {hand.street}
    initial = hand.observation()
    private_cards_hidden = initial["opponent_cards"] is None
    conservation_passed = True
    external_actuation_violations = 0
    actions = 0
    final: dict[str, object] | None = None
    while not hand.terminal:
        observation = hand.observation()
        stacks = sum((Decimal(value) for value in observation["stacks_bb"].values()), Decimal("0"))
        conservation_passed &= stacks + Decimal(observation["pot_bb"]) == Decimal("200")
        external_actuation_violations += int(
            observation["provider"]["external_actuation"] is not False
            or observation["provider"]["screen_or_input_control"] is not False
        )
        command = _command(policy, hand)
        started = time.perf_counter()
        final = hand.step(command)
        latencies.append((time.perf_counter() - started) * 1000.0)
        fingerprints.append(final["transition_fingerprint"])
        action_kinds.add(command.action)
        streets.add(final["state"]["street"])
        actions += 1
        if actions > 256:
            raise RuntimeError("hand exceeded the bounded action count")
    assert final is not None
    feedback = final["completed_hand_feedback"]
    rewards = sum((Decimal(value) for value in feedback["rewards_bb"].values()), Decimal("0"))
    cards = [str(card) for pair in hand.hole_cards for card in pair] + [str(card) for card in hand.runout]
    final_stacks = sum(
        (Decimal(value) for value in feedback["final_stacks_bb"].values()), Decimal("0")
    )
    row = {
        "seed": seed,
        "button_player": button,
        "policy": policy,
        "actions": actions,
        "action_kinds": sorted(action_kinds),
        "streets": sorted(streets),
        "terminal_reason": feedback["terminal_reason"],
        "final_pot_bb": feedback["final_pot_bb"],
        "private_cards_hidden_before_showdown": private_cards_hidden,
        "unique_dealt_cards": len(cards) == len(set(cards)) == 9,
        "chip_conservation_passed": conservation_passed and final_stacks == Decimal("200"),
        "zero_sum_rewards": rewards == 0,
        "external_actuation_violations": external_actuation_violations,
        "terminal_fingerprint": fingerprints[-1],
    }
    return row, fingerprints, latencies


def evaluate_full_hand_table(*, hands: int = 100, seed_start: int = 1000) -> dict[str, object]:
    if hands < 10:
        raise ValueError("hands must be at least 10")
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    replay_mismatches = 0
    for offset in range(hands):
        seed = seed_start + offset
        policy = POLICIES[offset % len(POLICIES)]
        row, trace, timings = _play(seed, offset % 2, policy)
        replay_row, replay_trace, _ = _play(seed, offset % 2, policy)
        if row["terminal_fingerprint"] != replay_row["terminal_fingerprint"] or trace != replay_trace:
            replay_mismatches += 1
        rows.append(row)
        latencies.extend(timings)
    action_kinds = sorted({action for row in rows for action in row["action_kinds"]})
    streets = sorted({street for row in rows for street in row["streets"]})
    passed = (
        replay_mismatches == 0
        and all(
            row["unique_dealt_cards"]
            and row["chip_conservation_passed"]
            and row["zero_sum_rewards"]
            and row["private_cards_hidden_before_showdown"]
            and row["external_actuation_violations"] == 0
            for row in rows
        )
        and set(action_kinds) == {"all_in", "bet", "call", "check", "fold", "raise"}
        and set(streets) == {"flop", "preflop", "river", "turn"}
    )
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "controlled_virtual_chip_full_hand_provider",
        "passed": passed,
        "promotion_eligible": False,
        "units": "BB",
        "metrics": {
            "hands": len(rows),
            "showdowns": sum(row["terminal_reason"] == "showdown" for row in rows),
            "folds": sum(row["terminal_reason"] == "fold" for row in rows),
            "actions": sum(row["actions"] for row in rows),
            "max_actions_per_hand": max(row["actions"] for row in rows),
            "action_kinds_covered": action_kinds,
            "streets_covered": streets,
            "replay_mismatches": replay_mismatches,
            "conservation_failures": sum(not row["chip_conservation_passed"] for row in rows),
            "card_uniqueness_failures": sum(not row["unique_dealt_cards"] for row in rows),
            "zero_sum_failures": sum(not row["zero_sum_rewards"] for row in rows),
            "external_actuation_violations": sum(row["external_actuation_violations"] for row in rows),
            "step_latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "rows": rows,
        "limitations": [
            "The environment is heads-up no-limit Hold'em with equal starting stacks and no rake.",
            "Multiway pots, side pots, antes, rake and tournament blind schedules are not implemented.",
            "Audit policies are deterministic coverage probes, not trained or promoted APC policies.",
            "This internal provider has no screen, mouse, keyboard or external table integration.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit APC's controlled full-hand provider.")
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_full_hand_table(hands=args.hands, seed_start=args.seed_start)
        if args.output:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 3
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
