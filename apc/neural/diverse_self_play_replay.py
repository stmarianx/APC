from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from apc.deadline import ActionCommand
from apc.full_hand_table import HeadsUpVirtualHand
from apc.neural.contract import ACTION_VOCABULARY
from apc.neural.replay_adapter import load_replay_temporal_corpus
from apc.neural.replay_buffer import APCReplayBuffer, split_for_group
from apc.neural.self_play_replay import _private_safe_state


POLICY_FAMILIES = ("scripted_mixed", "passive", "pressure", "selective")
TRAIN_OPPONENT_POLICIES = ("scripted_mixed", "passive", "pressure")
HELD_OUT_OPPONENT_POLICY = "selective"
DEFAULT_STACK_DEPTHS_BB = ("40", "100", "200")


def _source_fingerprint(configuration: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).resolve().parents[1] / "full_hand_table.py"):
        digest.update(source.read_bytes())
    digest.update(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _rank_value(card: object) -> int:
    return "23456789TJQKA".index(str(card)[0].upper()) + 2


def _selective_strength(hand: HeadsUpVirtualHand, player: int) -> bool:
    hole = hand.hole_cards[player]
    ranks = [_rank_value(card) for card in hole]
    if hand.street == "preflop":
        return ranks[0] == ranks[1] or max(ranks) >= 12
    visible_ranks = ranks + [_rank_value(card) for card in hand.board]
    paired = len(visible_ranks) != len(set(visible_ranks))
    return paired or max(ranks) >= 13


def select_policy_action(hand: HeadsUpVirtualHand, policy_family: str) -> ActionCommand:
    if policy_family not in POLICY_FAMILIES:
        raise ValueError("APC self-play policy family is unsupported")
    buttons = {row["action"]: row for row in hand.legal_action_buttons()}
    token = hand.seed * 17 + hand.revision * 31 + (hand.next_actor or 0) * 7
    if policy_family == "passive":
        if "check" in buttons:
            return ActionCommand("check")
        if token % 23 == 0:
            return ActionCommand("fold")
        return ActionCommand("call", amount_bb=buttons["call"]["amount_bb"])
    if policy_family == "pressure":
        if "bet" in buttons:
            return ActionCommand("bet", to_amount_bb=buttons["bet"]["minimum_to_bb"])
        if "raise" in buttons and token % 3:
            return ActionCommand("raise", to_amount_bb=buttons["raise"]["minimum_to_bb"])
        if "check" in buttons:
            return ActionCommand("check")
        return ActionCommand("call", amount_bb=buttons["call"]["amount_bb"])
    if policy_family == "selective":
        strong = _selective_strength(hand, int(hand.next_actor or 0))
        if strong and "bet" in buttons:
            return ActionCommand("bet", to_amount_bb=buttons["bet"]["minimum_to_bb"])
        if strong and "raise" in buttons:
            return ActionCommand("raise", to_amount_bb=buttons["raise"]["minimum_to_bb"])
        if "check" in buttons:
            return ActionCommand("check")
        if not strong and token % 3:
            return ActionCommand("fold")
        return ActionCommand("call", amount_bb=buttons["call"]["amount_bb"])
    if "check" in buttons:
        if "bet" in buttons and token % 9 == 0:
            return ActionCommand("bet", to_amount_bb=buttons["bet"]["minimum_to_bb"])
        return ActionCommand("check")
    if token % 17 == 0:
        return ActionCommand("fold")
    if "raise" in buttons and token % 13 == 0:
        return ActionCommand("raise", to_amount_bb=buttons["raise"]["minimum_to_bb"])
    return ActionCommand("call", amount_bb=buttons["call"]["amount_bb"])


def observed_profile_features(opponent_actions: list[str]) -> list[float]:
    if any(action not in ACTION_VOCABULARY for action in opponent_actions):
        raise ValueError("APC observed profile received an unsupported action")
    counts = Counter(opponent_actions)
    observations = len(opponent_actions)
    denominator = observations + 2
    continued = sum(counts[action] for action in ("call", "bet", "raise", "all_in"))
    aggressive = sum(counts[action] for action in ("bet", "raise", "all_in"))
    posterior = [(counts[action] + 1) / (observations + len(ACTION_VOCABULARY)) for action in ACTION_VOCABULARY]
    entropy = -sum(probability * math.log(probability) for probability in posterior) / math.log(len(ACTION_VOCABULARY))
    return [
        (continued + 1) / denominator,
        (counts["raise"] + counts["all_in"] + 1) / denominator,
        (aggressive + 1) / denominator,
        (counts["fold"] + 1) / denominator,
        (counts["check"] + 1) / denominator,
        min(observations / 32.0, 1.0),
        1.0 / math.sqrt(observations + 1),
        entropy,
    ]


def generate_diverse_virtual_replay(
    seed: int,
    *,
    session_id: str,
    hero_policy: str,
    opponent_policy: str,
    starting_stack_bb: str,
) -> dict[str, object]:
    if hero_policy not in POLICY_FAMILIES or opponent_policy not in POLICY_FAMILIES:
        raise ValueError("APC diverse self-play policy is unsupported")
    configuration = {
        "generator": "diverse_self_play_replay_v1",
        "hero_policy": hero_policy,
        "opponent_policy": opponent_policy,
        "starting_stack_bb": str(starting_stack_bb),
    }
    hand = HeadsUpVirtualHand(seed=seed, button_player=seed % 2, starting_stack_bb=starting_stack_bb)
    events = []
    opponent_actions: list[str] = []
    final = None
    while not hand.terminal:
        observation = hand.observation()
        actor = int(hand.next_actor or 0)
        family = hero_policy if actor == 0 else opponent_policy
        command = select_policy_action(hand, family)
        if actor == 1 and not events and command.action == "fold":
            visible = {row["action"]: row for row in hand.legal_action_buttons()}
            command = ActionCommand("call", amount_bb=visible["call"]["amount_bb"])
        if actor == 0:
            events.append({
                "observed_monotonic_ms": 1000 + hand.revision * 100,
                "state_fingerprint": observation["state_fingerprint"],
                "legal_action_keys": [str(row["action"]) for row in observation["action_buttons"]],
                "chosen_action_key": command.action,
                "chosen_action": command.payload(),
                "player_profile_features": observed_profile_features(opponent_actions),
                "canonical_state": _private_safe_state(observation),
            })
        final = hand.step(command)
        if actor == 1:
            opponent_actions.append(command.action)
    if not events or final is None:
        raise RuntimeError("APC diverse virtual hand produced no Hero decisions")
    feedback = final["completed_hand_feedback"]
    return {
        "schema_version": "1.0.0", "model_name": "APC", "units": "BB",
        "source_environment": "controlled_virtual_chips",
        "session_id": session_id, "hand_id": hand.hand_id, "split_group_id": session_id,
        "source_fingerprint": _source_fingerprint(configuration),
        "full_hand_completed": True, "events": events,
        "completed_hand_feedback": {"full_hand_completed": True, "hero_reward_bb": feedback["rewards_bb"]["Hero"]},
        "self_play_configuration": configuration,
        "external_actuation": False,
    }


def build_diverse_virtual_replay_buffer(
    destination: str | Path,
    *,
    hands: int,
    seed_start: int,
    hands_per_session: int = 3,
    stack_depths_bb: tuple[str, ...] = DEFAULT_STACK_DEPTHS_BB,
) -> dict[str, object]:
    if hands < 48 or hands_per_session <= 0 or not stack_depths_bb:
        raise ValueError("APC diverse replay build parameters are invalid")
    buffer = APCReplayBuffer(destination)
    configuration_counts: Counter[str] = Counter()
    split_configuration_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for offset, seed in enumerate(range(seed_start, seed_start + hands)):
        session_number = offset // hands_per_session
        session_id = f"diverse-self-play-{seed_start}-{session_number:06d}"
        split = split_for_group(session_id)
        hero_policy = POLICY_FAMILIES[session_number % len(POLICY_FAMILIES)]
        opponent_policy = (
            HELD_OUT_OPPONENT_POLICY
            if split != "train"
            else TRAIN_OPPONENT_POLICIES[(session_number // len(POLICY_FAMILIES)) % len(TRAIN_OPPONENT_POLICIES)]
        )
        stack = str(stack_depths_bb[(session_number // (len(POLICY_FAMILIES) * len(TRAIN_OPPONENT_POLICIES))) % len(stack_depths_bb)])
        replay = generate_diverse_virtual_replay(
            seed,
            session_id=session_id,
            hero_policy=hero_policy,
            opponent_policy=opponent_policy,
            starting_stack_bb=stack,
        )
        buffer.ingest(replay)
        key = f"hero={hero_policy}|opponent={opponent_policy}|stack={stack}"
        configuration_counts[key] += 1
        split_configuration_counts[split][key] += 1
    validation = buffer.validate()
    if not validation["valid"]:
        raise RuntimeError("APC diverse replay buffer failed validation")
    corpus = load_replay_temporal_corpus(buffer)
    split_hands = {
        split: len({corpus.replay_fingerprints[int(index)] for index in corpus.indices(split)})
        for split in ("train", "validation", "test")
    }
    profile_available = int(corpus.modality_available[:, 2].sum())
    observed_train = set(split_configuration_counts["train"])
    required_train = {
        f"hero={hero}|opponent={opponent}|stack={stack}"
        for hero in POLICY_FAMILIES
        for opponent in TRAIN_OPPONENT_POLICIES
        for stack in stack_depths_bb
    }
    observed_held_out = set(split_configuration_counts["validation"]) | set(split_configuration_counts["test"])
    required_held_out = {
        f"hero={hero}|opponent={HELD_OUT_OPPONENT_POLICY}|stack={stack}"
        for hero in POLICY_FAMILIES
        for stack in stack_depths_bb
    }
    passed = (
        all(split_hands.values())
        and profile_available == int(corpus.manifest["decisions"])
        and required_train <= observed_train
        and required_held_out <= observed_held_out
    )
    return {
        "schema_version": "1.0.0", "model_name": "APC",
        "status": "training_eligible" if passed else "diverse_replay_gate_failed",
        "hands": hands, "seed_start": seed_start, "seed_end": seed_start + hands - 1,
        "hands_per_session": hands_per_session,
        "stack_depths_bb": list(stack_depths_bb),
        "training_opponent_policies": list(TRAIN_OPPONENT_POLICIES),
        "held_out_opponent_policy": HELD_OUT_OPPONENT_POLICY,
        "split_complete_hands": split_hands,
        "decisions": corpus.manifest["decisions"],
        "profile_conditioned_decisions": profile_available,
        "required_training_configurations": len(required_train),
        "covered_training_configurations": len(required_train & observed_train),
        "required_held_out_configurations": len(required_held_out),
        "covered_held_out_configurations": len(required_held_out & observed_held_out),
        "configuration_counts": dict(sorted(configuration_counts.items())),
        "split_configuration_counts": {split: dict(sorted(counts.items())) for split, counts in sorted(split_configuration_counts.items())},
        "adapter_fingerprint": corpus.manifest["adapter_fingerprint"],
        "buffer_validation": validation,
        "training_eligible": passed,
        "external_actuation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diverse APC virtual-chip completed replay")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--hands", type=int, default=480)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--hands-per-session", type=int, default=3)
    parser.add_argument("--stack-depth-bb", action="append", dest="stacks")
    args = parser.parse_args()
    stacks = DEFAULT_STACK_DEPTHS_BB if not args.stacks else tuple(args.stacks)
    print(json.dumps(build_diverse_virtual_replay_buffer(args.destination, hands=args.hands, seed_start=args.seed_start, hands_per_session=args.hands_per_session, stack_depths_bb=stacks), indent=2))


if __name__ == "__main__":
    main()
