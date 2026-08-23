from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from apc.neural.contract import ACTION_VOCABULARY


STATE_TOKEN_COUNT = 16
STATE_TOKEN_DIMENSION = 24
PROFILE_FEATURE_DIMENSION = 8
FEATURE_SCHEMA_VERSION = "3.0.0"
ACTION_INDEX = {action: index for index, action in enumerate(ACTION_VOCABULARY)}
RANK_INDEX = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
SUIT_INDEX = {suit: index for index, suit in enumerate("cdhs")}
_ACTION_RE = re.compile(r"^(?P<actor>\S+)\s+(?P<action>[a-z_]+)(?::(?P<amount>-?\d+(?:\.\d+)?))?$")


@lru_cache(maxsize=100_000)
def _poker_math_features(
    hero_cards: tuple[str, ...], board_cards: tuple[str, ...]
) -> tuple[float, float, float, float, float]:
    """Verified public/hero poker invariants; never uses opponent cards."""
    all_cards = (*hero_cards, *board_cards)
    category = primary_kicker = 0.0
    if len(all_cards) >= 5:
        from apc.full_hand_table import _coach_types

        best_hand_rank, _ = _coach_types()
        from poker_coach.models import Card

        rank = best_hand_rank(
            Card(card[0].upper(), card[1].lower()) for card in all_cards
        )
        category = rank.category / 8.0
        primary_kicker = (rank.kickers[0] / 14.0) if rank.kickers else 0.0
    board_ranks = [RANK_INDEX[card[0]] for card in board_cards]
    board_paired = 1.0 if len(board_ranks) != len(set(board_ranks)) else 0.0
    suit_counts = [sum(card[1] == suit for card in all_cards) for suit in "cdhs"]
    flush_progress = min(max(suit_counts, default=0), 5) / 5.0
    values = {RANK_INDEX[card[0]] for card in all_cards}
    if 14 in values:
        values.add(1)
    straight_windows = [set(range(start, start + 5)) for start in range(1, 11)]
    straight_progress = max((len(values & window) for window in straight_windows), default=0) / 5.0
    return category, primary_kicker, board_paired, flush_progress, straight_progress


def _bounded_bb(value: object, scale: float = 25.0) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return math.tanh(number / scale)


def generic_action(action_key: str) -> str:
    if action_key in ACTION_INDEX:
        return action_key
    if action_key.startswith("bet_"):
        return "bet"
    if action_key.startswith("raise_"):
        return "raise"
    raise ValueError(f"unsupported APC action key: {action_key}")


def _action_size(
    action: dict[str, object], state: dict[str, object]
) -> tuple[float, float]:
    amount = action.get("to_amount_bb", action.get("amount_bb", 0.0))
    try:
        amount_bb = float(str(amount))
        pot_bb = float(str(state.get("pot_bb", 0.0)))
    except (TypeError, ValueError):
        amount_bb, pot_bb = 0.0, 0.0
    fraction = amount_bb / pot_bb if pot_bb > 0 else 0.0
    return _bounded_bb(amount_bb), math.tanh(fraction / 2.0)


@dataclass(frozen=True)
class EncodedDecision:
    state_tokens: np.ndarray
    state_padding_mask: np.ndarray
    profile_features: np.ndarray
    modality_available: np.ndarray
    legal_action_mask: np.ndarray
    candidate_action_index: int
    candidate_size_features: np.ndarray
    target_return_bb: float
    split: str
    group_id: str
    policy_state_id: str
    action_key: str


def encode_state(state: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Encode only decision-visible public/hero information into APC tokens."""
    if state.get("opponent_cards") is not None:
        raise ValueError("APC decision features cannot include opponent private cards")
    if state.get("units") != "BB":
        raise ValueError("APC state features require BB units")

    tokens = np.zeros((STATE_TOKEN_COUNT, STATE_TOKEN_DIMENSION), dtype=np.float32)
    padding = np.ones(STATE_TOKEN_COUNT, dtype=np.bool_)

    # Global table token.
    global_token = tokens[0]
    global_token[0] = 1.0
    global_token[4] = _bounded_bb(state.get("pot_bb"))
    global_token[5] = _bounded_bb(state.get("to_call_bb"))
    stacks = state.get("stacks_bb", {})
    contributions = state.get("street_contributions_bb", {})
    if isinstance(stacks, dict):
        global_token[6] = _bounded_bb(stacks.get("Hero"), 100.0)
        global_token[7] = _bounded_bb(stacks.get("Villain"), 100.0)
    if isinstance(contributions, dict):
        global_token[8] = _bounded_bb(contributions.get("Hero"))
        global_token[9] = _bounded_bb(contributions.get("Villain"))
    global_token[10] = 1.0 if state.get("hero_position") == "BTN" else -1.0
    global_token[11] = {"preflop": 0.0, "flop": 1 / 3, "turn": 2 / 3, "river": 1.0}.get(
        str(state.get("street")), 0.0
    )
    global_token[12] = min(len(state.get("board", [])) / 5.0, 1.0)
    buttons = state.get("action_buttons", [])
    if isinstance(buttons, list):
        for button in buttons:
            if not isinstance(button, dict):
                continue
            try:
                index = ACTION_INDEX[generic_action(str(button.get("action")))]
            except ValueError:
                continue
            global_token[13 + index] = 1.0
    hero_cards = tuple(str(card) for card in state.get("hero_cards", []))
    board_cards = tuple(str(card) for card in state.get("board", []))
    padding[0] = False

    cards = list(hero_cards) + list(board_cards)
    # Canonical first-seen suit IDs make strategically equivalent suit renamings
    # identical before they reach the network.
    canonical_suits: dict[str, int] = {}
    for offset, card in enumerate(cards[:7], start=1):
        if not isinstance(card, str) or len(card) != 2:
            raise ValueError("APC card token is invalid")
        rank, suit = card[0], card[1]
        if rank not in RANK_INDEX or suit not in SUIT_INDEX:
            raise ValueError("APC card token is invalid")
        token = tokens[offset]
        token[1] = 1.0
        token[4] = RANK_INDEX[rank] / 14.0
        if suit not in canonical_suits:
            canonical_suits[suit] = len(canonical_suits)
        token[5 + canonical_suits[suit]] = 1.0
        token[9] = 1.0 if offset <= 2 else 0.0
        token[10] = max(0, offset - 2) / 5.0
        padding[offset] = False
    global_token[19:24] = _poker_math_features(hero_cards, board_cards)

    history = state.get("action_history", [])
    history_rows = history[-8:] if isinstance(history, list) else []
    for offset, raw in enumerate(history_rows, start=8):
        if offset >= STATE_TOKEN_COUNT or not isinstance(raw, str):
            break
        match = _ACTION_RE.fullmatch(raw)
        if match is None:
            continue
        token = tokens[offset]
        token[2] = 1.0
        token[4] = 1.0 if match.group("actor") in {"Hero", str(state.get("hero_position"))} else -1.0
        raw_action = match.group("action")
        action = "raise" if raw_action == "raise_to" else raw_action
        if action in ACTION_INDEX:
            token[5 + ACTION_INDEX[action]] = 1.0
        token[11] = _bounded_bb(match.group("amount"))
        token[12] = offset / (STATE_TOKEN_COUNT - 1)
        padding[offset] = False
    return tokens, padding


def encode_raised_row(row: dict[str, object]) -> EncodedDecision:
    state = row.get("state")
    if not isinstance(state, dict):
        raise ValueError("APC training row lacks canonical state")
    tokens, padding = encode_state(state)
    action_key = str(row.get("counterfactual_action_key", ""))
    action = generic_action(action_key)
    action_payload = row.get("counterfactual_action")
    if not isinstance(action_payload, dict):
        raise ValueError("APC training row lacks counterfactual action")
    legal = np.zeros(len(ACTION_VOCABULARY), dtype=np.bool_)
    for button in state.get("action_buttons", []):
        if isinstance(button, dict):
            try:
                legal[ACTION_INDEX[generic_action(str(button.get("action")))]] = True
            except ValueError:
                pass
    action_index = ACTION_INDEX[action]
    if not legal[action_index]:
        raise ValueError("APC target action is not visibly legal")
    signal = row.get("learning_signal", {})
    target = float(str(signal.get("hero_return_bb"))) if isinstance(signal, dict) else math.nan
    if not math.isfinite(target):
        raise ValueError("APC target return is not finite")
    policy = str(row.get("opponent_policy", ""))
    policy_index = {
        "check_call": 0,
        "fold_to_pressure": 1,
        "made_hand_selective": 2,
    }.get(policy)
    if policy_index is None:
        raise ValueError("APC training row has unsupported opponent policy")
    profile = np.zeros(PROFILE_FEATURE_DIMENSION, dtype=np.float32)
    profile[policy_index] = 1.0
    profile[3] = 1.0  # Controlled policy is fully evidenced in this simulator corpus.
    profile[4] = 0.0  # Posterior uncertainty for the declared controlled policy.
    return EncodedDecision(
        state_tokens=tokens,
        state_padding_mask=padding,
        profile_features=profile,
        modality_available=np.asarray([False, True, True], dtype=np.bool_),
        legal_action_mask=legal,
        candidate_action_index=action_index,
        candidate_size_features=np.asarray(_action_size(action_payload, state), dtype=np.float32),
        target_return_bb=target,
        split=str(row.get("split", "")),
        group_id=str(row.get("group_id", "")),
        policy_state_id=str(row.get("policy_state_id", "")),
        action_key=action_key,
    )
