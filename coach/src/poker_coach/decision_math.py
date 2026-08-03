from __future__ import annotations

from decimal import Decimal
from typing import Iterable


D = Decimal


def _positive(value: Decimal | int | str, name: str) -> Decimal:
    result = D(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: Decimal | int | str, name: str) -> Decimal:
    result = D(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def call_break_even_equity(pot_before_bet: Decimal | int | str, bet: Decimal | int | str) -> Decimal:
    """Equity required to call B into pot P, ignoring future action, rake and ties."""
    pot = _nonnegative(pot_before_bet, "pot_before_bet")
    wager = _positive(bet, "bet")
    return wager / (pot + 2 * wager)


def call_ev(equity: Decimal | int | str, pot_before_bet: Decimal | int | str, bet: Decimal | int | str) -> Decimal:
    q = D(equity)
    if not D(0) <= q <= D(1):
        raise ValueError("equity must be between zero and one")
    pot = _nonnegative(pot_before_bet, "pot_before_bet")
    wager = _positive(bet, "bet")
    return q * (pot + 2 * wager) - wager


def bluff_break_even_fold_frequency(pot_before_bet: Decimal | int | str, bet: Decimal | int | str) -> Decimal:
    pot = _positive(pot_before_bet, "pot_before_bet")
    wager = _positive(bet, "bet")
    return wager / (pot + wager)


def pure_bluff_ev(fold_probability: Decimal | int | str, pot_before_bet: Decimal | int | str, bet: Decimal | int | str) -> Decimal:
    fold = D(fold_probability)
    if not D(0) <= fold <= D(1):
        raise ValueError("fold_probability must be between zero and one")
    pot = _positive(pot_before_bet, "pot_before_bet")
    wager = _positive(bet, "bet")
    return fold * pot - (D(1) - fold) * wager


def minimum_defense_frequency(pot_before_bet: Decimal | int | str, bet: Decimal | int | str) -> Decimal:
    pot = _positive(pot_before_bet, "pot_before_bet")
    wager = _positive(bet, "bet")
    return pot / (pot + wager)


def polar_bluff_to_value_ratio(pot_before_bet: Decimal | int | str, bet: Decimal | int | str) -> Decimal:
    return bluff_break_even_fold_frequency(pot_before_bet, bet)


def polar_bluff_share(pot_before_bet: Decimal | int | str, bet: Decimal | int | str) -> Decimal:
    return call_break_even_equity(pot_before_bet, bet)


def hit_probability_one_card(outs: int, unseen_cards: int) -> Decimal:
    if not 0 <= outs <= unseen_cards or unseen_cards <= 0:
        raise ValueError("outs must be between zero and unseen_cards")
    return D(outs) / D(unseen_cards)


def hit_probability_by_river(outs: int, unseen_on_flop: int = 47) -> Decimal:
    if not 0 <= outs <= unseen_on_flop - 1 or unseen_on_flop < 2:
        raise ValueError("invalid outs/unseen card count")
    miss_turn = D(unseen_on_flop - outs) / D(unseen_on_flop)
    miss_river = D(unseen_on_flop - 1 - outs) / D(unseen_on_flop - 1)
    return D(1) - miss_turn * miss_river


def stack_to_pot_ratio(effective_stack: Decimal | int | str, pot: Decimal | int | str) -> Decimal:
    stack = _nonnegative(effective_stack, "effective_stack")
    current_pot = _positive(pot, "pot")
    return stack / current_pot


def expected_value(outcomes: Iterable[tuple[Decimal | int | str, Decimal | int | str]]) -> Decimal:
    rows = [(D(probability), D(utility)) for probability, utility in outcomes]
    if not rows:
        raise ValueError("at least one outcome is required")
    if any(probability < 0 for probability, _ in rows):
        raise ValueError("probabilities cannot be negative")
    total_probability = sum((probability for probability, _ in rows), D(0))
    if abs(total_probability - D(1)) > D("0.000000001"):
        raise ValueError("outcome probabilities must sum to one")
    return sum((probability * utility for probability, utility in rows), D(0))

