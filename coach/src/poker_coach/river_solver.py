from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class HandBucket:
    name: str
    weight: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Hand bucket name cannot be empty")
        if not isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Hand bucket weight must be positive and finite")


@dataclass(frozen=True)
class RiverSubgame:
    pot_bb: float
    bet_bb: float
    ip_buckets: tuple[HandBucket, ...]
    oop_buckets: tuple[HandBucket, ...]
    showdown_equity: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isfinite(self.pot_bb) or self.pot_bb <= 0:
            raise ValueError("Pot must be positive and finite")
        if not isfinite(self.bet_bb) or self.bet_bb <= 0:
            raise ValueError("Bet must be positive and finite")
        if not self.ip_buckets or not self.oop_buckets:
            raise ValueError("Both players require at least one hand bucket")
        if len({bucket.name for bucket in self.ip_buckets}) != len(self.ip_buckets):
            raise ValueError("IP bucket names must be unique")
        if len({bucket.name for bucket in self.oop_buckets}) != len(self.oop_buckets):
            raise ValueError("OOP bucket names must be unique")
        if len(self.showdown_equity) != len(self.ip_buckets):
            raise ValueError("Equity matrix row count must match IP buckets")
        for row in self.showdown_equity:
            if len(row) != len(self.oop_buckets):
                raise ValueError("Equity matrix column count must match OOP buckets")
            if any(not isfinite(equity) or not 0 <= equity <= 1 for equity in row):
                raise ValueError("Showdown equity must be finite and between zero and one")

    @property
    def pair_probabilities(self) -> tuple[tuple[float, ...], ...]:
        total = sum(bucket.weight for bucket in self.ip_buckets) * sum(
            bucket.weight for bucket in self.oop_buckets
        )
        return tuple(
            tuple(ip.weight * oop.weight / total for oop in self.oop_buckets)
            for ip in self.ip_buckets
        )

    def terminal_utilities(self, ip_index: int, oop_index: int) -> tuple[float, float, float]:
        """Return IP utility for check, bet/fold, and bet/call terminal states."""
        equity = self.showdown_equity[ip_index][oop_index]
        check = (2 * equity - 1) * self.pot_bb / 2
        fold = self.pot_bb / 2
        call = (2 * equity - 1) * (self.pot_bb / 2 + self.bet_bb)
        return check, fold, call


@dataclass(frozen=True)
class RiverSolution:
    iterations: int
    ip_strategy: dict[str, dict[str, float]]
    oop_strategy: dict[str, dict[str, float]]
    expected_ip_ev_bb: float
    ip_best_response_ev_bb: float
    oop_best_response_ip_ev_bb: float
    nash_gap_bb: float
    exploitability_bb: float

    def to_dict(self) -> dict[str, object]:
        return {
            "solver": "vanilla_cfr",
            "iterations": self.iterations,
            "strategy": {"ip": self.ip_strategy, "oop": self.oop_strategy},
            "expected_ip_ev_bb": self.expected_ip_ev_bb,
            "ip_best_response_ev_bb": self.ip_best_response_ev_bb,
            "oop_best_response_ip_ev_bb": self.oop_best_response_ip_ev_bb,
            "nash_gap_bb": self.nash_gap_bb,
            "exploitability_bb": self.exploitability_bb,
        }


def _regret_strategy(regrets: list[float]) -> list[float]:
    positive = [max(0.0, regret) for regret in regrets]
    total = sum(positive)
    return [value / total for value in positive] if total > 0 else [0.5, 0.5]


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values] if total > 0 else [0.5, 0.5]


class RiverCFRSolver:
    """Solve a one-bet heads-up river abstraction using full-tree vanilla CFR.

    IP chooses check/bet by private hand bucket. After a bet, OOP chooses
    fold/call by private hand bucket. Utilities are zero-sum chip EV measured
    from the decision point; chance uses independent normalized bucket weights.
    """

    def solve(self, game: RiverSubgame, *, iterations: int = 50_000) -> RiverSolution:
        if iterations < 100 or iterations > 1_000_000:
            raise ValueError("iterations must be between 100 and 1,000,000")
        ip_regrets = [[0.0, 0.0] for _ in game.ip_buckets]
        oop_regrets = [[0.0, 0.0] for _ in game.oop_buckets]
        ip_sums = [[0.0, 0.0] for _ in game.ip_buckets]
        oop_sums = [[0.0, 0.0] for _ in game.oop_buckets]
        pair_probs = game.pair_probabilities
        ip_marginals = [sum(row) for row in pair_probs]
        oop_marginals = [sum(pair_probs[i][j] for i in range(len(game.ip_buckets))) for j in range(len(game.oop_buckets))]

        for _ in range(iterations):
            ip_current = [_regret_strategy(regrets) for regrets in ip_regrets]
            oop_current = [_regret_strategy(regrets) for regrets in oop_regrets]
            ip_updates = [[0.0, 0.0] for _ in game.ip_buckets]
            oop_updates = [[0.0, 0.0] for _ in game.oop_buckets]

            for i, _ip in enumerate(game.ip_buckets):
                for j, _oop in enumerate(game.oop_buckets):
                    chance = pair_probs[i][j]
                    check_u, fold_u, call_u = game.terminal_utilities(i, j)
                    ip_strategy = ip_current[i]
                    oop_strategy = oop_current[j]
                    bet_u = oop_strategy[0] * fold_u + oop_strategy[1] * call_u
                    node_u = ip_strategy[0] * check_u + ip_strategy[1] * bet_u
                    ip_updates[i][0] += chance * (check_u - node_u)
                    ip_updates[i][1] += chance * (bet_u - node_u)
                    # OOP maximizes negative IP utility; counterfactual reach includes IP betting.
                    oop_updates[j][0] += chance * ip_strategy[1] * (bet_u - fold_u)
                    oop_updates[j][1] += chance * ip_strategy[1] * (bet_u - call_u)

            for i in range(len(game.ip_buckets)):
                for action in range(2):
                    ip_regrets[i][action] += ip_updates[i][action]
                    ip_sums[i][action] += ip_marginals[i] * ip_current[i][action]
            for j in range(len(game.oop_buckets)):
                for action in range(2):
                    oop_regrets[j][action] += oop_updates[j][action]
                    oop_sums[j][action] += oop_marginals[j] * oop_current[j][action]

        ip_average = [_normalize(row) for row in ip_sums]
        oop_average = [_normalize(row) for row in oop_sums]
        expected = self._expected_value(game, ip_average, oop_average)
        ip_best = self._ip_best_response(game, oop_average)
        oop_best = self._oop_best_response_value(game, ip_average)
        gap = max(0.0, ip_best - oop_best)
        return RiverSolution(
            iterations=iterations,
            ip_strategy={
                bucket.name: {"check": row[0], "bet": row[1]}
                for bucket, row in zip(game.ip_buckets, ip_average)
            },
            oop_strategy={
                bucket.name: {"fold": row[0], "call": row[1]}
                for bucket, row in zip(game.oop_buckets, oop_average)
            },
            expected_ip_ev_bb=expected,
            ip_best_response_ev_bb=ip_best,
            oop_best_response_ip_ev_bb=oop_best,
            nash_gap_bb=gap,
            exploitability_bb=gap / 2,
        )

    @staticmethod
    def _expected_value(game: RiverSubgame, ip: list[list[float]], oop: list[list[float]]) -> float:
        value = 0.0
        for i in range(len(game.ip_buckets)):
            for j in range(len(game.oop_buckets)):
                check_u, fold_u, call_u = game.terminal_utilities(i, j)
                bet_u = oop[j][0] * fold_u + oop[j][1] * call_u
                value += game.pair_probabilities[i][j] * (ip[i][0] * check_u + ip[i][1] * bet_u)
        return value

    @staticmethod
    def _ip_best_response(game: RiverSubgame, oop: list[list[float]]) -> float:
        probabilities = game.pair_probabilities
        value = 0.0
        for i in range(len(game.ip_buckets)):
            marginal = sum(probabilities[i])
            check_value = bet_value = 0.0
            for j in range(len(game.oop_buckets)):
                conditional = probabilities[i][j] / marginal
                check_u, fold_u, call_u = game.terminal_utilities(i, j)
                check_value += conditional * check_u
                bet_value += conditional * (oop[j][0] * fold_u + oop[j][1] * call_u)
            value += marginal * max(check_value, bet_value)
        return value

    @staticmethod
    def _oop_best_response_value(game: RiverSubgame, ip: list[list[float]]) -> float:
        probabilities = game.pair_probabilities
        value = 0.0
        for j in range(len(game.oop_buckets)):
            check_branch = fold_branch = call_branch = 0.0
            for i in range(len(game.ip_buckets)):
                chance = probabilities[i][j]
                check_u, fold_u, call_u = game.terminal_utilities(i, j)
                check_branch += chance * ip[i][0] * check_u
                fold_branch += chance * ip[i][1] * fold_u
                call_branch += chance * ip[i][1] * call_u
            value += check_branch + min(fold_branch, call_branch)
        return value


def solve_akq_river(*, pot_bb: float = 10.0, bet_bb: float = 10.0, iterations: int = 50_000) -> tuple[RiverSubgame, RiverSolution]:
    """Classic value/bluff versus bluff-catcher river game.

    IP receives Value or Air with equal prior weight. OOP always holds a
    Bluff-catcher. Value has 100% showdown equity and Air has 0%.
    """
    game = RiverSubgame(
        pot_bb=pot_bb,
        bet_bb=bet_bb,
        ip_buckets=(HandBucket("Value", 1), HandBucket("Air", 1)),
        oop_buckets=(HandBucket("Bluff-catcher", 1),),
        showdown_equity=((1.0,), (0.0,)),
    )
    return game, RiverCFRSolver().solve(game, iterations=iterations)
