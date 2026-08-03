from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from math import sqrt


class Tendency(str, Enum):
    VPIP = "vpip"
    PFR = "pfr"
    THREE_BET = "three_bet"
    FOLD_TO_THREE_BET = "fold_to_three_bet"
    FLOP_CBET = "flop_cbet"
    FOLD_TO_FLOP_CBET = "fold_to_flop_cbet"
    AGGRESSIVE_ACTION = "aggressive_action"
    WENT_TO_SHOWDOWN = "went_to_showdown"


@dataclass(frozen=True)
class ContextKey:
    tendency: Tendency
    position: str | None = None
    stack_bucket: str | None = None


@dataclass(frozen=True)
class BetaPosterior:
    alpha: Decimal = Decimal("1")
    beta: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta parameters must be positive")

    @property
    def mean(self) -> Decimal:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> Decimal:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total * total * (total + 1))

    @property
    def effective_observations(self) -> Decimal:
        return self.alpha + self.beta - 2

    def update(self, success: bool) -> "BetaPosterior":
        return BetaPosterior(
            self.alpha + (1 if success else 0),
            self.beta + (0 if success else 1),
        )

    def approximate_interval(self, z: float = 1.96) -> tuple[Decimal, Decimal]:
        """Normal approximation for display; not an exact Beta credible interval."""
        center = float(self.mean)
        radius = z * sqrt(float(self.variance))
        return Decimal(str(max(0.0, center - radius))), Decimal(str(min(1.0, center + radius)))


@dataclass
class PlayerProfile:
    player: str
    priors: dict[Tendency, BetaPosterior] = field(default_factory=dict)
    estimates: dict[ContextKey, BetaPosterior] = field(default_factory=dict)

    def observe(
        self,
        tendency: Tendency,
        success: bool,
        *,
        position: str | None = None,
        stack_bucket: str | None = None,
    ) -> BetaPosterior:
        key = ContextKey(tendency, position, stack_bucket)
        posterior = self.estimates.get(key, self.priors.get(tendency, BetaPosterior()))
        updated = posterior.update(success)
        self.estimates[key] = updated
        return updated

    def estimate(
        self,
        tendency: Tendency,
        *,
        position: str | None = None,
        stack_bucket: str | None = None,
    ) -> BetaPosterior:
        key = ContextKey(tendency, position, stack_bucket)
        return self.estimates.get(key, self.priors.get(tendency, BetaPosterior()))

