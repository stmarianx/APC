from __future__ import annotations

from decimal import Decimal
from typing import Iterable

from .explanations import explain_decision
from .features import ProfileBook
from .models import HandHistory
from .profiles import BetaPosterior, PlayerProfile, Tendency
from .replay import HandReplayer
from .exploit import player_exploit_insights
from .trends import analyze_hero_trends


def _decimal(value: Decimal) -> str:
    return format(value, "f")


def _aggregate_tendency(profile: PlayerProfile, tendency: Tendency) -> BetaPosterior:
    rows = [
        posterior
        for key, posterior in profile.estimates.items()
        if key.tendency == tendency
    ]
    return BetaPosterior(
        Decimal("1") + sum((row.alpha - Decimal("1") for row in rows), Decimal("0")),
        Decimal("1") + sum((row.beta - Decimal("1") for row in rows), Decimal("0")),
    )


def _metric(posterior: BetaPosterior) -> dict[str, object]:
    low, high = posterior.approximate_interval()
    return {
        "posterior_mean": _decimal(posterior.mean),
        "opportunities": _decimal(posterior.effective_observations),
        "approximate_95_interval": [_decimal(low), _decimal(high)],
    }


def _profile_summary(profile: PlayerProfile) -> dict[str, object]:
    vpip = _aggregate_tendency(profile, Tendency.VPIP)
    pfr = _aggregate_tendency(profile, Tendency.PFR)
    aggression = _aggregate_tendency(profile, Tendency.AGGRESSIVE_ACTION)
    sample = min(vpip.effective_observations, pfr.effective_observations)
    if sample < 10:
        label = "Developing profile"
        confidence = "limited"
    else:
        looseness = "Loose" if vpip.mean >= Decimal("0.32") else "Tight" if vpip.mean <= Decimal("0.22") else "Balanced"
        aggression_ratio = pfr.mean / max(vpip.mean, Decimal("0.000001"))
        posture = "aggressive" if aggression_ratio >= Decimal("0.68") else "passive" if aggression_ratio <= Decimal("0.42") else "selective"
        label = f"{looseness}-{posture}"
        confidence = "developing" if sample < 100 else "established"
    return {
        "style_label": label,
        "confidence": confidence,
        "classification_sample": _decimal(sample),
        "metrics": {
            "vpip": _metric(vpip),
            "pfr": _metric(pfr),
            "aggressive_action": _metric(aggression),
        },
    }


def analyze_hands(hands: Iterable[HandHistory]) -> dict[str, object]:
    hand_list = tuple(hands)
    book = ProfileBook()
    replayer = HandReplayer()
    hand_reports: list[dict[str, object]] = []

    for hand in hand_list:
        book.observe_hand(hand)
        replay = replayer.replay(hand)
        actions_by_index = {action.index: action for action in hand.actions}
        hero_decisions = [
            explain_decision(snapshot, actions_by_index[snapshot.action_index])
            for snapshot in replay.decisions
            if snapshot.actor == hand.hero
        ]
        hand_reports.append(
            {
                "hand_id": hand.hand_id,
                "table": hand.table_name,
                "played_at": hand.played_at_raw,
                "players": len(hand.players),
                "hero": hand.hero,
                "board": [str(card) for card in hand.board],
                "actions": len(hand.actions),
                "decision_snapshots": len(replay.decisions),
                "hero_decisions": hero_decisions,
                "committed_pot": _decimal(replay.committed_pot),
                "awarded_total": _decimal(replay.awarded_total),
                "rake": _decimal(replay.rake),
                "reconciliation_error": _decimal(replay.reconciliation_error),
            }
        )

    profile_reports: dict[str, list[dict[str, object]]] = {}
    profile_summaries: dict[str, dict[str, object]] = {}
    exploit_insights: dict[str, list[dict[str, object]]] = {}
    for player, profile in sorted(book.profiles.items()):
        profile_summaries[player] = _profile_summary(profile)
        exploit_insights[player] = player_exploit_insights(profile)
        estimates = []
        for key, posterior in sorted(
            profile.estimates.items(),
            key=lambda item: (item[0].tendency.value, item[0].position or "", item[0].stack_bucket or ""),
        ):
            low, high = posterior.approximate_interval()
            estimates.append(
                {
                    "tendency": key.tendency.value,
                    "position": key.position,
                    "stack_bucket": key.stack_bucket,
                    "posterior_mean": _decimal(posterior.mean),
                    "opportunities": _decimal(posterior.effective_observations),
                    "approximate_95_interval": [_decimal(low), _decimal(high)],
                }
            )
        profile_reports[player] = estimates

    return {
        "schema_version": "0.3.0",
        "hands": len(hand_list),
        "hand_reports": hand_reports,
        "player_profiles": profile_reports,
        "profile_summaries": profile_summaries,
        "exploit_insights": exploit_insights,
        "hero_trends": analyze_hero_trends(hand_list),
    }
