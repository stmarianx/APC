from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from .isomorphism import suit_isomorphic
from .models import Card, HandHistory
from .range_inference import combo_label
from .range_strategy import public_node_fingerprint
from .solutions import SolvedSpot


D = Decimal
CALIBRATION_BINS = (
    (D("0"), D("0.10")),
    (D("0.10"), D("0.25")),
    (D("0.25"), D("0.50")),
    (D("0.50"), D("0.75")),
    (D("0.75"), D("1.0000000000000000001")),
)


def _fmt(value: Decimal) -> str:
    return format(value, "f")


def _revealed_opponents(hand: HandHistory) -> dict[str, tuple[Card, ...]]:
    return {
        holding.player: holding.cards
        for holding in hand.hole_cards
        if holding.shown and holding.player != hand.hero
    }


def _actual_combo_label(
    spots: tuple[SolvedSpot, ...],
    board: tuple[Card, ...],
    cards: tuple[Card, ...],
) -> tuple[str | None, str]:
    matches = [
        spot
        for spot in spots
        if suit_isomorphic(spot.key.board, spot.key.hero_cards, board, cards)
    ]
    if not matches:
        return None, "revealed_combo_outside_imported_support"
    if len(matches) > 1:
        return None, "revealed_combo_mapping_ambiguous"
    return combo_label(matches[0]), "mapped"


def _bin_index(probability: Decimal) -> int:
    for index, (low, high) in enumerate(CALIBRATION_BINS):
        if low <= probability < high:
            return index
    return len(CALIBRATION_BINS) - 1


def score_opponent_range_timelines(
    hands: Iterable[HandHistory],
    solutions: Iterable[SolvedSpot],
    timeline_review: dict[str, object],
) -> dict[str, object]:
    """Score range posteriors against private cards revealed in saved hands."""

    hand_rows = tuple(hands)
    hands_by_id = {hand.hand_id: hand for hand in hand_rows}
    grouped: dict[str, list[SolvedSpot]] = defaultdict(list)
    for spot in solutions:
        if len(spot.key.hero_cards) == 2:
            grouped[public_node_fingerprint(spot)].append(spot)
    groups = {key: tuple(value) for key, value in grouped.items()}

    predictions: list[dict[str, object]] = []
    timeline_scores: list[dict[str, object]] = []
    support_misses = 0
    mapping_ambiguities = 0
    unrevealed_timelines = 0
    bucket_rows = [
        {"forecast_total": D("0"), "outcomes": 0, "observations": 0}
        for _ in CALIBRATION_BINS
    ]

    timelines = timeline_review.get("timelines", [])
    if not isinstance(timelines, list):
        raise ValueError("timeline_review.timelines must be an array")
    for timeline in timelines:
        if not isinstance(timeline, dict):
            raise ValueError("timeline_review contains an invalid timeline")
        hand_id = str(timeline.get("hand_id", ""))
        opponent = str(timeline.get("opponent", ""))
        hand = hands_by_id.get(hand_id)
        if hand is None:
            raise ValueError(f"Timeline references unknown hand: {hand_id}")
        shown = _revealed_opponents(hand).get(opponent)
        if shown is None:
            unrevealed_timelines += 1
            timeline_scores.append(
                {
                    "hand_id": hand_id,
                    "opponent": opponent,
                    "status": "not_revealed",
                    "revealed_cards": [],
                    "predictions": [],
                }
            )
            continue
        scored_events: list[dict[str, object]] = []
        for event in timeline.get("events", []):
            if not isinstance(event, dict) or event.get("status") != "conditioned":
                continue
            match = event.get("match")
            posterior = event.get("posterior")
            if not isinstance(match, dict) or not isinstance(posterior, dict):
                raise ValueError("Conditioned timeline event is missing match/posterior")
            fingerprint = str(match.get("public_fingerprint", ""))
            spots = groups.get(fingerprint, ())
            board_value = event.get("board", [])
            if not isinstance(board_value, list):
                raise ValueError("Timeline event board must be an array")
            board = tuple(Card.parse(str(card)) for card in board_value)
            actual_label, mapping_status = _actual_combo_label(spots, board, shown)
            if mapping_status == "revealed_combo_mapping_ambiguous":
                mapping_ambiguities += 1
            elif mapping_status != "mapped":
                support_misses += 1
            combo_rows = posterior.get("combos", [])
            if not isinstance(combo_rows, list):
                raise ValueError("Timeline posterior combos must be an array")
            event_score: dict[str, object] = {
                "action_index": event.get("action_index"),
                "street": event.get("street"),
                "observed_action": event.get("observed_action"),
                "mapping_status": mapping_status,
                "actual_combo": actual_label,
                "support_combos": len(combo_rows),
            }
            if actual_label is None:
                event_score["scored"] = False
                scored_events.append(event_score)
                continue
            actual_row = next(
                (row for row in combo_rows if str(row.get("combo")) == actual_label),
                None,
            )
            if actual_row is None:
                support_misses += 1
                event_score["scored"] = False
                event_score["mapping_status"] = "posterior_missing_mapped_combo"
                scored_events.append(event_score)
                continue
            probabilities = {
                str(row["combo"]): D(str(row["posterior"])) for row in combo_rows
            }
            probability = probabilities[actual_label]
            prior_probability = D(str(actual_row["prior"]))
            brier = sum(
                (
                    (candidate_probability - (D("1") if label == actual_label else D("0")))
                    ** 2
                    for label, candidate_probability in probabilities.items()
                ),
                D("0"),
            )
            rank = 1 + sum(
                1 for candidate in probabilities.values() if candidate > probability
            )
            zero_probability = probability <= 0
            log_loss_bits = (
                None if zero_probability else D(str(-math.log2(float(probability))))
            )
            log_loss_nats = (
                None if zero_probability else D(str(-math.log(float(probability))))
            )
            baseline_bits = D(str(math.log2(len(probabilities))))
            event_score.update(
                {
                    "scored": True,
                    "actual_prior_probability": _fmt(prior_probability),
                    "actual_posterior_probability": _fmt(probability),
                    "probability_change": _fmt(probability - prior_probability),
                    "posterior_rank": rank,
                    "top_1": rank == 1,
                    "top_3": rank <= 3,
                    "zero_probability": zero_probability,
                    "log_loss_bits": "infinite"
                    if zero_probability
                    else _fmt(log_loss_bits),
                    "log_loss_nats": "infinite"
                    if zero_probability
                    else _fmt(log_loss_nats),
                    "uniform_baseline_log_loss_bits": _fmt(baseline_bits),
                    "information_gain_vs_uniform_bits": "-infinite"
                    if zero_probability
                    else _fmt(baseline_bits - log_loss_bits),
                    "multiclass_brier_score": _fmt(brier),
                }
            )
            predictions.append(event_score)
            scored_events.append(event_score)
            for label, candidate_probability in probabilities.items():
                bucket = bucket_rows[_bin_index(candidate_probability)]
                bucket["forecast_total"] = D(str(bucket["forecast_total"])) + candidate_probability
                bucket["outcomes"] = int(bucket["outcomes"]) + (
                    1 if label == actual_label else 0
                )
                bucket["observations"] = int(bucket["observations"]) + 1
        timeline_scores.append(
            {
                "hand_id": hand_id,
                "opponent": opponent,
                "status": "scored" if any(row.get("scored") for row in scored_events) else "not_scored",
                "revealed_cards": [str(card) for card in shown],
                "predictions": scored_events,
            }
        )

    finite = [row for row in predictions if not row["zero_probability"]]
    infinite_predictions = len(predictions) - len(finite)
    scored_count = len(predictions)
    possible = scored_count + support_misses + mapping_ambiguities
    calibration_buckets: list[dict[str, object]] = []
    calibration_error_weighted = D("0")
    calibration_observations = 0
    for index, ((low, high), bucket) in enumerate(zip(CALIBRATION_BINS, bucket_rows)):
        observations = int(bucket["observations"])
        if not observations:
            continue
        mean_forecast = D(str(bucket["forecast_total"])) / D(observations)
        hit_rate = D(int(bucket["outcomes"])) / D(observations)
        absolute_gap = abs(mean_forecast - hit_rate)
        calibration_error_weighted += absolute_gap * D(observations)
        calibration_observations += observations
        calibration_buckets.append(
            {
                "bin": index,
                "lower": _fmt(low),
                "upper": "1" if index == len(CALIBRATION_BINS) - 1 else _fmt(high),
                "observations": observations,
                "mean_forecast": _fmt(mean_forecast),
                "empirical_hit_rate": _fmt(hit_rate),
                "absolute_gap": _fmt(absolute_gap),
            }
        )

    def mean(key: str) -> str | None:
        if not finite:
            return None
        return _fmt(
            sum((D(str(row[key])) for row in finite), D("0")) / D(len(finite))
        )

    aggregate = {
        "revealed_timelines": len(timeline_scores) - unrevealed_timelines,
        "unrevealed_timelines": unrevealed_timelines,
        "scored_predictions": scored_count,
        "support_misses": support_misses,
        "mapping_ambiguities": mapping_ambiguities,
        "support_coverage": _fmt(D(scored_count) / D(possible)) if possible else "0",
        "infinite_log_loss_predictions": infinite_predictions,
        "mean_log_loss_bits": "infinite"
        if infinite_predictions
        else mean("log_loss_bits"),
        "mean_log_loss_nats": "infinite"
        if infinite_predictions
        else mean("log_loss_nats"),
        "mean_uniform_baseline_log_loss_bits": mean(
            "uniform_baseline_log_loss_bits"
        ),
        "mean_information_gain_vs_uniform_bits": "-infinite"
        if infinite_predictions
        else mean("information_gain_vs_uniform_bits"),
        "mean_multiclass_brier_score": mean("multiclass_brier_score"),
        "top_1_accuracy": _fmt(
            D(sum(1 for row in predictions if row["top_1"])) / D(scored_count)
        )
        if scored_count
        else None,
        "top_3_accuracy": _fmt(
            D(sum(1 for row in predictions if row["top_3"])) / D(scored_count)
        )
        if scored_count
        else None,
        "expected_calibration_error": _fmt(
            calibration_error_weighted / D(calibration_observations)
        )
        if calibration_observations
        else None,
    }
    return {
        "schema_version": "1.0.0",
        "hands": len(hand_rows),
        "aggregate": aggregate,
        "calibration_buckets": calibration_buckets,
        "timelines": timeline_scores,
        "caveat": (
            "Calibration scores only opponents whose cards were actually revealed and only "
            "within imported exact-combo support. A small showdown sample is descriptive, not "
            "evidence that the range model is population-calibrated."
        ),
    }
