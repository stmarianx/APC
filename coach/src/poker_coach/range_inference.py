from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping

from .equity import RANK_VALUE
from .range_strategy import hand_class, public_node_fingerprint
from .solutions import SolvedSpot


D = Decimal


def combo_label(spot: SolvedSpot) -> str:
    cards = sorted(
        spot.key.hero_cards,
        key=lambda card: (RANK_VALUE[card.rank], card.suit),
        reverse=True,
    )
    if len(cards) != 2:
        raise ValueError("Range inference requires exact two-card private nodes")
    return " ".join(str(card) for card in cards)


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Prior weight for {label} must be a decimal number")
    try:
        result = D(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Prior weight for {label} must be a decimal number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"Prior weight for {label} must be finite and non-negative")
    return result


def _entropy(probabilities: Iterable[Decimal]) -> Decimal:
    value = -sum(
        float(probability) * math.log2(float(probability))
        for probability in probabilities
        if probability > 0
    )
    return D(str(value))


def _effective_combos(probabilities: Iterable[Decimal]) -> Decimal:
    concentration = sum((probability * probability for probability in probabilities), D("0"))
    return D("0") if concentration == 0 else D("1") / concentration


def _fmt(value: Decimal) -> str:
    return format(value, "f")


def condition_solution_range(
    spots: Iterable[SolvedSpot],
    observed_action: str,
    *,
    prior_weights: Mapping[str, object] | None = None,
) -> dict[str, object]:
    nodes = tuple(spots)
    if not nodes:
        raise ValueError("Range inference requires at least one solved private node")
    action_id = observed_action.strip()
    if not action_id:
        raise ValueError("observed_action is required")
    fingerprints = {public_node_fingerprint(spot) for spot in nodes}
    if len(fingerprints) != 1:
        raise ValueError("All private nodes must belong to one public decision state")
    labels = [combo_label(spot) for spot in nodes]
    if len(labels) != len(set(labels)):
        raise ValueError("Range inference contains duplicate exact private combos")
    by_label = dict(zip(labels, nodes))

    supplied = prior_weights is not None
    if prior_weights is None:
        raw_prior = {label: D("1") for label in labels}
    else:
        unknown = sorted(set(prior_weights) - set(labels))
        if unknown:
            raise ValueError(f"Prior weights reference unknown combos: {', '.join(unknown)}")
        raw_prior = {
            label: _decimal(prior_weights.get(label, 0), label) for label in labels
        }
    prior_total = sum(raw_prior.values(), D("0"))
    if prior_total <= 0:
        raise ValueError("Prior weights must contain positive mass")
    prior = {label: weight / prior_total for label, weight in raw_prior.items()}

    known_actions = sorted(
        {action.action for spot in nodes for action in spot.actions}
    )
    if action_id not in known_actions:
        raise ValueError(f"Observed action is not covered at this node: {action_id}")
    likelihood: dict[str, Decimal] = {}
    action_ev: dict[str, Decimal | None] = {}
    for label, spot in by_label.items():
        try:
            action = spot.action(action_id)
        except KeyError:
            likelihood[label] = D("0")
            action_ev[label] = None
        else:
            likelihood[label] = action.frequency
            action_ev[label] = action.ev
    evidence = sum(
        (prior[label] * likelihood[label] for label in labels), D("0")
    )
    if evidence <= 0:
        raise ValueError(
            "Observed action has zero probability under the supplied range prior"
        )
    posterior = {
        label: prior[label] * likelihood[label] / evidence for label in labels
    }

    prior_entropy = _entropy(prior.values())
    posterior_entropy = _entropy(posterior.values())
    total_variation = sum(
        (abs(posterior[label] - prior[label]) for label in labels), D("0")
    ) / 2
    kl_bits = sum(
        (
            posterior[label]
            * D(str(math.log2(float(posterior[label] / prior[label]))))
            for label in labels
            if posterior[label] > 0 and prior[label] > 0
        ),
        D("0"),
    )

    class_prior: dict[str, Decimal] = defaultdict(lambda: D("0"))
    class_posterior: dict[str, Decimal] = defaultdict(lambda: D("0"))
    combo_rows: list[dict[str, object]] = []
    for label in labels:
        spot = by_label[label]
        class_name = hand_class(spot)
        assert class_name is not None
        class_prior[class_name] += prior[label]
        class_posterior[class_name] += posterior[label]
        combo_rows.append(
            {
                "combo": label,
                "hand_class": class_name,
                "prior": _fmt(prior[label]),
                "action_likelihood": _fmt(likelihood[label]),
                "posterior": _fmt(posterior[label]),
                "bayes_factor": _fmt(likelihood[label] / evidence),
                "action_ev_bb": None
                if action_ev[label] is None
                else _fmt(action_ev[label]),
                "node_id": spot.node_id,
            }
        )
    combo_rows.sort(
        key=lambda row: (-D(str(row["posterior"])), str(row["combo"]))
    )
    class_rows = [
        {
            "hand_class": class_name,
            "prior": _fmt(class_prior[class_name]),
            "posterior": _fmt(class_posterior[class_name]),
            "change": _fmt(class_posterior[class_name] - class_prior[class_name]),
        }
        for class_name in sorted(set(class_prior) | set(class_posterior))
    ]
    range_action_probabilities = []
    for candidate in known_actions:
        probability = D("0")
        for label, spot in by_label.items():
            try:
                frequency = spot.action(candidate).frequency
            except KeyError:
                frequency = D("0")
            probability += prior[label] * frequency
        range_action_probabilities.append(
            {"action": candidate, "probability": _fmt(probability)}
        )

    representative = nodes[0]
    key = representative.key
    return {
        "schema_version": "1.0.0",
        "public_fingerprint": next(iter(fingerprints)),
        "observed_action": action_id,
        "action_probability_under_prior": _fmt(evidence),
        "prior_source": "caller_supplied_exact_combo_weights"
        if supplied
        else "uniform_exact_node_coverage",
        "state": {
            "game": key.game,
            "players": key.players,
            "hero_position": key.hero_position,
            "board": [str(card) for card in key.board],
            "action_history": list(key.action_history),
            "pot_bb": _fmt(key.pot_bb),
            "effective_stack_bb": _fmt(key.effective_stack_bb),
        },
        "information": {
            "prior_entropy_bits": _fmt(prior_entropy),
            "posterior_entropy_bits": _fmt(posterior_entropy),
            "entropy_reduction_bits": _fmt(prior_entropy - posterior_entropy),
            "prior_effective_combos": _fmt(_effective_combos(prior.values())),
            "posterior_effective_combos": _fmt(
                _effective_combos(posterior.values())
            ),
            "total_variation_shift": _fmt(total_variation),
            "kl_divergence_bits": _fmt(kl_bits),
        },
        "combos": combo_rows,
        "classes": class_rows,
        "range_action_probabilities": range_action_probabilities,
        "provenance": {
            "source": representative.source,
            "source_version": representative.source_version,
            "method": "Bayes posterior proportional to prior combo weight times imported solver action frequency",
            "coverage": f"{len(nodes)} imported exact private combos",
            "caveat": "Missing private combos are never inferred. Solver frequencies are strategy likelihoods, not empirical population frequencies.",
        },
    }
