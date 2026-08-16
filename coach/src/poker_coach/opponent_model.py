from __future__ import annotations

import hashlib
import json
import math
import tempfile
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product
from pathlib import Path

from .exploit import aggregate_posterior
from .profiles import BetaPosterior, ContextKey, PlayerProfile, Tendency


SCHEMA_VERSION = "1.0.0"
POLICIES = ("check_call", "fold_to_pressure", "made_hand_selective")
MIXTURE_TENDENCIES = (
    Tendency.FOLD_TO_FLOP_CBET,
    Tendency.AGGRESSIVE_ACTION,
    Tendency.WENT_TO_SHOWDOWN,
)
REFERENCES = {
    Tendency.FOLD_TO_FLOP_CBET: Decimal("0.45"),
    Tendency.AGGRESSIVE_ACTION: Decimal("0.33"),
    Tendency.WENT_TO_SHOWDOWN: Decimal("0.28"),
}
SCALES = {
    Tendency.FOLD_TO_FLOP_CBET: Decimal("0.25"),
    Tendency.AGGRESSIVE_ACTION: Decimal("0.20"),
    Tendency.WENT_TO_SHOWDOWN: Decimal("0.15"),
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _decimal(value: Decimal) -> str:
    return format(value, "f")


def _profile_rows(profile: PlayerProfile) -> dict[str, object]:
    return {
        "profile_key": profile.player,
        "priors": {
            tendency.value: {"alpha": _decimal(posterior.alpha), "beta": _decimal(posterior.beta)}
            for tendency, posterior in sorted(profile.priors.items(), key=lambda row: row[0].value)
        },
        "estimates": [
            {
                "tendency": key.tendency.value,
                "position": key.position,
                "stack_bucket": key.stack_bucket,
                "alpha": _decimal(posterior.alpha),
                "beta": _decimal(posterior.beta),
            }
            for key, posterior in sorted(
                profile.estimates.items(),
                key=lambda row: (row[0].tendency.value, row[0].position or "", row[0].stack_bucket or ""),
            )
        ],
    }


@dataclass
class PersistentProfileStore:
    """Atomic, duplicate-event-safe persistence for uncertainty-aware player profiles."""

    profiles: dict[str, PlayerProfile] = field(default_factory=dict)
    processed_event_ids: set[str] = field(default_factory=set)
    revision: int = 0
    _persisted_revision: int | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def profile(self, profile_key: str) -> PlayerProfile:
        if not isinstance(profile_key, str) or not profile_key.strip() or len(profile_key) > 160:
            raise ValueError("profile_key must contain 1 to 160 characters")
        normalized = profile_key.strip()
        with self._lock:
            return self.profiles.setdefault(normalized, PlayerProfile(normalized))

    def observe(
        self,
        profile_key: str,
        event_id: str,
        tendency: Tendency,
        success: bool,
        *,
        position: str | None = None,
        stack_bucket: str | None = None,
    ) -> dict[str, object]:
        if not isinstance(event_id, str) or not event_id.strip() or len(event_id) > 200:
            raise ValueError("event_id must contain 1 to 200 characters")
        if not isinstance(tendency, Tendency) or not isinstance(success, bool):
            raise ValueError("tendency/success observation is invalid")
        identity = event_id.strip()
        with self._lock:
            if identity in self.processed_event_ids:
                return {
                    "status": "duplicate_ignored",
                    "event_id": identity,
                    "profile_key": profile_key.strip(),
                    "revision": self.revision,
                }
            profile = self.profile(profile_key)
            posterior = profile.observe(tendency, success, position=position, stack_bucket=stack_bucket)
            self.processed_event_ids.add(identity)
            self.revision += 1
            return {
                "status": "profile_updated",
                "event_id": identity,
                "profile_key": profile.player,
                "revision": self.revision,
                "posterior": {
                    "mean": _decimal(posterior.mean),
                    "effective_observations": _decimal(posterior.effective_observations),
                },
            }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            material: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "revision": self.revision,
                "processed_event_ids": sorted(self.processed_event_ids),
                "profiles": [_profile_rows(self.profiles[key]) for key in sorted(self.profiles)],
            }
            material["snapshot_sha256"] = _fingerprint(material)
            return material

    @staticmethod
    def _read_snapshot(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        material = dict(payload)
        observed = material.pop("snapshot_sha256", None)
        if payload.get("schema_version") != SCHEMA_VERSION or observed != _fingerprint(material):
            raise ValueError("profile store snapshot is invalid or corrupted")
        return payload

    def save(self, path: str | Path) -> Path:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if destination.exists():
                disk = self._read_snapshot(destination)
                if self._persisted_revision is None or int(disk["revision"]) != self._persisted_revision:
                    raise ValueError("profile store save rejected a stale writer")
            elif self._persisted_revision is not None:
                raise ValueError("profile store destination disappeared after load/save")
            payload = self.snapshot()
            with tempfile.TemporaryDirectory(prefix=f".{destination.stem}-", dir=destination.parent) as temporary:
                temporary_file = Path(temporary) / destination.name
                temporary_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                temporary_file.replace(destination)
            self._persisted_revision = self.revision
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PersistentProfileStore":
        source = Path(path).resolve()
        payload = cls._read_snapshot(source)
        store = cls(revision=int(payload["revision"]), _persisted_revision=int(payload["revision"]))
        event_ids = [str(value) for value in payload["processed_event_ids"]]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("profile store contains duplicate event identities")
        if store.revision < 0 or store.revision != len(event_ids):
            raise ValueError("profile store revision does not match unique event coverage")
        store.processed_event_ids = set(event_ids)
        for raw_profile in payload["profiles"]:
            profile_key = str(raw_profile["profile_key"])
            if profile_key in store.profiles:
                raise ValueError("profile store contains duplicate profile keys")
            priors = {
                Tendency(name): BetaPosterior(Decimal(str(row["alpha"])), Decimal(str(row["beta"])))
                for name, row in raw_profile["priors"].items()
            }
            profile = PlayerProfile(profile_key, priors=priors)
            for row in raw_profile["estimates"]:
                key = ContextKey(Tendency(str(row["tendency"])), row.get("position"), row.get("stack_bucket"))
                if key in profile.estimates:
                    raise ValueError("profile store contains duplicate context estimates")
                profile.estimates[key] = BetaPosterior(Decimal(str(row["alpha"])), Decimal(str(row["beta"])))
            store.profiles[profile_key] = profile
        return store


def _clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _weights(values: dict[Tendency, float], reliability: dict[Tendency, float]) -> dict[str, float]:
    signals = {
        tendency: _clamp((values[tendency] - float(REFERENCES[tendency])) / float(SCALES[tendency]))
        * reliability[tendency]
        for tendency in MIXTURE_TENDENCIES
    }
    fold = signals[Tendency.FOLD_TO_FLOP_CBET]
    aggression = signals[Tendency.AGGRESSIVE_ACTION]
    showdown = signals[Tendency.WENT_TO_SHOWDOWN]
    logits = {
        "check_call": -1.1 * fold - 0.8 * aggression + 0.3 * showdown,
        "fold_to_pressure": 1.2 * fold - 0.4 * aggression - 0.2 * showdown,
        "made_hand_selective": -0.1 * fold + 0.9 * aggression + 0.2 * showdown,
    }
    exponentials = {key: math.exp(value) for key, value in logits.items()}
    total = sum(exponentials.values())
    return {key: exponentials[key] / total for key in POLICIES}


def infer_opponent_policy_mixture(profile: PlayerProfile, *, profile_key: str | None = None) -> dict[str, object]:
    if not isinstance(profile, PlayerProfile):
        raise ValueError("profile must be a PlayerProfile")
    key = profile_key or profile.player
    if not isinstance(key, str) or not key.strip():
        raise ValueError("profile_key must be non-empty")
    posteriors = {tendency: aggregate_posterior(profile, tendency) for tendency in MIXTURE_TENDENCIES}
    reliability = {
        tendency: float(posterior.effective_observations / (posterior.effective_observations + Decimal("20")))
        if posterior.effective_observations > 0
        else 0.0
        for tendency, posterior in posteriors.items()
    }
    values = {tendency: float(posterior.mean) for tendency, posterior in posteriors.items()}
    weights = _weights(values, reliability)
    endpoints: dict[Tendency, tuple[float, float]] = {}
    posterior_rows = {}
    for tendency, posterior in posteriors.items():
        low, high = posterior.approximate_interval()
        endpoints[tendency] = (float(low), float(high))
        posterior_rows[tendency.value] = {
            "mean": _decimal(posterior.mean),
            "effective_observations": _decimal(posterior.effective_observations),
            "approximate_95_interval": [_decimal(low), _decimal(high)],
            "reference_rate": _decimal(REFERENCES[tendency]),
            "evidence_reliability": format(reliability[tendency], ".12g"),
        }
    corner_weights = []
    for choices in product((0, 1), repeat=len(MIXTURE_TENDENCIES)):
        corner = {
            tendency: endpoints[tendency][choice]
            for tendency, choice in zip(MIXTURE_TENDENCIES, choices)
        }
        corner_weights.append(_weights(corner, reliability))
    intervals = {
        policy: [
            format(min(row[policy] for row in corner_weights), ".12g"),
            format(max(row[policy] for row in corner_weights), ".12g"),
        ]
        for policy in POLICIES
    }
    max_span = max(float(high) - float(low) for low, high in intervals.values())
    total_observations = sum(float(row.effective_observations) for row in posteriors.values())
    separation = max(weights.values()) - min(weights.values())
    evidence_gate = total_observations >= 30 and max_span <= 0.35 and separation >= 0.10
    status = "profile_policy_mixture_evidence_supported" if evidence_gate else "profile_policy_mixture_observe_only"
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "profile_key": key.strip(),
        "model_scope": "synthetic_probe_policy_bridge_v1",
        "opponent_policy_weights": {policy: format(weights[policy], ".12g") for policy in POLICIES},
        "weight_uncertainty_approximate_95": intervals,
        "posterior_evidence": posterior_rows,
        "evidence_gate": {
            "passed": evidence_gate,
            "total_tendency_observations": format(total_observations, ".12g"),
            "maximum_weight_interval_span": format(max_span, ".12g"),
            "weight_separation": format(separation, ".12g"),
            "actionability": "directional_offline_only" if evidence_gate else "observe_only",
        },
        "confidence_calibrated": False,
        "recommendation_allowed": False,
        "activation_authorized": False,
        "limitations": [
            "This is an uncertainty-aware bridge to three synthetic probe policies, not a learned opponent classifier.",
            "Approximate intervals use the profile Beta normal approximation and corner propagation, not calibrated mixture posteriors.",
            "The mixture cannot authorize an exploit adjustment, recommendation or external action.",
        ],
    }
    result["mixture_fingerprint"] = _fingerprint(result)
    return result


def validate_opponent_policy_mixture(payload: dict[str, object]) -> dict[str, object]:
    issues = []
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_scope") != "synthetic_probe_policy_bridge_v1":
        issues.append("mixture schema/scope is invalid")
    if any(payload.get(key) is not False for key in ("confidence_calibrated", "recommendation_allowed", "activation_authorized")):
        issues.append("mixture cannot authorize confidence, recommendations or activation")
    try:
        weights = {key: float(payload["opponent_policy_weights"][key]) for key in POLICIES}
        weights_valid = set(payload["opponent_policy_weights"]) == set(POLICIES) and all(0 <= value <= 1 for value in weights.values()) and abs(sum(weights.values()) - 1) <= 1e-9
    except (KeyError, TypeError, ValueError):
        weights_valid = False
    if not weights_valid:
        issues.append("mixture weights are invalid")
    try:
        intervals = {
            policy: (float(payload["weight_uncertainty_approximate_95"][policy][0]), float(payload["weight_uncertainty_approximate_95"][policy][1]))
            for policy in POLICIES
        }
        intervals_valid = (
            set(payload["weight_uncertainty_approximate_95"]) == set(POLICIES)
            and all(0 <= low <= high <= 1 for low, high in intervals.values())
            and sum(low for low, _ in intervals.values()) <= 1 + 1e-9
            and sum(high for _, high in intervals.values()) >= 1 - 1e-9
        )
    except (KeyError, TypeError, ValueError, IndexError):
        intervals_valid = False
    if not intervals_valid:
        issues.append("mixture uncertainty intervals are invalid")
    gate = payload.get("evidence_gate", {})
    passed = gate.get("passed") if isinstance(gate, dict) else None
    expected_status = "profile_policy_mixture_evidence_supported" if passed is True else "profile_policy_mixture_observe_only"
    if passed not in (True, False) or payload.get("status") != expected_status:
        issues.append("mixture evidence gate/status is invalid")
    evidence = payload.get("posterior_evidence")
    try:
        evidence_valid = isinstance(evidence, dict) and set(evidence) == {row.value for row in MIXTURE_TENDENCIES} and all(
            float(row["effective_observations"]) >= 0
            and 0 <= float(row["mean"]) <= 1
            and len(row["approximate_95_interval"]) == 2
            and 0 <= float(row["approximate_95_interval"][0]) <= float(row["approximate_95_interval"][1]) <= 1
            for row in evidence.values()
        )
    except (KeyError, TypeError, ValueError, IndexError):
        evidence_valid = False
    if not evidence_valid:
        issues.append("mixture posterior evidence is invalid")
    material = dict(payload)
    observed = material.pop("mixture_fingerprint", None)
    if observed != _fingerprint(material):
        issues.append("mixture fingerprint mismatch")
    return {"schema_version": SCHEMA_VERSION, "valid": not issues, "issues": issues, "mixture_fingerprint": payload.get("mixture_fingerprint")}
