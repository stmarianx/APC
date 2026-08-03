from __future__ import annotations

import hashlib
import json
import math
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.0.0"


def normalize_player_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("player name must be text")
    display = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if not 1 <= len(display) <= 64 or any(ord(character) < 32 for character in display):
        raise ValueError("player name must contain 1 to 64 printable characters")
    return display.casefold()


def _identity_id(provider_namespace: str, normalized_name: str) -> str:
    return hashlib.sha256(f"{provider_namespace}\0{normalized_name}".encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class NameObservation:
    seat_no: int
    raw_name: str
    confidence: float
    frame_sha256: str
    observed_at_ms: int

    def __post_init__(self) -> None:
        if isinstance(self.seat_no, bool) or self.seat_no < 1 or self.seat_no > 10:
            raise ValueError("seat_no must be between 1 and 10")
        normalize_player_name(self.raw_name)
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("name confidence must be between zero and one")
        if len(self.frame_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.frame_sha256):
            raise ValueError("frame_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.observed_at_ms, bool) or self.observed_at_ms < 0:
            raise ValueError("observed_at_ms must be nonnegative")


@dataclass
class _Candidate:
    normalized_name: str
    weight: float = 0.0
    frames: int = 0
    raw_variants: dict[str, int] = field(default_factory=dict)

    def observe(self, raw_name: str, confidence: float) -> None:
        self.weight += confidence
        self.frames += 1
        display = " ".join(unicodedata.normalize("NFKC", raw_name).strip().split())
        self.raw_variants[display] = self.raw_variants.get(display, 0) + 1

    @property
    def display_name(self) -> str:
        return sorted(self.raw_variants.items(), key=lambda row: (-row[1], row[0]))[0][0]


@dataclass
class _SeatEvidence:
    candidates: dict[str, _Candidate] = field(default_factory=dict)
    seen_frames: set[str] = field(default_factory=set)


class PlayerIdentityRegistry:
    """Resolve persistent player keys from repeated, confidence-bearing name observations."""

    def __init__(
        self,
        provider_namespace: str,
        *,
        minimum_frames: int = 3,
        minimum_mean_confidence: float = 0.85,
        minimum_probability: float = 0.75,
        minimum_margin: float = 0.25,
    ) -> None:
        if not isinstance(provider_namespace, str) or not provider_namespace.strip():
            raise ValueError("provider_namespace must be non-empty")
        if minimum_frames < 1:
            raise ValueError("minimum_frames must be positive")
        for value, name in (
            (minimum_mean_confidence, "minimum_mean_confidence"),
            (minimum_probability, "minimum_probability"),
            (minimum_margin, "minimum_margin"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        self.provider_namespace = provider_namespace.strip()
        self.minimum_frames = minimum_frames
        self.minimum_mean_confidence = minimum_mean_confidence
        self.minimum_probability = minimum_probability
        self.minimum_margin = minimum_margin
        self._seats: dict[str, _SeatEvidence] = {}
        self._identities: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _seat_key(table_track_id: str, seat_no: int) -> str:
        if not isinstance(table_track_id, str) or not table_track_id.strip():
            raise ValueError("table_track_id must be non-empty")
        return f"{table_track_id.strip()}:{seat_no}"

    def observe_batch(
        self,
        table_track_id: str,
        observations: Iterable[NameObservation],
    ) -> list[dict[str, object]]:
        rows = list(observations)
        seats = [row.seat_no for row in rows]
        if len(seats) != len(set(seats)):
            raise ValueError("one frame batch cannot contain duplicate seats")
        normalized_to_seats: dict[str, list[int]] = {}
        for row in rows:
            normalized_to_seats.setdefault(normalize_player_name(row.raw_name), []).append(row.seat_no)
        collisions = {
            name
            for name, collision_seats in normalized_to_seats.items()
            if len(collision_seats) > 1
        }
        results = []
        with self._lock:
            for row in rows:
                seat_key = self._seat_key(table_track_id, row.seat_no)
                evidence = self._seats.setdefault(seat_key, _SeatEvidence())
                normalized = normalize_player_name(row.raw_name)
                duplicate = row.frame_sha256 in evidence.seen_frames
                if not duplicate:
                    evidence.seen_frames.add(row.frame_sha256)
                    candidate = evidence.candidates.setdefault(normalized, _Candidate(normalized))
                    candidate.observe(row.raw_name, row.confidence)
                result = self._resolve(seat_key, collision=normalized in collisions)
                result["seat_no"] = row.seat_no
                result["duplicate_frame"] = duplicate
                if result["status"] == "resolved":
                    identity_id = str(result["identity_id"])
                    identity = self._identities.setdefault(
                        identity_id,
                        {
                            "identity_id": identity_id,
                            "provider_namespace": self.provider_namespace,
                            "normalized_name": result["normalized_name"],
                            "display_name": result["display_name"],
                            "first_observed_at_ms": row.observed_at_ms,
                            "last_observed_at_ms": row.observed_at_ms,
                            "observation_frames": 0,
                        },
                    )
                    identity["last_observed_at_ms"] = max(int(identity["last_observed_at_ms"]), row.observed_at_ms)
                    if not duplicate:
                        identity["observation_frames"] = int(identity["observation_frames"]) + 1
                results.append(result)
        return results

    def _resolve(self, seat_key: str, *, collision: bool) -> dict[str, object]:
        evidence = self._seats[seat_key]
        candidates = sorted(evidence.candidates.values(), key=lambda item: (-item.weight, item.normalized_name))
        total = sum(candidate.weight + 0.1 for candidate in candidates)
        rows = []
        for candidate in candidates:
            probability = (candidate.weight + 0.1) / total if total else 0.0
            rows.append(
                {
                    "normalized_name": candidate.normalized_name,
                    "display_name": candidate.display_name,
                    "frames": candidate.frames,
                    "mean_confidence": candidate.weight / candidate.frames,
                    "posterior_probability": probability,
                }
            )
        best = rows[0]
        runner_probability = float(rows[1]["posterior_probability"]) if len(rows) > 1 else 0.0
        margin = float(best["posterior_probability"]) - runner_probability
        resolved = (
            not collision
            and int(best["frames"]) >= self.minimum_frames
            and float(best["mean_confidence"]) >= self.minimum_mean_confidence
            and float(best["posterior_probability"]) >= self.minimum_probability
            and margin >= self.minimum_margin
        )
        return {
            "seat_key": seat_key,
            "status": "ambiguous_collision" if collision else "resolved" if resolved else "developing",
            "identity_id": _identity_id(self.provider_namespace, str(best["normalized_name"])) if resolved else None,
            "profile_key": f"{self.provider_namespace}:{best['normalized_name']}" if resolved else None,
            "normalized_name": best["normalized_name"],
            "display_name": best["display_name"],
            "posterior_probability": best["posterior_probability"],
            "margin": margin,
            "frames": best["frames"],
            "mean_confidence": best["mean_confidence"],
            "candidates": rows,
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            material: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "provider_namespace": self.provider_namespace,
                "thresholds": {
                    "minimum_frames": self.minimum_frames,
                    "minimum_mean_confidence": self.minimum_mean_confidence,
                    "minimum_probability": self.minimum_probability,
                    "minimum_margin": self.minimum_margin,
                },
                "identities": [self._identities[key] for key in sorted(self._identities)],
                "seat_evidence": {
                    key: {
                        "seen_frames": sorted(evidence.seen_frames),
                        "candidates": {
                            name: {
                                "weight": candidate.weight,
                                "frames": candidate.frames,
                                "raw_variants": dict(sorted(candidate.raw_variants.items())),
                            }
                            for name, candidate in sorted(evidence.candidates.items())
                        },
                    }
                    for key, evidence in sorted(self._seats.items())
                },
            }
            material["snapshot_sha256"] = hashlib.sha256(
                json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            return material

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PlayerIdentityRegistry":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        expected = payload.get("snapshot_sha256")
        material = dict(payload)
        material.pop("snapshot_sha256", None)
        actual = hashlib.sha256(
            json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if payload.get("schema_version") != SCHEMA_VERSION or expected != actual:
            raise ValueError("identity registry snapshot is invalid or corrupted")
        thresholds = payload["thresholds"]
        registry = cls(
            str(payload["provider_namespace"]),
            minimum_frames=int(thresholds["minimum_frames"]),
            minimum_mean_confidence=float(thresholds["minimum_mean_confidence"]),
            minimum_probability=float(thresholds["minimum_probability"]),
            minimum_margin=float(thresholds["minimum_margin"]),
        )
        registry._identities = {str(row["identity_id"]): dict(row) for row in payload["identities"]}
        for key, raw_evidence in payload["seat_evidence"].items():
            evidence = _SeatEvidence(seen_frames=set(raw_evidence["seen_frames"]))
            for name, raw_candidate in raw_evidence["candidates"].items():
                evidence.candidates[name] = _Candidate(
                    name,
                    float(raw_candidate["weight"]),
                    int(raw_candidate["frames"]),
                    {str(raw): int(count) for raw, count in raw_candidate["raw_variants"].items()},
                )
            registry._seats[str(key)] = evidence
        return registry
