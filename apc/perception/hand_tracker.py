from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from apc.player_identity import PlayerIdentityRegistry
from apc.visual_identity_signature import registry_observations_from_state


STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


@dataclass
class _Track:
    hand_counter: int = -1
    hand_id: str | None = None
    revision: int = -1
    last_after: dict[str, object] | None = None
    history: list[dict[str, object]] = field(default_factory=list)
    seen_after_frames: set[str] = field(default_factory=set)


def _stack_signature(state: dict[str, object]) -> tuple[tuple[int, str], ...]:
    rows = state.get("seat_stacks_bb")
    if not isinstance(rows, list):
        raise ValueError("tracked visible state requires seat_stacks_bb")
    return tuple(
        sorted(
            (int(row["seat_no"]), format(Decimal(str(row["stack_bb"])), "f"))
            for row in rows
            if isinstance(row, dict)
        )
    )


def _cross_pair_audit(previous: dict[str, object], before: dict[str, object]) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(code: str, passed: bool, expected: object, observed: object) -> None:
        checks.append({"code": code, "passed": bool(passed), "expected": expected, "observed": observed})

    for key in ("layout_id", "theme_id", "hero_seat", "dealer_seat", "hero_cards"):
        check(f"{key}_continuous", previous.get(key) == before.get(key), previous.get(key), before.get(key))
    prior_board = previous.get("board_cards")
    current_board = before.get("board_cards")
    board_prefix = (
        isinstance(prior_board, list)
        and isinstance(current_board, list)
        and len(current_board) >= len(prior_board)
        and current_board[: len(prior_board)] == prior_board
    )
    check("board_prefix", board_prefix, prior_board, current_board)
    prior_street, current_street = previous.get("street"), before.get("street")
    street_progress = (
        prior_street in STREET_ORDER
        and current_street in STREET_ORDER
        and STREET_ORDER[str(current_street)] in {STREET_ORDER[str(prior_street)], STREET_ORDER[str(prior_street)] + 1}
    )
    check("street_progress", street_progress, prior_street, current_street)
    check("stacks_carried", _stack_signature(previous) == _stack_signature(before), _stack_signature(previous), _stack_signature(before))
    check("pot_carried", Decimal(str(previous.get("pot_bb"))) == Decimal(str(before.get("pot_bb"))), previous.get("pot_bb"), before.get("pot_bb"))
    violations = [row for row in checks if not row["passed"]]
    return {"status": "accepted" if not violations else "rejected", "checks": checks, "violations": violations}


def attach_player_identities(
    tracked_result: dict[str, object],
    resolutions: list[dict[str, object]],
) -> dict[str, object]:
    """Attach only fully resolved, collision-free identity evidence to a tracked state."""
    result = copy.deepcopy(tracked_result)
    state = result.get("state")
    if not isinstance(state, dict):
        raise ValueError("tracked result has no state to enrich")
    stack_rows = state.get("seat_stacks_bb")
    if not isinstance(stack_rows, list):
        raise ValueError("tracked state has no visible seat set")
    expected_seats = {
        int(row["seat_no"])
        for row in stack_rows
        if isinstance(row, dict) and isinstance(row.get("seat_no"), int)
    }
    by_seat: dict[int, dict[str, object]] = {}
    for resolution in resolutions:
        seat = resolution.get("seat_no")
        if isinstance(seat, bool) or not isinstance(seat, int) or seat in by_seat:
            raise ValueError("identity resolutions must contain unique integer seat_no values")
        by_seat[seat] = resolution
    all_resolved = expected_seats == set(by_seat) and all(
        row.get("status") == "resolved" and isinstance(row.get("identity_id"), str)
        for row in by_seat.values()
    )
    identity_ids = [str(row["identity_id"]) for row in by_seat.values() if row.get("status") == "resolved"]
    all_resolved = all_resolved and len(identity_ids) == len(set(identity_ids))
    state["player_identities"] = [
        {
            "seat_no": seat,
            "identity_id": row.get("identity_id"),
            "profile_key": row.get("profile_key"),
            "display_name": row.get("display_name"),
            "status": row.get("status"),
            "posterior_probability": row.get("posterior_probability"),
            "evidence_frames": row.get("frames"),
        }
        for seat, row in sorted(by_seat.items())
    ]
    missing = set(str(value) for value in result.get("missing_critical_fields", []))
    if all_resolved:
        missing.discard("player_identities")
        result["status"] = "state_tracked_identity_resolved"
    else:
        missing.add("player_identities")
        result["status"] = "state_tracked_identity_uncertain"
    result["missing_critical_fields"] = sorted(missing)
    result["identity_gate"] = {
        "status": "passed" if all_resolved else "unresolved",
        "expected_seats": sorted(expected_seats),
        "observed_seats": sorted(by_seat),
        "unique_resolved_identities": len(set(identity_ids)),
    }
    result["recommendation"] = None
    return result


def resolve_visual_player_identities(
    tracked_result: dict[str, object],
    registry: PlayerIdentityRegistry,
    *,
    observed_at_ms: int,
) -> dict[str, object]:
    """Resolve pseudonymous profiles from repeated name-band pixels, without an OCR claim."""
    state = tracked_result.get("state")
    track_id = tracked_result.get("track_id")
    if not isinstance(state, dict) or not isinstance(track_id, str) or not track_id:
        raise ValueError("tracked result requires state and track_id for visual identity resolution")
    observations = registry_observations_from_state(
        state,
        observed_at_ms=observed_at_ms,
    )
    resolutions = registry.observe_batch(track_id, observations)
    enriched = attach_player_identities(tracked_result, resolutions)
    enriched["identity_gate"]["evidence_kind"] = "visual_name_band_signature_not_ocr"
    enriched["identity_gate"]["human_readable_names"] = False
    return enriched


class TemporalHandTracker:
    """Accumulate adjacent-frame observations into an evidence-bounded hand state."""

    def __init__(self, *, minimum_boundary_confidence: float = 0.90) -> None:
        if not 0 <= minimum_boundary_confidence <= 1:
            raise ValueError("minimum_boundary_confidence must be between zero and one")
        self.minimum_boundary_confidence = float(minimum_boundary_confidence)
        self._tracks: dict[str, _Track] = {}
        self._lock = threading.RLock()

    def prior_history(self, track_id: str) -> list[dict[str, object]]:
        with self._lock:
            track = self._tracks.get(track_id)
            return copy.deepcopy(track.history if track is not None else [])

    def submit(
        self,
        track_id: str,
        temporal_result: dict[str, object],
        *,
        hand_start: bool,
        boundary_confidence: float,
    ) -> dict[str, object]:
        if not isinstance(track_id, str) or not track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if not 0 <= float(boundary_confidence) <= 1:
            raise ValueError("boundary_confidence must be between zero and one")
        if temporal_result.get("schema_version") != "1.0.0":
            raise ValueError("unsupported temporal result schema")
        transition = temporal_result.get("transition_audit")
        if not isinstance(transition, dict) or transition.get("status") != "accepted":
            return self._abstain("transition_rejected", track_id, temporal_result)
        before = temporal_result.get("previous_visible_state")
        after = temporal_result.get("visible_state")
        frames = temporal_result.get("frames")
        if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(frames, dict):
            raise ValueError("temporal result must contain previous/current visible states and frames")
        after_frame = frames.get("after")
        if not isinstance(after_frame, dict) or not isinstance(after_frame.get("image_sha256"), str):
            raise ValueError("temporal result after-frame fingerprint is missing")
        frame_sha = str(after_frame["image_sha256"])
        event = after.get("observed_action")
        if not isinstance(event, dict):
            raise ValueError("temporal result has no observed action")

        with self._lock:
            track = self._tracks.setdefault(track_id, _Track())
            if frame_sha in track.seen_after_frames:
                return {
                    "schema_version": "1.0.0",
                    "status": "duplicate_frame_ignored",
                    "changed": False,
                    "track_id": track_id,
                    "hand_id": track.hand_id,
                    "revision": track.revision if track.revision >= 0 else None,
                    "state": self._state(track, after) if track.hand_id is not None else None,
                    "recommendation": None,
                }

            boundary_audit: dict[str, object]
            if hand_start:
                evidence_ok = (
                    float(boundary_confidence) >= self.minimum_boundary_confidence
                    and before.get("street") == "preflop"
                    and before.get("board_cards") == []
                )
                boundary_audit = {
                    "status": "accepted" if evidence_ok else "rejected",
                    "kind": "new_hand",
                    "confidence": float(boundary_confidence),
                    "required_confidence": self.minimum_boundary_confidence,
                    "preflop": before.get("street") == "preflop",
                    "empty_board": before.get("board_cards") == [],
                }
                if not evidence_ok:
                    return self._abstain("hand_boundary_low_confidence", track_id, temporal_result, boundary_audit)
                track.hand_counter += 1
                track.hand_id = hashlib.sha256(
                    f"{track_id}:{track.hand_counter}:{frames['before']['image_sha256']}".encode("utf-8")
                ).hexdigest()[:24]
                track.revision = -1
                track.last_after = None
                track.history = []
                track.seen_after_frames = set()
            else:
                if track.hand_id is None or track.last_after is None:
                    return self._abstain("hand_boundary_unresolved", track_id, temporal_result)
                boundary_audit = _cross_pair_audit(track.last_after, before)
                boundary_audit["kind"] = "same_hand_progression"
                if boundary_audit["status"] != "accepted":
                    return self._abstain("cross_pair_transition_rejected", track_id, temporal_result, boundary_audit)

            track.history.append(copy.deepcopy(event))
            track.revision += 1
            current = copy.deepcopy(after)
            current["action_history"] = copy.deepcopy(track.history)
            current["history_complete"] = True
            current["hand_id"] = track.hand_id
            current["revision"] = track.revision
            track.last_after = current
            track.seen_after_frames.add(frame_sha)
            missing = ["player_identities"]
            if current.get("effective_stack_bb") is None:
                missing.append("effective_stack_bb")
            return {
                "schema_version": "1.0.0",
                "status": "state_tracked_incomplete_identity",
                "changed": True,
                "track_id": track_id,
                "hand_id": track.hand_id,
                "revision": track.revision,
                "state": copy.deepcopy(current),
                "boundary_audit": boundary_audit,
                "pair_transition_audit": copy.deepcopy(transition),
                "perception_evidence": {
                    "frames": copy.deepcopy(temporal_result.get("frames")),
                    "checkpoint_provenance": copy.deepcopy(temporal_result.get("checkpoint_provenance")),
                    "minimum_supported_confidence": temporal_result.get("minimum_supported_confidence"),
                    "event_confidence": copy.deepcopy(temporal_result.get("event_confidence")),
                },
                "missing_critical_fields": missing,
                "recommendation": None,
                "limitations": [
                    "Hand start evidence is supplied by a boundary detector and remains uncalibrated in the synthetic pipeline.",
                    "Player identities are unresolved; temporary seat aliases are not persistent profiles.",
                    "Multiway effective stacks remain opponent-specific and are not collapsed to a scalar.",
                ],
            }

    @staticmethod
    def _state(track: _Track, fallback: dict[str, object]) -> dict[str, object]:
        return copy.deepcopy(track.last_after if track.last_after is not None else fallback)

    @staticmethod
    def _abstain(
        reason: str,
        track_id: str,
        temporal_result: dict[str, object],
        audit: dict[str, object] | None = None,
    ) -> dict[str, object]:
        material = json.dumps(temporal_result.get("frames", {}), sort_keys=True, separators=(",", ":"))
        return {
            "schema_version": "1.0.0",
            "status": f"abstain_{reason}",
            "changed": False,
            "track_id": track_id,
            "observation_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
            "audit": audit,
            "state": None,
            "recommendation": None,
        }
