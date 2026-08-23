from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apc.neural.contract import ACTION_VOCABULARY, validate_completed_hand_replay
from apc.neural.features import (
    ACTION_INDEX,
    PROFILE_FEATURE_DIMENSION,
    PROFILE_FEATURE_NAMES,
    PROFILE_FEATURE_SCHEMA_VERSION,
    STATE_TOKEN_COUNT,
    STATE_TOKEN_DIMENSION,
    encode_state,
)
from apc.neural.replay_buffer import APCReplayBuffer


SCHEMA_VERSION = "1.0.0"
DEFAULT_MAX_EVENTS = 16


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ReplayTemporalCorpus:
    """Decision windows built only from validated, completed APC hands."""

    state_tokens: np.ndarray
    state_padding_mask: np.ndarray
    profile_features: np.ndarray
    modality_available: np.ndarray
    legal_action_mask: np.ndarray
    chosen_action_index: np.ndarray
    chosen_size_features: np.ndarray
    target_return_bb: np.ndarray
    temporal_consistency_target: np.ndarray
    split: np.ndarray
    replay_fingerprints: tuple[str, ...]
    decision_ids: tuple[str, ...]
    manifest: dict[str, object]

    def indices(self, split: str) -> np.ndarray:
        if split not in {"train", "validation", "test"}:
            raise ValueError("APC replay split is invalid")
        return np.flatnonzero(self.split == split)


def _profile(event: dict[str, object]) -> tuple[np.ndarray, bool]:
    raw = event.get("player_profile_features")
    if raw is None:
        return np.zeros(PROFILE_FEATURE_DIMENSION, dtype=np.float32), False
    if not isinstance(raw, list) or len(raw) != PROFILE_FEATURE_DIMENSION:
        raise ValueError("APC replay player profile must contain exactly eight features")
    values = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("APC replay player profile must contain finite unit-interval values")
    return values, True


def _size_features(event: dict[str, object], state: dict[str, object]) -> np.ndarray:
    command = event.get("chosen_action")
    if command is None:
        return np.zeros(2, dtype=np.float32)
    if not isinstance(command, dict):
        raise ValueError("APC replay chosen action payload is invalid")
    amount = command.get("to_amount_bb", command.get("amount_bb", 0))
    try:
        amount_bb = float(str(amount))
        pot_bb = float(str(state.get("pot_bb", 0)))
    except (TypeError, ValueError) as error:
        raise ValueError("APC replay chosen action size is not BB numeric") from error
    if not math.isfinite(amount_bb) or amount_bb < 0 or not math.isfinite(pot_bb) or pot_bb < 0:
        raise ValueError("APC replay chosen action size is outside the BB domain")
    return np.asarray((math.tanh(amount_bb / 25.0), math.tanh((amount_bb / pot_bb if pot_bb else 0.0) / 2.0)), dtype=np.float32)


def encode_completed_hand_replays(
    rows: list[tuple[dict[str, object], str]], *, max_events: int = DEFAULT_MAX_EVENTS
) -> ReplayTemporalCorpus:
    if max_events <= 0:
        raise ValueError("APC replay temporal window must be positive")
    samples: list[dict[str, object]] = []
    source_rows = []
    for replay, split in rows:
        validation = validate_completed_hand_replay(replay)
        if not validation["valid"]:
            raise ValueError("APC replay adapter rejected hand: " + "; ".join(validation["issues"]))
        if split not in {"train", "validation", "test"}:
            raise ValueError("APC replay adapter received an invalid split")
        fingerprint = str(validation["replay_fingerprint"])
        reward = float(replay["completed_hand_feedback"]["hero_reward_bb"])
        encoded_events: list[tuple[np.ndarray, np.ndarray]] = []
        for event_index, event in enumerate(replay["events"]):
            state = event["canonical_state"]
            tokens, padding = encode_state(state)
            encoded_events.append((tokens, padding))
            window = encoded_events[-max_events:]
            temporal_tokens = np.zeros((max_events, STATE_TOKEN_COUNT, STATE_TOKEN_DIMENSION), dtype=np.float32)
            temporal_padding = np.ones((max_events, STATE_TOKEN_COUNT), dtype=np.bool_)
            for time_index, (event_tokens, event_padding) in enumerate(window):
                temporal_tokens[time_index] = event_tokens
                temporal_padding[time_index] = event_padding
                visible = ~event_padding
                temporal_tokens[time_index, visible, 3] = (time_index + 1) / max_events
            legal = np.zeros(len(ACTION_VOCABULARY), dtype=np.bool_)
            for action in event["legal_action_keys"]:
                legal[ACTION_INDEX[str(action)]] = True
            chosen = ACTION_INDEX[str(event["chosen_action_key"])]
            profile, profile_available = _profile(event)
            samples.append({
                "tokens": temporal_tokens,
                "padding": temporal_padding,
                "profile": profile,
                "modalities": np.asarray((False, True, profile_available), dtype=np.bool_),
                "legal": legal,
                "action": chosen,
                "sizes": _size_features(event, state),
                "target": reward,
                "split": split,
                "fingerprint": fingerprint,
                "decision_id": f"{fingerprint}:{event_index}",
            })
        source_rows.append({"replay_fingerprint": fingerprint, "split": split, "events": len(replay["events"])})
    if not samples:
        raise ValueError("APC replay adapter requires at least one completed decision")
    manifest_material = {
        "schema_version": SCHEMA_VERSION,
        "model_name": "APC",
        "adapter_kind": "completed_hand_replay_to_temporal_neural_tensor",
        "units": "BB",
        "max_events": max_events,
        "feature_shape": [max_events, STATE_TOKEN_COUNT, STATE_TOKEN_DIMENSION],
        "profile_feature_schema_version": PROFILE_FEATURE_SCHEMA_VERSION,
        "profile_feature_names": list(PROFILE_FEATURE_NAMES),
        "completed_hands": len(source_rows),
        "decisions": len(samples),
        "sources": sorted(source_rows, key=lambda row: str(row["replay_fingerprint"])),
        "opponent_private_cards_used": False,
    }
    manifest = dict(manifest_material)
    manifest["adapter_fingerprint"] = _sha256(manifest_material)
    return ReplayTemporalCorpus(
        state_tokens=np.stack([row["tokens"] for row in samples]),
        state_padding_mask=np.stack([row["padding"] for row in samples]),
        profile_features=np.stack([row["profile"] for row in samples]),
        modality_available=np.stack([row["modalities"] for row in samples]),
        legal_action_mask=np.stack([row["legal"] for row in samples]),
        chosen_action_index=np.asarray([row["action"] for row in samples], dtype=np.int64),
        chosen_size_features=np.stack([row["sizes"] for row in samples]),
        target_return_bb=np.asarray([row["target"] for row in samples], dtype=np.float32),
        temporal_consistency_target=np.ones(len(samples), dtype=np.float32),
        split=np.asarray([row["split"] for row in samples]),
        replay_fingerprints=tuple(str(row["fingerprint"]) for row in samples),
        decision_ids=tuple(str(row["decision_id"]) for row in samples),
        manifest=manifest,
    )


def load_replay_temporal_corpus(
    buffer_or_path: APCReplayBuffer | str | Path, *, max_events: int = DEFAULT_MAX_EVENTS
) -> ReplayTemporalCorpus:
    buffer = buffer_or_path if isinstance(buffer_or_path, APCReplayBuffer) else APCReplayBuffer(buffer_or_path)
    report = buffer.validate()
    if not report["valid"]:
        raise ValueError("APC replay buffer is invalid: " + "; ".join(report["issues"]))
    manifest = buffer._load()
    rows = [
        (json.loads((buffer.root / str(entry["file"])).read_text(encoding="utf-8")), str(entry["split"]))
        for entry in manifest["entries"]
    ]
    corpus = encode_completed_hand_replays(rows, max_events=max_events)
    corpus.manifest["replay_buffer_content_fingerprint"] = manifest.get("content_fingerprint")
    return corpus
