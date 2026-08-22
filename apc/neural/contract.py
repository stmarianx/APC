from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


APC_MODEL_NAME = "APC"
SCHEMA_VERSION = "1.0.0"
ACTION_VOCABULARY = ("fold", "check", "call", "bet", "raise", "all_in")
OUTPUT_HEADS = {
    "perception_fields",
    "temporal_state_consistency",
    "legal_action_policy_logits",
    "action_value_bb",
    "state_value_bb",
    "opponent_tendency",
    "uncertainty_abstention",
}
EXPERIENCE_SOURCES = {
    "controlled_virtual_chips",
    "explicitly_permitted_virtual_chips",
    "offline_completed_hand_replay",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def validate_apc_neural_config(payload: dict[str, object]) -> dict[str, object]:
    issues = []
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("model_name") != APC_MODEL_NAME
        or payload.get("model_family") != "multimodal_temporal_neural_network"
        or payload.get("neural_network") is not True
        or payload.get("units") != "BB"
    ):
        issues.append("APC neural identity/schema/BB contract is invalid")
    framework = payload.get("framework", {})
    if (
        not isinstance(framework, dict)
        or framework.get("preferred") != "pytorch"
        or framework.get("dependency_install_required") is not True
    ):
        issues.append("APC framework contract is invalid")
    inputs = payload.get("inputs", {})
    try:
        action_mask = inputs["legal_action_mask"]
        input_valid = (
            inputs["visible_frame_sequence"]["enabled"] is True
            and inputs["canonical_state_sequence"]["enabled"] is True
            and inputs["canonical_state_sequence"]["chip_units"] == "BB"
            and inputs["canonical_state_sequence"][
                "opponent_private_cards_allowed_at_decision"
            ]
            is False
            and inputs["player_profile"]["uncertainty_features_required"] is True
            and action_mask["required"] is True
            and action_mask["action_vocabulary"] == list(ACTION_VOCABULARY)
            and set(action_mask["continuous_size_features"])
            == {"to_amount_bb", "pot_fraction"}
        )
    except (KeyError, TypeError):
        input_valid = False
    if not input_valid:
        issues.append("APC multimodal input/legal-action contract is invalid")
    architecture = payload.get("architecture", {})
    try:
        state_encoder = architecture["state_encoder"]
        hidden = int(state_encoder["hidden_dimension"])
        heads = int(state_encoder["attention_heads"])
        architecture_valid = (
            architecture["visual_encoder"]["kind"]
            == "compact_convolutional_encoder"
            and state_encoder["kind"] == "masked_transformer"
            and hidden > 0
            and heads > 0
            and hidden % heads == 0
            and set(architecture["output_heads"]) == OUTPUT_HEADS
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        architecture_valid = False
    if not architecture_valid:
        issues.append("APC neural architecture/output-head contract is invalid")
    training = payload.get("training", {})
    try:
        training_valid = (
            training["course_corpus_role"]
            == "explanation_retrieval_and_concept_auxiliary_only"
            and set(training["gto_label_sources"])
            == {"fingerprinted_solver_outputs", "reproducible_solver_self_play"}
            and "completed_live_observation_virtual_chip_hands"
            in training["experience_sources"]
            and training["group_exclusive_split_unit"]
            == "complete_hand_or_session"
            and training["replay_buffer"]["versioned"] is True
            and training["replay_buffer"]["content_addressed"] is True
        )
    except (KeyError, TypeError):
        training_valid = False
    if not training_valid:
        issues.append("APC training/replay provenance contract is invalid")
    live = payload.get("live_learning", {})
    try:
        live_valid = (
            live["experience_ingestion_during_session"] is True
            and live["profile_posterior_updates_during_session"] is True
            and live["policy_weight_updates_during_hand"] is False
            and live["candidate_training_after_completed_hands"] is True
            and live["candidate_evaluation_before_activation"] is True
            and live["automatic_promotion"] is False
            and live["rollback_required"] is True
        )
    except (KeyError, TypeError):
        live_valid = False
    if not live_valid:
        issues.append("APC live-learning isolation/promotion contract is invalid")
    inference = payload.get("inference", {})
    try:
        inference_valid = (
            inference["legal_action_mask_required"] is True
            and inference["auditable_bb_outputs_required"] is True
            and inference["uncertainty_abstention_required"] is True
            and inference["deadline_aware_fast_path_required"] is True
            and 0 < float(inference["target_strategy_p95_ms"]) <= 50
        )
    except (KeyError, TypeError, ValueError):
        inference_valid = False
    if not inference_valid:
        issues.append("APC inference/deadline contract is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not issues,
        "issues": issues,
        "model_name": payload.get("model_name"),
        "config_fingerprint": _fingerprint(payload),
    }


def load_apc_neural_config(path: str | Path | None = None) -> dict[str, object]:
    source = Path(path) if path is not None else Path(__file__).with_name("config_v1.json")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"APC neural config is unreadable: {error}") from error
    validation = validate_apc_neural_config(payload)
    if not validation["valid"]:
        raise ValueError("APC neural config is invalid: " + "; ".join(validation["issues"]))
    result = dict(payload)
    result["config_fingerprint"] = validation["config_fingerprint"]
    return result


def validate_completed_hand_replay(payload: dict[str, object]) -> dict[str, object]:
    issues = []
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("model_name") != APC_MODEL_NAME
        or payload.get("units") != "BB"
        or payload.get("full_hand_completed") is not True
        or payload.get("external_actuation") is not False
        or payload.get("source_environment") not in EXPERIENCE_SOURCES
    ):
        issues.append("APC completed-hand replay identity/scope is invalid")
    for field in ("session_id", "hand_id", "split_group_id", "source_fingerprint"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{field} must be non-empty")
    fingerprint = payload.get("source_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint.lower()
    ):
        issues.append("source_fingerprint must be SHA-256 hex")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        issues.append("completed-hand replay requires temporal events")
        events = []
    previous_time = -1
    for index, event in enumerate(events):
        label = f"event[{index}]"
        if not isinstance(event, dict):
            issues.append(f"{label} must be an object")
            continue
        try:
            observed_ms = int(event["observed_monotonic_ms"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{label} timestamp is invalid")
            continue
        if observed_ms <= previous_time:
            issues.append(f"{label} timestamp is not strictly increasing")
        previous_time = observed_ms
        if not isinstance(event.get("state_fingerprint"), str):
            issues.append(f"{label} state fingerprint is missing")
        legal = event.get("legal_action_keys")
        chosen = event.get("chosen_action_key")
        if (
            not isinstance(legal, list)
            or not legal
            or any(action not in ACTION_VOCABULARY for action in legal)
            or chosen not in legal
        ):
            issues.append(f"{label} legal/chosen action contract is invalid")
        state = event.get("canonical_state", {})
        if not isinstance(state, dict) or state.get("opponent_cards") is not None:
            issues.append(f"{label} leaks or lacks decision-state evidence")
        if state.get("units") != "BB":
            issues.append(f"{label} canonical state is not BB normalized")
    feedback = payload.get("completed_hand_feedback", {})
    try:
        reward = float(feedback["hero_reward_bb"])
        feedback_valid = (
            feedback["full_hand_completed"] is True
            and math.isfinite(reward)
            and -1000 <= reward <= 1000
        )
    except (KeyError, TypeError, ValueError):
        feedback_valid = False
    if not feedback_valid:
        issues.append("completed-hand BB feedback is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not issues,
        "issues": issues,
        "model_name": payload.get("model_name"),
        "event_count": len(events),
        "replay_fingerprint": _fingerprint(payload),
        "eligible_for_candidate_training": not issues,
    }
