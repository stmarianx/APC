from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path


D = Decimal
VISUAL_SCHEMA_VERSION = "1.0.0"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
REQUIRED_FIELDS = (
    "table_id",
    "hand_id",
    "game",
    "players",
    "hero_position",
    "effective_stack_bb",
    "pot_bb",
    "to_call_bb",
    "board",
    "hero_cards",
    "action_history",
    "legal_actions",
    "rake_model",
    "utility_model",
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _confidence(value: object, name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a confidence between zero and one")
    try:
        result = D(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a confidence between zero and one") from error
    if not result.is_finite() or not D("0") <= result <= D("1"):
        raise ValueError(f"{name} must be a confidence between zero and one")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class NormalizedBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_dict(cls, payload: object, name: str) -> "NormalizedBox":
        if not isinstance(payload, dict):
            raise ValueError(f"{name} must be an object")
        box = cls(
            _number(payload.get("x"), f"{name}.x"),
            _number(payload.get("y"), f"{name}.y"),
            _number(payload.get("width"), f"{name}.width"),
            _number(payload.get("height"), f"{name}.height"),
        )
        if (
            box.x < 0
            or box.y < 0
            or box.width <= 0
            or box.height <= 0
            or box.x + box.width > 1
            or box.y + box.height > 1
        ):
            raise ValueError(f"{name} must stay inside normalized image bounds")
        return box

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class VisualField:
    name: str
    value: object
    confidence: Decimal
    region: str


@dataclass(frozen=True)
class VisualObservation:
    provider: str
    provider_version: str
    profile_id: str
    frame_id: str
    image_sha256: str
    image_path: str | None
    image_bytes: int | None
    regions: dict[str, NormalizedBox]
    fields: dict[str, VisualField]

    @property
    def minimum_confidence(self) -> Decimal:
        return min(field.confidence for field in self.fields.values())

    @property
    def mean_confidence(self) -> Decimal:
        return sum(
            (field.confidence for field in self.fields.values()), D("0")
        ) / D(len(self.fields))

    @property
    def semantic_signature(self) -> str:
        values = {
            name: self.fields[name].value
            for name in sorted(self.fields)
        }
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def live_payload(self, revision: int) -> dict[str, object]:
        values = {name: self.fields[name].value for name in REQUIRED_FIELDS}
        return {
            "schema_version": "1.0.0",
            "table_id": values["table_id"],
            "hand_id": values["hand_id"],
            "revision": revision,
            "game": values["game"],
            "players": values["players"],
            "hero_position": values["hero_position"],
            "effective_stack_bb": values["effective_stack_bb"],
            "pot_bb": values["pot_bb"],
            "to_call_bb": values["to_call_bb"],
            "board": values["board"],
            "hero_cards": values["hero_cards"],
            "action_history": values["action_history"],
            "legal_actions": values["legal_actions"],
            "rake_model": values["rake_model"],
            "utility_model": values["utility_model"],
            "source": (
                f"visual_provider:{self.provider}:{self.provider_version}:"
                f"{self.profile_id}"
            ),
        }


@dataclass
class _VisualTrack:
    candidate_signature: str = ""
    candidate_frames: set[str] = field(default_factory=set)
    stable_signature: str = ""
    revision: int = -1


class VisualObservationAdapter:
    def __init__(
        self,
        *,
        minimum_confidence: Decimal | str = D("0.90"),
        stable_frames: int = 2,
    ) -> None:
        self.minimum_confidence = _confidence(
            minimum_confidence, "minimum_confidence"
        )
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        self.stable_frames = stable_frames
        self._tracks: dict[str, _VisualTrack] = {}
        self._lock = threading.RLock()

    def submit(
        self, payload: dict[str, object], *, base_path: str | Path | None = None
    ) -> dict[str, object]:
        observation = self._parse(payload, base_path=base_path)
        low = [
            name
            for name, field in observation.fields.items()
            if field.confidence < self.minimum_confidence
        ]
        evidence = self._evidence(observation)
        if low:
            return {
                "status": "low_confidence",
                "changed": False,
                "revision": None,
                "required_stable_frames": self.stable_frames,
                "observed_stable_frames": 0,
                "low_confidence_fields": sorted(low),
                "minimum_confidence": format(
                    observation.minimum_confidence, "f"
                ),
                "mean_confidence": format(observation.mean_confidence, "f"),
                "payload": None,
                "evidence": evidence,
            }

        table_id = str(observation.fields["table_id"].value)
        track_key = f"{table_id}:{observation.profile_id}"
        signature = observation.semantic_signature
        with self._lock:
            track = self._tracks.setdefault(track_key, _VisualTrack())
            if track.candidate_signature != signature:
                track.candidate_signature = signature
                track.candidate_frames = set()
            track.candidate_frames.add(observation.frame_id)
            observed_frames = len(track.candidate_frames)
            if observed_frames < self.stable_frames:
                return {
                    "status": "pending_stability",
                    "changed": False,
                    "revision": None,
                    "required_stable_frames": self.stable_frames,
                    "observed_stable_frames": observed_frames,
                    "low_confidence_fields": [],
                    "minimum_confidence": format(
                        observation.minimum_confidence, "f"
                    ),
                    "mean_confidence": format(
                        observation.mean_confidence, "f"
                    ),
                    "payload": None,
                    "evidence": evidence,
                }
            changed = track.stable_signature != signature
            if changed:
                track.revision += 1
                track.stable_signature = signature
            revision = track.revision
        return {
            "status": "state_ready",
            "changed": changed,
            "revision": revision,
            "required_stable_frames": self.stable_frames,
            "observed_stable_frames": observed_frames,
            "low_confidence_fields": [],
            "minimum_confidence": format(observation.minimum_confidence, "f"),
            "mean_confidence": format(observation.mean_confidence, "f"),
            "payload": observation.live_payload(revision),
            "evidence": evidence,
        }

    @staticmethod
    def _evidence(observation: VisualObservation) -> dict[str, object]:
        return {
            "provider": observation.provider,
            "provider_version": observation.provider_version,
            "profile_id": observation.profile_id,
            "frame_id": observation.frame_id,
            "image_sha256": observation.image_sha256,
            "image_path": observation.image_path,
            "image_bytes": observation.image_bytes,
            "regions": {
                name: box.to_dict()
                for name, box in sorted(observation.regions.items())
            },
            "field_confidence": {
                name: format(field.confidence, "f")
                for name, field in sorted(observation.fields.items())
            },
        }

    def _parse(
        self, payload: dict[str, object], *, base_path: str | Path | None
    ) -> VisualObservation:
        schema = str(payload.get("schema_version", VISUAL_SCHEMA_VERSION))
        if schema != VISUAL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported visual observation schema: {schema}")
        provider = _text(payload.get("provider"), "provider")
        provider_version = _text(
            payload.get("provider_version"), "provider_version"
        )
        frame = payload.get("frame")
        if not isinstance(frame, dict):
            raise ValueError("frame must be an object")
        frame_id = _text(frame.get("frame_id"), "frame.frame_id")
        image_path, image_sha256, image_bytes = self._image_evidence(
            frame, base_path=base_path
        )
        calibration = payload.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("calibration must be an object")
        profile_id = _text(
            calibration.get("profile_id"), "calibration.profile_id"
        )
        raw_regions = calibration.get("regions")
        if not isinstance(raw_regions, dict) or not raw_regions:
            raise ValueError("calibration.regions must be a non-empty object")
        regions = {
            _text(name, "calibration region name"): NormalizedBox.from_dict(
                box, f"calibration.regions.{name}"
            )
            for name, box in raw_regions.items()
        }
        raw_fields = payload.get("fields")
        if not isinstance(raw_fields, dict):
            raise ValueError("fields must be an object")
        missing = sorted(set(REQUIRED_FIELDS) - set(raw_fields))
        if missing:
            raise ValueError(f"Visual observation missing fields: {', '.join(missing)}")
        fields: dict[str, VisualField] = {}
        for name in REQUIRED_FIELDS:
            raw = raw_fields[name]
            if not isinstance(raw, dict) or "value" not in raw:
                raise ValueError(f"fields.{name} must contain value and confidence")
            region = _text(raw.get("region"), f"fields.{name}.region")
            if region not in regions:
                raise ValueError(
                    f"fields.{name}.region references unknown region: {region}"
                )
            fields[name] = VisualField(
                name,
                raw["value"],
                _confidence(raw.get("confidence"), f"fields.{name}.confidence"),
                region,
            )
        return VisualObservation(
            provider,
            provider_version,
            profile_id,
            frame_id,
            image_sha256,
            image_path,
            image_bytes,
            regions,
            fields,
        )

    @staticmethod
    def _image_evidence(
        frame: dict[str, object], *, base_path: str | Path | None
    ) -> tuple[str | None, str, int | None]:
        raw_path = frame.get("image_path")
        expected = str(frame.get("image_sha256", "")).lower().strip()
        if raw_path is None or str(raw_path).strip() == "":
            if len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise ValueError(
                    "frame.image_sha256 must be a 64-character hex digest when image_path is omitted"
                )
            return None, expected, None
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute() and base_path is not None:
            path = Path(base_path) / path
        path = path.resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"Visual evidence image does not exist: {path}")
        size = path.stat().st_size
        if size <= 0 or size > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Visual evidence image must contain 1 to {MAX_IMAGE_BYTES} bytes"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise ValueError("frame.image_sha256 does not match image_path")
        return str(path), digest, size
