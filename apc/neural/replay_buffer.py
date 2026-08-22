from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

from apc.neural.contract import validate_completed_hand_replay


SCHEMA_VERSION = "1.0.0"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _split(group_id: str) -> str:
    fraction = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    return "train" if fraction < 0.70 else "validation" if fraction < 0.85 else "test"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(_canonical(payload) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


class APCReplayBuffer:
    """Immutable completed-hand objects plus an atomic, versioned index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.objects = self.root / "objects"
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def _empty_manifest() -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_name": "APC",
            "buffer_kind": "content_addressed_completed_hand_replay",
            "revision": 0,
            "entries": [],
        }

    def _load(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"APC replay manifest is unreadable: {error}") from error
        report = self.validate(manifest_only=True)
        if not report["valid"]:
            raise ValueError("APC replay manifest is invalid: " + "; ".join(report["issues"]))
        return manifest

    def ingest(self, replay: dict[str, object]) -> dict[str, object]:
        report = validate_completed_hand_replay(replay)
        if not report["valid"]:
            raise ValueError("APC replay is ineligible: " + "; ".join(report["issues"]))
        fingerprint = str(report["replay_fingerprint"])
        manifest = self._load()
        entries = list(manifest["entries"])
        by_fingerprint = {str(entry["replay_fingerprint"]): entry for entry in entries}
        if fingerprint in by_fingerprint:
            object_path = self.root / str(by_fingerprint[fingerprint]["file"])
            if not object_path.exists() or hashlib.sha256(object_path.read_bytes()).hexdigest() != fingerprint:
                raise ValueError("existing APC replay object is missing or tampered")
            return {"added": False, "replay_fingerprint": fingerprint, "revision": manifest["revision"], "split": by_fingerprint[fingerprint]["split"]}
        identity = (str(replay["session_id"]), str(replay["hand_id"]))
        for entry in entries:
            if (str(entry["session_id"]), str(entry["hand_id"])) == identity:
                raise ValueError("session/hand identity already maps to different APC replay bytes")
        object_path = self.objects / f"{fingerprint}.json"
        if object_path.exists():
            if hashlib.sha256(object_path.read_bytes()).hexdigest() != fingerprint:
                raise ValueError("unindexed APC replay object is tampered")
        else:
            _atomic_bytes(object_path, _canonical(replay))
        if hashlib.sha256(object_path.read_bytes()).hexdigest() != fingerprint:
            object_path.unlink()
            raise RuntimeError("APC replay object fingerprint changed during write")
        reward = float(replay["completed_hand_feedback"]["hero_reward_bb"])
        event_count = int(report["event_count"])
        split = _split(str(replay["split_group_id"]))
        entry = {
            "replay_fingerprint": fingerprint,
            "file": object_path.relative_to(self.root).as_posix(),
            "session_id": replay["session_id"],
            "hand_id": replay["hand_id"],
            "split_group_id": replay["split_group_id"],
            "split": split,
            "event_count": event_count,
            "reward_bb": format(reward, ".12g"),
            "priority": format(1.0 + min(abs(reward), 100.0) / 100.0 + min(event_count, 64) / 64.0, ".12g"),
            "source_environment": replay["source_environment"],
        }
        entries.append(entry)
        entries.sort(key=lambda row: str(row["replay_fingerprint"]))
        updated = dict(manifest)
        updated["revision"] = int(manifest["revision"]) + 1
        updated["entries"] = entries
        updated["content_fingerprint"] = _sha256({key: value for key, value in updated.items() if key != "content_fingerprint"})
        _atomic_json(self.manifest_path, updated)
        return {"added": True, "replay_fingerprint": fingerprint, "revision": updated["revision"], "split": split}

    def sample_training_batch(
        self,
        limit: int,
        *,
        seed: int,
        incumbent_fingerprints: tuple[str, ...] = (),
        minimum_incumbent_fraction: float = 0.20,
    ) -> list[dict[str, object]]:
        if limit <= 0 or not 0 <= minimum_incumbent_fraction <= 1:
            raise ValueError("APC replay sample parameters are invalid")
        manifest = self._load()
        train = [entry for entry in manifest["entries"] if entry["split"] == "train"]
        if not train:
            return []
        incumbent_set = set(incumbent_fingerprints)
        incumbent = [entry for entry in train if entry["replay_fingerprint"] in incumbent_set]
        fresh = [entry for entry in train if entry["replay_fingerprint"] not in incumbent_set]
        incumbent_quota = min(len(incumbent), math.ceil(limit * minimum_incumbent_fraction))
        rng = random.Random(seed)

        def weighted(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
            ranked = sorted(
                rows,
                key=lambda row: (
                    -(rng.random() ** (1.0 / float(row["priority"]))),
                    str(row["replay_fingerprint"]),
                ),
            )
            return ranked[:count]

        selected = weighted(incumbent, incumbent_quota)
        selected.extend(weighted(fresh, min(limit - len(selected), len(fresh))))
        if len(selected) < min(limit, len(train)):
            selected_fingerprints = {row["replay_fingerprint"] for row in selected}
            remainder = [row for row in incumbent if row["replay_fingerprint"] not in selected_fingerprints]
            selected.extend(weighted(remainder, min(limit - len(selected), len(remainder))))
        return [json.loads((self.root / str(entry["file"])).read_text(encoding="utf-8")) for entry in selected]

    def validate(self, *, manifest_only: bool = False) -> dict[str, object]:
        issues = []
        if not self.manifest_path.exists():
            manifest = self._empty_manifest()
        else:
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                return {"schema_version": SCHEMA_VERSION, "valid": False, "issues": [f"manifest unreadable: {error}"]}
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("model_name") != "APC" or manifest.get("buffer_kind") != "content_addressed_completed_hand_replay":
            issues.append("replay manifest identity is invalid")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            issues.append("replay manifest entries are invalid")
            entries = []
        fingerprints = [str(entry.get("replay_fingerprint", "")) for entry in entries if isinstance(entry, dict)]
        if len(fingerprints) != len(entries) or len(fingerprints) != len(set(fingerprints)) or fingerprints != sorted(fingerprints):
            issues.append("replay manifest fingerprints are duplicated or unordered")
        group_splits: dict[str, str] = {}
        for index, entry in enumerate(entries):
            label = f"entry[{index}]"
            try:
                expected_split = _split(str(entry["split_group_id"]))
                if entry["split"] != expected_split:
                    issues.append(f"{label} split mismatch")
                prior = group_splits.setdefault(str(entry["split_group_id"]), str(entry["split"]))
                if prior != entry["split"]:
                    issues.append(f"{label} group leaks across splits")
                if not math.isfinite(float(entry["priority"])) or float(entry["priority"]) <= 0:
                    issues.append(f"{label} priority is invalid")
                if not manifest_only:
                    object_path = self.root / str(entry["file"])
                    payload = object_path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != entry["replay_fingerprint"]:
                        issues.append(f"{label} object fingerprint mismatch")
                    elif not validate_completed_hand_replay(json.loads(payload))["valid"]:
                        issues.append(f"{label} replay payload is invalid")
            except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                issues.append(f"{label} invalid: {error}")
        if self.manifest_path.exists():
            material = {key: value for key, value in manifest.items() if key != "content_fingerprint"}
            if manifest.get("content_fingerprint") != _sha256(material):
                issues.append("replay manifest content fingerprint mismatch")
        return {"schema_version": SCHEMA_VERSION, "valid": not issues, "issues": issues, "revision": manifest.get("revision", 0), "entries": len(entries)}
