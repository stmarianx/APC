"""Verified spatial supervision, without inventing unannotated subregions.

Targets are dependency-free JSON-compatible records. Partial/uncertain objects
retain localization supervision but never supervise identity or numeric values.
Absent objects are not automatically negative labels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from apc.tools.validate_dataset import validate_annotation, validate_manifest

CLASSES = ("table", "seat", "hero_card", "board_card", "pot", "action_button", "turn_clock")
VISIBILITIES = ("clear", "partial", "occluded", "uncertain")


def encode_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    """Convert a complete verified annotation to normalized XYXY targets."""
    if annotation.get("use_policy") is not None:
        raise ValueError("Reference/audit annotations are not training inputs")
    errors = validate_annotation(annotation, require_image=False)
    if errors:
        raise ValueError("Invalid annotation: " + "; ".join(errors))
    if annotation["provenance"].get("verified") is not True:
        raise ValueError("Spatial supervision requires verified annotations")
    targets: list[dict[str, Any]] = []

    def add(kind: str, obj: dict[str, Any], attributes: dict[str, Any]) -> None:
        box = obj["box"]
        x, y, w, h = (float(box[key]) for key in ("x", "y", "width", "height"))
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("Spatial boxes must be finite")
        visibility = obj.get("visibility", "clear")
        if visibility not in VISIBILITIES:
            raise ValueError("Invalid visibility")
        clear = visibility == "clear"
        targets.append({
            "class_id": CLASSES.index(kind), "class_name": kind,
            "box_xyxy": [x, y, x + w, y + h],
            "visibility": visibility,
            "box_mask": visibility != "occluded",
            "attributes": attributes,
            "attribute_masks": {key: clear and value is not None
                                for key, value in attributes.items()},
        })

    objects = annotation["objects"]
    add("table", {"box": objects["table"]}, {})
    for seat in objects["seats"]:
        # Stack/name/dealer are seat attributes: no separate boxes exist in v1.
        add("seat", seat, {key: seat.get(key) for key in (
            "seat_no", "occupied", "is_hero", "has_dealer_button", "status",
            "player_name", "stack_bb")})
    for collection, kind in (("hero_cards", "hero_card"), ("board_cards", "board_card")):
        for index, card in enumerate(objects[collection]):
            rank, suit = card["rank"], card["suit"]
            if rank not in tuple("23456789TJQKA") + ("back", "unknown"):
                raise ValueError("Invalid card rank")
            if suit not in ("c", "d", "h", "s", "none", "unknown"):
                raise ValueError("Invalid card suit")
            add(kind, card, {"index": index,
                            "rank": rank if rank not in ("unknown", "back") else None,
                            "suit": suit if suit not in ("unknown", "none") else None})
            targets[-1]["recognition_regions"] = {
                key.removesuffix("_box"): {"box": dict(card[key]),
                                          "supervised": card["visibility"] == "clear" and targets[-1]["attribute_masks"][key.removesuffix("_box")]}
                for key in ("rank_box", "suit_box") if key in card}
    add("pot", objects["pot"], {"amount_bb": objects["pot"]["amount_bb"]})
    for button in objects["action_buttons"]:
        if button["action"] not in ("fold", "check", "call", "bet", "raise", "all_in"):
            raise ValueError("Invalid button action")
        add("action_button", button, {key: button.get(key) for key in ("action", "enabled", "amount_bb")})
    if objects.get("turn_clock") is not None:
        clock = objects["turn_clock"]
        add("turn_clock", clock, {"remaining_ms": clock["remaining_ms"]})
    return {"target_version": "1.0.0", "sample_id": annotation["sample_id"],
            "capture_session_id": annotation["capture_session_id"],
            "sequence_index": annotation["sequence_index"],
            "image": dict(annotation["image"]), "objects": targets,
            "absence_supervised": False}


def load_split(manifest_path: str | Path, split: str) -> list[dict[str, Any]]:
    """Validate image hashes/session isolation before loading a declared split."""
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown dataset split")
    path = Path(manifest_path).resolve()
    report = validate_manifest(path, require_images=True)
    if not report["valid"]:
        raise ValueError("Invalid dataset: " + "; ".join(report["errors"]))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sessions = set(manifest["splits"][split])
    result = []
    for filename in manifest["annotation_files"]:
        annotation_path = (path.parent / filename).resolve()
        item = json.loads(annotation_path.read_text(encoding="utf-8"))
        if item["capture_session_id"] in sessions:
            target = encode_annotation(item)
            target["image"]["path"] = str((annotation_path.parent / item["image"]["path"]).resolve())
            result.append(target)
    return sorted(result, key=lambda item: (item["capture_session_id"], item["sequence_index"], item["sample_id"]))
