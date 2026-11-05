"""Import image-bound, stable browser acquisitions as unverified card suggestions."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from apc.annotator.project import AnnotationProject
from apc.perception.capture_stability import validate_capture_stability


def import_acquisition(project, manifest, session, timestamp_ms):
    manifest = Path(manifest).resolve()
    raw = manifest.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != "apc_browser_frame_acquisition_v1":
        raise ValueError("Expected browser frame acquisition manifest")
    if data.get("use_policy") == "sealed_audit_no_training":
        raise ValueError("Sealed audit captures must not enter the labeling/training project")
    if (data.get("eligible_for_draft_scoring") is not True
            or data.get("rejection_reason") is not None or data.get("readiness_credit") is not False):
        raise ValueError("Rejected or malformed acquisition cannot be imported")
    if project.config["environment"]["provider_id"] != "apc-full-hand-browser":
        raise ValueError("Use a dedicated apc-full-hand-browser annotation project")
    name = data.get("image")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".png"):
        raise ValueError("Acquisition image must be a local PNG filename")
    image_path = (manifest.parent / name).resolve()
    if image_path.parent != manifest.parent:
        raise ValueError("Acquisition image escapes manifest folder")
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != data.get("sha256"):
        raise ValueError("Acquisition image hash mismatch")
    with Image.open(image_path) as image:
        if image.format != "PNG" or list(image.size) != data.get("image_size"):
            raise ValueError("Acquisition image dimensions/format mismatch")
        image_width, image_height = image.size
    evidence = data.get("capture_stability")
    viewport = evidence.get("before", {}).get("viewport", []) if isinstance(evidence, dict) else []
    cards = data.get("cards")
    if not isinstance(cards, list):
        raise ValueError("Missing acquisition labels")
    validate_capture_stability(evidence, image_size=(image_width,image_height),
        image_sha256=data["sha256"], viewport=viewport, cards=cards)
    width, _ = viewport
    extent_height = image_height * width / image_width
    objects, labels = {"hero_cards": [], "board_cards": []}, {"hero_cards": [], "board": []}
    identities = []
    for observed in evidence["before"]["cards"]:
        role, identity, x, y, w, h = observed
        if (role not in {"hero", "board"} or not isinstance(identity, str) or len(identity) != 2
                or identity[0] not in "23456789TJQKA" or identity[1] not in "shdc"):
            raise ValueError("Invalid visible card identity/role")
        if not (0 <= x < x+w <= width and 0 <= y < y+h <= extent_height):
            raise ValueError("Card outside saved screenshot")
        key = "hero_cards" if role == "hero" else "board_cards"
        objects[key].append({"box":{"x":x/width,"y":y/extent_height,
                                     "width":w/width,"height":h/extent_height}})
        labels["hero_cards" if role == "hero" else "board"].append(identity)
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate visible identity")
    # All capture integrity/geometry checks precede project mutations.
    record, inserted = project.import_frame(image_path, capture_session_id=session, timestamp_ms=timestamp_ms)
    if project.load_annotation(record.sample_id) is not None or project.load_suggestion(record.sample_id) is not None:
        raise ValueError("Existing review work preserved")
    suggestion = {**project.annotation_template(record.sample_id),
        "kind":"apc_capture_metadata_suggestion", "review_required":True,
        "capture_import_version":"stable_browser_acquisition_v1",
        "auto_applied":False, "model_status":"visible_dom_draft_not_model_prediction",
        "metadata_sha256":hashlib.sha256(raw).hexdigest(),
        "screenshot_binding":"hash_dimensions_and_before_after_geometry_validated",
        "capture_stability":evidence, "suggested_objects":objects, "draft_card_labels":labels,
        "pixel_geometry_reviewed":False, "readiness_counts_added":False}
    path = project.save_suggestion(record.sample_id, suggestion)
    return {"sample_id":record.sample_id,"inserted":inserted,"suggestion":str(path),"verified":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--timestamp-ms", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(import_acquisition(AnnotationProject(args.project), args.manifest,
        args.session,args.timestamp_ms),indent=2))
