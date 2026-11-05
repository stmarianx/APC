"""Import image-bound, stable browser acquisitions as unverified card suggestions."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from PIL import Image

from apc.annotator.project import AnnotationProject
from apc.perception.capture_stability import validate_capture_stability
from apc.perception.pixel_alignment import PixelAlignment, validate_alignment_landmarks


def import_acquisition(project, manifest, session, timestamp_ms, *, pixel_alignment=None):
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
    if not isinstance(name, str) or Path(name).name != name or Path(name).suffix.lower() not in {".png", ".jpg"}:
        raise ValueError("Acquisition image must be a local PNG/JPEG filename")
    image_path = (manifest.parent / name).resolve()
    if image_path.parent != manifest.parent:
        raise ValueError("Acquisition image escapes manifest folder")
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != data.get("sha256"):
        raise ValueError("Acquisition image hash mismatch")
    with Image.open(image_path) as image:
        expected_format = "PNG" if image_path.suffix.lower() == ".png" else "JPEG"
        if (image.format != expected_format or list(image.size) != data.get("image_size")
                or data.get("image_format", expected_format) != expected_format):
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
    alignment, measured, measurement_digest = None, None, None
    if pixel_alignment is not None:
        measurement_raw = Path(pixel_alignment).read_bytes()
        measured = json.loads(measurement_raw)
        measurement_digest = hashlib.sha256(measurement_raw).hexdigest()
        alignment = PixelAlignment(measured, image_sha256=data["sha256"],
            image_size=(image_width,image_height), viewport=viewport)
        validate_alignment_landmarks(measured,evidence["before"])
    objects, labels = {"hero_cards": [], "board_cards": []}, {"hero_cards": [], "board": []}
    identities = []
    for observed in evidence["before"]["cards"]:
        role, identity, x, y, w, h = observed
        if (role not in {"hero", "board"} or not isinstance(identity, str) or len(identity) != 2
                or identity[0] not in "23456789TJQKA" or identity[1] not in "shdc"):
            raise ValueError("Invalid visible card identity/role")
        if alignment is not None:
            x1,y1,x2,y2 = alignment.box(x,y,w,h)
        else:
            if not (0 <= x < x+w <= width and 0 <= y < y+h <= extent_height):
                raise ValueError("Card outside saved screenshot")
            x1,y1,x2,y2 = x/width,y/extent_height,(x+w)/width,(y+h)/extent_height
        key = "hero_cards" if role == "hero" else "board_cards"
        objects[key].append({"box":{"x":x1,"y":y1,"width":x2-x1,"height":y2-y1}})
        labels["hero_cards" if role == "hero" else "board"].append(identity)
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate visible identity")
    public_labels = None
    if "public_regions" in evidence["before"]:
        from apc.annotator.public_table_labels import public_table_suggestions
        public_labels = public_table_suggestions(evidence["before"]["public_regions"],alignment)
        objects.update(public_labels[0])
    # All capture integrity/geometry checks precede project mutations.
    if expected_format == "JPEG":
        # Dataset v1 requires PNG. Preserve original acquisition/hash; import a
        # lossless encoding of decoded RGB pixels and record the derivation.
        with tempfile.TemporaryDirectory(prefix="apc-jpeg-import-") as directory:
            derived = Path(directory) / "decoded.png"
            with Image.open(image_path) as image:
                image.convert("RGB").save(derived,format="PNG")
            record, inserted = project.import_frame(derived, capture_session_id=session, timestamp_ms=timestamp_ms)
    else:
        record, inserted = project.import_frame(image_path, capture_session_id=session, timestamp_ms=timestamp_ms)
    if project.load_annotation(record.sample_id) is not None or project.load_suggestion(record.sample_id) is not None:
        raise ValueError("Existing review work preserved")
    suggestion = {**project.annotation_template(record.sample_id),
        "kind":"apc_capture_metadata_suggestion", "review_required":True,
        "capture_import_version":"stable_browser_acquisition_v1",
        "auto_applied":False, "model_status":"visible_dom_draft_not_model_prediction",
        "metadata_sha256":hashlib.sha256(raw).hexdigest(),
        "source_image_sha256":data["sha256"], "source_image_format":expected_format,
        "image_derivation":"decoded_rgb_lossless_png" if expected_format == "JPEG" else "original_png",
        "screenshot_binding":"hash_dimensions_and_before_after_geometry_validated",
        "capture_stability":evidence, "suggested_objects":objects, "draft_card_labels":labels,
        "pixel_geometry_reviewed":False, "readiness_counts_added":False}
    suggestion["screenshot_mapping"] = ("measured_pixel_alignment_v1" if alignment is not None
                                         else "uniform_width_scale_top_left_crop")
    if measured is not None:
        suggestion["pixel_alignment"] = measured
        suggestion["pixel_alignment_metadata_sha256"] = measurement_digest
        suggestion["capture_import_version"] = "measured_browser_acquisition_v1"
    if public_labels is not None:
        suggestion["suggested_visible_state"] = public_labels[1]
        suggestion["skipped_unmeasured_public_regions"] = public_labels[2]
        suggestion["public_label_source"] = "owned_practice_visible_dom_v1_not_model_prediction"
    path = project.save_suggestion(record.sample_id, suggestion)
    return {"sample_id":record.sample_id,"inserted":inserted,"suggestion":str(path),"verified":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--timestamp-ms", type=int, required=True)
    parser.add_argument("--pixel-alignment", type=Path,
                        help="Image-bound manual landmark measurements; still requires annotation review")
    args = parser.parse_args()
    print(json.dumps(import_acquisition(AnnotationProject(args.project), args.manifest,
        args.session,args.timestamp_ms,pixel_alignment=args.pixel_alignment),indent=2))
