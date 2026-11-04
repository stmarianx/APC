"""Import screenshot/DOM metadata pairs as unverified review suggestions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from apc.annotator.project import AnnotationProject


def suggested_objects(metadata):
    if metadata.get("source") != "apc_full_hand_browser" or metadata.get("draft_only") is not True or metadata.get("units") != "BB":
        raise ValueError("Expected draft APC browser metadata in BB")
    viewport = metadata["viewport"]
    width, height = float(viewport["width"]), float(viewport["height"])
    if not all(math.isfinite(v) and v > 0 for v in (width,height)):
        raise ValueError("Invalid viewport")
    objects, skipped = {"hero_cards": [], "board_cards": [], "action_buttons": []}, []
    for region in metadata["regions"]:
        identifier, box = region["id"], region["box"]
        x,y,w,h = [float(box[key]) for key in ("x","y","width","height")]
        if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
            raise ValueError("Invalid region geometry")
        if x < 0 or y < 0 or x+w > width+1e-6 or y+h > height+1e-6:
            skipped.append(identifier)
            continue
        normalized = {"x":x/width,"y":y/height,"width":w/width,"height":h/height}
        if identifier == "table":
            objects["table"] = normalized
        elif identifier == "pot":
            objects["pot"] = {"box":normalized}
        else:
            for prefix,collection in (("heroCards-","hero_cards"),("board-","board_cards"),("buttons-","action_buttons")):
                if identifier.startswith(prefix):
                    index = int(identifier[len(prefix):])
                    if index != len(objects[collection]):
                        skipped.append(identifier)
                        break
                    objects[collection].append({"box":normalized})
                    break
    return objects, skipped


def import_capture(project, image, metadata_path, session, timestamp_ms):
    raw = Path(metadata_path).read_bytes()
    metadata = json.loads(raw)
    objects, skipped = suggested_objects(metadata)
    if project.config["environment"]["provider_id"] != "apc-full-hand-browser":
        raise ValueError("Use a dedicated apc-full-hand-browser annotation project")
    record, inserted = project.import_frame(image, capture_session_id=session, timestamp_ms=timestamp_ms)
    state = metadata["public_state"]
    suggestion = {**project.annotation_template(record.sample_id),
        "kind":"apc_capture_metadata_suggestion", "review_required":True, "auto_applied":False,
        "model_status":"browser_metadata_draft_not_model_prediction",
        "metadata_sha256":hashlib.sha256(raw).hexdigest(),
        "screenshot_binding":"user_supplied_pair_not_independently_verified",
        "suggested_objects":objects, "skipped_offscreen_or_noncontiguous_regions":skipped,
        "suggested_visible_state":{key:state[key] for key in ("street","pot_bb","to_call_bb") if key in state},
        "draft_card_labels":{key:state.get(key,[]) for key in ("hero_cards","board")},
        "readiness_counts_added":False}
    if project.load_annotation(record.sample_id) is not None or project.load_suggestion(record.sample_id) is not None:
        raise ValueError("Frame already has review work; existing annotation/suggestion preserved")
    path = project.save_suggestion(record.sample_id, suggestion)
    return {"sample_id":record.sample_id,"inserted":inserted,"suggestion":str(path),"verified":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("project","image","metadata"):
        parser.add_argument(argument,type=Path)
    parser.add_argument("--session",required=True)
    parser.add_argument("--timestamp-ms",type=int,required=True)
    args = parser.parse_args()
    print(json.dumps(import_capture(AnnotationProject(args.project),args.image,args.metadata,args.session,args.timestamp_ms),indent=2))
