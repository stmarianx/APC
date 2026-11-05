"""Build one unapproved development frame from measured practice landmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from apc.perception.capture_stability import validate_capture_stability
from apc.perception.pixel_alignment import PixelAlignment


def prepare(capture, measurement, output):
    capture, measurement, output = map(lambda p: Path(p).resolve(), (capture, measurement, output))
    data = json.loads(capture.read_text(encoding="utf-8"))
    if data.get("use_policy") != "draft_review_no_training" or data.get("eligible_for_draft_scoring") is not True:
        raise ValueError("Only eligible unsealed development captures may be prepared")
    image = (capture.parent / data["image"]).resolve()
    if not image.is_relative_to(output.parent) or not image.is_relative_to(capture.parent):
        raise ValueError("Output must contain the original capture; image must remain inside acquisition")
    with Image.open(image) as opened:
        size = opened.size
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if data.get("sha256") != digest or data.get("image_size") != list(size):
        raise ValueError("Acquisition image binding mismatch")
    stability = data["capture_stability"]
    viewport = stability["before"]["viewport"]
    validate_capture_stability(stability, image_size=size, image_sha256=digest,
                               viewport=viewport, cards=data["cards"])
    measured = json.loads(measurement.read_text(encoding="utf-8"))
    alignment = PixelAlignment(measured, image_size=size, image_sha256=digest, viewport=viewport)
    declared = stability["before"]["alignment_landmarks"]
    if ([{k: a[k] for k in ("description", "dom_xy")} for a in measured["anchors"]] != declared):
        raise ValueError("Measured landmarks do not match captured DOM evidence")
    cards = stability["before"]["cards"]
    if not cards or len({tuple(c[4:]) for c in cards}) != 1:
        raise ValueError("Expected nonempty uniform practice card dimensions")
    card_size = cards[0][4:]
    # Broad upper/lower content bands of this owned fixture, not measured span
    # bounds or generic perception. Keep their draft status visible in review.
    offsets = {"rank_region_offset_xywh": [6,6,card_size[0]-12,(card_size[1]-12)/2],
               "suit_region_offset_xywh": [6,6+(card_size[1]-12)/2,card_size[0]-12,(card_size[1]-12)/2]}
    for _, _, x, y, w, h in cards:
        alignment.box(x,y,w,h)
        for dx,dy,rw,rh in offsets.values():
            alignment.box(x+dx,y+dy,rw,rh)
    row = {"image": os.path.relpath(image,output.parent).replace(os.sep,"/"), "sha256":digest,
           "hand_group":"alignment-setup-hand", "cards":data["cards"], "capture_stability":stability,
           "screenshot_mapping":"measured_pixel_alignment_v1", "pixel_alignment":measured}
    collection = {"schema_version":"apc_browser_card_development_v1", "draft_only":True,
                  "readiness_credit":False, "virtual_chips":True, "capture_stability_required":True,
                  "capture_session_id":"measured-alignment-demo-v1", "viewport":viewport,
                  "card_size":card_size, **offsets, "frames":[row]}
    with output.open("x",encoding="utf-8") as stream:
        json.dump(collection,stream,indent=2)
        stream.write("\n")
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("measurement")
    parser.add_argument("output")
    args = parser.parse_args()
    prepare(args.capture,args.measurement,args.output)
