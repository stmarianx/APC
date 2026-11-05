"""Bracket a browser screenshot with independently obtained visible geometry.

The browser adapter supplies async observe()/screenshot() callbacks. Observation
must return visible-only viewport, scroll and card geometry per CAPTURE_STABILITY.
No game actions, hidden state, model predictions or automatic retries are used.
"""
import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from apc.perception.capture_stability import validate_capture_stability


async def capture_browser_frame(observe, screenshot):
    """Return retained PNG/evidence plus eligibility; caller persists even rejects.

    Callback errors propagate: no frame is invented when acquisition fails.
    A valid PNG with unstable or malformed geometry is retained but ineligible.
    This endpoint check is not a pixel-review or readiness certification.
    """
    before = copy.deepcopy(await observe())
    png = bytes(await screenshot())
    after = copy.deepcopy(await observe())
    with Image.open(io.BytesIO(png)) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Browser collector requires PNG or JPEG screenshots")
        image_format = image.format
        image.load()
        size = list(image.size)
    digest = hashlib.sha256(png).hexdigest()
    evidence = {"schema_version": "apc_capture_stability_v1",
                "image_sha256": digest, "image_size": size,
                "before": before, "after": after}
    cards = ([row[:4] for row in before["cards"]]
             if isinstance(before, dict) and isinstance(before.get("cards"), list)
             and all(isinstance(row, list) for row in before["cards"]) else [])
    try:
        validate_capture_stability(evidence, image_size=size, image_sha256=digest,
            viewport=before.get("viewport", []) if isinstance(before, dict) else [], cards=cards)
        rejection = None
    except ValueError as error:
        rejection = str(error)
    return {"png": png, "sha256": digest, "image_size": size, "image_format": image_format,
            "capture_stability": evidence, "cards": cards,
            "eligible_for_draft_scoring": rejection is None,
            "rejection_reason": rejection, "pixel_geometry_reviewed": False,
            "readiness_credit": False}


def save_browser_frame(result, folder):
    """Retain one acquisition in a NEW folder, including rejected evidence.

    Existing folders are never overwritten. A caller must check the manifest's
    eligibility before adding this frame to a draft collection.
    """
    png = result["png"]
    if hashlib.sha256(png).hexdigest() != result["sha256"]:
        raise ValueError("Capture bytes changed before saving")
    metadata = {key: value for key, value in result.items() if key != "png"}
    metadata["image"] = "frame.jpg" if result.get("image_format") == "JPEG" else "frame.png"
    metadata["schema_version"] = "apc_browser_frame_acquisition_v1"
    serialized = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / metadata["image"]).write_bytes(png)
    (folder / "capture.json").write_text(serialized, encoding="utf-8")
    return folder / "capture.json"
