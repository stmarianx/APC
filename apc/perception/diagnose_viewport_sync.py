"""Describe a fixed setup capture experiment; never select evaluation frames."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def summarize(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan["captures_per_viewport"] != 2:
        raise ValueError("Expected two declared captures per viewport")
    pairs = []
    expected = set()
    for viewport in plan["viewports"]:
        captures = []
        for ordinal in ("first", "second"):
            key = f"w{viewport[0]}-{ordinal}"
            expected.add(key)
            folder = root / key
            metadata = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
            name = metadata["image"]
            if name not in ("frame.jpg", "frame.png"):
                raise ValueError("Unexpected image filename")
            image = folder / name
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            with Image.open(image) as opened:
                size = list(opened.size)
            evidence = metadata["capture_stability"]
            if (digest != metadata["sha256"] or size != metadata["image_size"]
                    or digest != evidence["image_sha256"] or size != evidence["image_size"]):
                raise ValueError("Image binding mismatch")
            if any(evidence[end]["viewport"] != viewport for end in ("before", "after")):
                raise ValueError("Unexpected viewport")
            if metadata["use_policy"] != plan["use_policy"]:
                raise ValueError("Use policy mismatch")
            captures.append({"frame_key": key, "image_size": size, "sha256": digest,
                             "endpoint_geometry_equal": evidence["before"] == evidence["after"],
                             "bounds_check_passed": metadata["eligible_for_draft_scoring"],
                             "rejection_reason": metadata["rejection_reason"],
                             "snapshot": evidence["before"]})
        pairs.append({"viewport": viewport, "captures": captures,
                      "same_image_dimensions": captures[0]["image_size"] == captures[1]["image_size"],
                      "same_dom_geometry": captures[0]["snapshot"] == captures[1]["snapshot"]})
    actual = {p.parent.name for p in root.glob("*/capture.json")}
    if actual != expected:
        raise ValueError("Capture schedule mismatch")
    return {"experiment": plan["experiment"], "pairs": pairs,
            "dimension_disagreement_pairs": sum(not p["same_image_dimensions"] for p in pairs),
            "all_captures_retained": True, "model_inference_performed": False,
            "pixel_alignment_proven": False, "readiness_credit": False,
            "interpretation": "Equal DOM endpoints and a passing bounds check do not prove raster alignment. "
                              "Repeated captures are diagnostic evidence, not replacement audit samples."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
