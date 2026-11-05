"""Render model-independent card/glyph overlays for draft capture review."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from apc.perception.evaluate_browser_collection import load_collection


def pixel_box(box, size):
    return [round(v * d) for v, d in zip(box, size * 2)]


def render_review(collection, output):
    """Use the scoring loader's exact mapping, never predictions or auto-approval."""
    collection = Path(collection).resolve()
    data, frames = load_collection(collection)
    output = Path(output).resolve()
    if output == collection.parent or output in collection.parents:
        raise ValueError("Review output must not contain the source collection")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, frame in enumerate(frames):
        with Image.open(frame["image"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        targets = []
        for target in frame["targets"]:
            parent = pixel_box(target["box_xyxy"], image.size)
            regions = {key: pixel_box(box, image.size) for key, box in target["regions"].items()}
            draw.rectangle(parent, outline="#ffdf00", width=2)
            for key, color in (("rank", "#00ffff"), ("suit", "#ff50ff")):
                draw.rectangle(regions[key], outline=color, width=1)
            text = target["role"] + ":" + target["identity"]
            label_xy = (parent[0], max(0, parent[1] - 14))
            draw.text(label_xy, text, fill="#ffdf00", stroke_width=1, stroke_fill="black")
            targets.append({"identity": target["identity"], "role": target["role"],
                            "card_xyxy": parent, "glyph_xyxy": regions})
        name = f"{index:03d}-{frame['image'].stem}.png"
        image.save(output / name)
        rows.append({"source_image": str(frame["image"]),
                     "source_sha256": hashlib.sha256(frame["image"].read_bytes()).hexdigest(),
                     "overlay": name, "image_size": list(image.size), "targets": targets,
                     "review_status": "pending", "mapping": frame["screenshot_mapping"]})
    report = {"schema_version": "apc_browser_geometry_review_v1",
              "collection_sha256": hashlib.sha256(collection.read_bytes()).hexdigest(),
              "capture_session_id": data["capture_session_id"],
              "legend": {"card": "yellow", "rank": "cyan", "suit": "magenta"},
              "model_used": False, "labels_changed": False, "readiness_credit": False,
              "independently_reviewed": False, "frames": rows}
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("output", type=Path, help="New review folder; existing folders refused")
    args = parser.parse_args()
    result = render_review(args.collection, args.output)
    print(json.dumps({"frames": len(result["frames"]), "review_status": "pending"}))
