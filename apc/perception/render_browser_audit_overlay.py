"""Render draft card boxes for visual geometry QA, never training labels."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from apc.perception.evaluate_browser_collection import load_collection


def render(collection, frame_index, output):
    _, frames = load_collection(collection)
    frame = frames[frame_index]
    with Image.open(frame["image"]) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for target in frame["targets"]:
        coords = [round(v*d) for v, d in zip(target["box_xyxy"], canvas.size*2)]
        draw.rectangle(coords, outline="#ff00ff", width=1)
        for kind in ("rank", "suit"):
            bounds = [round(v*d) for v, d in zip(target["regions"][kind], canvas.size*2)]
            draw.rectangle(bounds, outline="#00ffff", width=1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection")
    parser.add_argument("output")
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()
    print(render(args.collection, args.frame_index, args.output))
