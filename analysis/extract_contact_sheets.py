from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "contact_sheets"


def slug_for(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def frame_at(path: Path, seconds: float) -> Image.Image:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(seconds * av.time_base), stream=None, any_frame=False, backward=True)
        chosen = None
        for frame in container.decode(stream):
            chosen = frame
            frame_seconds = float(frame.time or 0.0)
            if frame_seconds >= seconds:
                break
        if chosen is None:
            raise RuntimeError(f"Could not decode {path.name} at {seconds:.2f}s")
        image = chosen.to_image().convert("RGB")
    return image


def fit_frame(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (20, 20, 20))
    canvas.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return canvas


def build_sheet(path: Path, sample_count: int) -> Path:
    with av.open(str(path)) as container:
        duration = float((container.duration or 0) / av.time_base)
    if duration <= 0:
        raise RuntimeError(f"No duration available for {path.name}")

    times = [duration * (index + 1) / (sample_count + 1) for index in range(sample_count)]
    tile_size = (320, 180)
    label_h = 24
    columns = 4
    rows = (sample_count + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size[0], rows * (tile_size[1] + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)

    for index, seconds in enumerate(times):
        image = fit_frame(frame_at(path, seconds), tile_size)
        x = (index % columns) * tile_size[0]
        y = (index // columns) * (tile_size[1] + label_h)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y + tile_size[1], x + tile_size[0], y + tile_size[1] + label_h), fill=(20, 20, 20))
        draw.text((x + 8, y + tile_size[1] + 3), format_time(seconds), fill=(245, 245, 245), font=font)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{slug_for(path)}.jpg"
    sheet.save(output, quality=88, optimize=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create evenly sampled contact sheets for local videos.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=38)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    videos = sorted(
        (path for path in ROOT.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name.lower(),
    )
    selected = videos[max(args.start - 1, 0) : min(args.end, len(videos))]
    for ordinal, path in enumerate(selected, start=args.start):
        output = OUTPUT_DIR / f"{slug_for(path)}.jpg"
        if output.exists() and not args.overwrite:
            print(f"SKIP {ordinal:02d}/38 {path.name}", flush=True)
            continue
        written = build_sheet(path, args.samples)
        print(f"DONE {ordinal:02d}/38 {written.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
