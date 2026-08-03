from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ANALYSIS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ANALYSIS_DIR / "contact_sheets"
OUTPUT_DIR = ANALYSIS_DIR / "contact_sheet_gallery"


def main() -> int:
    sources = sorted(SOURCE_DIR.glob("*.jpg"))
    if not sources:
        raise SystemExit("No contact sheets found")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    width = 960
    sheet_height = 459
    label_height = 30
    font = ImageFont.load_default(size=16)
    for group_index in range(0, len(sources), 2):
        group = sources[group_index : group_index + 2]
        canvas = Image.new("RGB", (width, len(group) * (label_height + sheet_height)), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for item_index, source in enumerate(group):
            y = item_index * (label_height + sheet_height)
            draw.text((8, y + 6), source.stem, fill=(245, 245, 245), font=font)
            with Image.open(source) as image:
                resized = image.convert("RGB").resize((width, sheet_height), Image.Resampling.LANCZOS)
            canvas.paste(resized, (0, y + label_height))
        output = OUTPUT_DIR / f"gallery-{group_index // 2 + 1:02d}.jpg"
        canvas.save(output, quality=88, optimize=True)
        print(output.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
