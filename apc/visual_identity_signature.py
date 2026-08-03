from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from apc.player_identity import NameObservation


SIGNATURE_WIDTH = 128
SIGNATURE_HEIGHT = 16


def _pil_numpy() -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Visual identity signatures require Pillow and NumPy; use the bundled workspace Python"
        ) from error
    return Image, np


def _box_pixels(box: dict[str, object], width: int, height: int) -> tuple[int, int, int, int]:
    try:
        x = float(box["x"])
        y = float(box["y"])
        w = float(box["width"])
        h = float(box["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("seat box must contain normalized x, y, width and height") from error
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        raise ValueError("seat box coordinates must be finite")
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
        raise ValueError("seat box must stay within normalized image bounds")
    left, top = round(x * width), round(y * height)
    right, bottom = round((x + w) * width), round((y + h) * height)
    if right - left < 20 or bottom - top < 20:
        raise ValueError("seat box is too small for a visual identity signature")
    return left, top, right, bottom


@dataclass(frozen=True)
class VisualSeatSignature:
    seat_no: int
    signature_sha256: str
    quality_score: float
    frame_sha256: str
    foreground_pixels: int
    normalized_shape: tuple[int, int] = (SIGNATURE_HEIGHT, SIGNATURE_WIDTH)

    def __post_init__(self) -> None:
        if isinstance(self.seat_no, bool) or not 1 <= self.seat_no <= 10:
            raise ValueError("seat_no must be between 1 and 10")
        for value, name in (
            (self.signature_sha256, "signature_sha256"),
            (self.frame_sha256, "frame_sha256"),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not math.isfinite(self.quality_score) or not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be between zero and one")

    @property
    def visual_token(self) -> str:
        return f"visual:{self.signature_sha256[:32]}"

    def registry_observation(self, *, observed_at_ms: int) -> NameObservation:
        """Create a pseudonymous registry observation; this is explicitly not OCR text."""
        return NameObservation(
            self.seat_no,
            self.visual_token,
            self.quality_score,
            self.frame_sha256,
            observed_at_ms,
        )


def extract_visual_seat_signature(
    image_path: str | Path,
    seat: dict[str, object],
) -> VisualSeatSignature:
    """Hash a normalized central name-band glyph mask without interpreting its text."""
    Image, np = _pil_numpy()
    source = Path(image_path).expanduser().resolve()
    raw = source.read_bytes()
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    pixels = np.asarray(image)
    return _extract_visual_seat_signature(
        pixels,
        seat,
        frame_sha256=hashlib.sha256(raw).hexdigest(),
        Image=Image,
        np=np,
    )


def _extract_visual_seat_signature(
    pixels: Any,
    seat: dict[str, object],
    *,
    frame_sha256: str,
    Image: Any,
    np: Any,
) -> VisualSeatSignature:
    height, width = pixels.shape[:2]
    box = seat.get("box")
    if not isinstance(box, dict):
        raise ValueError("seat must contain a normalized box")
    left, top, right, bottom = _box_pixels(box, width, height)
    seat_width, seat_height = right - left, bottom - top

    # The central band excludes the seat border, stack row and dealer button.
    crop_left = left + round(seat_width * 0.15)
    crop_right = right - round(seat_width * 0.15)
    crop_top = top + round(seat_height * 0.12)
    crop_bottom = top + round(seat_height * 0.40)
    crop = pixels[crop_top:crop_bottom, crop_left:crop_right]
    if crop.size == 0:
        raise ValueError("seat name band is empty")

    colors, counts = np.unique(crop.reshape(-1, 3), axis=0, return_counts=True)
    background = colors[int(counts.argmax())].astype("int32")
    distances = np.sqrt(((crop.astype("int32") - background) ** 2).sum(axis=2))
    mask = distances > 24.0
    ys, xs = np.where(mask)
    if len(xs) < 8:
        raise ValueError("seat name band has insufficient foreground evidence")
    tight = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    tight_height, tight_width = tight.shape
    target_height = SIGNATURE_HEIGHT - 2
    target_width = max(1, round(tight_width * target_height / tight_height))
    target_width = min(target_width, SIGNATURE_WIDTH - 4)
    resized = Image.fromarray(tight.astype("uint8") * 255).resize(
        (target_width, target_height),
        Image.Resampling.NEAREST,
    )
    normalized = np.zeros((SIGNATURE_HEIGHT, SIGNATURE_WIDTH), dtype=bool)
    x_offset = (SIGNATURE_WIDTH - target_width) // 2
    normalized[1 : 1 + target_height, x_offset : x_offset + target_width] = (
        np.asarray(resized) > 127
    )
    signature = hashlib.sha256(np.packbits(normalized).tobytes()).hexdigest()
    contrast = min(1.0, float(np.median(distances[mask])) / 100.0)
    support = min(1.0, len(xs) / 30.0)
    quality = min(0.99, contrast * support)
    return VisualSeatSignature(
        seat_no=int(seat["seat_no"]),
        signature_sha256=signature,
        quality_score=quality,
        frame_sha256=frame_sha256,
        foreground_pixels=len(xs),
    )


def extract_frame_signatures(
    image_path: str | Path,
    seats: Iterable[dict[str, object]],
) -> list[VisualSeatSignature]:
    Image, np = _pil_numpy()
    source = Path(image_path).expanduser().resolve()
    raw = source.read_bytes()
    with Image.open(source) as opened:
        pixels = np.asarray(opened.convert("RGB"))
    frame_sha = hashlib.sha256(raw).hexdigest()
    rows = [
        _extract_visual_seat_signature(
            pixels,
            seat,
            frame_sha256=frame_sha,
            Image=Image,
            np=np,
        )
        for seat in seats
    ]
    seat_numbers = [row.seat_no for row in rows]
    if len(seat_numbers) != len(set(seat_numbers)):
        raise ValueError("one frame cannot contain duplicate seat signatures")
    return rows


def registry_observations_from_state(
    state: dict[str, object],
    *,
    observed_at_ms: int,
) -> list[NameObservation]:
    """Convert state-carried visual tokens into pseudonymous identity evidence."""
    rows = state.get("visual_identity_signatures")
    if not isinstance(rows, list) or not rows:
        raise ValueError("visible state has no visual identity signatures")
    observations: list[NameObservation] = []
    seats: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"visual identity signature {index} is not structured")
        seat = row.get("seat_no")
        digest = row.get("signature_sha256")
        token = row.get("visual_token")
        frame_sha = row.get("frame_sha256")
        quality = row.get("quality_score")
        if isinstance(seat, bool) or not isinstance(seat, int) or seat in seats:
            raise ValueError("visual identity signatures require unique integer seats")
        if not isinstance(digest, str) or token != f"visual:{digest[:32]}":
            raise ValueError("visual identity token does not match its signature digest")
        observations.append(
            NameObservation(
                seat,
                str(token),
                float(quality),
                str(frame_sha),
                observed_at_ms,
            )
        )
        seats.add(seat)
    return observations
