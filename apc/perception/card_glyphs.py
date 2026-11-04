"""Light-card ink normalization and neural token recognition prototype.

No font templates or annotation identities enter preprocessing. This baseline
assumes a light card face with dark/colored ink; complex artwork needs a detector
for individual rank/suit glyphs before it can use this recognizer reliably.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn


def glyph_tensor(image: Image.Image, box_xyxy, *, border_margin=True, background_rgb=None):
    width, height = image.size
    left, top, right, bottom = [round(value * dimension) for value, dimension in
                               zip(box_xyxy, (width, height, width, height))]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Invalid card crop")
    pixels = np.asarray(image.convert("RGB").crop((left, top, right, bottom)), dtype=np.float32)
    margin = max(2, round(min(pixels.shape[:2]) * .08)) if border_margin else 0
    if min(pixels.shape[:2]) <= margin * 2:
        raise ValueError("Card crop is too small")
    inner = pixels[margin:-margin, margin:-margin] if margin else pixels
    background = np.median(inner.reshape(-1, 3), axis=0) if background_rgb is None else np.asarray(background_rgb)
    contrast = np.abs(inner - background)
    ink = contrast.max(2) > 60
    ys, xs = np.nonzero(ink)
    if not len(xs):
        return torch.zeros(3, 32, 32), False
    # Tight ink bounds suppress changes caused by a shifted card-face border.
    token = contrast[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    patch = Image.fromarray(np.clip(token, 0, 255).astype(np.uint8))
    scale = min(28 / patch.width, 28 / patch.height)
    patch = patch.resize((max(1, round(patch.width * scale)), max(1, round(patch.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (32, 32), "black")
    canvas.paste(patch, ((32 - patch.width) // 2, (32 - patch.height) // 2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy() / 255).permute(2, 0, 1), True


def corner_pair_tensor(image, box):
    """Find two vertically separated ink bands inside the upper-left corner."""
    w, h = image.size
    x, y, right, bottom = [round(v*d) for v, d in zip(box, (w,h,w,h))]
    face = np.asarray(image.crop((x,y,right,bottom)).convert("RGB"), dtype=np.float32)
    height, width = face.shape[:2]
    if height < 16 or width < 12:
        return torch.zeros(6,32,32), False
    bg = np.median(face.reshape(-1,3), axis=0)
    margin = max(2, round(width*.04))
    corner = face[margin:round(height*.60), margin:round(width*.45)]
    ink = np.abs(corner-bg).max(2) > 60
    rows = np.flatnonzero(ink.any(axis=1))
    bands = np.split(rows, np.flatnonzero(np.diff(rows)>1)+1)
    bands = [band for band in bands if len(band)>=2]
    if len(bands) != 2:
        return torch.zeros(6,32,32), False
    tokens = []
    for band in bands:
        cols = np.flatnonzero(ink[band].any(axis=0))
        bounds = [(x+margin+cols[0])/w,(y+margin+band[0])/h,
                  (x+margin+cols[-1]+1)/w,(y+margin+band[-1]+1)/h]
        token, present = glyph_tensor(image,bounds,border_margin=False,background_rgb=bg)
        if not present:
            return torch.zeros(6,32,32), False
        tokens.append(token)
    return torch.cat(tokens), True


def surface_pair_tensor(image, box):
    """Extract exactly two ink bands across a simple card face, without labels.

    Unlike the corner extractor this supports centered identities and lower suit
    positions. Artwork/extra bands are ambiguous and return missing ink rather
    than choosing bands by their expected identity.
    """
    w, h = image.size
    x, y, right, bottom = [round(v*d) for v, d in zip(box, (w, h, w, h))]
    if not (0 <= x < right <= w and 0 <= y < bottom <= h):
        raise ValueError("Invalid card crop")
    face = np.asarray(image.crop((x, y, right, bottom)).convert("RGB"), dtype=np.float32)
    height, width = face.shape[:2]
    if height < 16 or width < 12:
        return torch.zeros(6, 32, 32), False
    bg = np.median(face.reshape(-1, 3), axis=0)
    margin = max(2, round(min(width, height) * .08))
    inner = face[margin:-margin, margin:-margin]
    ink = np.abs(inner - bg).max(2) > 60
    rows = np.flatnonzero(ink.any(axis=1))
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    bands = [band for band in bands if len(band) >= 2]
    if len(bands) != 2:
        return torch.zeros(6, 32, 32), False
    tokens = []
    for band in bands:
        cols = np.flatnonzero(ink[band].any(axis=0))
        bounds = [(x + margin + cols[0]) / w, (y + margin + band[0]) / h,
                  (x + margin + cols[-1] + 1) / w, (y + margin + band[-1] + 1) / h]
        token, present = glyph_tensor(image, bounds, border_margin=False, background_rgb=bg)
        if not present:
            return torch.zeros(6, 32, 32), False
        tokens.append(token)
    return torch.cat(tokens), True


def adaptive_pair_tensor(image, box):
    """Prefer an unambiguous simple face, retaining corner art-card support."""
    token, present = surface_pair_tensor(image, box)
    return (token, present) if present else corner_pair_tensor(image, box)


def _glyph_encoder(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 8 * 8, 128), nn.GELU())


class CardGlyphNetwork(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.encoder = _glyph_encoder(input_channels)
        self.rank = nn.Linear(128, 13)
        self.suit = nn.Linear(128, 4)

    def forward(self, tokens):
        features = self.encoder(tokens)
        return {"rank": self.rank(features), "suit": self.suit(features)}


class SeparatedCardGlyphNetwork(nn.Module):
    """Independent rank/suit encoders prevent cross-glyph identity shortcuts."""
    def __init__(self):
        super().__init__()
        self.rank_encoder = _glyph_encoder(3)
        self.suit_encoder = _glyph_encoder(3)
        self.rank = nn.Linear(128, 13)
        self.suit = nn.Linear(128, 4)

    def forward(self, tokens):
        if tokens.ndim != 4 or tokens.shape[1:] != (6, 32, 32):
            raise ValueError("Expected Bx6x32x32 paired glyph tokens")
        return {"rank": self.rank(self.rank_encoder(tokens[:, :3])),
                "suit": self.suit(self.suit_encoder(tokens[:, 3:]))}
