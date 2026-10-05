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


def glyph_tensor(image: Image.Image, box_xyxy):
    width, height = image.size
    left, top, right, bottom = [round(value * dimension) for value, dimension in
                               zip(box_xyxy, (width, height, width, height))]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Invalid card crop")
    pixels = np.asarray(image.convert("RGB").crop((left, top, right, bottom)), dtype=np.float32)
    margin = max(2, round(min(pixels.shape[:2]) * .08))
    if min(pixels.shape[:2]) <= margin * 2:
        raise ValueError("Card crop is too small")
    inner = pixels[margin:-margin, margin:-margin]
    background = np.median(inner.reshape(-1, 3), axis=0)
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


class CardGlyphNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 128), nn.GELU())
        self.rank = nn.Linear(128, 13)
        self.suit = nn.Linear(128, 4)

    def forward(self, tokens):
        features = self.encoder(tokens)
        return {"rank": self.rank(features), "suit": self.suit(features)}
