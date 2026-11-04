"""Trainable full-frame card locator; no brightness threshold or known boxes at inference."""
import torch
from torch import nn
from torch.nn import functional as F


class CardLocator(nn.Module):
    stride = 8

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 5, 1))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected BCHW RGB frames")
        if image.shape[-1] % self.stride or image.shape[-2] % self.stride:
            raise ValueError("Frame dimensions must be divisible by eight")
        return self.network(image)


def locator_targets(boxes, height, width):
    """One center per cell. Collisions are explicit, never silently overwritten."""
    target = torch.zeros(5, height, width)
    for x1, y1, x2, y2 in boxes:
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Invalid normalized card bounds")
        cx, cy = (x1 + x2) * width / 2, (y1 + y2) * height / 2
        col, row = int(cx), int(cy)
        if target[0, row, col]:
            raise ValueError("Card center collision at this grid resolution")
        target[:, row, col] = torch.tensor([1, cx - col, cy - row, x2 - x1, y2 - y1])
    return target


def locator_loss(output, target, *, relative_size=False, hard_negative_ratio=None):
    if output.shape != target.shape or output.ndim != 4 or output.shape[1] != 5:
        raise ValueError("Expected aligned Bx5xHxW locator targets")
    if hard_negative_ratio is not None and (type(hard_negative_ratio) is not int or hard_negative_ratio < 1):
        raise ValueError("Hard-negative ratio must be a positive integer or None")
    positive = target[:, 0].bool()
    raw = F.binary_cross_entropy_with_logits(output[:, 0], target[:, 0], reduction="none")
    # Balance foreground/background separately; otherwise tiny card centers
    # disappear among thousands of negative cells.
    foreground = raw[positive].mean() if positive.any() else raw.sum() * 0
    negative = ~positive
    background = raw[negative].mean() if negative.any() else raw.sum() * 0
    if hard_negative_ratio is not None:
        # Do not dilute a few strong false positives among hundreds of easy cells.
        # Mine independently per frame, including blank frames; keep every positive.
        mined = []
        for index in range(len(raw)):
            losses = raw[index][negative[index]]
            if losses.numel():
                count = min(losses.numel(), max(32, hard_negative_ratio * int(positive[index].sum())))
                mined.append(losses.topk(count).values.mean())
        background = torch.stack(mined).mean() if mined else raw.sum() * 0
    geometry = F.smooth_l1_loss(output[:, 1:].sigmoid().permute(0, 2, 3, 1)[positive],
                              target[:, 1:].permute(0, 2, 3, 1)[positive]) if positive.any() else output.sum() * 0
    if relative_size and positive.any():
        predicted = output[:, 1:].sigmoid().permute(0, 2, 3, 1)[positive]
        truth = target[:, 1:].permute(0, 2, 3, 1)[positive]
        offsets = F.smooth_l1_loss(predicted[:, :2], truth[:, :2])
        size_error = predicted[:, 2:].clamp_min(1e-7).log() - truth[:, 2:].clamp_min(1e-7).log()
        geometry = offsets + F.smooth_l1_loss(size_error, torch.zeros_like(size_error))
    return foreground + background + geometry


def decode_locator(output, threshold=.5):
    """Uncalibrated proposals only; no card identity/readiness claims."""
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0,1]")
    if output.ndim != 3 or output.shape[0] != 5:
        raise ValueError("Expected 5xHxW output")
    values = output.detach().sigmoid()
    score = values[0]
    peaks = score == F.max_pool2d(score[None, None], 3, stride=1, padding=1)[0, 0]
    height, width = score.shape
    proposals = []
    for row, col in (peaks & (score >= threshold)).nonzero().tolist():
        dx, dy, w, h = values[1:, row, col].tolist()
        cx, cy = (col + dx) / width, (row + dy) / height
        box = [max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)]
        proposals.append({"box_xyxy": box, "score": float(score[row, col]), "kind": "neural_card_candidate"})
    return sorted(proposals, key=lambda item: -item["score"])
