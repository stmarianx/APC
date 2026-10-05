"""Spatially supervised region recognizer for APC.

Uses annotated/proposed XYXY regions, not an oracle-free table detector. Keeping
this boundary explicit prevents region recognition scores claiming end-to-end
perception readiness. Numeric/name OCR remains a separate component.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from apc.perception.spatial_targets import CLASSES, VISIBILITIES

RANKS = tuple("23456789TJQKA")
SUITS = ("c", "d", "h", "s")
ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")


class RegionRecognizer(nn.Module):
    def __init__(self, crop_size: int = 48):
        super().__init__()
        if crop_size < 8:
            raise ValueError("crop_size must be at least 8")
        self.crop_size = crop_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            # Preserve glyph position/shape instead of averaging the entire
            # card into one feature vector before rank/suit classification.
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.GELU())
        self.heads = nn.ModuleDict({name: nn.Linear(128, count) for name, count in (
            ("class", len(CLASSES)), ("visibility", len(VISIBILITIES)),
            ("rank", len(RANKS)), ("suit", len(SUITS)),
            ("action", len(ACTIONS)), ("dealer", 2), ("hero", 2), ("enabled", 2))})

    def forward(self, images, regions):
        """images: Bx3xHxW; regions: Nx5 [batch_index,x1,y1,x2,y2]."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB batch")
        if regions.ndim != 2 or regions.shape[1] != 5 or len(regions) == 0:
            raise ValueError("Expected nonempty Nx5 region tensor")
        if not torch.isfinite(regions).all():
            raise ValueError("Regions must be finite")
        indices = regions[:, 0].long()
        boxes = regions[:, 1:]
        if ((regions[:, 0] != indices).any() or (indices < 0).any()
                or (indices >= len(images)).any() or (boxes < 0).any()
                or (boxes > 1).any() or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("Invalid region bounds or batch index")
        axis = (torch.arange(self.crop_size, device=images.device, dtype=images.dtype) + .5) / self.crop_size
        x = boxes[:, 0, None] + axis * (boxes[:, 2] - boxes[:, 0])[:, None]
        y = boxes[:, 1, None] + axis * (boxes[:, 3] - boxes[:, 1])[:, None]
        grid = torch.stack((x[:, None, :].expand(-1, self.crop_size, -1),
                            y[:, :, None].expand(-1, -1, self.crop_size)), dim=-1) * 2 - 1
        # Sample all regions of an image in one tall grid. Advanced-indexing
        # images[indices] would replicate a full-resolution frame per region,
        # wasting hundreds of MB before extracting small crops.
        groups, positions = [], []
        for batch_index in indices.unique(sorted=True):
            selected = torch.nonzero(indices == batch_index).flatten()
            tall_grid = grid[selected].reshape(1, -1, self.crop_size, 2)
            sampled = F.grid_sample(images[batch_index:batch_index + 1], tall_grid,
                                    align_corners=False, padding_mode="border")
            groups.append(sampled.reshape(3, len(selected), self.crop_size, self.crop_size).permute(1, 0, 2, 3))
            positions.append(selected)
        crops = torch.cat(groups)[torch.cat(positions).argsort()]
        features = self.encoder(crops)
        return {name: head(features) for name, head in self.heads.items()}


def region_supervision(targets, *, device="cpu"):
    """Build crops/labels only for non-occluded annotated regions."""
    regions, labels = [], {name: [] for name in (
        "class", "visibility", "rank", "suit", "action", "dealer", "hero", "enabled")}
    for batch_index, target in enumerate(targets):
        for obj in target["objects"]:
            if not obj["box_mask"]:
                continue
            regions.append([batch_index, *obj["box_xyxy"]])
            values = {"class": obj["class_id"], "visibility": VISIBILITIES.index(obj["visibility"])}
            attributes, masks = obj["attributes"], obj["attribute_masks"]
            for head, attribute, vocabulary in (
                ("rank", "rank", RANKS), ("suit", "suit", SUITS),
                ("action", "action", ACTIONS), ("dealer", "has_dealer_button", (False, True)),
                ("hero", "is_hero", (False, True)), ("enabled", "enabled", (False, True))):
                values[head] = vocabulary.index(attributes[attribute]) if masks.get(attribute, False) else -100
            for name in labels:
                labels[name].append(values[name])
    if not regions:
        raise ValueError("Batch has no visible supervised regions")
    return (torch.tensor(regions, dtype=torch.float32, device=device),
            {name: torch.tensor(value, dtype=torch.long, device=device) for name, value in labels.items()})


def recognition_loss(predictions, labels):
    """Normalize each head over its supervised labels; fully masked heads are zero."""
    losses = {}
    for name, logits in predictions.items():
        truth = labels[name]
        if logits.shape[0] != truth.numel():
            raise ValueError("Prediction/target count mismatch")
        mask = truth != -100
        losses[name] = F.cross_entropy(logits[mask], truth[mask]) if mask.any() else logits.sum() * 0
    return sum(losses.values()), losses
