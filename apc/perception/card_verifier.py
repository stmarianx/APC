"""Small learned card-face proposal verifier; scores are not calibrated certainty."""
import math

import numpy as np
import torch
from PIL import Image
from torch import nn


ARCHITECTURE = "rgb-card-face-verifier-v1"


def proposal_tensor(image, box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError("Invalid verifier crop")
    coords = [round(v*d) for v, d in zip(box, image.size*2)]
    if coords[0] >= coords[2] or coords[1] >= coords[3]:
        raise ValueError("Verifier crop has no pixels")
    crop = image.convert("RGB").crop(coords)
    crop.thumbnail((48, 64), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (48, 64), "black")
    canvas.paste(crop, ((48-crop.width)//2, (64-crop.height)//2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()/255).permute(2, 0, 1)


class CardFaceVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(),
            nn.Linear(384, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 64, 48):
            raise ValueError("Expected Bx3x64x48 verifier inputs")
        return self.network(inputs).flatten()


class ProposalVerifier:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("architecture_version") != ARCHITECTURE or saved.get("model_kind") != "card_face_proposal_verifier":
            raise ValueError("Unsupported proposal verifier checkpoint")
        self.threshold = saved["threshold"]
        if not isinstance(self.threshold, (int, float)) or not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Invalid verifier threshold")
        self.model = CardFaceVerifier()
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()

    def filter(self, image, proposals):
        if not proposals:
            return [], []
        inputs = torch.stack([proposal_tensor(image, p["box_xyxy"]) for p in proposals])
        with torch.no_grad():
            scores = self.model(inputs).sigmoid().tolist()
        retained, rejected = [], []
        for proposal, score in zip(proposals, scores):
            item = dict(proposal, verifier_score=score, verifier_confidence_calibrated=False)
            (retained if math.isfinite(score) and score >= self.threshold else rejected).append(item)
        return retained, rejected
