from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from apc.neural.contract import ACTION_VOCABULARY, load_apc_neural_config
from apc.neural.features import PROFILE_FEATURE_DIMENSION, STATE_TOKEN_DIMENSION


WEIGHTS_MAGIC = b"APCNEURAL1\n"


@dataclass(frozen=True)
class APCArchitecture:
    hidden_dimension: int = 256
    transformer_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.1
    profile_hidden: int = 64
    visual_channels: tuple[int, int, int] = (16, 32, 64)

    @classmethod
    def from_contract(cls) -> "APCArchitecture":
        config = load_apc_neural_config()
        state = config["architecture"]["state_encoder"]
        profile = config["architecture"]["profile_encoder"]
        return cls(
            hidden_dimension=int(state["hidden_dimension"]),
            transformer_layers=int(state["layers"]),
            attention_heads=int(state["attention_heads"]),
            dropout=float(state["dropout"]),
            profile_hidden=int(profile["hidden_dimensions"][0]),
        )


class APCNetwork(nn.Module):
    """Multimodal APC policy/value network with explicit missing-modality masks."""

    def __init__(self, architecture: APCArchitecture | None = None) -> None:
        super().__init__()
        self.architecture = architecture or APCArchitecture.from_contract()
        hidden = self.architecture.hidden_dimension
        c1, c2, c3 = self.architecture.visual_channels
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, c1, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c3, hidden), nn.LayerNorm(hidden),
        )
        self.state_projection = nn.Linear(STATE_TOKEN_DIMENSION, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.architecture.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=self.architecture.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.architecture.transformer_layers, enable_nested_tensor=False
        )
        self.state_norm = nn.LayerNorm(hidden)
        self.profile_encoder = nn.Sequential(
            nn.Linear(PROFILE_FEATURE_DIMENSION, self.architecture.profile_hidden), nn.GELU(),
            nn.Linear(self.architecture.profile_hidden, hidden), nn.LayerNorm(hidden),
        )
        self.modality_score = nn.Linear(hidden, 1)
        self.fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        action_count = len(ACTION_VOCABULARY)
        self.policy_head = nn.Linear(hidden, action_count)
        self.action_value_head = nn.Linear(hidden, action_count)
        self.state_value_head = nn.Linear(hidden, 1)
        self.opponent_head = nn.Linear(hidden, 4)
        self.uncertainty_head = nn.Linear(hidden, 1)
        self.temporal_consistency_head = nn.Linear(hidden, 1)
        self.action_embedding = nn.Embedding(action_count, 32)
        self.sized_action_value = nn.Sequential(
            nn.Linear(hidden + 32 + 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.register_buffer("value_mean_bb", torch.tensor(0.0))
        self.register_buffer("value_scale_bb", torch.tensor(1.0))

    def forward(
        self,
        state_tokens: Tensor,
        state_padding_mask: Tensor,
        profile_features: Tensor,
        modality_available: Tensor,
        legal_action_mask: Tensor,
        *,
        visual_frames: Tensor | None = None,
        visual_frame_padding_mask: Tensor | None = None,
        candidate_action_index: Tensor | None = None,
        candidate_size_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = state_tokens.shape[0]
        hidden = self.architecture.hidden_dimension
        if visual_frames is None:
            visual = torch.zeros((batch, hidden), dtype=state_tokens.dtype, device=state_tokens.device)
        else:
            if visual_frames.ndim == 4:
                visual = self.visual_encoder(visual_frames)
            elif visual_frames.ndim == 5:
                if visual_frames.shape[0] != batch:
                    raise ValueError("visual frame sequence batch does not match state batch")
                frame_count = visual_frames.shape[1]
                encoded_frames = self.visual_encoder(
                    visual_frames.reshape(batch * frame_count, *visual_frames.shape[2:])
                ).reshape(batch, frame_count, hidden)
                if visual_frame_padding_mask is None:
                    visible_frames = torch.ones(
                        (batch, frame_count), dtype=encoded_frames.dtype, device=encoded_frames.device
                    )
                else:
                    if visual_frame_padding_mask.shape != (batch, frame_count):
                        raise ValueError("visual frame padding mask shape is invalid")
                    visible_frames = (~visual_frame_padding_mask).to(encoded_frames.dtype)
                visual = (encoded_frames * visible_frames.unsqueeze(-1)).sum(1) / visible_frames.sum(1).clamp_min(1.0).unsqueeze(-1)
            else:
                raise ValueError("visual frames must be [batch,channels,height,width] or [batch,frames,channels,height,width]")
        if state_tokens.ndim == 4:
            if state_padding_mask.shape != state_tokens.shape[:3]:
                raise ValueError("temporal state padding mask shape is invalid")
            state_tokens = state_tokens.flatten(1, 2)
            state_padding_mask = state_padding_mask.flatten(1, 2)
        elif state_tokens.ndim != 3 or state_padding_mask.shape != state_tokens.shape[:2]:
            raise ValueError("state tokens must be [batch,tokens,features] or [batch,time,tokens,features]")
        # Replay windows are right-padded to a fixed event count. Removing token
        # columns that are padding for the entire batch is semantically exact and
        # keeps the live temporal path proportional to observed history length.
        retained = ~state_padding_mask.all(dim=0)
        if not bool(retained.any()):
            raise ValueError("state sequence contains no visible tokens")
        state_tokens = state_tokens[:, retained]
        state_padding_mask = state_padding_mask[:, retained]
        encoded = self.state_encoder(
            self.state_projection(state_tokens), src_key_padding_mask=state_padding_mask
        )
        visible = (~state_padding_mask).to(encoded.dtype).unsqueeze(-1)
        state = self.state_norm((encoded * visible).sum(1) / visible.sum(1).clamp_min(1.0))
        profile = self.profile_encoder(profile_features)
        modalities = torch.stack((visual, state, profile), dim=1)
        scores = self.modality_score(modalities).squeeze(-1)
        scores = scores.masked_fill(~modality_available, -1e9)
        weights = torch.softmax(scores, dim=1)
        fused = self.fusion((modalities * weights.unsqueeze(-1)).sum(1))
        policy_logits = self.policy_head(fused).masked_fill(~legal_action_mask, -1e9)
        generic_action_value = self.action_value_head(fused)
        result = {
            "policy_logits": policy_logits,
            "action_value_bb": generic_action_value * self.value_scale_bb + self.value_mean_bb,
            "state_value_bb": self.state_value_head(fused).squeeze(-1) * self.value_scale_bb + self.value_mean_bb,
            "opponent_tendency_logits": self.opponent_head(fused),
            "uncertainty": torch.sigmoid(self.uncertainty_head(fused).squeeze(-1)),
            "temporal_consistency": torch.sigmoid(self.temporal_consistency_head(fused).squeeze(-1)),
            "modality_weights": weights,
            "embedding": fused,
        }
        if candidate_action_index is not None:
            if candidate_size_features is None:
                raise ValueError("candidate size features are required with candidate action")
            action_embedding = self.action_embedding(candidate_action_index)
            residual = self.sized_action_value(
                torch.cat((fused, action_embedding, candidate_size_features), dim=-1)
            ).squeeze(-1)
            generic = generic_action_value.gather(1, candidate_action_index[:, None]).squeeze(1)
            result["candidate_action_value_bb"] = (
                generic + residual
            ) * self.value_scale_bb + self.value_mean_bb
        return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def save_apc_weights(model: APCNetwork, path: str | Path) -> dict[str, object]:
    """Write deterministic, pickle-free APC tensor bytes."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite APC weights: {target}")
    tensors: list[tuple[str, np.ndarray]] = []
    offset = 0
    entries = []
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        payload = array.tobytes(order="C")
        entries.append({"name": name, "shape": list(array.shape), "dtype": "float32", "offset": offset, "bytes": len(payload)})
        tensors.append((name, array))
        offset += len(payload)
    header = {"schema_version": "1.0.0", "model_name": "APC", "architecture": asdict(model.architecture), "tensors": entries}
    header_bytes = _canonical_json(header)
    body = b"".join(array.tobytes(order="C") for _, array in tensors)
    complete = WEIGHTS_MAGIC + struct.pack("<Q", len(header_bytes)) + header_bytes + body
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(complete)
    return {"weights_sha256": hashlib.sha256(complete).hexdigest(), "weights_bytes": len(complete), "tensor_count": len(entries)}


def load_apc_weights(path: str | Path, expected_sha256: str | None = None) -> APCNetwork:
    payload = Path(path).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError("APC weights fingerprint mismatch")
    if not payload.startswith(WEIGHTS_MAGIC):
        raise ValueError("APC weights magic is invalid")
    cursor = len(WEIGHTS_MAGIC)
    header_size = struct.unpack("<Q", payload[cursor:cursor + 8])[0]
    cursor += 8
    header = json.loads(payload[cursor:cursor + header_size])
    cursor += header_size
    if header.get("model_name") != "APC":
        raise ValueError("APC weights identity is invalid")
    architecture = APCArchitecture(**header["architecture"])
    model = APCNetwork(architecture)
    state = {}
    for entry in header["tensors"]:
        start = cursor + int(entry["offset"])
        stop = start + int(entry["bytes"])
        array = np.frombuffer(payload[start:stop], dtype="<f4").reshape(entry["shape"]).copy()
        state[entry["name"]] = torch.from_numpy(array)
    if cursor + sum(int(row["bytes"]) for row in header["tensors"]) != len(payload):
        raise ValueError("APC weights byte count is invalid")
    model.load_state_dict(state, strict=True)
    return model
