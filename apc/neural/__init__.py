"""APC neural-model contracts and executable PyTorch implementation."""

from apc.neural.contract import (
    APC_MODEL_NAME,
    load_apc_neural_config,
    validate_apc_neural_config,
    validate_completed_hand_replay,
)
from apc.neural.features import EncodedDecision, encode_raised_row, encode_state
from apc.neural.model import APCArchitecture, APCNetwork, load_apc_weights, save_apc_weights
from apc.neural.replay_buffer import APCReplayBuffer

__all__ = [
    "APC_MODEL_NAME",
    "load_apc_neural_config",
    "validate_apc_neural_config",
    "validate_completed_hand_replay",
    "APCArchitecture",
    "APCNetwork",
    "APCReplayBuffer",
    "EncodedDecision",
    "encode_raised_row",
    "encode_state",
    "load_apc_weights",
    "save_apc_weights",
]
