"""Uncalibrated action-value inspection for a known internal simulator policy."""
import time
import json
from pathlib import Path
import numpy as np
import torch

from apc.neural.model import load_apc_weights
from apc.neural.train_candidate import validate_checkpoint
from apc.neural.features import encode_state, _action_size, ACTION_INDEX, generic_action
from apc.self_learning.raised_postflop_rollout_dataset import _commands


def preview_inputs(state, *, internal_virtual_chips, opponent_policy):
    if internal_virtual_chips is not True or opponent_policy != "check_call":
        raise ValueError("Preview requires a known internal check/call simulator opponent")
    if state.get("terminal") or state.get("next_actor") != "Hero" or state.get("street") not in {"flop","turn","river"}:
        raise ValueError("Preview supports Hero postflop decisions only")
    if state.get("opponent_cards") is not None:
        raise ValueError("Decision input must not reveal opponent cards")
    # The current policy has only been trained on a 100 BB, 2.5 BB-open trunk.
    history = state.get("action_history", [])
    if len(history) < 2 or history[:2] != ["BTN raise_to:2.5", "BB call"]:
        raise ValueError("Unrecognized preflop trunk; no model preview")
    if (any(not item.endswith(" check") for item in history[2:]) or
            state.get("pot_bb") != "5" or
            state.get("stacks_bb") != {"Hero":"97.5","Villain":"97.5"}):
        raise ValueError("Preview requires the exact unbet 100 BB raised-pot trunk")
    if state.get("hero_position") not in {"BTN","BB"}:
        raise ValueError("Unsupported position")
    if float(state.get("to_call_bb",0)) != 0:
        raise ValueError("Only the trained lead-node action set is enabled for preview")
    commands = _commands(state, "lead")
    tokens, padding = encode_state(state)
    legal = np.zeros(6,dtype=np.bool_)
    for button in state["action_buttons"]:
        legal[ACTION_INDEX[generic_action(button["action"])]] = True
    profile = np.zeros(8,dtype=np.float32)
    profile[0],profile[3] = 1,1
    count = len(commands)
    inputs = {"state_tokens":torch.from_numpy(np.repeat(tokens[None],count,axis=0)),
              "state_padding_mask":torch.from_numpy(np.repeat(padding[None],count,axis=0)),
              "profile_features":torch.from_numpy(np.repeat(profile[None],count,axis=0)),
              "modality_available":torch.tensor([[False,True,True]]*count),
              "legal_action_mask":torch.from_numpy(np.repeat(legal[None],count,axis=0)),
              "candidate_action_index":torch.tensor([ACTION_INDEX[generic_action(key)] for key in commands]),
              "candidate_size_features":torch.tensor([_action_size(command.payload(),state) for command in commands.values()])}
    return commands,inputs


class ControlledModelPreview:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        validation = validate_checkpoint(checkpoint)
        if not validation["valid"]:
            raise ValueError("Invalid checkpoint: " + "; ".join(validation["issues"]))
        record = json.loads(checkpoint.read_text())
        self.fingerprint = record["checkpoint_fingerprint"]
        self.model = load_apc_weights(checkpoint.parent/record["weights"]["file"],record["weights"]["weights_sha256"])
        self.model.eval()

    def inspect(self, state, *, internal_virtual_chips=False, opponent_policy=None):
        started = time.perf_counter()
        base = {"recommendation":None,"activation_authorized":False,"units":"BB",
                "checkpoint_fingerprint":self.fingerprint,"confidence_calibrated":False}
        try:
            commands,inputs = preview_inputs(state,internal_virtual_chips=internal_virtual_chips,opponent_policy=opponent_policy)
        except (ValueError,KeyError,TypeError) as error:
            return {**base,"status":"unsupported_state","reason":str(error)}
        with torch.inference_mode():
            output = self.model(**inputs)
        values = output["candidate_action_value_bb"].tolist()
        return {**base,"status":"experimental_estimates_only",
                "estimates":[{"action_key":key,"command":command.payload(),"estimated_return_bb":value}
                             for (key,command),value in zip(commands.items(),values)],
                "elapsed_ms":(time.perf_counter()-started)*1000,
                "limitation":"Known simulator-policy features, not evidence about a human opponent. Not a GTO solution."}
