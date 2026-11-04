import copy
import unittest
from apc.neural.controlled_preview import preview_inputs
from apc.self_learning.raised_postflop_rollout_dataset import _raised_trunk, _node


class ControlledPreviewTests(unittest.TestCase):
    def test_supported_state_has_four_candidates_without_private_cards(self):
        _,state,_ = _node(_raised_trunk(11,"BTN"),"BTN","lead")
        commands,inputs = preview_inputs(state,internal_virtual_chips=True,opponent_policy="check_call")
        self.assertEqual(len(commands),4)
        self.assertEqual(inputs["state_tokens"].shape[0],4)

    def test_unknown_environment_policy_and_out_of_scope_state_are_rejected(self):
        _,state,_ = _node(_raised_trunk(11,"BTN"),"BTN","lead")
        for internal,policy in ((False,"check_call"),(True,None),(True,"pressure")):
            with self.assertRaises(ValueError):
                preview_inputs(state,internal_virtual_chips=internal,opponent_policy=policy)
        for field,value in (("opponent_cards",["As","Ad"]),("pot_bb","10"),("street","preflop")):
            changed=copy.deepcopy(state);changed[field]=value
            with self.assertRaises(ValueError):
                preview_inputs(changed,internal_virtual_chips=True,opponent_policy="check_call")
