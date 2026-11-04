import copy
import unittest
from apc.neural.compare_strategy import matched_conditions


class ComparisonConditionsTests(unittest.TestCase):
    def test_paired_interval_keeps_complete_hands_and_actions(self):
        import numpy as np
        from types import SimpleNamespace
        from apc.neural.compare_strategy import paired_regret_interval
        corpus = SimpleNamespace(policy_state_ids=["one"]*4+["two"]*4,
            group_ids=["hand-a"]*4+["hand-b"]*4, action_keys=list("abcd")*2,
            target=np.asarray([0,1,2,3]*2,dtype=float))
        report = paired_regret_interval(corpus,np.arange(8),np.asarray([4,3,2,1]*2),np.asarray([0,1,2,3]*2),samples=100)
        self.assertEqual(report["hands"],2)
        self.assertEqual(report["regret_improvement_bb"],3)
        self.assertEqual(report["lower_95_bb"],3)
        with self.assertRaises(ValueError):
            paired_regret_interval(corpus,np.arange(7),np.arange(7),np.arange(7),samples=100)

    def test_matching_and_missing_conditions_are_explicit(self):
        record = {"training": {"seed": 1, "epochs": 3, "batch_size": 128, "learning_rate": "0.0003"},
                  "dataset": {"dataset_fingerprint": "same"},
                  "framework": {"name": "pytorch", "version": "test", "device": "cpu"},
                  "architecture": {"layers": 4}}
        changed = copy.deepcopy(record)
        changed["training"]["paired_expected_regret_weight"] = 1
        self.assertTrue(matched_conditions(record, changed)["matched"])
        changed["training"]["batch_size"] = 384
        self.assertEqual(matched_conditions(record, changed)["mismatches"], ["training.batch_size"])
        self.assertFalse(matched_conditions({}, {})["matched"])
