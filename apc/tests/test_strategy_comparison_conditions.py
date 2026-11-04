import copy
import unittest
from apc.neural.compare_strategy import matched_conditions


class ComparisonConditionsTests(unittest.TestCase):
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
