import unittest

from apc.neural.replay_distribution import summarize


class ReplayDistributionTests(unittest.TestCase):
    def test_hand_and_decision_weighting_are_distinguished(self):
        report = summarize([{"reward_bb": "-10", "event_count": 1},
                            {"reward_bb": "10", "event_count": 9}])
        self.assertEqual(report["mean_reward_bb_hand_weighted"], 0)
        self.assertEqual(report["mean_reward_bb_decision_weighted"], 8)
        self.assertEqual(report["hands"], 2)
        self.assertEqual(report["decisions"], 10)
        self.assertEqual(summarize([]), {"hands": 0})
