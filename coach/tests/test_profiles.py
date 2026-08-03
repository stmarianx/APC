from decimal import Decimal
import unittest

from poker_coach.profiles import PlayerProfile, Tendency


class ProfileTests(unittest.TestCase):
    def test_beta_shrinkage_and_context_separation(self) -> None:
        profile = PlayerProfile("Villain")
        for success in [True] * 8 + [False] * 2:
            posterior = profile.observe(Tendency.FOLD_TO_FLOP_CBET, success, position="BB")
        self.assertEqual(posterior.mean, Decimal("0.75"))
        self.assertEqual(posterior.effective_observations, Decimal("10"))
        self.assertEqual(profile.estimate(Tendency.FOLD_TO_FLOP_CBET, position="BTN").mean, Decimal("0.5"))
        low, high = posterior.approximate_interval()
        self.assertLess(low, posterior.mean)
        self.assertGreater(high, posterior.mean)


if __name__ == "__main__":
    unittest.main()

