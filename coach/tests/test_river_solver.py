import unittest

from poker_coach.river_solver import HandBucket, RiverCFRSolver, RiverSubgame, solve_akq_river


class RiverSolverTests(unittest.TestCase):
    def test_classic_pot_sized_akq_equilibrium(self) -> None:
        _, solution = solve_akq_river(iterations=40_000)
        self.assertAlmostEqual(solution.ip_strategy["Value"]["bet"], 1.0, delta=0.01)
        self.assertAlmostEqual(solution.ip_strategy["Air"]["bet"], 0.5, delta=0.03)
        self.assertAlmostEqual(solution.oop_strategy["Bluff-catcher"]["call"], 0.5, delta=0.03)
        self.assertLess(solution.exploitability_bb, 0.02)

    def test_half_pot_changes_equilibrium_frequencies(self) -> None:
        _, solution = solve_akq_river(pot_bb=10, bet_bb=5, iterations=40_000)
        self.assertAlmostEqual(solution.ip_strategy["Air"]["bet"], 1 / 3, delta=0.03)
        self.assertAlmostEqual(solution.oop_strategy["Bluff-catcher"]["call"], 2 / 3, delta=0.03)

    def test_invalid_game_and_iteration_count_fail(self) -> None:
        with self.assertRaises(ValueError):
            RiverSubgame(
                pot_bb=0,
                bet_bb=1,
                ip_buckets=(HandBucket("Value", 1),),
                oop_buckets=(HandBucket("Catch", 1),),
                showdown_equity=((1.0,),),
            )
        game, _ = solve_akq_river(iterations=100)
        with self.assertRaises(ValueError):
            RiverCFRSolver().solve(game, iterations=10)


if __name__ == "__main__":
    unittest.main()
