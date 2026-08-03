from __future__ import annotations

import unittest

from apc.perception.baseline import fit_centroids, predict_centroids


class PerceptionBaselineMathTests(unittest.TestCase):
    def test_nearest_centroid_fits_and_predicts_deterministically(self) -> None:
        model = fit_centroids(
            [
                ([0.0, 0.0], "dark"),
                ([0.2, 0.1], "dark"),
                ([0.8, 0.9], "light"),
                ([1.0, 1.0], "light"),
            ]
        )
        self.assertEqual(predict_centroids([0.1, 0.05], model)[0], "dark")
        self.assertEqual(predict_centroids([0.9, 0.95], model)[0], "light")

    def test_centroid_model_rejects_dimension_mismatch(self) -> None:
        model = fit_centroids([([0.0, 0.0], "only")])
        with self.assertRaisesRegex(ValueError, "dimension"):
            predict_centroids([0.0], model)

    def test_training_rows_cannot_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            fit_centroids([])


if __name__ == "__main__":
    unittest.main()
