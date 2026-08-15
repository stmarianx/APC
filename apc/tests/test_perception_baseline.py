from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from apc.perception.baseline import (
    clear_feature_cache,
    extract_feature,
    feature_cache_info,
    fit_centroids,
    predict_centroids,
)


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

    def test_feature_cache_decodes_immutable_image_once(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (640, 480), (20, 40, 60)).save(path)
            clear_feature_cache()
            with mock.patch("PIL.Image.open", wraps=Image.open) as opened:
                first = extract_feature(
                    path,
                    {"crop": [0, 0, 1, 1], "size": [8, 8]},
                )
                second = extract_feature(
                    path,
                    {"crop": [0.1, 0.1, 0.9, 0.9], "size": [8, 8]},
                )
                repeated = extract_feature(
                    path,
                    {"crop": [0, 0, 1, 1], "size": [8, 8]},
                )
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(first, repeated)
            self.assertNotEqual(id(first), id(repeated))
            self.assertEqual(len(second), 8 * 8 * 3)
            self.assertGreaterEqual(feature_cache_info()["features"]["hits"], 1)


if __name__ == "__main__":
    unittest.main()
