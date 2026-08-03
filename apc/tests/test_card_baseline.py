from __future__ import annotations

import unittest

from apc.perception.card_baseline import _mean_box, _sub_box, fit_exemplars, predict_exemplars


class CardBaselineGeometryTests(unittest.TestCase):
    def test_exemplar_prediction_can_enforce_numeric_alphabet(self) -> None:
        model = fit_exemplars([([0.0], "."), ([1.0], "7"), ([2.0], "8")])
        unrestricted, _ = predict_exemplars([0.0], model)
        numeric, _ = predict_exemplars(
            [0.0], model, allowed_labels=set("0123456789")
        )
        self.assertEqual(unrestricted, ".")
        self.assertEqual(numeric, "7")

    def test_sub_box_stays_relative_to_parent(self) -> None:
        result = _sub_box(
            {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2},
            (0.25, 0.25, 0.75, 0.75),
        )
        self.assertAlmostEqual(result["x"], 0.3)
        self.assertAlmostEqual(result["y"], 0.35)
        self.assertAlmostEqual(result["width"], 0.2)
        self.assertAlmostEqual(result["height"], 0.1)

    def test_mean_box_is_deterministic(self) -> None:
        result = _mean_box(
            [
                {"x": 0.0, "y": 0.2, "width": 0.1, "height": 0.2},
                {"x": 0.2, "y": 0.4, "width": 0.3, "height": 0.4},
            ]
        )
        self.assertEqual(result, {"x": 0.1, "y": 0.30000000000000004, "width": 0.2, "height": 0.30000000000000004})

    def test_mean_box_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero boxes"):
            _mean_box([])

    def test_exemplar_classifier_preserves_distinct_glyphs(self) -> None:
        model = fit_exemplars([([0.0, 1.0], "A"), ([1.0, 0.0], "K")])
        self.assertEqual(predict_exemplars([0.0, 0.9], model)[0], "A")
        self.assertEqual(predict_exemplars([0.9, 0.0], model)[0], "K")


if __name__ == "__main__":
    unittest.main()
