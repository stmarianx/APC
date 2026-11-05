from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from apc.perception.benchmark_reader_candidates import benchmark_readers


class ReaderBenchmarkTests(unittest.TestCase):
    def test_rotating_order_sample_counts_and_prediction_repeatability(self):
        calls = []
        def reader(name):
            def read(path):
                calls.append((name,path.name))
                return {"cards":[{"candidate":name,"card":None}]}
            return SimpleNamespace(read=read)
        collection = Path(__file__)
        frames = [{"image":collection.parent/"one.png"},{"image":collection.parent/"two.png"}]
        with patch("apc.perception.benchmark_reader_candidates.load_collection",return_value=({},frames)):
            report = benchmark_readers(collection,{"a":reader("a"),"b":reader("b")},repeats=2)
        self.assertEqual(len(calls),10)  # Two warmups plus eight measured reads.
        self.assertEqual(calls[2:6],[("a","one.png"),("b","one.png"),("b","two.png"),("a","two.png")])
        self.assertEqual(report["summary"]["a"]["samples"],4)
        self.assertTrue(report["prediction_repeatability_passed"])
        self.assertFalse(report["cross_variant_predictions_equal"])
        self.assertFalse(report["full_table_readiness"])
        for bad in (0,True,1.5):
            with self.assertRaises(ValueError):
                benchmark_readers(collection,{"a":reader("a"),"b":reader("b")},bad)

    def test_same_basename_distinct_paths_are_not_conflated(self):
        collection = Path(__file__)
        frames = [{"image":collection.parent/part/"frame.jpg"} for part in ("left","right")]
        reader = SimpleNamespace(read=lambda path:{"cards":[{"candidate":path.parent.name}]})
        with patch("apc.perception.benchmark_reader_candidates.load_collection",return_value=({},frames)):
            report = benchmark_readers(collection,{"a":reader,"b":reader},repeats=2)
        self.assertTrue(report["prediction_repeatability_passed"])
        self.assertTrue(report["cross_variant_predictions_equal"])
        self.assertEqual({row["frame_key"] for row in report["observations"]},{"left/frame.jpg","right/frame.jpg"})
        with patch("apc.perception.benchmark_reader_candidates.load_collection",return_value=({},[frames[0],frames[0]])):
            with self.assertRaisesRegex(ValueError,"Duplicate frame"):
                benchmark_readers(collection,{"a":reader,"b":reader})
