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
        frames = [{"image":Path("one.png")},{"image":Path("two.png")}]
        with patch("apc.perception.benchmark_reader_candidates.load_collection",return_value=({},frames)):
            report = benchmark_readers(collection,{"a":reader("a"),"b":reader("b")},repeats=2)
        self.assertEqual(len(calls),10)  # Two warmups plus eight measured reads.
        self.assertEqual(calls[2:6],[("a","one.png"),("b","one.png"),("b","two.png"),("a","two.png")])
        self.assertEqual(report["summary"]["a"]["samples"],4)
        self.assertTrue(report["prediction_repeatability_passed"])
        self.assertFalse(report["full_table_readiness"])
        for bad in (0,True,1.5):
            with self.assertRaises(ValueError):
                benchmark_readers(collection,{"a":reader("a"),"b":reader("b")},bad)
