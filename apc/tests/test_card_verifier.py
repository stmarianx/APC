import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from apc.perception.card_verifier import CardFaceVerifier, ProposalVerifier, proposal_tensor
from apc.perception.train_card_verifier import control_patch, patch_dataset, train


class CardVerifierTests(unittest.TestCase):
    def test_crop_shape_empty_and_invalid_geometry(self):
        image = Image.new("RGB", (100, 100), "white")
        self.assertEqual(proposal_tensor(image, [.2, .1, .6, .9]).shape, (3, 64, 48))
        for box in ([0, 0, 0, 1], [float("nan"), 0, 1, 1], [-1, 0, 1, 1]):
            with self.assertRaises(ValueError):
                proposal_tensor(image, box)
        with self.assertRaises(ValueError):
            CardFaceVerifier()(torch.zeros(1, 3, 48, 64))

    def test_synthetic_control_generation_is_deterministic(self):
        left, right = control_patch(random.Random(42)), control_patch(random.Random(42))
        self.assertEqual(left.size, right.size)
        self.assertEqual(left.tobytes(), right.tobytes())

    def test_audit_material_is_rejected_before_loading_training_images(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory)/"audit.json"
            manifest.write_text(json.dumps({"use_policy": "sealed_audit_no_training", "source_policy": {"allowed_source_kinds": ["synthetic_render"]}}))
            with patch("apc.perception.train_card_verifier.load_split") as loader, self.assertRaises(ValueError):
                patch_dataset(manifest, "train", 42)
            loader.assert_not_called()

    def test_filter_preserves_input_and_returns_uncalibrated_evidence(self):
        verifier = object.__new__(ProposalVerifier)
        verifier.threshold = .5
        verifier.model = lambda inputs: torch.tensor([10., -10.])
        proposals = [{"box_xyxy": [.1, .1, .4, .8]}, {"box_xyxy": [.5, .1, .9, .8]}]
        retained, rejected = verifier.filter(Image.new("RGB", (100, 100)), proposals)
        self.assertEqual(len(retained), 1)
        self.assertEqual(len(rejected), 1)
        self.assertNotIn("verifier_score", proposals[0])
        self.assertFalse(retained[0]["verifier_confidence_calibrated"])
        self.assertEqual(verifier.filter(None, []), ([], []))

    def test_training_report_keeps_test_out_of_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/"manifest.json"
            manifest.write_text("{}")
            data = (torch.zeros(2, 3, 64, 48), torch.tensor([1., 0.]), ["card_exact", "generated_ui_background"])
            with patch("apc.perception.train_card_verifier.patch_dataset", return_value=data):
                report = train(manifest, root/"run", epochs=1)
            self.assertEqual(report["test"]["samples"], 2)
            self.assertEqual(report["selected_epoch"], 1)
            self.assertFalse(report["audit_images_used"])
            self.assertFalse(report["full_table_readiness"])
            loaded = ProposalVerifier(root/"run/weights.pt")
            self.assertEqual(loaded.threshold, .5)

    def test_verified_locator_filters_before_glyph_recognition(self):
        from types import SimpleNamespace
        from apc.perception.verified_card_reader import VerifyingLocator, VerifiedCardReader
        boxes = [{"box_xyxy": [.1, .1, .4, .8]}, {"box_xyxy": [.5, .1, .9, .8]}]
        base = SimpleNamespace(propose=lambda path: {"proposals": boxes, "full_table_readiness": False})
        verifier = SimpleNamespace(filter=lambda image, proposals: (proposals[:1], proposals[1:]))
        with patch("apc.perception.verified_card_reader.Image.open", return_value=Image.new("RGB", (100, 100))):
            result = VerifyingLocator(base, verifier).propose("unused.png")
        self.assertEqual(result["proposals"], boxes[:1])
        self.assertEqual(result["verifier_rejected"], 1)
        self.assertFalse(result["full_table_readiness"])
        with self.assertRaisesRegex(ValueError, "explicit neural"):
            VerifiedCardReader("missing.pt", verifier_checkpoint="missing.pt")

    def test_matched_benchmark_warms_and_times_both_variants(self):
        from apc.perception.benchmark_card_verifier import benchmark
        fixture = Path(__file__)
        with patch("apc.perception.benchmark_card_verifier.load_collection", return_value=({}, [{"image": fixture}])), \
             patch("apc.perception.benchmark_card_verifier.CardReader") as baseline, \
             patch("apc.perception.benchmark_card_verifier.VerifiedCardReader") as verified:
            baseline.return_value.read.return_value = verified.return_value.read.return_value = {"cards": []}
            report = benchmark(fixture, fixture, fixture, fixture, repeats=3)
        for reader in (baseline, verified):
            self.assertEqual(reader.return_value.read.call_count, 4)
        self.assertEqual(report["summary"]["baseline"]["samples"], 3)
        self.assertEqual(report["summary"]["verified"]["samples"], 3)
        self.assertFalse(report["full_table_readiness"])
        with self.assertRaises(ValueError):
            benchmark(fixture, fixture, fixture, fixture, repeats=0)
