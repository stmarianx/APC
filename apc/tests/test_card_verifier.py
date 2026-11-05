import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageFont

from apc.perception.card_verifier import CardFaceVerifier, ProposalVerifier, proposal_tensor
from apc.perception.train_card_verifier import control_patch, control_scene_patch, generated_card_patch, patch_dataset, train


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

    def test_composite_controls_are_deterministic(self):
        left, right = control_scene_patch(random.Random(52)), control_scene_patch(random.Random(52))
        self.assertEqual(left.size, right.size)
        self.assertEqual(left.tobytes(), right.tobytes())
        self.assertGreaterEqual(min(left.size), 30)

    def test_generated_card_faces_are_deterministic_with_explicit_fonts(self):
        with patch("apc.perception.train_card_verifier._face_font", return_value=ImageFont.load_default(size=20)):
            left = generated_card_patch(random.Random(61), ("unit-font.ttf",))
            right = generated_card_patch(random.Random(61), ("unit-font.ttf",))
        self.assertEqual(left.size, right.size)
        self.assertEqual(left.tobytes(), right.tobytes())
        self.assertTrue(40 <= left.width <= 76)
        self.assertTrue(64 <= left.height <= 100)

    def test_per_kind_scoring_handles_positive_only_and_negative_only_groups(self):
        from apc.perception.evaluate_card_verifier import binary_counts
        positive = binary_counts(torch.tensor([1, 1]), torch.tensor([1, 0]))
        negative = binary_counts(torch.tensor([0, 0]), torch.tensor([1, 0]))
        self.assertEqual(positive["positive_recall"], .5)
        self.assertIsNone(positive["negative_rejection"])
        self.assertIsNone(negative["positive_recall"])
        self.assertEqual(negative["negative_rejection"], .5)

    def test_composite_examples_do_not_change_original_training_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"source_policy": {"allowed_source_kinds": ["synthetic_render"]}}))
            image = root/"image.png"
            Image.new("RGB", (200, 200), "white").save(image)
            row = {"image": {"path": str(image)}, "objects": [{"class_name": "hero_card", "box_mask": True,
                "visibility": "clear", "box_xyxy": [.2, .2, .4, .6]}]}
            with patch("apc.perception.train_card_verifier.load_split", return_value=[row, row]):
                old = patch_dataset(manifest, "train", 55)
                new = patch_dataset(manifest, "train", 55, negative_scenes=3)
                varied = patch_dataset(manifest, "train", 55, negative_scenes=3, positive_aspect_variants=2)
                with patch("apc.synthetic.render_table.validate_symbol_font"), \
                     patch("apc.perception.train_card_verifier.generated_card_patch", return_value=Image.new("RGB", (50, 70))):
                    faces = patch_dataset(manifest, "train", 55, negative_scenes=3, positive_aspect_variants=2,
                                          generated_positive_faces=2, positive_fonts=("unit-font.ttf",))
            original_indices = [i for i, kind in enumerate(new[2]) if kind != "generated_composite_ui"]
            torch.testing.assert_close(old[0], new[0][original_indices])
            torch.testing.assert_close(old[1], new[1][original_indices])
            self.assertEqual(new[2].count("generated_composite_ui"), 6)
            same = [i for i, kind in enumerate(varied[2]) if kind != "card_aspect_variant"]
            torch.testing.assert_close(new[0], varied[0][same])
            self.assertEqual(varied[2].count("card_aspect_variant"), 4)
            self.assertEqual(int(varied[1].sum())-int(new[1].sum()), 4)
            same = [i for i, kind in enumerate(faces[2]) if kind != "generated_card_face"]
            torch.testing.assert_close(varied[0], faces[0][same])
            self.assertEqual(faces[2].count("generated_card_face"), 4)
        for count in (-1, 65, True, 1.5):
            with self.assertRaises(ValueError):
                patch_dataset("missing.json", "train", 55, negative_scenes=count)
        for count in (-1, 9, True, 1.5):
            with self.assertRaises(ValueError):
                patch_dataset("missing.json", "train", 55, positive_aspect_variants=count)
        with self.assertRaisesRegex(ValueError, "explicit fonts"):
            patch_dataset("missing.json", "train", 55, generated_positive_faces=2)

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

    def test_surface_routing_preserves_order_inputs_and_expert_choices(self):
        from types import SimpleNamespace
        from apc.perception.verified_card_reader import SurfaceRoutedVerifier, VerifiedCardReader
        calls = []
        def expert(name):
            def filtering(image, proposals):
                calls.append((name, [p["score"] for p in proposals]))
                return proposals[:1], proposals[1:]
            return SimpleNamespace(filter=filtering)
        proposals = [{"box_xyxy": [0, 0, 1, 1], "score": .9, "refinement": "adaptive_no_bounded_surface"},
                     {"box_xyxy": [0, 0, 1, 1], "score": .8, "refinement": "adaptive_bounded_surface"},
                     {"box_xyxy": [0, 0, 1, 1], "score": .7, "refinement": "adaptive_bounded_surface"}]
        kept, dropped = SurfaceRoutedVerifier(expert("ambiguous"), expert("bounded")).filter(None, proposals)
        self.assertEqual(calls, [("bounded", [.8, .7]), ("ambiguous", [.9])])
        self.assertEqual([p["score"] for p in kept], [.9, .8])
        self.assertEqual(len(dropped), 1)
        self.assertNotIn("_route_index", kept[0])
        self.assertNotIn("verifier_route", proposals[0])
        with self.assertRaisesRegex(ValueError, "adaptive surface"):
            VerifiedCardReader("missing.pt", verifier_checkpoint="missing.pt", bounded_verifier_checkpoint="missing.pt", locator_checkpoint="missing.pt")

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
