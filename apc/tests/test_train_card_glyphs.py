import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from apc.perception.train_card_glyphs import train, augment_tokens, audit_source_isolation, validation_views
from apc.perception.infer_cards import CardReader


class GlyphTrainingTests(unittest.TestCase):
    def test_fixed_validation_views_preserve_labels_inputs_and_training_rng(self):
        data = (torch.rand(4, 6, 32, 32), torch.arange(4), torch.arange(4))
        unchanged = data[0].clone()
        rng = torch.get_rng_state().clone()
        first = validation_views([data], 2, 99)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        second = validation_views([data], 2, 99)
        self.assertEqual(len(first), 3)
        for before, after in zip(first, second):
            torch.testing.assert_close(before[0], after[0])
            torch.testing.assert_close(before[1], data[1])
            torch.testing.assert_close(before[2], data[2])
        torch.testing.assert_close(data[0], unchanged)
        self.assertFalse(torch.equal(first[0][0], first[1][0]))
        for invalid in (-1, 17, True):
            with self.assertRaises(ValueError):
                validation_views([data], invalid, 99)

    def test_source_audit_rejects_duplicate_images_and_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one.json", Path(directory) / "two.json"
            first.write_text('{"dataset_id":"one"}')
            second.write_text('{"dataset_id":"two"}')
            def frames(source, split):
                digest = "shared" if (source == second and split == "validation") or (source == first and split == "train") else str(source) + split
                return [{"image": {"sha256": digest}, "capture_session_id": str(source) + split}]
            with patch("apc.perception.train_card_glyphs.load_split", side_effect=frames):
                with self.assertRaisesRegex(ValueError, "cross-source split leakage"):
                    audit_source_isolation([first, second])
                with self.assertRaisesRegex(ValueError, "Duplicate training source manifest"):
                    audit_source_isolation([first, first])

    def test_source_audit_rejects_capture_session_leakage_without_duplicate_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one.json", Path(directory) / "two.json"
            first.write_text('{"dataset_id":"one"}')
            second.write_text('{"dataset_id":"two"}')
            def frames(source, split):
                session = "shared" if (source == first and split == "train") or (source == second and split == "test") else str(source) + split
                return [{"image": {"sha256": str(source) + split}, "capture_session_id": session}]
            with patch("apc.perception.train_card_glyphs.load_split", side_effect=frames):
                with self.assertRaisesRegex(ValueError, "session crosses"):
                    audit_source_isolation([first, second])

    def test_independent_synthetic_sources_namespace_generated_session_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "one.json", Path(directory) / "two.json"]
            for index, path in enumerate(paths):
                path.write_text(json.dumps({"dataset_id": str(index), "source_policy": {"allowed_source_kinds": ["synthetic_render"]}}))
            def frames(source, split):
                return [{"image": {"sha256": str(source) + split}, "capture_session_id": "synthetic-" + split}]
            with patch("apc.perception.train_card_glyphs.load_split", side_effect=frames):
                audit = audit_source_isolation(paths)
            self.assertEqual(len(audit), 2)
            self.assertEqual(audit[0]["frames_by_split"], {"train": 1, "validation": 1, "test": 1})

    def test_rehearsal_selection_protects_the_worst_validation_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one.json", root / "two.json"
            first.write_text("{}")
            second.write_text('{"different":true}')
            dataset = (torch.rand(4, 6, 32, 32), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
            def score(correct, cards=4):
                return {"cards": cards, "rank_correct": correct, "suit_correct": correct, "exact_correct": correct}
            scores = [score(4), score(0), score(2), score(2), score(4, 8), score(2), score(2)]
            with patch("apc.perception.train_card_glyphs.audit_source_isolation", return_value=[{}, {}]), \
                 patch("apc.perception.train_card_glyphs.token_dataset", return_value=dataset), \
                 patch("apc.perception.train_card_glyphs.metrics", side_effect=scores):
                report = train(first, root / "run", epochs=2, glyph_pair=True, rehearsal_manifests=[second])
            self.assertEqual(report["selected_epoch"], 2)
            self.assertEqual(len(report["test_by_source"]), 2)
            self.assertEqual(report["history"][0]["validation"]["exact_correct"], 4)
            self.assertEqual(report["history"][1]["validation"]["exact_correct"], 4)

    def test_augmentation_is_seeded_bounded_and_preserves_blank_and_source(self):
        original = torch.rand(4, 6, 32, 32)
        original[0] = 0
        unchanged = original.clone()
        torch.manual_seed(123)
        first = augment_tokens(original)
        torch.manual_seed(123)
        second = augment_tokens(original)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(original, unchanged)
        self.assertEqual(first.shape, original.shape)
        self.assertEqual(first[0].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreaterEqual(first.min().item(), 0)
        self.assertLessEqual(first.max().item(), 1)
        self.assertFalse(torch.equal(first, original))
        with self.assertRaises(ValueError):
            augment_tokens(torch.zeros(4, 4, 32, 32))

    def test_pair_checkpoint_cannot_be_loaded_as_whole_card_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            dataset = (torch.rand(4, 6, 32, 32), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
            with patch("apc.perception.train_card_glyphs.token_dataset", return_value=dataset):
                report = train(manifest, root / "run", epochs=1, glyph_pair=True, augment=True)
            self.assertEqual(report["model_kind"], "annotated_card_glyph_pair_recognizer")
            self.assertEqual(report["token_augmentation"], "color_scale_translation_v1")
            with self.assertRaises(ValueError):
                CardReader(root / "run" / "glyph_weights.pt")

    def test_runner_checkpoint_selection_and_reader_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            dataset = (torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
            with patch("apc.perception.train_card_glyphs.token_dataset", return_value=dataset):
                report = train(manifest, root / "run", epochs=1)
            self.assertEqual(report["test_exact_box"]["cards"], 4)
            self.assertFalse(report["full_table_readiness"])
            reader = CardReader(root / "run" / "glyph_weights.pt")
            self.assertTrue(reader.glyph_model)
            self.assertTrue((root / "run" / "evaluation.json").is_file())
            with self.assertRaises(ValueError):
                train(manifest, root / "run")
