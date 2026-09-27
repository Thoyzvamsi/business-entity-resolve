import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import data_loader, eda


class PhaseZeroTests(unittest.TestCase):
    """Check Phase 0 loading and EDA behavior on a small local fixture."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.train_dir = self.root / "train"
        self.test_dir = self.root / "test"
        self.train_dir.mkdir()
        self.test_dir.mkdir()
        self._write_source_files("train", self.train_dir, "US")
        self._write_source_files("test", self.test_dir, "France")
        pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-1", ""],
        }).to_csv(self.train_dir / "train_ground_truth.tsv", sep="\t", index=False)
        self.patches = [
            patch.object(data_loader, "TRAIN_DIR", self.train_dir),
            patch.object(data_loader, "TEST_DIR", self.test_dir),
            patch.object(eda, "TRAIN_DIR", self.train_dir),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()
        self.temp_dir.cleanup()

    def _write_source_files(self, split, directory, country):
        for source in ("source1", "source2", "source3"):
            pd.DataFrame({
                "entity_id": [f"{source}-{split}"],
                "business_name": ["Test Business"],
                "business_address": ["1 Main Street"],
                "country": [country],
            }).to_csv(directory / f"{split}_{source}.tsv", sep="\t", index=False)

    def test_loaders_and_eda_summarize_expected_data(self):
        """Load source data and verify country and singleton summaries."""
        loaded = data_loader.load_split("train")
        ground_truth = data_loader.load_ground_truth()
        summary = eda.run_eda()

        self.assertEqual(set(loaded), {"source1", "source2", "source3"})
        self.assertEqual(ground_truth.columns.tolist(), ["source1_entity_id", "matched_entity_ids"])
        self.assertEqual(summary["ground_truth_size"], 2)
        self.assertEqual(summary["singleton_count"], 1)
        self.assertFalse(summary["france_in_train"])
        self.assertTrue(summary["france_in_test"])


if __name__ == "__main__":
    unittest.main()
