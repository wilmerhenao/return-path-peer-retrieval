"""Check relocated inputs and fail-closed provenance for learned fusion."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_learned_fusion_followup as fusion


class LearnedFusionProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.repo = self.base / "repository"
        self.historical = self.base / "inputs" / "historical"
        self.modern = self.base / "inputs" / "modern"
        self.vintage = self.base / "inputs" / "filings"
        for patch in (
            mock.patch.object(fusion, "ROOT", self.repo),
            mock.patch.object(fusion.core, "WINDOWS_HISTORICAL_DIR", self.historical),
            mock.patch.object(fusion.core, "WINDOWS_MODERN_DIR", self.modern),
            mock.patch.object(fusion.core, "VINTAGE_DIR", self.vintage),
        ):
            patch.start()
            self.addCleanup(patch.stop)

        self.paths = {
            self.repo / "scripts" / name
            for name in (
                "metrics.py", "run_acceptance_robustness.py",
                "run_learned_fusion_followup.py",
            )
        }
        for root, prefix, count in (
            (self.historical, "H", 13), (self.modern, "W", 7)
        ):
            for index in range(1, count + 1):
                self.paths.update(
                    root / f"{prefix}{index}" / filename
                    for filename in (
                        "ts_features.npy", "corr_target.npy", "universe.parquet",
                        "meta.json",
                    )
                )
                if index < count:
                    self.paths.add(
                        self.vintage / f"{prefix}{index}_descriptions.parquet"
                    )
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Synthetic provenance fixture; not training data.\n")

    def test_external_inputs_have_complete_unambiguous_logical_names(self):
        self.assertEqual(set(fusion.input_paths()), self.paths)
        hashes = fusion.hash_inputs()
        self.assertEqual(len(hashes), len(self.paths))
        self.assertIn("results/ea_historical/windows/H13/corr_target.npy", hashes)
        self.assertIn("results/windows_v3/W7/corr_target.npy", hashes)
        self.assertIn("results/vintage_descriptions/H12_descriptions.parquet", hashes)
        self.assertIn("scripts/run_learned_fusion_followup.py", hashes)
        self.assertFalse(any(name.startswith("/") for name in hashes))
        self.assertNotIn("results/vintage_descriptions/H13_descriptions.parquet", hashes)
        self.assertNotIn("results/vintage_descriptions/W7_descriptions.parquet", hashes)
        fusion.verify_unchanged_inputs(hashes)

    def test_missing_source_array_or_vintage_input_fails(self):
        for path in (
            self.repo / "scripts" / "metrics.py",
            self.historical / "H13" / "corr_target.npy",
            self.vintage / "W1_descriptions.parquet",
        ):
            with self.subTest(path=path):
                content = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(FileNotFoundError) as raised:
                        fusion.hash_inputs()
                    self.assertIn(str(path), str(raised.exception))
                finally:
                    path.write_bytes(content)

    def test_changed_input_prevents_successful_provenance_check(self):
        hashes = fusion.hash_inputs()
        path = self.modern / "W6" / "ts_features.npy"
        path.write_bytes(b"Changed input after training started.\n")
        with self.assertRaisesRegex(AssertionError, "W6/ts_features.npy"):
            fusion.verify_unchanged_inputs(hashes)


if __name__ == "__main__":
    unittest.main()
