"""Check required inputs and optional upstream-source provenance."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_adversarial_followup as followup


class FollowupProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.vintage = self.root / "results" / "vintage_descriptions"
        self.cache = self.root / "results" / "embedding_cache"
        self.source = self.root / "scripts" / "run_adversarial_followup.py"

        patches = (
            mock.patch.object(followup, "ROOT", self.root),
            mock.patch.object(followup, "__file__", str(self.source)),
            mock.patch.object(followup, "CACHE_DIR", self.cache),
            mock.patch.object(followup.core, "VINTAGE_DIR", self.vintage),
            mock.patch.object(
                followup.core, "window_directory", side_effect=self.window_directory
            ),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        self.required = {
            self.source,
            self.root / "scripts" / "run_acceptance_robustness.py",
            self.root / "scripts" / "metrics.py",
            self.root / "data" / "universe.parquet",
        }
        for name in followup.SEQUENCE:
            for filename in (
                "ts_features.npy", "corr_target.npy", "universe.parquet", "meta.json"
            ):
                self.required.add(self.window_directory(name) / filename)
        for feature, _ in followup.core.MODERN_TRANSITIONS:
            self.required.add(self.vintage / f"{feature}_descriptions.parquet")
            self.required.update(
                self.cache / filename
                for filename in followup.DENSE_CACHE_FILES[feature].values()
            )
        for path in self.required:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic provenance fixture\n")

    def window_directory(self, name):
        return self.root / "results" / "windows" / name

    def test_missing_upstream_sources_do_not_block_cached_inputs(self):
        self.assertEqual(set(followup.input_paths()), self.required)

    def test_existing_upstream_source_is_included_with_its_actual_hash(self):
        path = self.root / "src" / "config.py"
        path.parent.mkdir(parents=True)
        content = b"# Optional upstream source supplied with these inputs.\n"
        path.write_bytes(content)

        hashes = {
            followup.logical_relative(input_path): followup.sha256_file(input_path)
            for input_path in followup.input_paths()
        }
        self.assertEqual(hashes["src/config.py"], hashlib.sha256(content).hexdigest())
        self.assertNotIn("src/s01_universe.py", hashes)
        self.assertNotIn("scripts/exp_vintage_all_encoders.py", hashes)

    def test_missing_numerical_corpus_or_cache_input_still_fails(self):
        paths = (
            self.window_directory("H1") / "ts_features.npy",
            self.vintage / "W1_descriptions.parquet",
            self.cache / followup.DENSE_CACHE_FILES["W1"]["MiniLM"],
        )
        for path in paths:
            with self.subTest(path=path.relative_to(self.root).as_posix()):
                content = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(FileNotFoundError) as raised:
                        followup.input_paths()
                    self.assertIn(str(path), str(raised.exception))
                finally:
                    path.write_bytes(content)


if __name__ == "__main__":
    unittest.main()
