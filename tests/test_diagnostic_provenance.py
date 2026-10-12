"""Check that diagnostic provenance covers the files the runners read."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_pairwise_alignment as pairwise
from scripts import run_same_window_text_diagnostic as same_window


class DiagnosticProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.vintage = self.root / "results" / "vintage_descriptions"
        self.cache = self.root / "results" / "embedding_cache"
        self.selection = self.root / "results" / "tfidf_selection.json"
        self.wf_robustness = self.root / "results" / "wf_robustness.json"
        self.wf_dense = self.root / "results" / "wf_dense.json"
        self.dense_paths = {
            label: self.cache / filename
            for label, filename in pairwise.DENSE_CACHE_FILES.items()
        }

        patches = [
            mock.patch.object(same_window, "CACHE_DIR", self.cache),
            mock.patch.object(same_window.core, "VINTAGE_DIR", self.vintage),
            mock.patch.object(
                same_window.core, "window_directory", side_effect=self.window_directory
            ),
        ]
        for module in (same_window, pairwise):
            patches.extend(
                (
                    mock.patch.object(module, "ROOT", self.root),
                    mock.patch.object(
                        module, "__file__", str(self.root / "scripts" / Path(module.__file__).name)
                    ),
                )
            )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        paths = {
            self.selection, self.wf_robustness, self.wf_dense,
            self.root / "scripts" / "metrics.py",
            self.root / "scripts" / "run_acceptance_robustness.py",
            self.root / "scripts" / "run_adversarial_followup.py",
            self.root / "scripts" / "run_same_window_text_diagnostic.py",
            self.root / "scripts" / "run_pairwise_alignment.py",
            *self.dense_paths.values(),
        }
        for name in same_window.core.MODERN_WINDOWS:
            for filename in (
                "ts_features.npy", "corr_target.npy", "universe.parquet", "meta.json"
            ):
                paths.add(self.window_directory(name) / filename)
        for feature, _ in same_window.core.MODERN_TRANSITIONS:
            paths.add(self.vintage / f"{feature}_descriptions.parquet")
            for method in ("Granite-r2 (512 tok.)", "MiniLM"):
                paths.add(self.cache / same_window.DENSE_CACHE_FILES[feature][method])
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic diagnostic provenance fixture\n")

    def window_directory(self, name):
        return self.root / "results" / "windows" / name

    def same_window_paths(self):
        return same_window.input_paths(self.wf_robustness, self.wf_dense)

    def pairwise_paths(self):
        return pairwise.collect_input_paths(self.selection, self.dense_paths)

    def assert_missing_inputs_fail(self, collect_paths, paths):
        for path in paths:
            with self.subTest(path=path.relative_to(self.root).as_posix()):
                content = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(FileNotFoundError) as raised:
                        collect_paths()
                    self.assertIn(str(path), str(raised.exception))
                finally:
                    path.write_bytes(content)

    def test_same_window_records_last_target_arrays_and_imported_source(self):
        paths = self.same_window_paths()
        self.assertIn(self.window_directory("W7") / "ts_features.npy", paths)
        self.assertIn(self.window_directory("W7") / "corr_target.npy", paths)
        self.assertIn(self.root / "scripts" / "run_adversarial_followup.py", paths)

    def test_same_window_missing_required_inputs_fail(self):
        self.assert_missing_inputs_fail(
            self.same_window_paths,
            (
                self.window_directory("W7") / "ts_features.npy",
                self.window_directory("W7") / "corr_target.npy",
                self.root / "scripts" / "run_adversarial_followup.py",
                self.vintage / "W1_descriptions.parquet",
            ),
        )

    def test_pairwise_records_target_path_array_and_existing_dependencies(self):
        paths = self.pairwise_paths()
        self.assertIn(self.window_directory("W2") / "ts_features.npy", paths)
        self.assertIn(self.selection, paths)
        self.assertTrue(set(self.dense_paths.values()).issubset(paths))

    def test_pairwise_missing_required_inputs_fail(self):
        self.assert_missing_inputs_fail(
            self.pairwise_paths,
            (
                self.window_directory("W2") / "ts_features.npy",
                self.selection,
                self.dense_paths["Granite-r2 (512 tok.)"],
            ),
        )

    def test_internal_labels_normalize_parent_components(self):
        path = self.selection.parent / ".." / "results" / self.selection.name
        for module in (pairwise, same_window):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.logical_path(path), "results/tfidf_selection.json")

    def test_pairwise_accepts_external_selection_input_without_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "selection.json"
            content = b'{"selected_config_id": "b"}\n'
            external.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            paths = pairwise.collect_input_paths(external, self.dense_paths)
            hashes = {
                pairwise.logical_path(path): pairwise.sha256_file(path)
                for path in paths
            }

            self.assertIn(external, paths)
            self.assertEqual(hashes[f"external/{digest[:16]}/selection.json"], digest)
            self.assertTrue(all(not Path(label).is_absolute() for label in hashes))
            self.assertTrue(all(directory not in label for label in hashes))
            self.assertTrue(all(str(self.root) not in label for label in hashes))

    def test_same_window_external_summaries_with_same_name_do_not_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            external_root = Path(directory)
            robustness = external_root / "robustness" / "summary.json"
            dense = external_root / "dense" / "summary.json"
            contents = {
                robustness: b'{"result": "walk-forward blend"}\n',
                dense: b'{"result": "walk-forward dense"}\n',
            }
            for path, content in contents.items():
                path.parent.mkdir()
                path.write_bytes(content)

            paths = same_window.input_paths(robustness, dense)
            hashes = {
                same_window.logical_path(path): same_window.sha256_file(path)
                for path in paths
            }
            labels = [same_window.logical_path(path) for path in contents]

            self.assertEqual(len(set(labels)), 2)
            self.assertEqual(len(hashes), len(paths))
            for path, content in contents.items():
                digest = hashlib.sha256(content).hexdigest()
                self.assertEqual(hashes[f"external/{digest[:16]}/summary.json"], digest)
                self.assertIn(path, paths)
            self.assertTrue(all(not Path(label).is_absolute() for label in hashes))
            self.assertTrue(all(directory not in label for label in hashes))
            self.assertTrue(all(str(self.root) not in label for label in hashes))


if __name__ == "__main__":
    unittest.main()
