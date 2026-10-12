import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_full_return_census as census


class FullReturnProvenanceTests(unittest.TestCase):
    def test_internal_and_external_inputs_have_private_labels_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "checkout"
            internal = root / "data" / "universe.parquet"
            external = workspace / "private-directory" / "universe.parquet"
            internal.parent.mkdir(parents=True)
            external.parent.mkdir(parents=True)
            internal.write_bytes(b"internal input")
            external.write_bytes(b"external input")

            with patch.object(census, "ROOT", root):
                hashes = {
                    census.logical_path(path): census.sha256_file(path)
                    for path in (internal, external)
                }

            self.assertEqual(
                hashes,
                {
                    "data/universe.parquet": hashlib.sha256(b"internal input").hexdigest(),
                    "external/universe.parquet": hashlib.sha256(b"external input").hexdigest(),
                },
            )
            self.assertTrue(all(not Path(label).is_absolute() for label in hashes))
            self.assertTrue(all(str(workspace) not in label for label in hashes))

    def test_external_label_is_stable_and_normalizes_parent_components(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "checkout"
            root.mkdir()
            external = workspace / "universe.parquet"
            external.write_bytes(b"external input")

            with patch.object(census, "ROOT", root):
                self.assertEqual(
                    census.logical_path(root / ".." / external.name),
                    "external/universe.parquet",
                )
                self.assertEqual(
                    census.logical_path(workspace / "another-directory" / external.name),
                    "external/universe.parquet",
                )


if __name__ == "__main__":
    unittest.main()
