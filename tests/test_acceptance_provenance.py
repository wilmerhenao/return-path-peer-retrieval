import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_acceptance_robustness as robustness


class AcceptanceProvenanceTests(unittest.TestCase):
    def test_internal_output_keeps_relative_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            output_root = root / "results" / "acceptance_robustness"
            with patch.object(robustness, "PROJECT_ROOT", root):
                run_dir = robustness.create_run_directory(output_root, "run_test")
                label = robustness.relative(run_dir)
                parent_path = run_dir.parent / ".." / run_dir.parent.name / run_dir.name
                self.assertEqual(robustness.relative(parent_path), label)

            self.assertEqual(label, "results/acceptance_robustness/run_test")
            self.assertTrue(run_dir.is_dir())
            self.assertFalse(Path(label).is_absolute())
            self.assertNotIn(directory, label)

    def test_external_output_directory_has_private_label_without_file_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "checkout"
            output_root = workspace / "private-directory" / "runs"
            with patch.object(robustness, "PROJECT_ROOT", root):
                run_dir = robustness.create_run_directory(output_root, "run_test")
                label = robustness.relative(run_dir)

            self.assertEqual(label, "external/run_test")
            self.assertTrue(run_dir.is_dir())
            self.assertFalse(Path(label).is_absolute())
            self.assertNotIn(directory, label)
            self.assertNotIn("private-directory", label)


if __name__ == "__main__":
    unittest.main()
