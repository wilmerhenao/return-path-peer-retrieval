"""Published summaries must retain their arithmetic and evaluation partitions."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.verify_paper_results import ROOT, verify


class PaperResultsTests(unittest.TestCase):
    def test_published_results(self):
        report = verify(ROOT / "paper_results")
        self.assertEqual(report["held_out_transitions"], 7)
        self.assertEqual(report["cnn_continuation_transitions_inside_selection"], 6)

    def test_detects_corrupt_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "paper_results", root, dirs_exist_ok=True)
            path = root / "full19_frozen_blend.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify(root)

    def test_rejects_selection_leakage_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "paper_results", root, dirs_exist_ok=True)
            path = root / "full19_frozen_blend.json"
            original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = json.loads(path.read_text())
            result["configuration"]["held_out_transitions"][0] = "H12->H13"
            path.write_text(json.dumps(result))
            new_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS.txt"
            manifest.write_text(manifest.read_text().replace(original_digest, new_digest))
            with self.assertRaisesRegex(ValueError, "Held-out partition changed"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
