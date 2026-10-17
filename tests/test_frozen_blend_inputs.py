import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import frozen_blend_inputs as inputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenBlendInputsTests(unittest.TestCase):
    def test_all_published_extract_checksums_and_source_identities(self):
        for name in inputs.AUTHORITY_SOURCES:
            payload = inputs.verify_artifact(inputs.PROJECT_ROOT / "evidence/frozen_blend/authorities" / name)
            self.assertIn("sha256", payload["provenance"])

    def test_checksum_and_source_identity_are_separate_checks(self):
        name = "01_tfidf_historical_tuning.json"
        source = inputs.PROJECT_ROOT / "evidence/frozen_blend/authorities" / name
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_bytes(source.read_bytes())
            manifest = path.parent / "SHA256SUMS.txt"
            manifest.write_text(f"{sha(path)}  {name}\n")
            wrapper = json.loads(path.read_text())
            wrapper["source_artifact_sha256"] = "0" * 64
            path.write_text(json.dumps(wrapper))
            with self.assertRaisesRegex(AssertionError, "checksum mismatch"):
                inputs.verify_artifact(path)
            manifest.write_text(f"{sha(path)}  {name}\n")
            with self.assertRaisesRegex(AssertionError, "source identity"):
                inputs.verify_artifact(path)

    def test_missing_or_changed_numerical_data_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "results/windows_v3/W1/ts_features.npy"
            data.parent.mkdir(parents=True)
            data.write_bytes(b"frozen numerical array")
            recorded = {name: policy["archived_sha256"] for name, policy in inputs.SOURCE_COMPATIBILITY.items()}
            recorded[str(data.relative_to(root))] = sha(data)
            authority = {"provenance": {"sha256": recorded}}
            with patch.object(inputs, "INPUT_ROOT", root):
                result = inputs.verify_inputs(authority, authority)
                self.assertEqual(result["numerical_input_paths_checked"], 1)
                data.write_bytes(b"changed numerical array")
                with self.assertRaisesRegex(AssertionError, "numerical input changed"):
                    inputs.verify_inputs(authority, authority)
                data.unlink()
                with self.assertRaises(FileNotFoundError):
                    inputs.verify_inputs(authority, authority)

    def test_unreviewed_execution_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "scripts/metrics.py"
            path = root / name
            path.parent.mkdir()
            path.write_text("unreviewed code")
            recorded = {name: "original"}
            policy = {name: {"archived_sha256": "original", "execution_sha256": "approved"}}
            authority = {"provenance": {"sha256": recorded}}
            with patch.object(inputs, "PROJECT_ROOT", root), patch.object(inputs, "SOURCE_COMPATIBILITY", policy):
                with self.assertRaisesRegex(AssertionError, "unreviewed execution source"):
                    inputs.verify_inputs(authority, authority)

    def test_external_provenance_labels_do_not_expose_directories(self):
        self.assertEqual(inputs.rel(Path("/private/sensitive/location/result.json")), "external/result.json")


if __name__ == "__main__":
    unittest.main()
