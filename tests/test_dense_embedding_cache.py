"""Cache provenance checks with local fixtures, without loading a model."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.build_dense_embedding_cache import (
    CPU_DTYPE_POLICY,
    cache_key,
    model_tree_hashes,
    sha256_file,
    validate_cache_reuse,
)


class CacheReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "model.safetensors").write_bytes(b"fixed test weights")
        (self.model_dir / "tokenizer_config.json").write_text(
            json.dumps({"do_lower_case": True}), encoding="utf-8"
        )
        self.model = self.model_record()
        self.key = self.content_key(self.model)
        self.path = self.root / f"W1_minilm_{self.key}.npy"
        self.values = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.save(self.path, self.values)
        self.inputs = {
            "window_universe_sha256": "universe fixture hash",
            "vintage_text_sha256": "filing fixture hash",
            "window_meta_sha256": "window fixture hash",
        }
        self.manifest = {
            "schema_version": "dense-embedding-cache-v1",
            "device": "cpu",
            "cpu_dtype_policy": CPU_DTYPE_POLICY,
            "models": {"MiniLM": copy.deepcopy(self.model)},
            "artifacts": {
                self.path.name: {
                    "sha256": sha256_file(self.path),
                    "shape": list(self.values.shape),
                    **self.inputs,
                }
            },
        }

    def model_record(self):
        hashes, tree_hash = model_tree_hashes(self.model_dir)
        return {
            "model_directory": "all-MiniLM-L12-v2",
            "loader": "sentence_transformer",
            "max_length": 128,
            "batch_size": 128,
            "weight_file": "model.safetensors",
            "weight_sha256": hashes["model.safetensors"],
            "model_tree_sha256": tree_hash,
            "model_file_sha256": hashes,
        }

    @staticmethod
    def content_key(model):
        return cache_key(
            "W1", "MiniLM", 128, model["weight_sha256"],
            ["AAA", "BBB"], ["first filing", "second filing"],
        )

    def reuse(self, **overrides):
        kwargs = {
            "previous_manifest": self.manifest,
            "method": "MiniLM",
            "model_record": self.model,
            "input_hashes": self.inputs,
            "device": "cpu",
            "n_rows": 2,
        }
        kwargs.update(overrides)
        return validate_cache_reuse(self.path, **kwargs)

    def test_matching_provenance_reuses_the_saved_values(self):
        np.testing.assert_array_equal(self.reuse(), self.values)

    def test_changed_tokenizer_with_same_weights_rejects_legacy_key(self):
        (self.model_dir / "tokenizer_config.json").write_text(
            json.dumps({"do_lower_case": False}), encoding="utf-8"
        )
        changed = self.model_record()
        self.assertEqual(changed["weight_sha256"], self.model["weight_sha256"])
        self.assertEqual(self.content_key(changed), self.key)
        with self.assertRaisesRegex(ValueError, "model snapshot or configuration"):
            self.reuse(model_record=changed)

    def test_device_mismatch_rejects_cpu_cache(self):
        with self.assertRaisesRegex(ValueError, "recorded device"):
            self.reuse(device="cuda")

    def test_missing_manifest_requires_rebuilding(self):
        with self.assertRaisesRegex(ValueError, "--force"):
            self.reuse(previous_manifest=None)

    def test_altered_cached_file_is_rejected(self):
        np.save(self.path, self.values + 1)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.reuse()

    def test_changed_input_hash_is_rejected(self):
        changed = {**self.inputs, "window_meta_sha256": "changed cutoff hash"}
        with self.assertRaisesRegex(ValueError, "window_meta_sha256"):
            self.reuse(input_hashes=changed)

    def test_recorded_shape_must_match_cached_array(self):
        self.manifest["artifacts"][self.path.name]["shape"] = [2, 4]
        with self.assertRaisesRegex(ValueError, "cached shape"):
            self.reuse()


if __name__ == "__main__":
    unittest.main()
