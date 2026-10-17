"""Synthetic-data tests for the historical-only loading boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from experiments.multimodal_peer_encoder.build_historical_embeddings import (
    _load_frozen_granite,
    build_historical_embedding_cache,
    model_tree_hashes,
)
from experiments.multimodal_peer_encoder.historical_data import (
    CACHE_SCHEMA_VERSION,
    GRANITE_CPU_DTYPE_POLICY,
    GRANITE_INFERENCE_PARAMETER_DTYPE,
    GRANITE_MODEL_EXCLUDED_FILES,
    GRANITE_MODEL_TREE_SHA256,
    GRANITE_SOURCE_PARAMETER_DTYPE,
    GRANITE_WEIGHT_SHA256,
    HISTORICAL_FEATURE_WINDOWS,
    PINNED_EMBEDDING_SOFTWARE,
    load_eligible_vintage_window,
    load_historical_transition,
    sha256_file,
)


def _write_window(
    root: Path,
    name: str,
    tickers: list[str],
    *,
    start: str,
    end: str,
    ts_features: np.ndarray | None = None,
    corr_target: np.ndarray | None = None,
) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    pl.DataFrame(
        {"ticker": tickers, "company_idx": list(range(len(tickers)))}
    ).write_parquet(directory / "universe.parquet")
    (directory / "meta.json").write_text(
        json.dumps({"window": name, "start": start, "end": end}),
        encoding="utf-8",
    )
    if ts_features is not None:
        np.save(directory / "ts_features.npy", np.asarray(ts_features, dtype=np.float32))
    if corr_target is not None:
        np.save(directory / "corr_target.npy", np.asarray(corr_target, dtype=np.float32))


def _write_vintage(root: Path, name: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    normalized = []
    for index, row in enumerate(rows):
        value = {
            "ticker": row["ticker"],
            "cik": index + 1,
            "window": row.get("window", name),
            "filing_date": row.get("filing_date"),
            "item1_text": row.get("item1_text"),
            "item1_len": len(row.get("item1_text") or ""),
            "source": row.get("source", "10K"),
        }
        normalized.append(value)
    pl.DataFrame(normalized).write_parquet(root / f"{name}_descriptions.parquet")


def _write_mock_cache(
    cache_dir: Path,
    window: str,
    tickers: list[str],
    embeddings: np.ndarray,
    *,
    historical_root: Path,
    vintage_root: Path,
) -> None:
    cache_dir.mkdir(parents=True)
    ticker_path = cache_dir / f"{window}_eligible_tickers.json"
    embedding_path = cache_dir / f"{window}_granite_512.npy"
    ticker_path.write_text(json.dumps(tickers), encoding="utf-8")
    np.save(embedding_path, np.asarray(embeddings, dtype=np.float32))
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "feature_windows": list(HISTORICAL_FEATURE_WINDOWS),
        "protocol": {
            "text_source": "feature-window eligible vintage 10-K Item 1",
            "vintage_rule": "filing_date strictly before feature-window start",
            "ticker_order": "sorted feature-universe/text intersection",
            "deduplication": "exact Python string equality across H1--H12",
            "max_sequence_length": 512,
            "normalize_embeddings": True,
            "output_dtype": "float32",
        },
        "model": {
            "name": "ibm-granite/granite-embedding-english-r2",
            "loader": "sentence_transformers.SentenceTransformer",
            "weight_file": "model.safetensors",
            "weight_sha256": GRANITE_WEIGHT_SHA256,
            "expected_weight_sha256": GRANITE_WEIGHT_SHA256,
            "frozen_weight_match_enforced": True,
            "model_tree_sha256": GRANITE_MODEL_TREE_SHA256,
            "expected_model_tree_sha256": GRANITE_MODEL_TREE_SHA256,
            "model_tree_match_enforced": True,
            "model_tree_excluded_files": sorted(GRANITE_MODEL_EXCLUDED_FILES),
            "source_parameter_dtype": GRANITE_SOURCE_PARAMETER_DTYPE,
            "inference_parameter_dtype": GRANITE_INFERENCE_PARAMETER_DTYPE,
            "cpu_dtype_policy": GRANITE_CPU_DTYPE_POLICY,
        },
        "inputs": {
            window: {
                f"historical_windows/{window}/universe.parquet": sha256_file(
                    historical_root / window / "universe.parquet"
                ),
                f"historical_windows/{window}/meta.json": sha256_file(
                    historical_root / window / "meta.json"
                ),
                f"vintage_descriptions/{window}_descriptions.parquet": sha256_file(
                    vintage_root / f"{window}_descriptions.parquet"
                ),
            }
        },
        "source_code_sha256": {
            "build_historical_embeddings.py": sha256_file(
                Path(__file__).resolve().parents[1] / "experiments" / "multimodal_peer_encoder" / "build_historical_embeddings.py"
            ),
            "historical_data.py": sha256_file(
                Path(__file__).resolve().parents[1] / "experiments" / "multimodal_peer_encoder" / "historical_data.py"
            ),
        },
        "software": {
            "python": "test",
            "packages": dict(PINNED_EMBEDDING_SOFTWARE),
            "pinned_packages": dict(PINNED_EMBEDDING_SOFTWARE),
            "version_match_enforced": True,
        },
        "artifacts": {
            ticker_path.name: {
                "window": window,
                "sha256": sha256_file(ticker_path),
            },
            embedding_path.name: {
                "window": window,
                "sha256": sha256_file(embedding_path),
                "shape": list(embeddings.shape),
                "dtype": "float32",
                "row_order_file": ticker_path.name,
            },
        },
    }
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_transition_uses_sorted_intersection_and_independent_row_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    historical = tmp_path / "historical"
    vintage = tmp_path / "vintage"
    cache = tmp_path / "cache"
    feature_tickers = ["B", "C", "A"]
    target_tickers = ["C", "A", "B"]
    feature_values = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.float32)
    target_values = np.array(
        [[1.0, 0.31, 0.32], [0.31, 1.0, 0.12], [0.32, 0.12, 1.0]],
        dtype=np.float32,
    )
    _write_window(
        historical,
        "H1",
        feature_tickers,
        start="2020-02-01",
        end="2020-04-30",
        ts_features=feature_values,
    )
    _write_window(
        historical,
        "H2",
        target_tickers,
        start="2020-05-01",
        end="2020-08-31",
        corr_target=target_values,
    )
    _write_vintage(
        vintage,
        "H1",
        [
            {"ticker": "C", "filing_date": "2019-03-01", "item1_text": "text C"},
            {"ticker": "A", "filing_date": "2019-03-02", "item1_text": "text A"},
            {"ticker": "B", "filing_date": "2019-03-03", "item1_text": "text B"},
            {"ticker": "OUT", "filing_date": "2019-03-04", "item1_text": "outside"},
        ],
    )
    embeddings = np.zeros((3, 768), dtype=np.float32)
    embeddings[np.arange(3), np.arange(3)] = 1.0
    _write_mock_cache(
        cache,
        "H1",
        ["A", "B", "C"],
        embeddings,
        historical_root=historical,
        vintage_root=vintage,
    )

    opened: list[Path] = []
    original_load = np.load

    def audited_load(path, *args, **kwargs):
        opened.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_load)
    transition = load_historical_transition(
        1,
        historical_root=historical,
        vintage_root=vintage,
        cache_dir=cache,
    )

    assert transition.transition == "H1->H2"
    assert transition.tickers == ("A", "B", "C")
    np.testing.assert_array_equal(
        transition.ts_features,
        np.array([[30, 31], [10, 11], [20, 21]], dtype=np.float32),
    )
    np.testing.assert_array_equal(transition.feature_indices, [2, 0, 1])
    np.testing.assert_array_equal(transition.target_indices, [1, 2, 0])
    np.testing.assert_allclose(
        transition.corr_target,
        target_values[np.ix_([1, 2, 0], [1, 2, 0])],
    )
    np.testing.assert_array_equal(transition.text_embeddings, embeddings)
    assert transition.texts == ("text A", "text B", "text C")
    loaded_return_files = [path for path in opened if path.name != "H1_granite_512.npy"]
    assert loaded_return_files == [
        historical / "H1" / "ts_features.npy",
        historical / "H2" / "corr_target.npy",
    ]

    manifest_path = cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(manifest))
    tamper_cases = (
        ("model", "frozen_weight_match_enforced", False, "cache model mismatch"),
        ("model", "model_tree_sha256", "a" * 64, "cache model mismatch"),
        (
            "model",
            "inference_parameter_dtype",
            "torch.bfloat16",
            "cache model mismatch",
        ),
        ("software", "packages", {"numpy": "0"}, "software-version mismatch"),
    )
    for section, key, value, message in tamper_cases:
        changed = json.loads(json.dumps(original))
        changed[section][key] = value
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_historical_transition(
                1,
                historical_root=historical,
                vintage_root=vintage,
                cache_dir=cache,
            )
    manifest_path.write_text(json.dumps(original), encoding="utf-8")


def test_vintage_loader_rejects_non_vintage_duplicate_and_wrong_label(tmp_path: Path):
    historical = tmp_path / "historical"
    vintage = tmp_path / "vintage"
    _write_window(
        historical,
        "H1",
        ["A"],
        start="2020-02-01",
        end="2020-04-30",
    )

    _write_vintage(
        vintage,
        "H1",
        [{"ticker": "A", "filing_date": "2020-02-01", "item1_text": "late"}],
    )
    with pytest.raises(ValueError, match="is not before"):
        load_eligible_vintage_window(
            "H1", historical_root=historical, vintage_root=vintage
        )

    _write_vintage(
        vintage,
        "H1",
        [
            {"ticker": "A", "filing_date": "2019-01-01", "item1_text": "one"},
            {"ticker": "A", "filing_date": "2019-02-01", "item1_text": "two"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate eligible"):
        load_eligible_vintage_window(
            "H1", historical_root=historical, vintage_root=vintage
        )

    _write_vintage(
        vintage,
        "H1",
        [
            {
                "ticker": "A",
                "window": "H2",
                "filing_date": "2019-01-01",
                "item1_text": "wrong label",
            }
        ],
    )
    with pytest.raises(ValueError, match="wrong window label"):
        load_eligible_vintage_window(
            "H1", historical_root=historical, vintage_root=vintage
        )


def test_cache_builder_deduplicates_exact_text_and_writes_full_hash_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    # Synthetic encoding exercises cache structure, not this host's package pins.
    monkeypatch.setattr(
        "experiments.multimodal_peer_encoder.build_historical_embeddings._verified_software_versions",
        lambda: dict(PINNED_EMBEDDING_SOFTWARE),
    )
    historical = tmp_path / "historical"
    vintage = tmp_path / "vintage"
    model_dir = tmp_path / "granite-embedding-english-r2"
    output = tmp_path / "output"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"synthetic weights")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / ".gitattributes").write_text("ignored", encoding="utf-8")
    (model_dir / "README.md").write_text("ignored", encoding="utf-8")
    (model_dir / "pytorch_model.bin").write_bytes(b"redundant")

    for number in range(1, 13):
        name = f"H{number}"
        _write_window(
            historical,
            name,
            ["B", "A"],
            start=f"{2019 + number:04d}-02-01",
            end=f"{2019 + number:04d}-04-30",
        )
        _write_vintage(
            vintage,
            name,
            [
                {
                    "ticker": "A",
                    "filing_date": f"{2018 + number:04d}-01-01",
                    "item1_text": "shared exact text",
                },
                {
                    "ticker": "B",
                    "filing_date": f"{2018 + number:04d}-01-02",
                    "item1_text": f"alternating text {number % 2}",
                },
            ],
        )

    calls: list[tuple[str, ...]] = []

    def mock_encode(texts, batch_size):
        assert batch_size == 4
        calls.append(tuple(texts))
        values = np.zeros((len(texts), 768), dtype=np.float32)
        values[np.arange(len(texts)), np.arange(len(texts))] = 1.0
        return values

    manifest = build_historical_embedding_cache(
        historical_root=historical,
        vintage_root=vintage,
        model_dir=model_dir,
        output_dir=output,
        batch_size=4,
        encode_override=mock_encode,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert manifest["counts"] == {
        "eligible_window_rows": 24,
        "unique_exact_texts": 3,
        "duplicate_encodes_avoided": 21,
    }
    assert len(manifest["model"]["model_tree_sha256"]) == 64
    assert manifest["model"]["expected_model_tree_sha256"] == GRANITE_MODEL_TREE_SHA256
    assert manifest["model"]["model_tree_match_enforced"] is False
    assert manifest["model"]["model_tree_excluded_files"] == sorted(
        GRANITE_MODEL_EXCLUDED_FILES
    )
    assert manifest["model"]["source_parameter_dtype"] == "test_override"
    assert manifest["model"]["inference_parameter_dtype"] == "test_override"
    assert set(manifest["model"]["model_file_sha256"]) == {
        "config.json",
        "model.safetensors",
    }
    assert manifest["software"]["packages"] == PINNED_EMBEDDING_SOFTWARE
    assert manifest["software"]["version_match_enforced"] is True
    assert json.loads((output / "H1_eligible_tickers.json").read_text()) == ["A", "B"]
    h1_embeddings = np.load(output / "H1_granite_512.npy")
    assert h1_embeddings.shape == (2, 768)
    assert h1_embeddings.dtype == np.float32
    assert len(manifest["artifacts"]) == 24
    checksums = (output / "SHA256SUMS.txt").read_text().splitlines()
    assert len(checksums) == 25
    assert any(line.endswith("  cache_manifest.json") for line in checksums)

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_historical_embedding_cache(
            historical_root=historical,
            vintage_root=vintage,
            model_dir=model_dir,
            output_dir=output,
            encode_override=mock_encode,
        )

    with pytest.raises(ValueError, match="CPU-only"):
        build_historical_embedding_cache(
            historical_root=historical,
            vintage_root=vintage,
            model_dir=model_dir,
            output_dir=tmp_path / "non_cpu",
            device="cuda",
            encode_override=mock_encode,
        )


def test_model_tree_hash_excludes_only_declared_presentation_and_duplicate_files(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weights")
    (tmp_path / "config.json").write_text("{}")
    before_files, before_hash = model_tree_hashes(tmp_path)
    for name in GRANITE_MODEL_EXCLUDED_FILES:
        (tmp_path / name).write_text("excluded")
    after_files, after_hash = model_tree_hashes(tmp_path)
    assert after_files == before_files
    assert after_hash == before_hash
    (tmp_path / "config.json").write_text('{"changed": true}')
    assert model_tree_hashes(tmp_path)[1] != before_hash


def test_granite_loader_explicitly_upcasts_bfloat16_to_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sentence_transformers

    instances = []

    class FakeSentenceTransformer:
        def __init__(self, _path, *, device):
            assert device == "cpu"
            self.parameter = torch.nn.Parameter(
                torch.ones(2, dtype=torch.bfloat16)
            )
            self.float_called = False
            self.max_seq_length = None
            instances.append(self)

        def parameters(self):
            return (self.parameter,)

        def float(self):
            self.float_called = True
            self.parameter = torch.nn.Parameter(self.parameter.detach().float())
            return self

        def eval(self):
            return self

        def get_sentence_embedding_dimension(self):
            return 768

        def encode(self, texts, **kwargs):
            assert self.float_called
            assert self.max_seq_length == 512
            assert kwargs["normalize_embeddings"] is True
            values = np.zeros((len(texts), 768), dtype=np.float32)
            values[:, 0] = 1.0
            return values

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    loaded = _load_frozen_granite(tmp_path, device="cpu")
    values = loaded.encode(["one", "two"], 2)

    assert instances[0].float_called
    assert loaded.source_parameter_dtype == "torch.bfloat16"
    assert loaded.inference_parameter_dtype == "torch.float32"
    assert values.shape == (2, 768)

    with pytest.raises(ValueError, match="CPU-only"):
        _load_frozen_granite(tmp_path, device="cuda")


def test_prelock_modules_have_no_broad_helpers_or_modern_target_root():
    experiment_dir = Path(__file__).resolve().parents[1] / "experiments" / "multimodal_peer_encoder"
    for filename in ("historical_data.py", "build_historical_embeddings.py"):
        source = (experiment_dir / filename).read_text(encoding="utf-8")
        assert "run_acceptance_robustness" not in source
        assert "check_data_contract" not in source
        assert "results.provenance" not in source
        assert "windows_v3" not in source
