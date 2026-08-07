#!/usr/bin/env python3
"""Build the frozen Granite-r2 cache for H1--H12 feature windows.

Exact filing text is deduplicated before encoding.  Each output matrix is then
expanded back to a sorted, explicit ticker list, so downstream alignment never
depends on a nonzero-vector heuristic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .historical_data import (
        CACHE_SCHEMA_VERSION,
        GRANITE_CPU_DTYPE_POLICY,
        GRANITE_EMBEDDING_DIMENSION,
        GRANITE_INFERENCE_PARAMETER_DTYPE,
        GRANITE_MODEL_EXCLUDED_FILES,
        GRANITE_MODEL_TREE_SHA256,
        GRANITE_SOURCE_PARAMETER_DTYPE,
        GRANITE_TOKEN_CAP,
        GRANITE_WEIGHT_SHA256,
        HISTORICAL_FEATURE_WINDOWS,
        PINNED_EMBEDDING_SOFTWARE,
        load_eligible_vintage_window,
        sha256_file,
    )
else:  # Permit direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from historical_data import (  # type: ignore[no-redef]
        CACHE_SCHEMA_VERSION,
        GRANITE_CPU_DTYPE_POLICY,
        GRANITE_EMBEDDING_DIMENSION,
        GRANITE_INFERENCE_PARAMETER_DTYPE,
        GRANITE_MODEL_EXCLUDED_FILES,
        GRANITE_MODEL_TREE_SHA256,
        GRANITE_SOURCE_PARAMETER_DTYPE,
        GRANITE_TOKEN_CAP,
        GRANITE_WEIGHT_SHA256,
        HISTORICAL_FEATURE_WINDOWS,
        PINNED_EMBEDDING_SOFTWARE,
        load_eligible_vintage_window,
        sha256_file,
    )


MAX_SEQUENCE_LENGTH = GRANITE_TOKEN_CAP
EMBEDDING_DIMENSION = GRANITE_EMBEDDING_DIMENSION
DEFAULT_BATCH_SIZE = 32
EncodeFunction = Callable[[Sequence[str], int], np.ndarray]


@dataclass(frozen=True)
class LoadedGraniteEncoder:
    """Frozen callable plus the parameter dtypes observed around CPU upcasting."""

    encode: EncodeFunction
    source_parameter_dtype: str
    inference_parameter_dtype: str


def build_historical_embedding_cache(
    *,
    historical_root: Path,
    vintage_root: Path,
    model_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    encode_override: EncodeFunction | None = None,
) -> dict[str, Any]:
    """Encode all and only H1--H12 filing features and write a hashed cache."""

    historical_root = Path(historical_root)
    vintage_root = Path(vintage_root)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device != "cpu":
        raise ValueError("historical Granite cache generation is CPU-only")
    software_versions = _verified_software_versions()
    if output_dir.resolve() in {
        historical_root.resolve(),
        vintage_root.resolve(),
        model_dir.resolve(),
    }:
        raise ValueError("output directory must be separate from every input root")

    expected_outputs = _expected_output_names()
    _prepare_output_directory(output_dir, expected_outputs, overwrite=overwrite)

    model_file_hashes, model_tree_sha256 = model_tree_hashes(model_dir)
    weight_path = model_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    weight_sha256 = sha256_file(weight_path)
    if encode_override is None and weight_sha256 != GRANITE_WEIGHT_SHA256:
        raise ValueError(
            "Granite model.safetensors does not match the frozen experiment weight"
        )
    if encode_override is None and model_tree_sha256 != GRANITE_MODEL_TREE_SHA256:
        raise ValueError(
            "Granite selected model tree does not match the frozen experiment snapshot"
        )

    prepared = {
        window: load_eligible_vintage_window(
            window,
            historical_root=historical_root,
            vintage_root=vintage_root,
        )
        for window in HISTORICAL_FEATURE_WINDOWS
    }

    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}
    for window in HISTORICAL_FEATURE_WINDOWS:
        for text in prepared[window].texts:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
    if not unique_texts:
        raise ValueError("historical filing corpus has no eligible text")

    if encode_override is None:
        loaded_encoder = _load_frozen_granite(model_dir, device=device)
    else:
        loaded_encoder = LoadedGraniteEncoder(
            encode=encode_override,
            source_parameter_dtype="test_override",
            inference_parameter_dtype="test_override",
        )
    unique_embeddings = np.asarray(
        loaded_encoder.encode(unique_texts, batch_size), dtype=np.float32
    )
    expected_shape = (len(unique_texts), EMBEDDING_DIMENSION)
    if unique_embeddings.shape != expected_shape:
        raise ValueError(
            f"Granite output shape {unique_embeddings.shape}, expected {expected_shape}"
        )
    if not np.isfinite(unique_embeddings).all():
        raise ValueError("Granite produced a non-finite embedding")
    norms = np.linalg.norm(unique_embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError("Granite embeddings are not row-normalized")

    artifacts: dict[str, dict[str, Any]] = {}
    input_files: dict[str, dict[str, str]] = {}
    total_rows = 0
    for window in HISTORICAL_FEATURE_WINDOWS:
        window_data = prepared[window]
        total_rows += len(window_data.tickers)
        positions = np.fromiter(
            (text_to_index[text] for text in window_data.texts), dtype=np.int64
        )
        window_embeddings = np.asarray(unique_embeddings[positions], dtype=np.float32)

        ticker_path = output_dir / f"{window}_eligible_tickers.json"
        embedding_path = output_dir / f"{window}_granite_512.npy"
        _atomic_json(ticker_path, list(window_data.tickers))
        _atomic_numpy(embedding_path, window_embeddings)
        artifacts[ticker_path.name] = {
            "window": window,
            "kind": "eligible_tickers",
            "sha256": sha256_file(ticker_path),
            "rows": len(window_data.tickers),
            "row_order": "lexicographically sorted eligible feature-universe ticker",
        }
        artifacts[embedding_path.name] = {
            "window": window,
            "kind": "granite_embedding",
            "sha256": sha256_file(embedding_path),
            "shape": list(window_embeddings.shape),
            "dtype": str(window_embeddings.dtype),
            "row_order_file": ticker_path.name,
        }
        input_files[window] = {
            _input_label(path, historical_root, vintage_root): sha256_file(path)
            for path in window_data.input_paths
        }

    source_files = (Path(__file__).resolve(), Path(__file__).with_name("historical_data.py"))
    manifest: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "feature_windows": list(HISTORICAL_FEATURE_WINDOWS),
        "protocol": {
            "text_source": "feature-window eligible vintage 10-K Item 1",
            "vintage_rule": "filing_date strictly before feature-window start",
            "ticker_order": "sorted feature-universe/text intersection",
            "deduplication": "exact Python string equality across H1--H12",
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "normalize_embeddings": True,
            "output_dtype": "float32",
            "batch_size": batch_size,
            "device": device,
            "cpu_dtype_policy": GRANITE_CPU_DTYPE_POLICY,
        },
        "counts": {
            "eligible_window_rows": total_rows,
            "unique_exact_texts": len(unique_texts),
            "duplicate_encodes_avoided": total_rows - len(unique_texts),
        },
        "unique_text_sequence_sha256": _hash_strings(unique_texts),
        "model": {
            "name": "ibm-granite/granite-embedding-english-r2",
            "local_directory_name": model_dir.name,
            "loader": "sentence_transformers.SentenceTransformer",
            "weight_file": weight_path.name,
            "weight_sha256": weight_sha256,
            "expected_weight_sha256": GRANITE_WEIGHT_SHA256,
            "frozen_weight_match_enforced": encode_override is None,
            "model_tree_sha256": model_tree_sha256,
            "expected_model_tree_sha256": GRANITE_MODEL_TREE_SHA256,
            "model_tree_match_enforced": encode_override is None,
            "model_tree_excluded_files": sorted(GRANITE_MODEL_EXCLUDED_FILES),
            "model_file_sha256": model_file_hashes,
            "source_parameter_dtype": loaded_encoder.source_parameter_dtype,
            "inference_parameter_dtype": loaded_encoder.inference_parameter_dtype,
            "cpu_dtype_policy": GRANITE_CPU_DTYPE_POLICY,
        },
        "inputs": input_files,
        "source_code_sha256": {
            path.name: sha256_file(path) for path in source_files
        },
        "software": {
            "python": platform.python_version(),
            "packages": software_versions,
            "pinned_packages": dict(PINNED_EMBEDDING_SOFTWARE),
            "version_match_enforced": True,
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "cache_manifest.json"
    _atomic_json(manifest_path, manifest)
    checksum_paths = [output_dir / name for name in sorted(artifacts)] + [manifest_path]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
    )
    _atomic_text(output_dir / "SHA256SUMS.txt", checksum_text)
    return manifest


def model_tree_hashes(model_dir: Path) -> tuple[dict[str, str], str]:
    """Hash the authoritative selected model tree used by the W-cache launcher."""

    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    file_hashes: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        if relative in GRANITE_MODEL_EXCLUDED_FILES:
            continue
        file_hashes[relative] = sha256_file(path)
    if not file_hashes:
        raise ValueError(f"model snapshot is empty: {model_dir}")
    tree_hash = _hash_strings(
        [f"{name}:{digest}" for name, digest in sorted(file_hashes.items())]
    )
    return file_hashes, tree_hash


def _load_frozen_granite(
    model_dir: Path, *, device: str
) -> LoadedGraniteEncoder:
    import torch
    from sentence_transformers import SentenceTransformer

    if device != "cpu":
        raise ValueError("historical Granite cache generation is CPU-only")
    model = SentenceTransformer(str(model_dir), device=device)
    source_dtype = _single_parameter_dtype(model)
    if source_dtype != GRANITE_SOURCE_PARAMETER_DTYPE:
        raise ValueError(
            f"Granite source parameter dtype {source_dtype} does not match "
            f"{GRANITE_SOURCE_PARAMETER_DTYPE}"
        )
    # This is the exact policy used to produce the active W1--W6 Granite caches.
    model = model.float()
    inference_dtype = _single_parameter_dtype(model)
    if inference_dtype != GRANITE_INFERENCE_PARAMETER_DTYPE:
        raise ValueError(
            f"Granite inference parameter dtype {inference_dtype} does not match "
            f"{GRANITE_INFERENCE_PARAMETER_DTYPE}"
        )
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dimension = model.get_sentence_embedding_dimension()
    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(f"Granite reports embedding dimension {dimension}")

    def encode(texts: Sequence[str], batch_size: int) -> np.ndarray:
        with torch.inference_mode():
            values = model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return np.asarray(values, dtype=np.float32)

    return LoadedGraniteEncoder(
        encode=encode,
        source_parameter_dtype=source_dtype,
        inference_parameter_dtype=inference_dtype,
    )


def _single_parameter_dtype(model: Any) -> str:
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if len(dtypes) != 1:
        raise ValueError(f"Granite parameters have mixed dtypes: {sorted(dtypes)}")
    return next(iter(dtypes))


def _verified_software_versions() -> dict[str, str]:
    observed = {
        "numpy": np.__version__,
        "torch": importlib.metadata.version("torch"),
        "sentence-transformers": importlib.metadata.version("sentence-transformers"),
        "transformers": importlib.metadata.version("transformers"),
    }
    if observed != PINNED_EMBEDDING_SOFTWARE:
        raise RuntimeError(
            "embedding software differs from the pinned environment: "
            f"observed={observed}, expected={PINNED_EMBEDDING_SOFTWARE}"
        )
    return observed


def _prepare_output_directory(
    output_dir: Path,
    expected_names: set[str],
    *,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = list(output_dir.iterdir())
    if not entries:
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory is nonempty; pass --overwrite to replace this cache: {output_dir}"
        )
    unexpected = [path for path in entries if not path.is_file() or path.name not in expected_names]
    if unexpected:
        raise FileExistsError(
            "safe overwrite refused because output contains unexpected entries: "
            + ", ".join(path.name for path in unexpected)
        )


def _expected_output_names() -> set[str]:
    names = {"cache_manifest.json", "SHA256SUMS.txt"}
    for window in HISTORICAL_FEATURE_WINDOWS:
        names.add(f"{window}_eligible_tickers.json")
        names.add(f"{window}_granite_512.npy")
    return names


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _input_label(path: Path, historical_root: Path, vintage_root: Path) -> str:
    resolved = path.resolve()
    for prefix, label in (
        (historical_root.resolve(), "historical_windows"),
        (vintage_root.resolve(), "vintage_descriptions"),
    ):
        try:
            return f"{label}/{resolved.relative_to(prefix).as_posix()}"
        except ValueError:
            continue
    raise ValueError(f"input path lies outside declared historical roots: {path}")


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=repository_root / "results" / "ea_historical" / "windows",
    )
    parser.add_argument(
        "--vintage-root",
        type=Path,
        default=repository_root / "results" / "vintage_descriptions",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repository_root / "models" / "granite-embedding-english-r2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root
        / "experiments"
        / "multimodal_peer_encoder"
        / "artifacts"
        / "historical_text_cache",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only recognized files in an existing cache directory",
    )
    args = parser.parse_args()
    manifest = build_historical_embedding_cache(
        historical_root=args.historical_root,
        vintage_root=args.vintage_root,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(
        f"cache={args.output_dir} rows={manifest['counts']['eligible_window_rows']} "
        f"unique_texts={manifest['counts']['unique_exact_texts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
