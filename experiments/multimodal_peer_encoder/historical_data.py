"""Historical-only inputs for the pre-lock multimodal experiment.

This module intentionally knows only the H1--H13 data tree.  A transition uses
the residual path from H_t, the filing available before H_t begins, and the
realized residual-correlation matrix from H_{t+1}.  The ticker intersection is
sorted, while feature and target row positions are resolved independently.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


HISTORICAL_FEATURE_WINDOWS = tuple(f"H{i}" for i in range(1, 13))
HISTORICAL_WINDOWS = tuple(f"H{i}" for i in range(1, 14))
CACHE_SCHEMA_VERSION = "historical-granite-cache-v1"
GRANITE_WEIGHT_SHA256 = (
    "23bd9625b32680f180703408ed9d8d7b3a9c1efe4c950111e3ae9d9695b21b29"
)
GRANITE_MODEL_TREE_SHA256 = (
    "198d7df8b72a9fd77f9823e5e278f60df77c4577c08dd54e6df9888ade0c6d99"
)
# Match scripts/launch_vintage_granite_longctx.py exactly.  The binary checkpoint
# is redundant with model.safetensors; the remaining two files are repository
# presentation metadata and do not affect SentenceTransformer inference.
GRANITE_MODEL_EXCLUDED_FILES = frozenset(
    {".gitattributes", "README.md", "pytorch_model.bin"}
)
GRANITE_EMBEDDING_DIMENSION = 768
GRANITE_TOKEN_CAP = 512
GRANITE_SOURCE_PARAMETER_DTYPE = "torch.bfloat16"
GRANITE_INFERENCE_PARAMETER_DTYPE = "torch.float32"
GRANITE_CPU_DTYPE_POLICY = (
    "CPU-only; explicitly call model.float() before encoding"
)
PINNED_EMBEDDING_SOFTWARE = {
    "numpy": "2.4.2",
    "torch": "2.11.0",
    "sentence-transformers": "5.3.0",
    "transformers": "5.5.0",
}
_WINDOW_PATTERN = re.compile(r"H([1-9]|1[0-3])\Z")


@dataclass(frozen=True)
class EligibleVintageWindow:
    """Feature-window securities with a strictly vintage filing excerpt."""

    window: str
    start: date
    tickers: tuple[str, ...]
    texts: tuple[str, ...]
    filing_dates: tuple[str, ...]
    input_paths: tuple[Path, ...]


@dataclass(frozen=True)
class HistoricalTransition:
    """Ticker-aligned inputs and realized target for one H_t to H_{t+1} step."""

    transition: str
    feature_window: str
    target_window: str
    tickers: tuple[str, ...]
    ts_features: np.ndarray
    text_embeddings: np.ndarray | None
    corr_target: np.ndarray
    texts: tuple[str, ...]
    filing_dates: tuple[str, ...]
    feature_indices: np.ndarray
    target_indices: np.ndarray
    input_paths: tuple[Path, ...]

    @property
    def returns(self) -> np.ndarray:
        return self.ts_features

    @property
    def realized(self) -> np.ndarray:
        return self.corr_target

    @property
    def X(self) -> np.ndarray:  # noqa: N802 - X/Y mirror the protocol notation.
        return self.ts_features

    @property
    def Y(self) -> np.ndarray:  # noqa: N802
        return self.corr_target


@dataclass(frozen=True)
class _WindowIndex:
    name: str
    start: date
    end: date
    tickers: tuple[str, ...]
    ticker_to_index: dict[str, int]
    universe_path: Path
    meta_path: Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_eligible_vintage_window(
    window: str,
    *,
    historical_root: Path,
    vintage_root: Path,
) -> EligibleVintageWindow:
    """Load feature-universe filings and enforce label and timing invariants."""

    _require_feature_window(window)
    historical_root = Path(historical_root)
    vintage_root = Path(vintage_root)
    index = _load_window_index(window, historical_root)
    vintage_path = vintage_root / f"{window}_descriptions.parquet"
    if not vintage_path.is_file():
        raise FileNotFoundError(vintage_path)

    frame = pl.read_parquet(vintage_path)
    required = {"ticker", "window", "source", "item1_text", "filing_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{window}: vintage file lacks columns {missing}")

    labels = frame.get_column("window").to_list()
    if any(label is None or str(label) != window for label in labels):
        raise ValueError(f"{window}: vintage file contains a wrong window label")

    eligible = frame.filter(
        (pl.col("source") == "10K")
        & pl.col("item1_text").is_not_null()
        & (pl.col("item1_text").str.len_chars() > 0)
    )
    duplicate_rows = eligible.group_by("ticker").len().filter(pl.col("len") > 1)
    if duplicate_rows.height:
        raise ValueError(f"{window}: duplicate eligible vintage ticker")

    text_by_ticker: dict[str, str] = {}
    filing_by_ticker: dict[str, str] = {}
    for row in eligible.select(
        "ticker", "item1_text", "filing_date"
    ).iter_rows(named=True):
        if row["ticker"] is None or row["filing_date"] is None:
            raise ValueError(f"{window}: eligible filing has a null ticker or date")
        ticker = str(row["ticker"])
        filing = date.fromisoformat(str(row["filing_date"]))
        if filing >= index.start:
            raise ValueError(
                f"{window}: {ticker} filing {filing} is not before {index.start}"
            )
        text_by_ticker[ticker] = str(row["item1_text"])
        filing_by_ticker[ticker] = filing.isoformat()

    tickers = tuple(sorted(set(index.tickers) & set(text_by_ticker)))
    texts = tuple(text_by_ticker[ticker] for ticker in tickers)
    filing_dates = tuple(filing_by_ticker[ticker] for ticker in tickers)
    return EligibleVintageWindow(
        window=window,
        start=index.start,
        tickers=tickers,
        texts=texts,
        filing_dates=filing_dates,
        input_paths=(index.universe_path, index.meta_path, vintage_path),
    )


def load_historical_transition(
    index: int | str,
    *,
    historical_root: Path,
    vintage_root: Path,
    cache_dir: Path | None,
    verify_cache: bool = True,
) -> HistoricalTransition:
    """Build one H-only transition with independent feature/target indexing."""

    feature_number = _feature_number(index)
    feature_name = f"H{feature_number}"
    target_name = f"H{feature_number + 1}"
    historical_root = Path(historical_root)

    feature = _load_window_index(feature_name, historical_root)
    target = _load_window_index(target_name, historical_root)
    if target.start <= feature.start:
        raise ValueError(f"{feature_name}->{target_name}: non-increasing window dates")
    vintage = load_eligible_vintage_window(
        feature_name,
        historical_root=historical_root,
        vintage_root=Path(vintage_root),
    )

    tickers = tuple(
        sorted(set(feature.tickers) & set(target.tickers) & set(vintage.tickers))
    )
    if not tickers:
        raise ValueError(f"{feature_name}->{target_name}: empty eligible intersection")
    feature_indices = np.fromiter(
        (feature.ticker_to_index[ticker] for ticker in tickers), dtype=np.int64
    )
    target_indices = np.fromiter(
        (target.ticker_to_index[ticker] for ticker in tickers), dtype=np.int64
    )

    feature_path = historical_root / feature_name / "ts_features.npy"
    target_path = historical_root / target_name / "corr_target.npy"
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    feature_values = np.load(feature_path, mmap_mode="r")
    target_values = np.load(target_path, mmap_mode="r")
    if feature_values.ndim != 2 or feature_values.shape[0] != len(feature.tickers):
        raise ValueError(f"{feature_name}: ts_features shape/universe mismatch")
    if target_values.shape != (len(target.tickers), len(target.tickers)):
        raise ValueError(f"{target_name}: corr_target shape/universe mismatch")

    ts_features = np.asarray(feature_values[feature_indices], dtype=np.float32)
    corr_target = np.asarray(
        target_values[np.ix_(target_indices, target_indices)], dtype=np.float32
    )
    if not np.isfinite(ts_features).all() or not np.isfinite(corr_target).all():
        raise ValueError(f"{feature_name}->{target_name}: non-finite return input")

    vintage_position = {ticker: i for i, ticker in enumerate(vintage.tickers)}
    texts = tuple(vintage.texts[vintage_position[ticker]] for ticker in tickers)
    filing_dates = tuple(
        vintage.filing_dates[vintage_position[ticker]] for ticker in tickers
    )
    text_embeddings: np.ndarray | None = None
    cache_paths: tuple[Path, ...] = ()
    if cache_dir is not None:
        expected_cache_inputs = {
            f"historical_windows/{feature_name}/universe.parquet": sha256_file(
                feature.universe_path
            ),
            f"historical_windows/{feature_name}/meta.json": sha256_file(
                feature.meta_path
            ),
            f"vintage_descriptions/{feature_name}_descriptions.parquet": sha256_file(
                Path(vintage_root) / f"{feature_name}_descriptions.parquet"
            ),
        }
        text_embeddings, cache_paths = load_cached_embeddings(
            feature_name,
            tickers,
            cache_dir=Path(cache_dir),
            verify_manifest=verify_cache,
            expected_input_sha256=expected_cache_inputs,
        )

    input_paths = _unique_paths(
        (
            feature_path,
            target_path,
            feature.universe_path,
            feature.meta_path,
            target.universe_path,
            target.meta_path,
            *vintage.input_paths,
            *cache_paths,
        )
    )
    return HistoricalTransition(
        transition=f"{feature_name}->{target_name}",
        feature_window=feature_name,
        target_window=target_name,
        tickers=tickers,
        ts_features=ts_features,
        text_embeddings=text_embeddings,
        corr_target=corr_target,
        texts=texts,
        filing_dates=filing_dates,
        feature_indices=feature_indices,
        target_indices=target_indices,
        input_paths=input_paths,
    )


def load_all_historical_transitions(
    *,
    historical_root: Path,
    vintage_root: Path,
    cache_dir: Path | None,
    verify_cache: bool = True,
) -> tuple[HistoricalTransition, ...]:
    """Load H1->H2 through H12->H13 in chronological order."""

    return tuple(
        load_historical_transition(
            index,
            historical_root=historical_root,
            vintage_root=vintage_root,
            cache_dir=cache_dir,
            verify_cache=verify_cache,
        )
        for index in range(1, 13)
    )


def load_cached_embeddings(
    window: str,
    tickers: tuple[str, ...],
    *,
    cache_dir: Path,
    verify_manifest: bool = True,
    expected_input_sha256: dict[str, str] | None = None,
) -> tuple[np.ndarray, tuple[Path, ...]]:
    """Align an eligible-window Granite cache to transition tickers."""

    _require_feature_window(window)
    cache_dir = Path(cache_dir)
    ticker_path = cache_dir / f"{window}_eligible_tickers.json"
    embedding_path = cache_dir / f"{window}_granite_512.npy"
    for path in (ticker_path, embedding_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    cached_tickers_raw = json.loads(ticker_path.read_text(encoding="utf-8"))
    if not isinstance(cached_tickers_raw, list) or not all(
        isinstance(ticker, str) for ticker in cached_tickers_raw
    ):
        raise ValueError(f"{window}: eligible ticker cache must be a JSON string list")
    cached_tickers = tuple(cached_tickers_raw)
    if cached_tickers != tuple(sorted(set(cached_tickers))):
        raise ValueError(f"{window}: cache tickers must be sorted and unique")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(cached_tickers):
        raise ValueError(f"{window}: embedding cache row mismatch")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError(f"{window}: embeddings must be finite float32")
    cache_index = {ticker: i for i, ticker in enumerate(cached_tickers)}
    missing = sorted(set(tickers) - set(cache_index))
    if missing:
        raise ValueError(f"{window}: transition tickers absent from cache: {missing[:5]}")

    manifest_path = cache_dir / "cache_manifest.json"
    if verify_manifest:
        _verify_cache_artifacts(
            manifest_path,
            window=window,
            artifact_paths=(ticker_path, embedding_path),
            expected_input_sha256=expected_input_sha256,
        )
    selected = np.fromiter((cache_index[ticker] for ticker in tickers), dtype=np.int64)
    aligned = np.asarray(embeddings[selected], dtype=np.float32)
    paths = (ticker_path, embedding_path)
    if verify_manifest:
        paths = (*paths, manifest_path)
    return aligned, paths


def _load_window_index(name: str, historical_root: Path) -> _WindowIndex:
    _require_historical_window(name)
    directory = historical_root / name
    universe_path = directory / "universe.parquet"
    meta_path = directory / "meta.json"
    for path in (universe_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    universe = pl.read_parquet(universe_path)
    required = {"ticker", "company_idx"}
    missing = sorted(required - set(universe.columns))
    if missing:
        raise ValueError(f"{name}: universe lacks columns {missing}")
    raw_tickers = universe.get_column("ticker").to_list()
    if any(ticker is None for ticker in raw_tickers):
        raise ValueError(f"{name}: null ticker in universe")
    tickers = tuple(str(ticker) for ticker in raw_tickers)
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"{name}: duplicate ticker in universe")

    company_indices = universe.get_column("company_idx").to_numpy()
    if not np.array_equal(company_indices, np.arange(len(tickers))):
        raise ValueError(f"{name}: company_idx does not match stored row order")

    metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("window", name) != name:
        raise ValueError(f"{name}: metadata window label mismatch")
    try:
        start = date.fromisoformat(str(metadata["start"]))
        end = date.fromisoformat(str(metadata["end"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{name}: invalid start/end metadata") from exc
    if end < start:
        raise ValueError(f"{name}: end date precedes start date")

    return _WindowIndex(
        name=name,
        start=start,
        end=end,
        tickers=tickers,
        ticker_to_index={ticker: i for i, ticker in enumerate(tickers)},
        universe_path=universe_path,
        meta_path=meta_path,
    )


def _verify_cache_artifacts(
    manifest_path: Path,
    *,
    window: str,
    artifact_paths: tuple[Path, ...],
    expected_input_sha256: dict[str, str] | None,
) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("embedding-cache schema mismatch")
    if manifest.get("feature_windows") != list(HISTORICAL_FEATURE_WINDOWS):
        raise ValueError("embedding cache does not cover exactly H1--H12")
    protocol = manifest.get("protocol", {})
    expected_protocol = {
        "text_source": "feature-window eligible vintage 10-K Item 1",
        "vintage_rule": "filing_date strictly before feature-window start",
        "ticker_order": "sorted feature-universe/text intersection",
        "deduplication": "exact Python string equality across H1--H12",
        "max_sequence_length": GRANITE_TOKEN_CAP,
        "normalize_embeddings": True,
        "output_dtype": "float32",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"embedding cache protocol mismatch: {key}")
    model = manifest.get("model", {})
    expected_model = {
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
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"embedding cache model mismatch: {key}")

    software = manifest.get("software", {})
    if software.get("packages") != PINNED_EMBEDDING_SOFTWARE:
        raise ValueError("embedding cache software-version mismatch")
    if software.get("pinned_packages") != PINNED_EMBEDDING_SOFTWARE:
        raise ValueError("embedding cache pinned-software declaration mismatch")
    if software.get("version_match_enforced") is not True:
        raise ValueError("embedding cache did not enforce pinned software versions")

    expected_source = {
        "build_historical_embeddings.py": sha256_file(
            Path(__file__).with_name("build_historical_embeddings.py")
        ),
        "historical_data.py": sha256_file(Path(__file__)),
    }
    if manifest.get("source_code_sha256") != expected_source:
        raise ValueError("embedding cache source-code identity mismatch")
    if expected_input_sha256 is not None:
        window_inputs = manifest.get("inputs", {}).get(window)
        if window_inputs != expected_input_sha256:
            raise ValueError(f"{window}: embedding-cache input identity mismatch")

    artifacts = manifest.get("artifacts", {})
    for path in artifact_paths:
        record = artifacts.get(path.name)
        if not isinstance(record, dict) or record.get("window") != window:
            raise ValueError(f"manifest lacks artifact record for {path.name}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"cache hash mismatch: {path.name}")
        if path.name.endswith("_granite_512.npy"):
            shape = record.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or shape[1] != GRANITE_EMBEDDING_DIMENSION
            ):
                raise ValueError(f"{window}: embedding-cache dimension mismatch")
            if record.get("dtype") != "float32":
                raise ValueError(f"{window}: embedding-cache dtype mismatch")
            if record.get("row_order_file") != f"{window}_eligible_tickers.json":
                raise ValueError(f"{window}: embedding-cache row-order record mismatch")


def _feature_number(index: int | str) -> int:
    if isinstance(index, str):
        _require_feature_window(index)
        number = int(index[1:])
    elif isinstance(index, int) and not isinstance(index, bool):
        number = index
    else:
        raise TypeError("transition index must be an integer or H-window label")
    if number < 1 or number > 12:
        raise ValueError("historical transition index must lie in [1, 12]")
    return number


def _require_historical_window(window: str) -> None:
    if not isinstance(window, str) or _WINDOW_PATTERN.fullmatch(window) is None:
        raise ValueError(f"invalid historical window: {window!r}")


def _require_feature_window(window: str) -> None:
    _require_historical_window(window)
    if window not in HISTORICAL_FEATURE_WINDOWS:
        raise ValueError(f"embedding feature window must be H1--H12, received {window}")


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = Path(path)
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)
