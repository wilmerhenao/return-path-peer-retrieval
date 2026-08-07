"""Run the actual-vintage Granite long-context census inside SageMaker.

This script is intentionally narrow:

* Granite Embedding English R2 only;
* actual point-in-time 10-K Item 1 excerpts for feature windows W1--W6;
* ``max_seq_length=8192``, normalized embeddings, and batch size one;
* every eligible query under both the legacy symmetric-coverage universe and
  the feature-time-only universe; and
* the repository's canonical positive-correlation NDCG@5 implementation.

Expected SageMaker channels
---------------------------
``windows``
    W1--W7 ``universe.parquet`` and ``meta.json`` files, W1--W6
    ``ts_features.npy`` files, and W2--W7 ``corr_target.npy`` files.
``vintage``
    W1--W7 ``W*_descriptions.parquet`` files. W7 is used only to reproduce
    the legacy symmetric-coverage rule; only W1--W6 texts are embedded.
``models``
    A tarball containing ``granite-embedding-english-r2/``.
``code``
    This script and ``metrics.py``.

Outputs in ``SM_OUTPUT_DATA_DIR``
---------------------------------
``granite_longctx_census.json``
    Complete protocol, summaries, transition results, and provenance.
``per_transition.csv``
    Transition-level descriptive results.
``per_query_ndcg.csv``
    All-query method scores (no query subsampling).
``token_length_stats.csv``
    Actual tokenizer-length audit by feature window and in aggregate.
``provenance_manifest.json``
    Full SHA-256 input and output manifest.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# This must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import boto3
import numpy as np
import polars as pl
import torch
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Fixed protocol
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 8192
BATCH_SIZE = 1
K = 5
MODEL_NAME = "granite-embedding-english-r2"
MODEL_LABEL = "Granite-r2-longctx"
RETURN_LABEL = "ReturnPath"
WINDOWS = [f"W{i}" for i in range(1, 8)]
FEATURE_WINDOWS = WINDOWS[:-1]
TRANSITIONS = list(zip(WINDOWS[:-1], WINDOWS[1:]))
PROTOCOLS = ("symmetric", "feature_time_only")
SCHEMA_VERSION = 1

SM_INPUT = Path(os.environ.get("SM_INPUT_DIR", "/opt/ml/input"))
SM_OUTPUT = Path(
    os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
)
WINDOWS_DIR = SM_INPUT / "data" / "windows"
VINTAGE_DIR = SM_INPUT / "data" / "vintage"
MODELS_DIR = SM_INPUT / "data" / "models"
CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = MODELS_DIR / MODEL_NAME
TMP_EMBEDDINGS = Path("/tmp/granite_longctx_embeddings")

S3_BUCKET = os.environ.get("SAGEMAKER_S3_BUCKET")
S3_CHECKPOINT_PREFIX = os.environ.get(
    "GRANITE_CHECKPOINT_PREFIX",
    "icaif2026-granite-longctx-census-v1/checkpoints",
).strip("/")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
RUN_ID = os.environ.get("GRANITE_RUN_ID", "unknown")
TRAINING_JOB_NAME = os.environ.get(
    "TRAINING_JOB_NAME",
    os.environ.get("SAGEMAKER_JOB_NAME", "unknown"),
)

sys.path.insert(0, str(CODE_DIR))
from metrics import ndcg_at_k as canonical_ndcg_at_k  # noqa: E402


def utc_now() -> str:
    """Return a compact, timezone-aware UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(parts: Iterable[str]) -> str:
    """Hash a sequence with unambiguous separators."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def relative_channel_name(path: Path) -> str:
    """Return an anonymous, stable channel-relative provenance label."""
    for root, label in (
        (WINDOWS_DIR, "windows"),
        (VINTAGE_DIR, "vintage"),
        (MODELS_DIR, "models"),
        (CODE_DIR, "code"),
    ):
        try:
            return f"{label}/{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return path.name


def ensure_model_extracted() -> None:
    """Extract the one model tarball if the launcher did not do so."""
    if MODEL_DIR.is_dir():
        return
    archives = sorted(MODELS_DIR.glob("*.tar.gz"))
    if len(archives) != 1:
        raise FileNotFoundError(
            f"expected one Granite model tarball in {MODELS_DIR}, found {archives}"
        )
    import tarfile

    print(f"Extracting {archives[0].name} ...", flush=True)
    with tarfile.open(archives[0], "r:gz") as archive:
        destination = MODELS_DIR.resolve()
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"unsafe model archive member: {member.name}")
        archive.extractall(MODELS_DIR)
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"archive did not create {MODEL_DIR}")


def required_input_files() -> list[Path]:
    """List every data/code file that can affect the census."""
    files: list[Path] = []
    for window in WINDOWS:
        files.append(WINDOWS_DIR / window / "universe.parquet")
        files.append(WINDOWS_DIR / window / "meta.json")
    for window in FEATURE_WINDOWS:
        files.append(WINDOWS_DIR / window / "ts_features.npy")
    for window in WINDOWS:
        files.append(VINTAGE_DIR / f"{window}_descriptions.parquet")
    for window in WINDOWS[1:]:
        files.append(WINDOWS_DIR / window / "corr_target.npy")
    files.extend([CODE_DIR / "train_vintage_granite_longctx.py", CODE_DIR / "metrics.py"])
    return files


def validate_inputs() -> None:
    missing = [path for path in required_input_files() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing required SageMaker inputs:\n{formatted}")
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing {MODEL_DIR / 'model.safetensors'}")


def model_file_hashes() -> dict[str, str]:
    """Hash the exact extracted model tree used by SentenceTransformer."""
    result: dict[str, str] = {}
    for path in sorted(MODEL_DIR.rglob("*")):
        if path.is_file():
            result[path.relative_to(MODEL_DIR).as_posix()] = sha256_file(path)
    return result


def package_versions() -> dict[str, str]:
    names = (
        "boto3",
        "numpy",
        "polars",
        "pyarrow",
        "sentence-transformers",
        "torch",
        "transformers",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def load_window_inputs(window: str) -> dict[str, Any]:
    """Load universe order and the one vintage 10-K excerpt per covered ticker."""
    universe_path = WINDOWS_DIR / window / "universe.parquet"
    meta_path = WINDOWS_DIR / window / "meta.json"
    vintage_path = VINTAGE_DIR / f"{window}_descriptions.parquet"

    tickers = (
        pl.read_parquet(universe_path)
        .get_column("ticker")
        .cast(pl.String)
        .to_list()
    )
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"{window}: duplicate tickers in universe")

    vintage = pl.read_parquet(vintage_path)
    required_columns = {
        "ticker",
        "window",
        "filing_date",
        "item1_text",
        "source",
    }
    if not required_columns.issubset(vintage.columns):
        raise ValueError(
            f"{window}: vintage file lacks {sorted(required_columns - set(vintage.columns))}"
        )
    vintage = vintage.filter(
        (pl.col("source") == "10K")
        & pl.col("item1_text").is_not_null()
        & (pl.col("item1_text").str.len_chars() > 0)
    )
    wrong_window = vintage.filter(pl.col("window") != window).height
    if wrong_window:
        raise ValueError(
            f"{window}: {wrong_window} rows carry a different window label"
        )
    dated = vintage.with_columns(
        pl.col("filing_date").str.to_date(strict=True).alias("_filing_date")
    )
    cutoff = dt.date.fromisoformat(
        json.loads(meta_path.read_text(encoding="utf-8"))["start"]
    )
    late = dated.filter(pl.col("_filing_date") >= cutoff).height
    if late:
        raise ValueError(
            f"{window}: {late} filings are not strictly before "
            f"the feature-window start {cutoff.isoformat()}"
        )
    duplicate_count = (
        vintage.group_by("ticker")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(f"{window}: {duplicate_count} duplicate 10-K ticker rows")

    text_map = {
        str(row["ticker"]): str(row["item1_text"])
        for row in vintage.select("ticker", "item1_text").iter_rows(named=True)
    }
    covered = set(text_map).intersection(tickers)
    texts = [text_map.get(ticker, "") for ticker in tickers]
    covered_texts = [text_map[ticker] for ticker in tickers if ticker in covered]

    return {
        "tickers": tickers,
        "covered": covered,
        "texts": texts,
        "covered_texts": covered_texts,
        "n_vintage_10k": vintage.height,
        "universe_sha256": sha256_file(universe_path),
        "vintage_sha256": sha256_file(vintage_path),
    }


def tokenizer_lengths(tokenizer: Any, texts: list[str]) -> np.ndarray:
    """Tokenize without truncation and return special-token-inclusive lengths."""
    lengths: list[int] = []
    audit_batch_size = 64
    for start in range(0, len(texts), audit_batch_size):
        batch = texts[start : start + audit_batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        if isinstance(batch_lengths, int):
            batch_lengths = [batch_lengths]
        lengths.extend(int(value) for value in batch_lengths)
    return np.asarray(lengths, dtype=np.int64)


def length_summary(
    window: str,
    token_lengths: np.ndarray,
    char_lengths: np.ndarray,
) -> dict[str, Any]:
    """Create one auditable tokenizer-length row."""
    if token_lengths.size == 0:
        raise ValueError(f"{window}: no covered vintage 10-K texts")
    return {
        "window": window,
        "n_texts": int(token_lengths.size),
        "token_min": int(token_lengths.min()),
        "token_mean": float(token_lengths.mean()),
        "token_p50": float(np.percentile(token_lengths, 50)),
        "token_p90": float(np.percentile(token_lengths, 90)),
        "token_p95": float(np.percentile(token_lengths, 95)),
        "token_p99": float(np.percentile(token_lengths, 99)),
        "token_max": int(token_lengths.max()),
        "n_over_512": int(np.sum(token_lengths > 512)),
        "fraction_over_512": float(np.mean(token_lengths > 512)),
        "n_over_8192": int(np.sum(token_lengths > MAX_SEQ_LENGTH)),
        "char_min": int(char_lengths.min()),
        "char_mean": float(char_lengths.mean()),
        "char_p50": float(np.percentile(char_lengths, 50)),
        "char_p90": float(np.percentile(char_lengths, 90)),
        "char_max": int(char_lengths.max()),
    }


def audit_token_lengths(
    tokenizer: Any,
    window_inputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audit every actual-vintage feature text before any embedding is computed."""
    rows: list[dict[str, Any]] = []
    all_tokens: list[np.ndarray] = []
    all_chars: list[np.ndarray] = []

    for window in FEATURE_WINDOWS:
        texts = window_inputs[window]["covered_texts"]
        token_lengths = tokenizer_lengths(tokenizer, texts)
        char_lengths = np.asarray([len(text) for text in texts], dtype=np.int64)
        row = length_summary(window, token_lengths, char_lengths)
        rows.append(row)
        all_tokens.append(token_lengths)
        all_chars.append(char_lengths)
        print(
            f"{window}: n={row['n_texts']} tokens "
            f"p50={row['token_p50']:.0f} p90={row['token_p90']:.0f} "
            f"max={row['token_max']} >512={row['fraction_over_512']:.1%}",
            flush=True,
        )

    aggregate = length_summary(
        "ALL",
        np.concatenate(all_tokens),
        np.concatenate(all_chars),
    )
    rows.append(aggregate)
    if aggregate["n_over_8192"]:
        raise RuntimeError(
            f"{aggregate['n_over_8192']} actual Item 1 excerpts exceed "
            f"the {MAX_SEQ_LENGTH}-token model context; refusing to label this full-context"
        )
    return rows


def load_granite_model() -> tuple[Any, dict[str, Any]]:
    """Load Granite in FP16 on the T4 and explicitly enable its full context."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this dedicated SageMaker census; "
            "refusing an accidental CPU run"
        )

    from sentence_transformers import SentenceTransformer

    print(f"Loading {MODEL_NAME} on CPU before FP16 CUDA transfer ...", flush=True)
    model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    configured_sentence_length = int(model.max_seq_length)
    first_module = model._first_module()
    max_position_embeddings = int(
        getattr(first_module.auto_model.config, "max_position_embeddings", 0)
    )
    if max_position_embeddings < MAX_SEQ_LENGTH:
        raise RuntimeError(
            f"model supports {max_position_embeddings} positions, expected at least "
            f"{MAX_SEQ_LENGTH}"
        )

    model.max_seq_length = MAX_SEQ_LENGTH
    model.tokenizer.model_max_length = MAX_SEQ_LENGTH
    model.half()
    model.to("cuda")
    model.eval()
    parameter_dtype = str(next(model.parameters()).dtype)

    details = {
        "model_name": MODEL_NAME,
        "sentence_transformer_configured_max_seq_length": configured_sentence_length,
        "effective_max_seq_length": int(model.max_seq_length),
        "model_max_position_embeddings": max_position_embeddings,
        "batch_size": BATCH_SIZE,
        "normalize_embeddings": True,
        "inference_dtype": parameter_dtype,
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }
    print(
        f"Loaded on {details['gpu_name']}; dtype={parameter_dtype}; "
        f"max_seq_length={model.max_seq_length}",
        flush=True,
    )
    return model, details


def checkpoint_fingerprint(
    window: str,
    window_data: dict[str, Any],
    model_tree_sha256: str,
    versions: dict[str, str],
    inference_dtype: str,
) -> str:
    """Fingerprint every input that can change a window embedding matrix."""
    return sha256_strings(
        (
            "granite-longctx-checkpoint-v1",
            window,
            model_tree_sha256,
            window_data["universe_sha256"],
            window_data["vintage_sha256"],
            str(MAX_SEQ_LENGTH),
            str(BATCH_SIZE),
            "normalize_embeddings=True",
            inference_dtype,
            versions["sentence-transformers"],
            versions["transformers"],
            versions["torch"],
        )
    )


def checkpoint_key(window: str, fingerprint: str) -> str:
    return (
        f"{S3_CHECKPOINT_PREFIX}/{MODEL_NAME}/"
        f"{fingerprint[:2]}/{window}-{fingerprint}.npy"
    )


def validate_embedding_array(
    embeddings: np.ndarray,
    expected_rows: int,
    window: str,
) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{window}: embedding shape {embeddings.shape}, expected ({expected_rows}, D)"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{window}: embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3, rtol=0.0):
        raise ValueError(
            f"{window}: normalized-embedding check failed; "
            f"norm range [{norms.min():.6f}, {norms.max():.6f}]"
        )


def try_restore_checkpoint(
    s3: Any,
    window: str,
    fingerprint: str,
    expected_rows: int,
) -> tuple[np.ndarray, str] | None:
    if not S3_BUCKET:
        return None
    key = checkpoint_key(window, fingerprint)
    local_path = TMP_EMBEDDINGS / f"{window}-{fingerprint}.npy"
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise

    print(f"{window}: restoring content-addressed checkpoint ...", flush=True)
    s3.download_file(S3_BUCKET, key, str(local_path))
    expected_sha = head.get("Metadata", {}).get("sha256")
    actual_sha = sha256_file(local_path)
    if expected_sha and expected_sha != actual_sha:
        print(f"{window}: checkpoint SHA mismatch; recomputing", flush=True)
        local_path.unlink(missing_ok=True)
        return None
    try:
        embeddings = np.load(local_path, allow_pickle=False)
        validate_embedding_array(embeddings, expected_rows, window)
    except (OSError, ValueError) as exc:
        print(f"{window}: invalid checkpoint ({exc}); recomputing", flush=True)
        local_path.unlink(missing_ok=True)
        return None
    return embeddings.astype(np.float32, copy=False), actual_sha


def save_checkpoint(
    s3: Any,
    embeddings: np.ndarray,
    window: str,
    fingerprint: str,
) -> tuple[Path, str]:
    local_path = TMP_EMBEDDINGS / f"{window}-{fingerprint}.npy"
    np.save(local_path, embeddings.astype(np.float32))
    digest = sha256_file(local_path)
    if S3_BUCKET:
        key = checkpoint_key(window, fingerprint)
        print(f"{window}: uploading checkpoint s3://[bucket]/{key}", flush=True)
        s3.upload_file(
            str(local_path),
            S3_BUCKET,
            key,
            ExtraArgs={
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "sha256": digest,
                    "fingerprint": fingerprint,
                    "maxseqlength": str(MAX_SEQ_LENGTH),
                    "normalized": "true",
                },
            },
        )
    return local_path, digest


def encode_feature_windows(
    model: Any,
    window_inputs: dict[str, dict[str, Any]],
    model_tree_sha256: str,
    versions: dict[str, str],
    inference_dtype: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Encode or restore all six feature-window matrices."""
    TMP_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    s3 = (
        boto3.client("s3", region_name=AWS_REGION)
        if S3_BUCKET
        else None
    )
    embeddings_by_window: dict[str, np.ndarray] = {}
    checkpoint_provenance: dict[str, Any] = {}

    for window in FEATURE_WINDOWS:
        data = window_inputs[window]
        fingerprint = checkpoint_fingerprint(
            window,
            data,
            model_tree_sha256,
            versions,
            inference_dtype,
        )
        restored = (
            try_restore_checkpoint(s3, window, fingerprint, len(data["tickers"]))
            if s3 is not None
            else None
        )
        if restored is not None:
            embeddings, embedding_sha = restored
            source = "restored_content_addressed_s3_checkpoint"
        else:
            print(
                f"{window}: encoding {len(data['texts'])} universe rows "
                f"at max_seq_length={MAX_SEQ_LENGTH}, batch_size={BATCH_SIZE} ...",
                flush=True,
            )
            started = time.monotonic()
            embeddings = model.encode(
                data["texts"],
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            validate_embedding_array(embeddings, len(data["tickers"]), window)
            _, embedding_sha = save_checkpoint(
                s3,
                embeddings,
                window,
                fingerprint,
            )
            source = "computed"
            print(
                f"{window}: encoded shape={embeddings.shape} "
                f"in {time.monotonic() - started:.1f}s",
                flush=True,
            )

        embeddings_by_window[window] = embeddings
        checkpoint_provenance[window] = {
            "fingerprint": fingerprint,
            "embedding_npy_sha256": embedding_sha,
            "source": source,
            "shape": [int(value) for value in embeddings.shape],
            "s3_key": checkpoint_key(window, fingerprint),
        }
    return embeddings_by_window, checkpoint_provenance


def protocol_universe(
    protocol: str,
    current: dict[str, Any],
    future: dict[str, Any],
) -> list[str]:
    return sorted(
        set(current["tickers"])
        & set(future["tickers"])
        & current["covered"]
        & (future["covered"] if protocol == "symmetric" else set(future["tickers"]))
    )


def evaluate_census(
    window_inputs: dict[str, dict[str, Any]],
    embeddings_by_window: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Evaluate every eligible query under both coverage definitions."""
    transition_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for protocol in PROTOCOLS:
        print(f"\n=== Protocol: {protocol} ===", flush=True)
        for current_window, future_window in TRANSITIONS:
            current = window_inputs[current_window]
            future = window_inputs[future_window]
            common = protocol_universe(protocol, current, future)
            n_companies = len(common)
            if n_companies <= K:
                raise RuntimeError(
                    f"{protocol} {current_window}->{future_window}: only "
                    f"{n_companies} eligible companies"
                )

            current_index = {
                ticker: index for index, ticker in enumerate(current["tickers"])
            }
            future_index = {
                ticker: index for index, ticker in enumerate(future["tickers"])
            }
            idx_current = np.asarray(
                [current_index[ticker] for ticker in common],
                dtype=np.int64,
            )
            idx_future = np.asarray(
                [future_index[ticker] for ticker in common],
                dtype=np.int64,
            )

            target_full = np.load(
                WINDOWS_DIR / future_window / "corr_target.npy",
                allow_pickle=False,
            )
            target = target_full[np.ix_(idx_future, idx_future)].astype(
                np.float32,
                copy=False,
            )

            return_features = np.load(
                WINDOWS_DIR / current_window / "ts_features.npy",
                allow_pickle=False,
            )[idx_current]
            return_norms = np.linalg.norm(return_features, axis=1, keepdims=True)
            if np.any(return_norms <= 0.0):
                raise ValueError(
                    f"{current_window}: zero-norm return path in evaluation universe"
                )
            return_features = return_features / return_norms
            return_scores = (return_features @ return_features.T).astype(np.float32)

            granite = embeddings_by_window[current_window][idx_current]
            granite_scores = (granite @ granite.T).astype(np.float32)

            return_values: list[float] = []
            granite_values: list[float] = []
            for query_index, ticker in enumerate(common):
                realized = target[query_index].astype(float, copy=False)
                return_ndcg = canonical_ndcg_at_k(
                    return_scores[query_index],
                    realized,
                    query_index,
                    k=K,
                )
                granite_ndcg = canonical_ndcg_at_k(
                    granite_scores[query_index],
                    realized,
                    query_index,
                    k=K,
                )
                if (return_ndcg is None) != (granite_ndcg is None):
                    raise AssertionError(
                        "IDCG validity differs by method despite a shared target"
                    )
                valid = return_ndcg is not None
                if valid:
                    return_values.append(float(return_ndcg))
                    granite_values.append(float(granite_ndcg))
                query_rows.append(
                    {
                        "protocol": protocol,
                        "transition": f"{current_window}->{future_window}",
                        "query_index": query_index,
                        "query_ticker": ticker,
                        "n_candidates_including_query": n_companies,
                        "valid_positive_idcg": valid,
                        "return_path_ndcg5": (
                            float(return_ndcg) if return_ndcg is not None else None
                        ),
                        "granite_longctx_ndcg5": (
                            float(granite_ndcg) if granite_ndcg is not None else None
                        ),
                    }
                )

            if not return_values:
                raise RuntimeError(
                    f"{protocol} {current_window}->{future_window}: no valid queries"
                )
            return_mean = float(np.mean(return_values))
            granite_mean = float(np.mean(granite_values))
            row = {
                "protocol": protocol,
                "transition": f"{current_window}->{future_window}",
                "feature_window": current_window,
                "target_window": future_window,
                "universe_n": n_companies,
                "all_queries_n": n_companies,
                "valid_queries_n": len(return_values),
                "return_path_ndcg5": return_mean,
                "granite_longctx_ndcg5": granite_mean,
                "granite_minus_return": granite_mean - return_mean,
            }
            transition_rows.append(row)
            print(
                f"{row['transition']}: N={n_companies}, "
                f"ReturnPath={return_mean:.6f}, "
                f"Granite={granite_mean:.6f}, "
                f"delta={granite_mean - return_mean:+.6f}",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        selected = [row for row in transition_rows if row["protocol"] == protocol]
        return_means = np.asarray(
            [row["return_path_ndcg5"] for row in selected],
            dtype=float,
        )
        granite_means = np.asarray(
            [row["granite_longctx_ndcg5"] for row in selected],
            dtype=float,
        )
        summary[protocol] = {
            "n_transitions": len(selected),
            "query_sampling": "all eligible queries",
            "total_query_rows": int(sum(row["all_queries_n"] for row in selected)),
            "total_valid_queries": int(sum(row["valid_queries_n"] for row in selected)),
            "universe_n_min": int(min(row["universe_n"] for row in selected)),
            "universe_n_max": int(max(row["universe_n"] for row in selected)),
            "return_path_mean_across_transitions": float(return_means.mean()),
            "return_path_transition_sd_ddof0": float(return_means.std(ddof=0)),
            "granite_longctx_mean_across_transitions": float(granite_means.mean()),
            "granite_longctx_transition_sd_ddof0": float(granite_means.std(ddof=0)),
            "granite_minus_return_mean": float((granite_means - return_means).mean()),
            "return_path_transition_wins": int(np.sum(return_means > granite_means)),
            "granite_transition_wins": int(np.sum(granite_means > return_means)),
        }

    by_key = {
        (row["protocol"], row["transition"]): row for row in transition_rows
    }
    reconciliation: list[dict[str, Any]] = []
    for current_window, future_window in TRANSITIONS:
        transition = f"{current_window}->{future_window}"
        symmetric = by_key[("symmetric", transition)]
        feature_time = by_key[("feature_time_only", transition)]
        reconciliation.append(
            {
                "transition": transition,
                "symmetric_n": symmetric["universe_n"],
                "feature_time_only_n": feature_time["universe_n"],
                "feature_time_only_added_n": (
                    feature_time["universe_n"] - symmetric["universe_n"]
                ),
                "return_path_ndcg_delta_feature_time_minus_symmetric": (
                    feature_time["return_path_ndcg5"]
                    - symmetric["return_path_ndcg5"]
                ),
                "granite_ndcg_delta_feature_time_minus_symmetric": (
                    feature_time["granite_longctx_ndcg5"]
                    - symmetric["granite_longctx_ndcg5"]
                ),
            }
        )
    return transition_rows, query_rows, summary, reconciliation


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def input_hash_manifest(
    model_hashes: dict[str, str],
) -> dict[str, str]:
    result = {
        relative_channel_name(path): sha256_file(path)
        for path in required_input_files()
    }
    result.update(
        {
            f"models/{MODEL_NAME}/{name}": digest
            for name, digest in model_hashes.items()
        }
    )
    return dict(sorted(result.items()))


def run() -> None:
    started_at = utc_now()
    wall_start = time.monotonic()
    SM_OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    ensure_model_extracted()
    validate_inputs()
    versions = package_versions()
    model_hashes = model_file_hashes()
    model_tree_sha256 = sha256_strings(
        f"{name}:{digest}" for name, digest in sorted(model_hashes.items())
    )
    inputs_sha256 = input_hash_manifest(model_hashes)

    window_inputs = {
        window: load_window_inputs(window)
        for window in WINDOWS
    }
    model, model_details = load_granite_model()
    token_rows = audit_token_lengths(model.tokenizer, window_inputs)
    embeddings_by_window, checkpoints = encode_feature_windows(
        model,
        window_inputs,
        model_tree_sha256,
        versions,
        model_details["inference_dtype"],
    )

    transition_rows, query_rows, summary, reconciliation = evaluate_census(
        window_inputs,
        embeddings_by_window,
    )

    transition_csv = SM_OUTPUT / "per_transition.csv"
    query_csv = SM_OUTPUT / "per_query_ndcg.csv"
    token_csv = SM_OUTPUT / "token_length_stats.csv"
    result_json = SM_OUTPUT / "granite_longctx_census.json"
    provenance_json = SM_OUTPUT / "provenance_manifest.json"

    write_csv(
        transition_csv,
        transition_rows,
        [
            "protocol",
            "transition",
            "feature_window",
            "target_window",
            "universe_n",
            "all_queries_n",
            "valid_queries_n",
            "return_path_ndcg5",
            "granite_longctx_ndcg5",
            "granite_minus_return",
        ],
    )
    write_csv(
        query_csv,
        query_rows,
        [
            "protocol",
            "transition",
            "query_index",
            "query_ticker",
            "n_candidates_including_query",
            "valid_positive_idcg",
            "return_path_ndcg5",
            "granite_longctx_ndcg5",
        ],
    )
    write_csv(
        token_csv,
        token_rows,
        list(token_rows[0].keys()),
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Actual-vintage Granite-r2 full-context W1-W7 walk-forward census"
        ),
        "protocol": {
            "feature_text_windows": FEATURE_WINDOWS,
            "target_windows": WINDOWS[1:],
            "transitions": [
                f"{current}->{future}" for current, future in TRANSITIONS
            ],
            "text_source": (
                "one non-empty source=10K Item 1 excerpt available in each feature window"
            ),
            "coverage_definitions": {
                "symmetric": (
                    "return presence at t and t+1; vintage 10-K text at t and t+1"
                ),
                "feature_time_only": (
                    "return presence at t and t+1; vintage 10-K text at t only"
                ),
            },
            "query_sampling": "none; every eligible ticker is a query",
            "metric": (
                "canonical positive-correlation NDCG@5 from scripts/metrics.py; "
                "query excluded from retrieved and ideal lists; deterministic tie jitter"
            ),
            "relevance": "max(0, next-window LW-shrunk residual correlation)",
            "return_path_scores": (
                "cosine similarity of current-window ts_features.npy paths"
            ),
            "granite_scores": "cosine of normalized current-window Granite embeddings",
            "inference": (
                "descriptive finite-panel census; no IID query bootstrap or p-value"
            ),
        },
        "model": model_details,
        "token_length_stats": token_rows,
        "per_transition": transition_rows,
        "summary": summary,
        "coverage_reconciliation": reconciliation,
        "provenance": {
            "run_id": RUN_ID,
            "training_job_name": TRAINING_JOB_NAME,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "wall_seconds": float(time.monotonic() - wall_start),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": versions,
            "model_tree_sha256": model_tree_sha256,
            "input_sha256": inputs_sha256,
            "embedding_checkpoints": checkpoints,
            "aws_region": AWS_REGION,
            "instance_type": os.environ.get("SM_CURRENT_INSTANCE_TYPE", "unknown"),
            "s3_bucket_recorded": False,
            "absolute_local_paths_recorded": False,
        },
    }
    with result_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    output_sha256 = {
        path.name: sha256_file(path)
        for path in (result_json, transition_csv, query_csv, token_csv)
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "training_job_name": TRAINING_JOB_NAME,
        "created_at_utc": utc_now(),
        "model_tree_sha256": model_tree_sha256,
        "input_sha256": inputs_sha256,
        "output_sha256": output_sha256,
        "checkpoint_provenance": checkpoints,
        "notes": (
            "No account identifiers, bucket name, role ARN, or workstation paths "
            "are recorded in this artifact."
        ),
    }
    with provenance_json.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nWrote census outputs to {SM_OUTPUT}", flush=True)
    for protocol, values in summary.items():
        print(
            f"{protocol}: ReturnPath="
            f"{values['return_path_mean_across_transitions']:.6f}, "
            f"Granite={values['granite_longctx_mean_across_transitions']:.6f}, "
            f"ReturnPath wins={values['return_path_transition_wins']}/"
            f"{values['n_transitions']}",
            flush=True,
        )


def main() -> None:
    try:
        run()
    except Exception as exc:
        SM_OUTPUT.mkdir(parents=True, exist_ok=True)
        failure = {
            "failed_at_utc": utc_now(),
            "run_id": RUN_ID,
            "training_job_name": TRAINING_JOB_NAME,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with (SM_OUTPUT / "failure.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2)
            handle.write("\n")
        raise


if __name__ == "__main__":
    main()
