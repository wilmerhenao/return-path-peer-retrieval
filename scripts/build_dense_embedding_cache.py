#!/usr/bin/env python3
"""Build the frozen short-context filing-embedding caches used by the paper.

Each model embeds the Item 1 text available before each modern feature window.
The saved matrix follows that window's return-universe row order. Later
scripts evaluate retrieval; this script does not train the models.

Inputs
------
``results/windows_v3/W*/universe.parquet`` supplies the row order.
``results/vintage_descriptions/W*_descriptions.parquet`` supplies filing text,
source, and filing date. Local model snapshots go beneath ``models/``.
``MODEL_CONFIGS`` lists the snapshot directory names and token limits.

Protocol
--------
MiniLM uses 128 tokens. BGE, SEC-BERT, and Granite-r2 use 512. Sentence
Transformer models use their packaged pooling modules and normalized outputs.
SEC-BERT uses attention-mask mean pooling followed by L2 normalization. The
submitted caches were produced on CPU; reduced-precision checkpoint weights
were upcast to float32 before inference.

Outputs
-------
Each ``.npy`` file has shape ``(N_window, embedding_dimension)`` and follows
the window's ticker order. The filename includes the text, ticker order,
token limit, and primary weight-file hash. ``cache_manifest.json`` records
model-file and input hashes. Reusing a cache also requires a matching model,
device, input files, and cached-file checksum in the previous manifest.

Securities without an eligible filing are excluded from feature-time text
experiments. Different devices can produce small floating-point
differences; use ``--device cpu`` to match the recorded setup.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, NoReturn

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parent.parent
WINDOWS_DIR = ROOT / "results" / "windows_v3"
VINTAGE_DIR = ROOT / "results" / "vintage_descriptions"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "vintage_all_encoders_v3" / "embedding_cache"
)
WINDOWS = tuple(f"W{i}" for i in range(1, 7))
CPU_DTYPE_POLICY = "cpu_non_float32_upcast_to_float32"

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "MiniLM": {
        "slug": "minilm",
        "directory": "all-MiniLM-L12-v2",
        "weight_candidates": ("pytorch_model.bin", "model.safetensors"),
        "max_length": 128,
        "batch_size": 128,
        "loader": "sentence_transformer",
    },
    "BGE": {
        "slug": "bge",
        "directory": "bge-base-en-v1.5",
        "weight_candidates": ("model.safetensors", "pytorch_model.bin"),
        "max_length": 512,
        "batch_size": 64,
        "loader": "sentence_transformer",
    },
    "SEC-BERT": {
        "slug": "sec_bert",
        "directory": "sec-bert-base",
        "weight_candidates": ("pytorch_model.bin", "model.safetensors"),
        "max_length": 512,
        "batch_size": 32,
        "loader": "sec_bert_mean_pool",
    },
    "Granite-r2": {
        "slug": "granite_r2",
        "directory": "granite-embedding-english-r2",
        "weight_candidates": ("model.safetensors", "pytorch_model.bin"),
        "max_length": 512,
        "batch_size": 32,
        "loader": "sentence_transformer",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_parts(parts: list[str]) -> str:
    """Hash variable-length strings without ambiguous concatenation."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def model_tree_hashes(model_dir: Path) -> tuple[dict[str, str], str]:
    """Hash the model's weights, tokenizer, pooling, and configuration files."""
    hashes = {
        path.relative_to(model_dir).as_posix(): sha256_file(path)
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }
    if not hashes:
        raise FileNotFoundError(f"model snapshot is empty: {model_dir}")
    tree = sha256_parts(
        [f"{name}:{digest}" for name, digest in sorted(hashes.items())]
    )
    return hashes, tree


def weight_path(model_dir: Path, candidates: tuple[str, ...]) -> Path:
    matches = [model_dir / name for name in candidates if (model_dir / name).is_file()]
    if not matches:
        raise FileNotFoundError(
            f"no recognized model weight file in {model_dir}: {candidates}"
        )
    return matches[0]


def window_inputs(window: str) -> tuple[list[str], list[str], dict[str, str]]:
    """Return text in universe order, keeping filings before the window start."""
    universe_path = WINDOWS_DIR / window / "universe.parquet"
    vintage_path = VINTAGE_DIR / f"{window}_descriptions.parquet"
    meta_path = WINDOWS_DIR / window / "meta.json"
    for path in (universe_path, vintage_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tickers = (
        pl.read_parquet(universe_path)
        .get_column("ticker")
        .cast(pl.String)
        .to_list()
    )
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"{window}: duplicate return-universe ticker")
    start = date.fromisoformat(
        json.loads(meta_path.read_text(encoding="utf-8"))["start"]
    )

    vintage = pl.read_parquet(vintage_path)
    required = {"ticker", "source", "item1_text", "filing_date"}
    if not required.issubset(vintage.columns):
        raise ValueError(
            f"{window}: vintage file lacks {sorted(required - set(vintage.columns))}"
        )
    vintage = vintage.filter(
        (pl.col("source") == "10K")
        & pl.col("item1_text").is_not_null()
        & (pl.col("item1_text").str.len_chars() > 0)
    )
    duplicates = vintage.group_by("ticker").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"{window}: duplicate eligible vintage ticker rows")

    text_map: dict[str, str] = {}
    filing_dates: dict[str, str] = {}
    for row in vintage.select(
        "ticker", "item1_text", "filing_date"
    ).iter_rows(named=True):
        filing_date = date.fromisoformat(str(row["filing_date"]))
        if filing_date >= start:
            raise ValueError(
                f"{window}: {row['ticker']} filing {filing_date} is not before {start}"
            )
        ticker = str(row["ticker"])
        text_map[ticker] = str(row["item1_text"])
        filing_dates[ticker] = filing_date.isoformat()
    texts = [text_map.get(ticker, "") for ticker in tickers]
    return tickers, texts, filing_dates


def cache_key(
    window: str,
    method: str,
    max_length: int,
    weight_sha256: str,
    tickers: list[str],
    texts: list[str],
) -> str:
    """Reproduce the content key used by the submitted CPU caches."""
    digest = hashlib.sha256()
    digest.update(window.encode())
    digest.update(method.encode())
    digest.update(f"max_length={max_length}".encode())
    digest.update(CPU_DTYPE_POLICY.encode())
    digest.update(weight_sha256.encode())
    for ticker, text in zip(tickers, texts):
        digest.update(ticker.encode())
        digest.update(b"\0")
        digest.update(text.encode(errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def validate_cache_reuse(
    path: Path,
    *,
    previous_manifest: dict[str, Any] | None,
    method: str,
    model_record: dict[str, Any],
    input_hashes: dict[str, str],
    device: str,
    n_rows: int,
) -> np.ndarray:
    """Check a saved cache against its recorded model, inputs, and checksum."""

    def refuse(reason: str) -> NoReturn:
        raise ValueError(
            f"Cannot reuse {path.name}: {reason}. "
            "Use --force to rebuild it or --output-dir for a new cache directory."
        )

    if not isinstance(previous_manifest, dict):
        refuse("cache_manifest.json is missing or unreadable")
    if previous_manifest.get("schema_version") != "dense-embedding-cache-v1":
        refuse("the manifest schema is unsupported")
    if previous_manifest.get("device") != device:
        refuse("the recorded device differs from the requested device")
    if previous_manifest.get("cpu_dtype_policy") != CPU_DTYPE_POLICY:
        refuse("the recorded dtype policy differs")
    models = previous_manifest.get("models")
    if not isinstance(models, dict) or models.get(method) != model_record:
        refuse("the model snapshot or configuration differs")
    artifacts = previous_manifest.get("artifacts")
    artifact = artifacts.get(path.name) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        refuse("the manifest has no entry for this cache")
    for name, expected_hash in input_hashes.items():
        if artifact.get(name) != expected_hash:
            refuse(f"the recorded input hash differs or is missing: {name}")
    try:
        current_hash = sha256_file(path)
    except OSError as exc:
        refuse(f"the cached file cannot be read ({exc})")
    if artifact.get("sha256") != current_hash:
        refuse("the cached file checksum differs")
    try:
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        refuse(f"the cached array cannot be read ({exc})")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != n_rows
        or artifact.get("shape") != list(embeddings.shape)
    ):
        refuse("the cached shape differs from the manifest or window universe")
    if embeddings.dtype != np.dtype(np.float32):
        refuse("the cached array is not float32")
    if not np.isfinite(embeddings).all():
        refuse("the cached array contains non-finite values")
    return embeddings


def load_sentence_transformer(
    model_dir: Path,
    *,
    device: str,
    max_length: int,
) -> Callable[[list[str], int], np.ndarray]:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device=device)
    if device == "cpu" and next(model.parameters()).dtype != torch.float32:
        model = model.float()
    model.max_seq_length = max_length

    def encode(texts: list[str], batch_size: int) -> np.ndarray:
        values = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(values, dtype=np.float32)

    return encode


def load_sec_bert(
    model_dir: Path,
    *,
    device: str,
    max_length: int,
) -> Callable[[list[str], int], np.ndarray]:
    """Load SEC-BERT with the mean-pooling rule used in the submission."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    if device == "cpu" and next(model.parameters()).dtype != torch.float32:
        model = model.float()
    model.to(device)
    model.eval()

    def encode(texts: list[str], batch_size: int) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            with torch.no_grad():
                output = model(**tokens)
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            batches.append(pooled.float().cpu().numpy())
        return np.vstack(batches).astype(np.float32)

    return encode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build universe-indexed short-context filing embeddings."
    )
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--windows",
        default=",".join(WINDOWS),
        help="comma-separated feature windows; default W1,...,W6",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_CONFIGS),
        help="comma-separated model labels",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="CPU is required for the submitted numerical cache identity",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected_windows = tuple(value.strip() for value in args.windows.split(","))
    selected_models = tuple(value.strip() for value in args.models.split(","))
    unknown_windows = sorted(set(selected_windows) - set(WINDOWS))
    unknown_models = sorted(set(selected_models) - set(MODEL_CONFIGS))
    if unknown_windows or unknown_models:
        parser.error(
            f"unknown windows={unknown_windows}, unknown models={unknown_models}"
        )
    if args.device != "cpu":
        print(
            "NOTE: a non-CPU cache is a fresh-device rerun and may not be "
            "bitwise identical to the submitted caches.",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "cache_manifest.json"
    previous_manifest = None
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    prepared_inputs = {
        window: window_inputs(window) for window in selected_windows
    }
    manifest: dict[str, Any] = {
        "schema_version": "dense-embedding-cache-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "cpu_dtype_policy": CPU_DTYPE_POLICY,
        "software": {"python": platform.python_version()},
        "models": {},
        "artifacts": {},
    }

    for method in selected_models:
        config = MODEL_CONFIGS[method]
        model_dir = args.models_dir / config["directory"]
        hashes, tree_hash = model_tree_hashes(model_dir)
        weights = weight_path(model_dir, config["weight_candidates"])
        weight_hash = sha256_file(weights)
        loader = (
            load_sentence_transformer
            if config["loader"] == "sentence_transformer"
            else load_sec_bert
        )
        encode = loader(
            model_dir,
            device=args.device,
            max_length=int(config["max_length"]),
        )
        manifest["models"][method] = {
            "model_directory": config["directory"],
            "loader": config["loader"],
            "max_length": config["max_length"],
            "batch_size": config["batch_size"],
            "weight_file": weights.name,
            "weight_sha256": weight_hash,
            "model_tree_sha256": tree_hash,
            "model_file_sha256": hashes,
        }

        for window in selected_windows:
            tickers, texts, filing_dates = prepared_inputs[window]
            universe_path = WINDOWS_DIR / window / "universe.parquet"
            vintage_path = VINTAGE_DIR / f"{window}_descriptions.parquet"
            meta_path = WINDOWS_DIR / window / "meta.json"
            input_hashes = {
                "window_universe_sha256": sha256_file(universe_path),
                "vintage_text_sha256": sha256_file(vintage_path),
                "window_meta_sha256": sha256_file(meta_path),
            }
            key = cache_key(
                window,
                method,
                int(config["max_length"]),
                weight_hash,
                tickers,
                texts,
            )
            filename = f"{window}_{config['slug']}_{key}.npy"
            output_path = args.output_dir / filename
            if output_path.exists() and not args.force:
                embeddings = validate_cache_reuse(
                    output_path,
                    previous_manifest=previous_manifest,
                    method=method,
                    model_record=manifest["models"][method],
                    input_hashes=input_hashes,
                    device=args.device,
                    n_rows=len(tickers),
                )
                print(f"reuse {filename}: shape={embeddings.shape}")
            else:
                started = time.perf_counter()
                embeddings = encode(texts, int(config["batch_size"]))
                if embeddings.shape[0] != len(tickers):
                    raise ValueError(f"{method}/{window}: output row mismatch")
                if not np.isfinite(embeddings).all():
                    raise ValueError(f"{method}/{window}: non-finite embedding")
                np.save(output_path, embeddings.astype(np.float32))
                print(
                    f"wrote {filename}: shape={embeddings.shape}, "
                    f"seconds={time.perf_counter() - started:.1f}"
                )
            manifest["artifacts"][filename] = {
                "sha256": sha256_file(output_path),
                "shape": list(np.load(output_path, mmap_mode="r").shape),
                **input_hashes,
                "eligible_filing_count": len(filing_dates),
                "row_order": f"results/windows_v3/{window}/universe.parquet",
            }

        del encode
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
