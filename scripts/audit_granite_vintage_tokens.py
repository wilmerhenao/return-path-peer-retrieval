"""Audit Granite token lengths for the actual-vintage W1--W6 Item 1 corpus.

The audit is tokenizer-only and offline. Each invocation creates a new immutable
``results/granite_vintage_token_audit/run_TIMESTAMP`` directory; existing runs
are never modified.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VINTAGE_DIR = PROJECT_ROOT / "results" / "vintage_descriptions"
MODEL_DIR = PROJECT_ROOT / "models" / "granite-embedding-english-r2"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "granite_vintage_token_audit"
WINDOWS = [f"W{i}" for i in range(1, 7)]
CONTEXT_LENGTH = 8192
COLLECTION_CAP_CHARACTERS = 8000
AUDIT_BATCH_SIZE = 64
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentence_bert_config.json",
    "config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "polars", "pyarrow", "tokenizers", "transformers"):
        result[package] = importlib.metadata.version(package)
    return result


def distribution(values: np.ndarray, prefix: str) -> dict[str, Any]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{prefix}: expected a nonempty one-dimensional array")
    return {
        f"{prefix}_min": int(values.min()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_sd_ddof0": float(values.std(ddof=0)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_max": int(values.max()),
    }


def token_lengths(tokenizer: Any, texts: list[str]) -> np.ndarray:
    lengths: list[int] = []
    for start in range(0, len(texts), AUDIT_BATCH_SIZE):
        encoded = tokenizer(
            texts[start : start + AUDIT_BATCH_SIZE],
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


def summarize(
    label: str,
    tokens: np.ndarray,
    characters: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "window": label,
        "n_texts": int(tokens.size),
        "n_tokens_gt_512": int(np.sum(tokens > 512)),
        "fraction_tokens_gt_512": float(np.mean(tokens > 512)),
        "n_tokens_gt_8192": int(np.sum(tokens > CONTEXT_LENGTH)),
        "fraction_tokens_gt_8192": float(np.mean(tokens > CONTEXT_LENGTH)),
        "n_chars_at_collection_cap": int(
            np.sum(characters == COLLECTION_CAP_CHARACTERS)
        ),
        "fraction_chars_at_collection_cap": float(
            np.mean(characters == COLLECTION_CAP_CHARACTERS)
        ),
    }
    row.update(distribution(tokens, "tokens"))
    row.update(distribution(characters, "chars"))
    return row


def load_window(window: str) -> tuple[list[str], dict[str, Any]]:
    path = VINTAGE_DIR / f"{window}_descriptions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path)
    required = {"ticker", "source", "item1_text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path.name}: missing {sorted(required - set(frame.columns))}")
    selected = frame.filter(
        (pl.col("source") == "10K")
        & pl.col("item1_text").is_not_null()
        & (pl.col("item1_text").str.len_chars() > 0)
    )
    duplicate_tickers = (
        selected.group_by("ticker").len().filter(pl.col("len") > 1).height
    )
    if duplicate_tickers:
        raise ValueError(
            f"{path.name}: {duplicate_tickers} duplicate source=10K ticker rows"
        )
    texts = selected.get_column("item1_text").cast(pl.String).to_list()
    metadata = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "parquet_rows": frame.height,
        "selected_source_10k_nonempty_rows": selected.height,
        "selected_unique_tickers": selected.get_column("ticker").n_unique(),
    }
    return texts, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestamp",
        help="UTC run timestamp in YYYYmmddTHHMMSSZ form; defaults to current UTC",
    )
    args = parser.parse_args()
    if args.timestamp and not re.fullmatch(r"\d{8}T\d{6}Z", args.timestamp):
        parser.error("--timestamp must match YYYYmmddTHHMMSSZ")
    return args


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = OUTPUT_ROOT / f"run_{timestamp}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)

    tokenizer_paths = [MODEL_DIR / name for name in TOKENIZER_FILES]
    missing = [path for path in tokenizer_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing tokenizer inputs: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=True,
    )

    rows: list[dict[str, Any]] = []
    all_tokens: list[np.ndarray] = []
    all_characters: list[np.ndarray] = []
    input_files: dict[str, Any] = {}
    for window in WINDOWS:
        texts, metadata = load_window(window)
        tokens = token_lengths(tokenizer, texts)
        characters = np.asarray([len(text) for text in texts], dtype=np.int64)
        if tokens.size != characters.size:
            raise AssertionError(f"{window}: tokenizer/text row-count mismatch")
        rows.append(summarize(window, tokens, characters))
        all_tokens.append(tokens)
        all_characters.append(characters)
        input_files[window] = metadata

    aggregate_tokens = np.concatenate(all_tokens)
    aggregate_characters = np.concatenate(all_characters)
    rows.append(summarize("ALL", aggregate_tokens, aggregate_characters))

    stats_path = run_dir / "token_length_stats.csv"
    summary_path = run_dir / "summary.json"
    checksums_path = run_dir / "SHA256SUMS.txt"
    write_csv(stats_path, rows)

    summary = {
        "schema_version": 2,
        "run_timestamp_utc": timestamp,
        "observation_unit": (
            "window-ticker source=10K nonempty Item 1 excerpt; repeated filings "
            "across windows remain separate point-in-time observations"
        ),
        "windows": WINDOWS,
        "tokenization": {
            "model": "granite-embedding-english-r2",
            "model_path": MODEL_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "tokenizer_class": type(tokenizer).__name__,
            "is_fast": bool(tokenizer.is_fast),
            "add_special_tokens": True,
            "truncation": False,
            "padding": False,
            "configured_model_max_length": int(tokenizer.model_max_length),
            "audit_context_threshold": CONTEXT_LENGTH,
            "audit_batch_size": AUDIT_BATCH_SIZE,
            "character_length_definition": "Python Unicode code points via len(text)",
            "collection_cap_characters": COLLECTION_CAP_CHARACTERS,
            "percentile_method": "numpy.percentile default linear interpolation",
        },
        "aggregate": rows[-1],
        "per_window": rows[:-1],
        "input_files": input_files,
        "model_tokenizer_file_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in tokenizer_paths
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "package_versions": package_versions(),
        "absolute_paths_recorded": False,
    }
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in (stats_path, summary_path)
    ]
    with checksums_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(checksum_lines) + "\n")

    aggregate = rows[-1]
    print(run_dir.relative_to(PROJECT_ROOT))
    print(
        "aggregate: "
        f"n={aggregate['n_texts']}, "
        f"median={aggregate['tokens_p50']:.6g}, "
        f">512={aggregate['fraction_tokens_gt_512']:.9%}, "
        f"at_char_cap={aggregate['fraction_chars_at_collection_cap']:.9%}, "
        f"max={aggregate['tokens_max']}, "
        f">8192={aggregate['n_tokens_gt_8192']}"
    )


if __name__ == "__main__":
    main()
