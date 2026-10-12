#!/usr/bin/env python3
"""Compare return-path, spectral, SIC-2, and random retrieval without text filtering.

For each modern transition, align feature-window paths and correlations with
next-window correlations by ticker. Every security with complete returns in
both windows is a query. The frozen candidate table supplies SIC codes.

Return-path scores are inner products of row-normalized residual paths.
Spectral-dot scores use the positive part of the 128 leading eigenpairs of C_t.
SIC-2 gives equal scores to names sharing the first two SIC digits. Random
scores use a fixed seed reset for each transition. All methods use the shared
positive-correlation NDCG@5 metric.

Each run writes transition-level CSV and JSON, a manifest, and SHA-256 checksums
to a new directory. Results are descriptive; no query-level IID inference is
reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars
import scipy
import sklearn


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.metrics import ndcg_at_k  # noqa: E402


WINDOWS_DIR = ROOT / "results" / "windows_v3"
OUTPUT_ROOT = ROOT / "results" / "full_return_census"
TRANSITIONS = tuple((f"W{i}", f"W{i + 1}") for i in range(1, 7))
METHODS = (
    "ReturnPath",
    "Spectral-dot-128",
    "Spectral-cos-128",
    "SIC-2",
    "Random",
)
K = 5
SPECTRAL_RANK = 128
RANDOM_SEED = 42
SCHEMA_VERSION = "full-return-census-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    # Keep local directories and symlink targets out of provenance labels.
    absolute_path = Path(os.path.abspath(path))
    try:
        return absolute_path.relative_to(Path(os.path.abspath(ROOT))).as_posix()
    except ValueError:
        return f"external/{absolute_path.name}"


def load_window(name: str) -> dict[str, Any]:
    directory = WINDOWS_DIR / name
    tickers = (
        polars.read_parquet(directory / "universe.parquet")
        .get_column("ticker")
        .cast(polars.String)
        .to_list()
    )
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"{name}: duplicate return-universe ticker")
    ts = np.load(directory / "ts_features.npy", mmap_mode="r")
    correlation = np.load(directory / "corr_target.npy", mmap_mode="r")
    if ts.ndim != 2 or ts.shape[0] != len(tickers):
        raise ValueError(f"{name}: residual-path shape mismatch")
    if correlation.shape != (len(tickers), len(tickers)):
        raise ValueError(f"{name}: correlation shape mismatch")
    return {
        "tickers": tickers,
        "index": {ticker: index for index, ticker in enumerate(tickers)},
        "ts": ts,
        "correlation": correlation,
    }


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    # Keep float32: upcasting before normalization changes some near-tied scores.
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("zero-norm embedding row")
    return (values / norms).astype(np.float32)


def spectral_embeddings(correlation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return dot-product and row-normalized rank-128 embeddings of C_t."""
    values = np.asarray(correlation, dtype=np.float32)
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1][:SPECTRAL_RANK]
    retained = np.clip(eigenvalues[order], 0.0, None)
    dot = (
        eigenvectors[:, order] * np.sqrt(retained)[None, :]
    ).astype(np.float32)
    return dot, normalized_rows(dot)


def evaluate(scores: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if scores.shape != target.shape:
        raise ValueError("score and target matrices must have equal shape")
    values = [
        ndcg_at_k(scores[index], target[index], index, k=K)
        for index in range(scores.shape[0])
    ]
    valid = np.asarray([value for value in values if value is not None], dtype=float)
    if valid.size == 0:
        raise ValueError("no query has positive ideal gain")
    return {
        "n_queries": scores.shape[0],
        "n_valid": int(valid.size),
        "mean_ndcg": float(valid.mean()),
        "population_std_across_queries": float(valid.std(ddof=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the modern all-query full-return baseline census."
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "data" / "universe.parquet",
        help="frozen candidate table containing ticker and SIC",
    )
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not all(
        char.isalnum() or char in "._-" for char in args.run_id
    ):
        parser.error("--run-id must be a safe label beginning with run_")
    run_dir = OUTPUT_ROOT / args.run_id
    if run_dir.exists():
        parser.error(f"refusing to overwrite {run_dir}")

    candidate = polars.read_parquet(args.universe)
    if not {"ticker", "sic"}.issubset(candidate.columns):
        raise ValueError("candidate universe requires ticker and sic columns")
    if candidate["ticker"].n_unique() != candidate.height:
        raise ValueError("candidate-universe tickers are not unique")
    sic_map = {
        str(row["ticker"]): (
            str(row["sic"])[:2] if row["sic"] is not None else "XX"
        )
        for row in candidate.select("ticker", "sic").iter_rows(named=True)
    }

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for feature_name, target_name in TRANSITIONS:
        feature = load_window(feature_name)
        target = load_window(target_name)
        tickers = tuple(sorted(set(feature["tickers"]) & set(target["tickers"])))
        feature_idx = np.fromiter(
            (feature["index"][ticker] for ticker in tickers), dtype=np.int64
        )
        target_idx = np.fromiter(
            (target["index"][ticker] for ticker in tickers), dtype=np.int64
        )
        ts = np.asarray(feature["ts"][feature_idx], dtype=np.float32)
        feature_correlation = np.asarray(
            feature["correlation"][np.ix_(feature_idx, feature_idx)],
            dtype=np.float32,
        )
        realized = np.asarray(
            target["correlation"][np.ix_(target_idx, target_idx)],
            dtype=np.float32,
        )
        path = normalized_rows(ts)
        spectral_dot, spectral_cos = spectral_embeddings(feature_correlation)
        sic_codes = np.asarray([sic_map.get(ticker, "XX") for ticker in tickers])
        sic_scores = (sic_codes[:, None] == sic_codes[None, :]).astype(np.float32)
        random_scores = np.random.default_rng(RANDOM_SEED).random(
            (len(tickers), len(tickers))
        ).astype(np.float32)
        score_matrices = {
            "ReturnPath": path @ path.T,
            "Spectral-dot-128": spectral_dot @ spectral_dot.T,
            "Spectral-cos-128": spectral_cos @ spectral_cos.T,
            "SIC-2": sic_scores,
            "Random": random_scores,
        }
        for method, scores in score_matrices.items():
            evaluation = evaluate(scores, realized)
            rows.append(
                {
                    "transition": f"{feature_name}->{target_name}",
                    "method": method,
                    "n_universe": len(tickers),
                    **evaluation,
                }
            )
        print(f"{feature_name}->{target_name}: N={len(tickers)}", flush=True)

    aggregates: dict[str, Any] = {}
    for method in METHODS:
        values = [
            float(row["mean_ndcg"]) for row in rows if row["method"] == method
        ]
        aggregates[method] = {
            "macro_mean_ndcg": statistics.fmean(values),
            "population_std_across_transition_means": statistics.pstdev(values),
            "n_transitions": len(values),
        }

    input_paths = [args.universe.absolute(), Path(__file__).resolve(), ROOT / "scripts" / "metrics.py"]
    for name in tuple(f"W{i}" for i in range(1, 8)):
        directory = WINDOWS_DIR / name
        input_paths.extend(
            [
                directory / "ts_features.npy",
                directory / "corr_target.npy",
                directory / "universe.parquet",
                directory / "meta.json",
            ]
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "metric": "canonical positive-correlation NDCG@5",
            "query_policy": "every full-return-eligible security",
            "spectral_rank": SPECTRAL_RANK,
            "spectral_eigenvalue_policy": "negative retained eigenvalues clipped to zero",
            "sic": "first two digits of frozen candidate-table SIC",
            "random_seed_reset_per_transition": RANDOM_SEED,
            "inference": "descriptive finite-panel census; no IID inference",
        },
        "per_transition": rows,
        "summary": {
            "aggregates": aggregates,
            "n_universe_min": min(row["n_universe"] for row in rows),
            "n_universe_max": max(row["n_universe"] for row in rows),
        },
        "input_sha256": {
            logical_path(path): sha256_file(path)
            for path in sorted(set(input_paths), key=logical_path)
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": polars.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
    }

    run_dir.mkdir(parents=True)
    result_path = run_dir / "full_return_census.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = run_dir / "per_transition.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "artifact_sha256": {
            result_path.name: sha256_file(result_path),
            csv_path.name: sha256_file(csv_path),
        },
        "source_sha256": {
            "scripts/run_full_return_census.py": sha256_file(Path(__file__).resolve()),
            "scripts/metrics.py": sha256_file(ROOT / "scripts" / "metrics.py"),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (result_path, csv_path, manifest_path)
        ),
        encoding="utf-8",
    )
    print(f"run_dir={logical_path(run_dir)}")
    for method in METHODS:
        summary = aggregates[method]
        print(
            f"{method}: {summary['macro_mean_ndcg']:.6f} "
            f"+/- {summary['population_std_across_transition_means']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
