#!/usr/bin/env python3
"""Leakage-safe downstream peer-replication experiment.

For each modern W_t -> W_{t+1} transition, this script:

1. builds the paper's feature-time universe (returns in both windows and a
   non-empty vintage 10-K Item 1 filed strictly before W_t);
2. selects top-5 peers from W_t ReturnPath, vintage TF-IDF, or the historically
   frozen 0.8/0.2 rank blend;
3. estimates a residual-return hedge using only W_t raw FF5 residuals; and
4. freezes the peers and weights before measuring hedge variance reduction on
   the disjoint W_{t+1} raw-residual path.

The primary weight estimator is a six-security Ledoit-Wolf covariance estimate
(query plus five peers) in W_t standardized-residual coordinates, followed by
the minimum-variance coefficients and a conversion back to raw-return weights.
This is scale-invariant and matches the paper's path geometry.  Unregularized
OLS is reported as a robustness check.  The exact no-fit equal-weight top-5
basket is also evaluated through annualized next-window tracking error.  A
deterministic random peer set is a negative control.  No W_{t+1} correlation
matrix is read, and no modern target is used to select k, TF-IDF settings,
blend weights, or estimator.

Every invocation writes a new immutable directory under
``results/downstream_peer_hedging`` containing a compact census JSON, complete
query-level metrics, an artifact manifest, and checksums.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
import sklearn
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_acceptance_robustness as core  # noqa: E402


SCHEMA_VERSION = "downstream-peer-hedging-v2"
WINDOWS_DIR = ROOT / "results" / "windows_v3"
VINTAGE_DIR = ROOT / "results" / "vintage_descriptions"
OUTPUT_ROOT = ROOT / "results" / "downstream_peer_hedging"
SELECTION_ARTIFACT = (
    ROOT
    / "results"
    / "acceptance_robustness"
    / "run_20260730T032543Z"
    / "02_blend_historical_tuning_modern_frozen.json"
)
SELECTION_MANIFEST = SELECTION_ARTIFACT.parent / "artifact_manifest.json"
TRANSITIONS = tuple((f"W{i}", f"W{i + 1}") for i in range(1, 7))
K_PEERS = 5
RANDOM_SEED = 20260730
METHODS = ("ReturnPath", "TF-IDF", "FrozenBlend", "Random")
ESTIMATORS = ("EqualWeight", "LedoitWolf", "OLS")


@dataclass(frozen=True)
class Window:
    name: str
    start: date
    end: date
    tickers: tuple[str, ...]
    ticker_to_idx: dict[str, int]
    raw_residuals: np.ndarray
    standardized_residuals: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return relative(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_window(name: str) -> Window:
    directory = WINDOWS_DIR / name
    paths = {
        "raw": directory / "raw_resid.npy",
        "standardized": directory / "ts_features.npy",
        "universe": directory / "universe.parquet",
        "meta": directory / "meta.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{name}: missing inputs {missing}")

    raw = np.load(paths["raw"], mmap_mode="r")
    standardized = np.load(paths["standardized"], mmap_mode="r")
    tickers = tuple(
        str(ticker)
        for ticker in pl.read_parquet(paths["universe"])["ticker"].to_list()
    )
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

    if raw.shape != standardized.shape:
        raise AssertionError(f"{name}: raw/standardized shape mismatch")
    if raw.ndim != 2 or raw.shape[0] != len(tickers):
        raise AssertionError(f"{name}: residual/universe shape mismatch")
    if len(tickers) != len(set(tickers)):
        raise AssertionError(f"{name}: duplicate ticker")
    if not np.isfinite(raw).all() or not np.isfinite(standardized).all():
        raise AssertionError(f"{name}: non-finite residual")
    raw64 = np.asarray(raw, dtype=np.float64)
    reconstructed = (
        (raw64 - raw64.mean(axis=1, keepdims=True))
        / raw64.std(axis=1, keepdims=True)
    )
    max_error = float(
        np.max(np.abs(reconstructed - np.asarray(standardized, dtype=np.float64)))
    )
    if max_error > 2e-5:
        raise AssertionError(
            f"{name}: raw residuals do not reconstruct ts_features "
            f"(max error {max_error})"
        )

    return Window(
        name=name,
        start=date.fromisoformat(meta["start"]),
        end=date.fromisoformat(meta["end"]),
        tickers=tickers,
        ticker_to_idx={ticker: idx for idx, ticker in enumerate(tickers)},
        raw_residuals=raw,
        standardized_residuals=standardized,
    )


def load_vintage_text(name: str, feature_start: date) -> dict[str, str]:
    path = VINTAGE_DIR / f"{name}_descriptions.parquet"
    frame = pl.read_parquet(path)
    required = {"ticker", "source", "item1_text", "filing_date"}
    if not required.issubset(frame.columns):
        raise AssertionError(
            f"{name}: missing vintage columns {sorted(required - set(frame.columns))}"
        )
    frame = frame.filter(
        (pl.col("source") == "10K")
        & pl.col("item1_text").is_not_null()
        & (pl.col("item1_text").str.len_chars() > 0)
    )
    result: dict[str, str] = {}
    for row in frame.select(
        ["ticker", "item1_text", "filing_date"]
    ).iter_rows(named=True):
        filing_date = date.fromisoformat(str(row["filing_date"]))
        if filing_date >= feature_start:
            raise AssertionError(
                f"{name}: {row['ticker']} filing is not strictly vintage"
            )
        ticker = str(row["ticker"])
        if ticker in result:
            raise AssertionError(f"{name}: duplicate vintage text for {ticker}")
        result[ticker] = str(row["item1_text"])
    return result


def frozen_selection() -> tuple[dict[str, Any], float]:
    manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["artifact_sha256"][SELECTION_ARTIFACT.name]
    observed = sha256_file(SELECTION_ARTIFACT)
    if observed != expected:
        raise AssertionError("frozen blend-selection artifact hash mismatch")
    bundle = json.loads(SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    summary = bundle["summary"]
    if summary["modern_data_used_for_selection"] is not False:
        raise AssertionError("blend selection unexpectedly used modern targets")
    alpha = float(summary["selected_alpha_return_path"])
    selected = summary["tfidf_config_frozen_from_historical"]
    if selected["config_id"] != "b" or alpha != 0.8:
        raise AssertionError("unexpected frozen TF-IDF/blend configuration")
    kwargs = dict(selected["kwargs"])
    kwargs["ngram_range"] = tuple(kwargs["ngram_range"])
    return kwargs, alpha


def feature_time_arrays(
    feature: Window,
    target: Window,
    text_map: dict[str, str],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tickers = tuple(
        sorted(set(feature.tickers) & set(target.tickers) & set(text_map))
    )
    if len(tickers) < 100:
        raise AssertionError(
            f"{feature.name}->{target.name}: implausibly small universe"
        )
    feature_idx = np.fromiter(
        (feature.ticker_to_idx[ticker] for ticker in tickers), dtype=np.int64
    )
    target_idx = np.fromiter(
        (target.ticker_to_idx[ticker] for ticker in tickers), dtype=np.int64
    )
    texts = tuple(text_map[ticker] for ticker in tickers)
    return (
        tickers,
        texts,
        np.asarray(feature.raw_residuals[feature_idx], dtype=np.float64),
        np.asarray(target.raw_residuals[target_idx], dtype=np.float64),
        np.asarray(
            feature.standardized_residuals[feature_idx], dtype=np.float32
        ),
    )


def top_k_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Return deterministic top-k indices; sorted index breaks exact ties."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise AssertionError("score matrix must be square")
    n = scores.shape[0]
    if k >= n:
        raise ValueError("k must be smaller than universe")
    indices = np.arange(n)
    result = np.empty((n, k), dtype=np.int32)
    for query in range(n):
        row = scores[query].copy()
        row[query] = -np.inf
        order = np.lexsort((indices, -row))
        selected = order[:k]
        if query in selected or len(set(selected.tolist())) != k:
            raise AssertionError("invalid deterministic peer selection")
        result[query] = selected
    return result


def random_peer_rows(n: int, k: int, transition: str) -> np.ndarray:
    result = np.empty((n, k), dtype=np.int32)
    for query in range(n):
        token = f"{RANDOM_SEED}|{transition}|{query}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        candidates = np.concatenate(
            (np.arange(query, dtype=np.int32), np.arange(query + 1, n, dtype=np.int32))
        )
        result[query] = np.sort(rng.choice(candidates, size=k, replace=False))
    return result


def fit_weights(
    query_returns: np.ndarray,
    peer_returns: np.ndarray,
    estimator: str,
) -> tuple[np.ndarray, float, float | None]:
    """Fit W_t-only hedge weights and intercept.

    ``peer_returns`` has shape (k,T).  The optional third return is the
    Ledoit-Wolf shrinkage coefficient.
    """
    y = np.asarray(query_returns, dtype=np.float64)
    x = np.asarray(peer_returns, dtype=np.float64).T
    y_mean = float(y.mean())
    x_mean = x.mean(axis=0)
    y_scale = float(y.std(ddof=0))
    x_scale = x.std(axis=0, ddof=0)
    if y_scale <= 1e-12 or np.any(x_scale <= 1e-12):
        raise AssertionError("degenerate W_t residual scale")
    y_standardized = (y - y_mean) / y_scale
    x_standardized = (x - x_mean) / x_scale
    if estimator == "LedoitWolf":
        # Estimate dependence, rather than raw scale, with shrinkage.  Convert
        # standardized coefficients back into raw residual-return weights
        # using W_t volatilities only.
        panel = np.column_stack((y_standardized, x_standardized))
        fitted = LedoitWolf(assume_centered=False).fit(panel)
        covariance = fitted.covariance_
        standardized_weights = np.linalg.solve(
            covariance[1:, 1:], covariance[1:, 0]
        )
        weights = standardized_weights * y_scale / x_scale
        shrinkage: float | None = float(fitted.shrinkage_)
    elif estimator == "EqualWeight":
        # Exact no-fit diagnostic: an equal-notional basket of the five peers.
        # It deliberately uses neither feature-window covariance estimates nor
        # any target-window quantity.
        weights = np.full(x.shape[1], 1.0 / x.shape[1], dtype=np.float64)
        shrinkage = None
    elif estimator == "OLS":
        standardized_weights = np.linalg.lstsq(
            x_standardized, y_standardized, rcond=None
        )[0]
        weights = standardized_weights * y_scale / x_scale
        shrinkage = None
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    intercept = (
        0.0
        if estimator == "EqualWeight"
        else float(y_mean - x_mean @ weights)
    )
    return np.asarray(weights, dtype=np.float64), intercept, shrinkage


def evaluate_frozen_hedge(
    train_query: np.ndarray,
    train_peers: np.ndarray,
    target_query: np.ndarray,
    target_peers: np.ndarray,
    estimator: str,
) -> dict[str, float | None]:
    weights, intercept, shrinkage = fit_weights(
        train_query, train_peers, estimator
    )
    train_prediction = intercept + train_peers.T @ weights
    target_prediction = intercept + target_peers.T @ weights
    train_error = train_query - train_prediction
    target_error = target_query - target_prediction
    train_var = float(np.var(train_query, ddof=0))
    target_var = float(np.var(target_query, ddof=0))
    train_error_var = float(np.var(train_error, ddof=0))
    target_error_var = float(np.var(target_error, ddof=0))
    if train_var <= 0.0 or target_var <= 0.0:
        raise AssertionError("non-positive residual variance")
    prediction_var = float(np.var(target_prediction, ddof=0))
    correlation: float | None
    if prediction_var <= 1e-24:
        correlation = None
    else:
        correlation = float(np.corrcoef(target_query, target_prediction)[0, 1])
    target_mse = float(np.mean(np.square(target_error)))
    zero_baseline_mse = float(np.mean(np.square(target_query)))
    return {
        "train_variance_reduction": 1.0 - train_error_var / train_var,
        "target_variance": target_var,
        "target_hedge_error_variance": target_error_var,
        "target_variance_reduction": 1.0 - target_error_var / target_var,
        "target_tracking_error_ratio": float(
            np.sqrt(target_error_var / target_var)
        ),
        "target_annualized_tracking_error": float(
            np.sqrt(252.0 * target_error_var)
        ),
        "target_replication_correlation": correlation,
        "target_oos_r2_vs_zero": 1.0 - target_mse / zero_baseline_mse,
        "gross_exposure": float(np.sum(np.abs(weights))),
        "net_exposure": float(np.sum(weights)),
        "max_abs_weight": float(np.max(np.abs(weights))),
        "intercept": intercept,
        "lw_shrinkage": shrinkage,
        "weights": weights,
    }


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variance_reductions = finite_values(rows, "target_variance_reduction")
    correlations = finite_values(rows, "target_replication_correlation")
    error_variance = sum(
        float(row["target_hedge_error_variance"]) for row in rows
    )
    target_variance = sum(float(row["target_variance"]) for row in rows)
    return {
        "n_queries": len(rows),
        "mean_target_variance_reduction": statistics.fmean(
            variance_reductions
        ),
        "median_target_variance_reduction": statistics.median(
            variance_reductions
        ),
        "pooled_target_variance_reduction": 1.0
        - error_variance / target_variance,
        "share_queries_positive_variance_reduction": statistics.fmean(
            float(value > 0.0) for value in variance_reductions
        ),
        "mean_target_tracking_error_ratio": statistics.fmean(
            finite_values(rows, "target_tracking_error_ratio")
        ),
        "mean_target_annualized_tracking_error": statistics.fmean(
            finite_values(rows, "target_annualized_tracking_error")
        ),
        "pooled_target_annualized_tracking_error": float(
            np.sqrt(252.0 * error_variance / len(rows))
        ),
        "mean_target_replication_correlation": statistics.fmean(correlations),
        "median_target_replication_correlation": statistics.median(
            correlations
        ),
        "mean_target_oos_r2_vs_zero": statistics.fmean(
            finite_values(rows, "target_oos_r2_vs_zero")
        ),
        "mean_train_variance_reduction": statistics.fmean(
            finite_values(rows, "train_variance_reduction")
        ),
        "mean_gross_exposure": statistics.fmean(
            finite_values(rows, "gross_exposure")
        ),
        "mean_max_abs_weight": statistics.fmean(
            finite_values(rows, "max_abs_weight")
        ),
        "mean_lw_shrinkage": (
            statistics.fmean(finite_values(rows, "lw_shrinkage"))
            if finite_values(rows, "lw_shrinkage")
            else None
        ),
    }


def aggregate_results(
    query_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    transition_rows: list[dict[str, Any]] = []
    for transition in (f"{a}->{b}" for a, b in TRANSITIONS):
        for estimator in ESTIMATORS:
            for method in METHODS:
                group = [
                    row
                    for row in query_rows
                    if row["transition"] == transition
                    and row["estimator"] == estimator
                    and row["method"] == method
                ]
                if not group:
                    raise AssertionError(
                        f"missing group {transition}/{estimator}/{method}"
                    )
                transition_rows.append(
                    {
                        "transition": transition,
                        "estimator": estimator,
                        "method": method,
                        **summarize_group(group),
                    }
                )

    macro_rows: list[dict[str, Any]] = []
    for estimator in ESTIMATORS:
        for method in METHODS:
            group = [
                row
                for row in transition_rows
                if row["estimator"] == estimator and row["method"] == method
            ]
            pooled = [
                float(row["pooled_target_variance_reduction"]) for row in group
            ]
            equal_query = [
                float(row["mean_target_variance_reduction"]) for row in group
            ]
            correlations = [
                float(row["mean_target_replication_correlation"]) for row in group
            ]
            annualized_tracking_errors = [
                float(row["pooled_target_annualized_tracking_error"])
                for row in group
            ]
            macro_rows.append(
                {
                    "estimator": estimator,
                    "method": method,
                    "n_transitions": len(group),
                    "macro_mean_pooled_target_variance_reduction": statistics.fmean(
                        pooled
                    ),
                    "population_std_pooled_target_variance_reduction": statistics.pstdev(
                        pooled
                    ),
                    "macro_mean_equal_query_target_variance_reduction": statistics.fmean(
                        equal_query
                    ),
                    "macro_mean_target_replication_correlation": statistics.fmean(
                        correlations
                    ),
                    "macro_mean_pooled_target_annualized_tracking_error": statistics.fmean(
                        annualized_tracking_errors
                    ),
                    "transitions_positive_pooled_variance_reduction": sum(
                        value > 0.0 for value in pooled
                    ),
                }
            )

    comparison_rows: list[dict[str, Any]] = []
    pairs = (
        ("ReturnPath", "TF-IDF"),
        ("FrozenBlend", "ReturnPath"),
        ("FrozenBlend", "TF-IDF"),
        ("ReturnPath", "Random"),
    )
    for estimator in ESTIMATORS:
        for left, right in pairs:
            deltas = []
            tracking_error_deltas = []
            per_transition = []
            for transition in (f"{a}->{b}" for a, b in TRANSITIONS):
                left_row = next(
                    row
                    for row in transition_rows
                    if row["transition"] == transition
                    and row["estimator"] == estimator
                    and row["method"] == left
                )
                right_row = next(
                    row
                    for row in transition_rows
                    if row["transition"] == transition
                    and row["estimator"] == estimator
                    and row["method"] == right
                )
                delta = float(
                    left_row["pooled_target_variance_reduction"]
                    - right_row["pooled_target_variance_reduction"]
                )
                tracking_error_delta = float(
                    left_row["pooled_target_annualized_tracking_error"]
                    - right_row["pooled_target_annualized_tracking_error"]
                )
                deltas.append(delta)
                tracking_error_deltas.append(tracking_error_delta)
                per_transition.append(
                    {
                        "transition": transition,
                        "pooled_vr_delta": delta,
                        "pooled_annualized_te_delta": tracking_error_delta,
                    }
                )
            comparison_rows.append(
                {
                    "estimator": estimator,
                    "left_method": left,
                    "right_method": right,
                    "metric": "pooled target variance reduction",
                    "macro_mean_delta": statistics.fmean(deltas),
                    "population_std_delta": statistics.pstdev(deltas),
                    "left_wins": sum(delta > 0.0 for delta in deltas),
                    "ties": sum(delta == 0.0 for delta in deltas),
                    "macro_mean_pooled_annualized_te_delta": statistics.fmean(
                        tracking_error_deltas
                    ),
                    "left_lower_tracking_error_wins": sum(
                        delta < 0.0 for delta in tracking_error_deltas
                    ),
                    "n_transitions": len(deltas),
                    "per_transition": per_transition,
                }
            )
    return transition_rows, macro_rows, comparison_rows


def input_files() -> list[Path]:
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_acceptance_robustness.py",
        ROOT / "requirements.lock",
        SELECTION_ARTIFACT,
        SELECTION_MANIFEST,
    ]
    for index in range(1, 8):
        name = f"W{index}"
        directory = WINDOWS_DIR / name
        files.extend(
            [
                directory / "raw_resid.npy",
                directory / "ts_features.npy",
                directory / "universe.parquet",
                directory / "meta.json",
            ]
        )
        if index <= 6:
            files.append(VINTAGE_DIR / f"{name}_descriptions.parquet")
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"provenance inputs missing: {missing}")
    if any(path.name == "corr_target.npy" for path in files):
        raise AssertionError("future correlation target entered input provenance")
    return files


CSV_FIELDS = (
    "transition",
    "feature_window",
    "target_window",
    "query_ticker",
    "method",
    "estimator",
    "k_peers",
    "n_universe",
    "n_feature_days",
    "n_target_days",
    "peer_tickers",
    "weights",
    "intercept",
    "lw_shrinkage",
    "train_variance_reduction",
    "target_variance",
    "target_hedge_error_variance",
    "target_variance_reduction",
    "target_tracking_error_ratio",
    "target_annualized_tracking_error",
    "target_replication_correlation",
    "target_oos_r2_vs_zero",
    "gross_exposure",
    "net_exposure",
    "max_abs_weight",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="parent directory for immutable timestamped runs",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="optional explicit run directory name (must not already exist)",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    generated = utc_now()
    run_id = args.run_id or (
        "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)

    tfidf_kwargs, blend_alpha = frozen_selection()
    query_rows: list[dict[str, Any]] = []
    transition_metadata: list[dict[str, Any]] = []

    print(
        f"Downstream peer hedging: k={K_PEERS}, alpha={blend_alpha:.1f}",
        flush=True,
    )
    for feature_name, target_name in TRANSITIONS:
        transition_started = time.perf_counter()
        transition = f"{feature_name}->{target_name}"
        feature = load_window(feature_name)
        target = load_window(target_name)
        text_map = load_vintage_text(feature_name, feature.start)
        tickers, texts, train_raw, target_raw, train_standardized = (
            feature_time_arrays(feature, target, text_map)
        )

        return_scores = core.return_path_scores(train_standardized)
        tfidf_scores, vocab_size = core.fit_tfidf_scores(texts, tfidf_kwargs)
        blend_scores = (
            blend_alpha * core.average_rank_matrix(return_scores)
            + (1.0 - blend_alpha) * core.average_rank_matrix(tfidf_scores)
        )
        peer_rows = {
            "ReturnPath": top_k_rows(return_scores, K_PEERS),
            "TF-IDF": top_k_rows(tfidf_scores, K_PEERS),
            "FrozenBlend": top_k_rows(blend_scores, K_PEERS),
            "Random": random_peer_rows(len(tickers), K_PEERS, transition),
        }

        # Peer sets and weights are completely fixed from W_t before the target
        # raw-residual arrays are passed to evaluate_frozen_hedge.
        for method in METHODS:
            for query, query_ticker in enumerate(tickers):
                peers = peer_rows[method][query]
                peer_tickers = tuple(tickers[int(peer)] for peer in peers)
                train_query = train_raw[query]
                train_peers = train_raw[peers]
                for estimator in ESTIMATORS:
                    result = evaluate_frozen_hedge(
                        train_query,
                        train_peers,
                        target_raw[query],
                        target_raw[peers],
                        estimator,
                    )
                    query_rows.append(
                        {
                            "transition": transition,
                            "feature_window": feature_name,
                            "target_window": target_name,
                            "query_ticker": query_ticker,
                            "method": method,
                            "estimator": estimator,
                            "k_peers": K_PEERS,
                            "n_universe": len(tickers),
                            "n_feature_days": train_raw.shape[1],
                            "n_target_days": target_raw.shape[1],
                            "peer_tickers": ";".join(peer_tickers),
                            "weights": ";".join(
                                f"{float(value):.12g}"
                                for value in result.pop("weights")
                            ),
                            **result,
                        }
                    )

        transition_metadata.append(
            {
                "transition": transition,
                "n_universe": len(tickers),
                "n_feature_days": train_raw.shape[1],
                "n_target_days": target_raw.shape[1],
                "tfidf_vocab_size": vocab_size,
                "feature_start": feature.start.isoformat(),
                "feature_end": feature.end.isoformat(),
                "target_start": target.start.isoformat(),
                "target_end": target.end.isoformat(),
                "runtime_seconds": time.perf_counter() - transition_started,
            }
        )
        print(
            f"  {transition}: N={len(tickers)}, "
            f"T={train_raw.shape[1]}->{target_raw.shape[1]}, "
            f"{transition_metadata[-1]['runtime_seconds']:.1f}s",
            flush=True,
        )

    transition_rows, macro_rows, comparison_rows = aggregate_results(query_rows)

    per_query_path = run_dir / "peer_hedging_per_query.csv.gz"
    if per_query_path.exists():
        raise FileExistsError(f"refusing to overwrite {per_query_path}")
    with gzip.open(per_query_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                key: "" if row.get(key) is None else row.get(key)
                for key in CSV_FIELDS
            }
            for row in query_rows
        )

    provenance_files = input_files()
    census = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "experiment": "out-of-sample residual-return peer replication/hedging",
        "configuration": {
            "modern_transitions": [f"{a}->{b}" for a, b in TRANSITIONS],
            "universe": (
                "lexicographically sorted intersection of W_t and W_{t+1} "
                "return availability and non-empty W_t vintage 10-K Item 1"
            ),
            "text_timing": "filing_date strictly before W_t start",
            "k_peers": K_PEERS,
            "methods": METHODS,
            "return_path_scores": (
                "cosine similarity of W_t standardized FF5 residual paths"
            ),
            "tfidf": {
                "selection": (
                    "configuration b selected on H1->H2 through H12->H13 only"
                ),
                "kwargs": tfidf_kwargs,
            },
            "frozen_blend": {
                "definition": (
                    "alpha*per-query percentile-rank(ReturnPath) + "
                    "(1-alpha)*per-query percentile-rank(TF-IDF)"
                ),
                "alpha_return_path": blend_alpha,
                "alpha_tfidf": 1.0 - blend_alpha,
                "selection": "alpha selected on historical transitions only",
            },
            "deterministic_ties": (
                "descending score, then ascending index in sorted-ticker universe"
            ),
            "random_negative_control_seed": RANDOM_SEED,
            "weight_estimators": {
                "no_fit_diagnostic": (
                    "EqualWeight: beta_j=1/5 and intercept=0; next-window "
                    "annualized tracking error is sqrt(252)*sd(epsilon_query "
                    "- mean(epsilon_peers))"
                ),
                "primary": (
                    "Ledoit-Wolf covariance on W_t standardized residuals for "
                    "query plus five peers; gamma=Sigma_peers^{-1} "
                    "Sigma_peer,query; beta_j=gamma_j*sigma_query/sigma_peer_j"
                ),
                "robustness": (
                    "unregularized centered OLS with five W_t peer regressors"
                ),
                "intercept": "W_t sample means only",
            },
            "primary_outcome": (
                "1 - Var(W_{t+1} query residual - frozen peer hedge) / "
                "Var(W_{t+1} query residual)"
            ),
            "aggregation": (
                "transition-level pooled variance reduction is primary; "
                "equal-query mean/median and replication correlation are secondary; "
                "macro summaries weight six transitions equally"
            ),
        },
        "anti_leakage": {
            "peer_selection_inputs": "W_t paths and W_t vintage text only",
            "weight_fit_inputs": "W_t raw FF5 residuals only",
            "target_use": (
                "W_{t+1} raw FF5 residuals used only after peers and weights freeze"
            ),
            "corr_target_loaded": False,
            "modern_targets_used_for_hyperparameter_selection": False,
            "target_return_availability_used_for_eligibility": True,
        },
        "transition_metadata": transition_metadata,
        "per_transition": transition_rows,
        "macro_summary": macro_rows,
        "comparisons": comparison_rows,
        "limitations": [
            (
                "This is a residual-risk hedge diagnostic, not an investable "
                "strategy or a return forecast; it omits costs, constraints, "
                "borrow availability, and factor-leg implementation."
            ),
            (
                "The experiment inherits the paper's ex-post survivor/news-visible "
                "candidate cohort and conditions on W_{t+1} return availability."
            ),
            (
                "Six adjacent modern transitions are dependent; transition means "
                "and win counts are descriptive and no IID p-values are reported."
            ),
            (
                "The same top-5 depth used by the paper is fixed in advance; no "
                "modern target is used to tune peer count or weight estimator."
            ),
        ],
        "validations": {
            "all_eligible_securities_are_queries": True,
            "query_excluded_from_each_peer_set": True,
            "five_unique_peers_per_query": True,
            "raw_residuals_reconstruct_stored_standardized_paths": True,
            "future_correlation_matrix_absent_from_inputs": True,
            "complete_query_rows": len(query_rows),
            "expected_query_rows": sum(
                row["n_universe"] for row in transition_metadata
            )
            * len(METHODS)
            * len(ESTIMATORS),
        },
        "provenance": {
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "polars": pl.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "input_sha256": {
                relative(path): sha256_file(path) for path in provenance_files
            },
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    if (
        census["validations"]["complete_query_rows"]
        != census["validations"]["expected_query_rows"]
    ):
        raise AssertionError("query-row census is incomplete")

    census_path = run_dir / "peer_hedging_census.json"
    write_json_new(census_path, census)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "run_directory": relative(run_dir),
        "artifacts": {
            census_path.name: sha256_file(census_path),
            per_query_path.name: sha256_file(per_query_path),
        },
        "input_count": len(provenance_files),
        "future_corr_target_loaded": False,
    }
    manifest_path = run_dir / "artifact_manifest.json"
    write_json_new(manifest_path, manifest)

    checksum_paths = (census_path, per_query_path, manifest_path)
    checksum_path = run_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
        ),
        encoding="utf-8",
    )

    print(f"Wrote immutable run: {relative(run_dir)}", flush=True)
    for row in macro_rows:
        if row["estimator"] == "LedoitWolf":
            print(
                f"  {row['method']}: pooled VR="
                f"{row['macro_mean_pooled_target_variance_reduction']:.4f}, "
                f"corr={row['macro_mean_target_replication_correlation']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
