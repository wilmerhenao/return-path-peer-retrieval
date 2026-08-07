#!/usr/bin/env python3
"""Strict feature-time-only peer-basket application audit.

For each modern ``W_t -> W_{t+1}`` transition, peer selection is completed
before any target-window file is loaded.  The selection universe is the
lexicographically sorted intersection of securities with complete ``W_t``
returns and a non-empty vintage 10-K Item 1 filed before ``W_t``.  Target
availability is deliberately absent from this universe definition.

Top-five peer lists are selected from ``W_t`` ReturnPath scores, the
historically frozen TF-IDF configuration, the historically frozen 80/20 rank
blend, a two-digit SIC baseline, and a deterministic random negative control.
Lists are frozen.  Evaluation then retains only queries for which the query
and all five already-selected peers have complete ``W_{t+1}`` residual paths;
missing names cause the query-method observation to be dropped, never
reranked.  The application outcomes are next-window query--equal-notional
peer-basket correlation, pooled variance reduction, and mean signed
query--peer sample correlation.

Every run creates a new immutable directory under
``results/strict_feature_time_application`` with a full selection/evaluation
census, query-level peer identities and metrics, provenance hashes, a
manifest, and checksums.
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


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_acceptance_robustness as core  # noqa: E402
from scripts.metrics import TIE_RNG_BASE, top_k_indices  # noqa: E402


SCHEMA_VERSION = "strict-feature-time-application-v1"
WINDOWS_DIR = ROOT / "results" / "windows_v3"
VINTAGE_DIR = ROOT / "results" / "vintage_descriptions"
OUTPUT_ROOT = ROOT / "results" / "strict_feature_time_application"
SIC_UNIVERSE = ROOT / "data" / "universe.parquet"
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
METHODS = ("ReturnPath", "TF-IDF", "FrozenBlend", "SIC-2", "Random")


@dataclass(frozen=True)
class FeatureWindow:
    name: str
    start: date
    end: date
    tickers: tuple[str, ...]
    ticker_to_idx: dict[str, int]
    raw_residuals: np.ndarray
    standardized_residuals: np.ndarray


@dataclass(frozen=True)
class TargetWindow:
    name: str
    start: date
    end: date
    tickers: tuple[str, ...]
    ticker_to_idx: dict[str, int]
    raw_residuals: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    # Preserve the repository spelling even on filesystems that resolve a
    # data file through a clone/deduplication backing path outside ROOT.
    return path.absolute().relative_to(ROOT.absolute()).as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return relative(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def window_metadata(name: str) -> tuple[date, date]:
    meta_path = WINDOWS_DIR / name / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])


def load_feature_window(name: str) -> FeatureWindow:
    """Load only feature-time files used to define and rank the universe."""
    directory = WINDOWS_DIR / name
    raw_path = directory / "raw_resid.npy"
    standardized_path = directory / "ts_features.npy"
    universe_path = directory / "universe.parquet"
    raw = np.load(raw_path, mmap_mode="r")
    standardized = np.load(standardized_path, mmap_mode="r")
    tickers = tuple(
        str(value) for value in pl.read_parquet(universe_path)["ticker"].to_list()
    )
    start, end = window_metadata(name)
    if raw.shape != standardized.shape:
        raise AssertionError(f"{name}: raw/standardized shape mismatch")
    if raw.ndim != 2 or raw.shape[0] != len(tickers):
        raise AssertionError(f"{name}: feature residual/universe mismatch")
    if len(tickers) != len(set(tickers)):
        raise AssertionError(f"{name}: duplicate feature ticker")
    if not np.isfinite(raw).all() or not np.isfinite(standardized).all():
        raise AssertionError(f"{name}: non-finite feature residual")
    raw64 = np.asarray(raw, dtype=np.float64)
    reconstructed = (
        (raw64 - raw64.mean(axis=1, keepdims=True))
        / raw64.std(axis=1, keepdims=True, ddof=0)
    )
    max_error = float(
        np.max(np.abs(reconstructed - np.asarray(standardized, dtype=np.float64)))
    )
    if max_error > 2e-5:
        raise AssertionError(
            f"{name}: raw residuals do not reconstruct paths ({max_error})"
        )
    return FeatureWindow(
        name=name,
        start=start,
        end=end,
        tickers=tickers,
        ticker_to_idx={ticker: idx for idx, ticker in enumerate(tickers)},
        raw_residuals=raw,
        standardized_residuals=standardized,
    )


def load_target_window(name: str) -> TargetWindow:
    """Load target raw residuals only, after peer lists have been frozen."""
    directory = WINDOWS_DIR / name
    raw = np.load(directory / "raw_resid.npy", mmap_mode="r")
    tickers = tuple(
        str(value)
        for value in pl.read_parquet(directory / "universe.parquet")[
            "ticker"
        ].to_list()
    )
    start, end = window_metadata(name)
    if raw.ndim != 2 or raw.shape[0] != len(tickers):
        raise AssertionError(f"{name}: target residual/universe mismatch")
    if len(tickers) != len(set(tickers)):
        raise AssertionError(f"{name}: duplicate target ticker")
    if not np.isfinite(raw).all():
        raise AssertionError(f"{name}: non-finite target residual")
    return TargetWindow(
        name=name,
        start=start,
        end=end,
        tickers=tickers,
        ticker_to_idx={ticker: idx for idx, ticker in enumerate(tickers)},
        raw_residuals=raw,
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


def load_sic2_map() -> dict[str, str]:
    frame = pl.read_parquet(SIC_UNIVERSE)
    if not {"ticker", "sic"}.issubset(frame.columns):
        raise AssertionError("SIC universe lacks ticker/sic columns")
    if frame["ticker"].n_unique() != frame.height:
        raise AssertionError("duplicate ticker in SIC universe")
    result = {
        str(row["ticker"]): str(row["sic"])[:2]
        for row in frame.select(["ticker", "sic"]).iter_rows(named=True)
    }
    if any(len(value) != 2 for value in result.values()):
        raise AssertionError("invalid SIC-2 code")
    return result


def frozen_selection() -> tuple[dict[str, Any], float]:
    manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["artifact_sha256"][SELECTION_ARTIFACT.name]
    observed = sha256_file(SELECTION_ARTIFACT)
    if observed != expected:
        raise AssertionError("frozen selection artifact hash mismatch")
    bundle = json.loads(SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    summary = bundle["summary"]
    if summary["modern_data_used_for_selection"] is not False:
        raise AssertionError("selection unexpectedly used modern targets")
    selected = summary["tfidf_config_frozen_from_historical"]
    alpha = float(summary["selected_alpha_return_path"])
    if selected["config_id"] != "b" or alpha != 0.8:
        raise AssertionError("unexpected frozen TF-IDF/blend selection")
    kwargs = dict(selected["kwargs"])
    kwargs["ngram_range"] = tuple(kwargs["ngram_range"])
    return kwargs, alpha


def selection_arrays(
    feature: FeatureWindow,
    text_map: dict[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Build the universe without inspecting a target-window object."""
    tickers = tuple(sorted(set(feature.tickers) & set(text_map)))
    if len(tickers) < 100:
        raise AssertionError(f"{feature.name}: implausibly small selection universe")
    indices = np.fromiter(
        (feature.ticker_to_idx[ticker] for ticker in tickers), dtype=np.int64
    )
    texts = tuple(text_map[ticker] for ticker in tickers)
    paths = np.asarray(feature.standardized_residuals[indices], dtype=np.float32)
    if not all(texts):
        raise AssertionError(f"{feature.name}: empty text entered selection")
    return tickers, texts, paths


def score_top_k_rows(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise AssertionError("score matrix must be square")
    rows = np.empty((scores.shape[0], k), dtype=np.int32)
    for query in range(scores.shape[0]):
        rows[query] = top_k_indices(
            scores[query], k, excluded=query, query_seed=query
        )
    return rows


def sic2_top_k_rows(tickers: tuple[str, ...], sic2_map: dict[str, str]) -> np.ndarray:
    missing = sorted(set(tickers) - set(sic2_map))
    if missing:
        raise AssertionError(f"selection tickers missing SIC: {missing[:10]}")
    codes = np.asarray([sic2_map[ticker] for ticker in tickers], dtype=object)
    rows = np.empty((len(tickers), K_PEERS), dtype=np.int32)
    for query in range(len(tickers)):
        scores = (codes == codes[query]).astype(np.float64)
        rows[query] = top_k_indices(
            scores, K_PEERS, excluded=query, query_seed=query
        )
    return rows


def random_top_k_rows(n: int, feature_name: str) -> np.ndarray:
    rows = np.empty((n, K_PEERS), dtype=np.int32)
    all_indices = np.arange(n, dtype=np.int32)
    for query in range(n):
        token = f"{RANDOM_SEED}|{feature_name}|{query}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        candidates = all_indices[all_indices != query]
        rows[query] = np.sort(
            rng.choice(candidates, size=K_PEERS, replace=False)
        )
    return rows


def freeze_peer_lists(
    feature: FeatureWindow,
    text_map: dict[str, str],
    tfidf_kwargs: dict[str, Any],
    blend_alpha: float,
    sic2_map: dict[str, str],
) -> tuple[tuple[str, ...], dict[str, np.ndarray], int]:
    """Complete all ranking using feature-time inputs only."""
    tickers, texts, paths = selection_arrays(feature, text_map)
    return_scores = core.return_path_scores(paths)
    tfidf_scores, vocab_size = core.fit_tfidf_scores(texts, tfidf_kwargs)
    blend_scores = (
        blend_alpha * core.average_rank_matrix(return_scores)
        + (1.0 - blend_alpha) * core.average_rank_matrix(tfidf_scores)
    )
    peers = {
        "ReturnPath": score_top_k_rows(return_scores, K_PEERS),
        "TF-IDF": score_top_k_rows(tfidf_scores, K_PEERS),
        "FrozenBlend": score_top_k_rows(blend_scores, K_PEERS),
        "SIC-2": sic2_top_k_rows(tickers, sic2_map),
        "Random": random_top_k_rows(len(tickers), feature.name),
    }
    for method, rows in peers.items():
        if rows.shape != (len(tickers), K_PEERS):
            raise AssertionError(f"{method}: incomplete peer rows")
        for query, selected in enumerate(rows):
            if query in selected or len(set(selected.tolist())) != K_PEERS:
                raise AssertionError(f"{method}: invalid peer set for query {query}")
    return tickers, peers, vocab_size


def sample_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.ndim != 1 or right64.shape != left64.shape:
        raise AssertionError("correlation path shape mismatch")
    if left64.std(ddof=0) <= 1e-15 or right64.std(ddof=0) <= 1e-15:
        raise AssertionError("degenerate residual path in correlation")
    result = float(np.corrcoef(left64, right64)[0, 1])
    if not np.isfinite(result) or result < -1.000001 or result > 1.000001:
        raise AssertionError("invalid sample correlation")
    return result


def evaluate_frozen_lists(
    feature_name: str,
    target: TargetWindow,
    selection_tickers: tuple[str, ...],
    peer_rows: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Apply a complete-case evaluation filter without changing peer lists."""
    output: list[dict[str, Any]] = []
    target_available = set(target.tickers)
    for method in METHODS:
        for query_idx, query_ticker in enumerate(selection_tickers):
            peer_tickers = tuple(
                selection_tickers[int(idx)] for idx in peer_rows[method][query_idx]
            )
            required = (query_ticker, *peer_tickers)
            missing = tuple(ticker for ticker in required if ticker not in target_available)
            row: dict[str, Any] = {
                "transition": f"{feature_name}->{target.name}",
                "feature_window": feature_name,
                "target_window": target.name,
                "query_ticker": query_ticker,
                "method": method,
                "k_peers": K_PEERS,
                "n_selection_universe": len(selection_tickers),
                "peer_tickers": ";".join(peer_tickers),
                "evaluable": not missing,
                "missing_target_tickers": ";".join(missing),
                "n_target_days": target.raw_residuals.shape[1],
                "query_variance": None,
                "basket_error_variance": None,
                "query_basket_correlation": None,
                "mean_signed_query_peer_correlation": None,
            }
            if not missing:
                query_target_idx = target.ticker_to_idx[query_ticker]
                peer_target_idx = np.fromiter(
                    (target.ticker_to_idx[ticker] for ticker in peer_tickers),
                    dtype=np.int64,
                )
                query_path = np.asarray(
                    target.raw_residuals[query_target_idx], dtype=np.float64
                )
                peer_paths = np.asarray(
                    target.raw_residuals[peer_target_idx], dtype=np.float64
                )
                basket_path = peer_paths.mean(axis=0)
                error_path = query_path - basket_path
                peer_correlations = [
                    sample_correlation(query_path, peer_path)
                    for peer_path in peer_paths
                ]
                row.update(
                    {
                        "query_variance": float(np.var(query_path, ddof=0)),
                        "basket_error_variance": float(
                            np.var(error_path, ddof=0)
                        ),
                        "query_basket_correlation": sample_correlation(
                            query_path, basket_path
                        ),
                        "mean_signed_query_peer_correlation": statistics.fmean(
                            peer_correlations
                        ),
                    }
                )
            output.append(row)
    return output


def summarize_transition_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AssertionError("empty transition-method group")
    evaluable = [row for row in rows if row["evaluable"]]
    if not evaluable:
        raise AssertionError("no evaluable queries")
    basket_corr = [float(row["query_basket_correlation"]) for row in evaluable]
    peer_corr = [
        float(row["mean_signed_query_peer_correlation"]) for row in evaluable
    ]
    query_variance = sum(float(row["query_variance"]) for row in evaluable)
    error_variance = sum(
        float(row["basket_error_variance"]) for row in evaluable
    )
    return {
        "n_selection_queries": len(rows),
        "n_evaluable_queries": len(evaluable),
        "n_dropped_without_reranking": len(rows) - len(evaluable),
        "evaluation_coverage": len(evaluable) / len(rows),
        "mean_query_basket_correlation": statistics.fmean(basket_corr),
        "population_std_query_basket_correlation": statistics.pstdev(basket_corr),
        "median_query_basket_correlation": statistics.median(basket_corr),
        "pooled_equal_notional_variance_reduction": 1.0
        - error_variance / query_variance,
        "mean_signed_query_peer_sample_correlation": statistics.fmean(peer_corr),
        "population_std_signed_query_peer_sample_correlation": statistics.pstdev(
            peer_corr
        ),
    }


def aggregate(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transition_rows: list[dict[str, Any]] = []
    for feature_name, target_name in TRANSITIONS:
        transition = f"{feature_name}->{target_name}"
        for method in METHODS:
            group = [
                row
                for row in rows
                if row["transition"] == transition and row["method"] == method
            ]
            transition_rows.append(
                {
                    "transition": transition,
                    "method": method,
                    **summarize_transition_group(group),
                }
            )

    macro_rows: list[dict[str, Any]] = []
    for method in METHODS:
        group = [row for row in transition_rows if row["method"] == method]
        correlations = [
            float(row["mean_query_basket_correlation"]) for row in group
        ]
        pooled_vr = [
            float(row["pooled_equal_notional_variance_reduction"])
            for row in group
        ]
        peer_corr = [
            float(row["mean_signed_query_peer_sample_correlation"])
            for row in group
        ]
        coverage = [float(row["evaluation_coverage"]) for row in group]
        macro_rows.append(
            {
                "method": method,
                "n_transitions": len(group),
                "macro_equal_transition_mean_query_basket_correlation": statistics.fmean(
                    correlations
                ),
                "population_std_transition_mean_query_basket_correlation": statistics.pstdev(
                    correlations
                ),
                "macro_equal_transition_pooled_variance_reduction": statistics.fmean(
                    pooled_vr
                ),
                "population_std_transition_pooled_variance_reduction": statistics.pstdev(
                    pooled_vr
                ),
                "macro_equal_transition_mean_signed_query_peer_sample_correlation": statistics.fmean(
                    peer_corr
                ),
                "macro_equal_transition_evaluation_coverage": statistics.fmean(
                    coverage
                ),
            }
        )

    def transition_metric(transition: str, method: str, metric: str) -> float:
        return float(
            next(
                row
                for row in transition_rows
                if row["transition"] == transition and row["method"] == method
            )[metric]
        )

    metric_names = (
        "mean_query_basket_correlation",
        "pooled_equal_notional_variance_reduction",
        "mean_signed_query_peer_sample_correlation",
    )
    highest_wins: dict[str, dict[str, int]] = {}
    for metric in metric_names:
        counts = {method: 0 for method in METHODS}
        for feature_name, target_name in TRANSITIONS:
            transition = f"{feature_name}->{target_name}"
            values = {
                method: transition_metric(transition, method, metric)
                for method in METHODS
            }
            maximum = max(values.values())
            for method, value in values.items():
                if value == maximum:
                    counts[method] += 1
        highest_wins[metric] = counts

    pairs = (
        ("FrozenBlend", "ReturnPath"),
        ("FrozenBlend", "TF-IDF"),
        ("FrozenBlend", "SIC-2"),
        ("ReturnPath", "TF-IDF"),
        ("ReturnPath", "SIC-2"),
    )
    pairwise: list[dict[str, Any]] = []
    for metric in metric_names:
        for left, right in pairs:
            deltas = []
            per_transition = []
            for feature_name, target_name in TRANSITIONS:
                transition = f"{feature_name}->{target_name}"
                delta = transition_metric(transition, left, metric) - transition_metric(
                    transition, right, metric
                )
                deltas.append(delta)
                per_transition.append({"transition": transition, "delta": delta})
            pairwise.append(
                {
                    "metric": metric,
                    "left_method": left,
                    "right_method": right,
                    "left_wins": sum(delta > 0.0 for delta in deltas),
                    "ties": sum(delta == 0.0 for delta in deltas),
                    "right_wins": sum(delta < 0.0 for delta in deltas),
                    "macro_equal_transition_mean_delta": statistics.fmean(deltas),
                    "per_transition": per_transition,
                }
            )
    return transition_rows, macro_rows, {
        "highest_method_win_counts": highest_wins,
        "pairwise_win_counts": pairwise,
    }


def provenance_inputs() -> list[Path]:
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_acceptance_robustness.py",
        ROOT / "scripts" / "metrics.py",
        ROOT / "requirements.lock",
        SIC_UNIVERSE,
        SELECTION_ARTIFACT,
        SELECTION_MANIFEST,
    ]
    for index in range(1, 8):
        directory = WINDOWS_DIR / f"W{index}"
        files.extend([directory / "raw_resid.npy", directory / "universe.parquet", directory / "meta.json"])
        if index <= 6:
            files.extend(
                [
                    directory / "ts_features.npy",
                    VINTAGE_DIR / f"W{index}_descriptions.parquet",
                ]
            )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing provenance inputs: {missing}")
    if any(path.name == "corr_target.npy" for path in files):
        raise AssertionError("future correlation matrix entered provenance")
    return files


CSV_FIELDS = (
    "transition",
    "feature_window",
    "target_window",
    "query_ticker",
    "method",
    "k_peers",
    "n_selection_universe",
    "peer_tickers",
    "evaluable",
    "missing_target_tickers",
    "n_target_days",
    "query_variance",
    "basket_error_variance",
    "query_basket_correlation",
    "mean_signed_query_peer_correlation",
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
    tfidf_kwargs, blend_alpha = frozen_selection()
    sic2_map = load_sic2_map()
    query_rows: list[dict[str, Any]] = []
    transition_metadata: list[dict[str, Any]] = []

    print(
        f"Strict feature-time application audit: k={K_PEERS}, "
        f"alpha={blend_alpha:.1f}",
        flush=True,
    )
    for feature_name, target_name in TRANSITIONS:
        transition_started = time.perf_counter()
        feature = load_feature_window(feature_name)
        text_map = load_vintage_text(feature_name, feature.start)

        # Selection finishes here.  No target file or target availability set
        # has been loaded or inspected at this point.
        selection_tickers, peer_rows, vocab_size = freeze_peer_lists(
            feature, text_map, tfidf_kwargs, blend_alpha, sic2_map
        )
        frozen_peer_hash = hashlib.sha256(
            json.dumps(
                {
                    method: peer_rows[method].tolist() for method in METHODS
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        # Only after the peer lists are frozen do target raw residuals and
        # target availability enter the process.
        target = load_target_window(target_name)
        transition_query_rows = evaluate_frozen_lists(
            feature_name, target, selection_tickers, peer_rows
        )
        query_rows.extend(transition_query_rows)
        target_available_count = len(set(selection_tickers) & set(target.tickers))
        transition_metadata.append(
            {
                "transition": f"{feature_name}->{target_name}",
                "selection_n": len(selection_tickers),
                "feature_complete_return_n": len(feature.tickers),
                "nonempty_strictly_vintage_10k_n": len(text_map),
                "selection_names_with_target_returns": target_available_count,
                "selection_names_without_target_returns": len(selection_tickers)
                - target_available_count,
                "feature_days": feature.raw_residuals.shape[1],
                "target_days": target.raw_residuals.shape[1],
                "feature_start": feature.start.isoformat(),
                "feature_end": feature.end.isoformat(),
                "target_start": target.start.isoformat(),
                "target_end": target.end.isoformat(),
                "tfidf_vocab_size": vocab_size,
                "frozen_peer_list_sha256": frozen_peer_hash,
                "runtime_seconds": time.perf_counter() - transition_started,
            }
        )
        method_counts = {
            method: sum(
                row["evaluable"]
                for row in transition_query_rows
                if row["method"] == method
            )
            for method in METHODS
        }
        print(
            f"  {feature_name}->{target_name}: selection N={len(selection_tickers)}, "
            f"target-present N={target_available_count}, evaluable={method_counts}, "
            f"{transition_metadata[-1]['runtime_seconds']:.1f}s",
            flush=True,
        )

    transition_rows, macro_rows, comparisons = aggregate(query_rows)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or (
        "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)

    per_query_path = run_dir / "strict_application_per_query.csv.gz"
    with gzip.open(per_query_path, "xt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                field: "" if row.get(field) is None else row.get(field)
                for field in CSV_FIELDS
            }
            for row in query_rows
        )

    inputs = provenance_inputs()
    expected_rows = sum(row["selection_n"] for row in transition_metadata) * len(
        METHODS
    )
    census = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "experiment": "strict feature-time-only frozen-peer application audit",
        "configuration": {
            "transitions": [f"{left}->{right}" for left, right in TRANSITIONS],
            "selection_universe": (
                "lexicographically sorted intersection of W_t complete-return "
                "universe and non-empty W_t vintage 10-K Item 1; W_{t+1} "
                "availability is not inspected"
            ),
            "evaluation_universe": (
                "frozen-list complete case: query and all five selected peers "
                "must have W_{t+1} residuals; otherwise drop without reranking"
            ),
            "k_peers": K_PEERS,
            "methods": METHODS,
            "return_path": "cosine similarity of W_t standardized FF5 residual paths",
            "tfidf": {
                "selection": "configuration b selected on historical transitions only",
                "kwargs": tfidf_kwargs,
            },
            "frozen_blend": {
                "definition": (
                    "0.8*per-query average percentile rank(ReturnPath) + "
                    "0.2*per-query average percentile rank(TF-IDF)"
                ),
                "alpha_return_path": blend_alpha,
                "alpha_tfidf": 1.0 - blend_alpha,
                "selection": "weight selected on historical transitions only",
            },
            "sic2": (
                "score 1 for same current two-digit SIC and 0 otherwise; "
                "canonical deterministic exact-tie jitter"
            ),
            "score_tie_handling": (
                f"scripts.metrics.top_k_indices with TIE_RNG_BASE={TIE_RNG_BASE} "
                "and query_seed equal to sorted-universe query index"
            ),
            "random_negative_control": (
                f"five samples without replacement from W_t selection universe; "
                f"SHA-256-derived per-query seed based on base seed {RANDOM_SEED}, "
                "feature-window label, and sorted-universe query index"
            ),
            "peer_basket": "equal-notional arithmetic mean of five raw FF5 residual paths",
            "outcomes": {
                "query_basket_correlation": (
                    "sample correlation between W_{t+1} query residual and "
                    "frozen equal-notional peer basket"
                ),
                "pooled_variance_reduction": (
                    "1 - sum_q Var(epsilon_q - basket_q) / sum_q Var(epsilon_q)"
                ),
                "signed_peer_correlation": (
                    "within-query mean of five W_{t+1} sample residual correlations"
                ),
            },
            "aggregation": (
                "query census within transition; macro values weight six transitions equally"
            ),
        },
        "anti_leakage": {
            "selection_universe_uses_target_availability": False,
            "selection_ranking_uses_target_data": False,
            "target_return_availability_used_for_peer_selection": False,
            "target_window_loaded_after_all_transition_peer_lists_frozen": True,
            "target_use": (
                "complete-case evaluation eligibility and outcome measurement only"
            ),
            "missing_target_names_trigger_reranking": False,
            "corr_target_loaded": False,
            "modern_targets_used_for_hyperparameter_selection": False,
            "tfidf_and_blend_hyperparameters_frozen_historically": True,
        },
        "transition_metadata": transition_metadata,
        "per_transition": transition_rows,
        "macro_summary": macro_rows,
        "comparisons": comparisons,
        "limitations": [
            (
                "Complete-case evaluation is method-specific because a frozen peer "
                "missing W_{t+1} returns causes that query-method pair to be dropped."
            ),
            (
                "The equal-notional residual basket is an application diagnostic, "
                "not an investable hedge, return forecast, or profitability test."
            ),
            (
                "The experiment inherits the paper's ex-post survivor/news-visible "
                "candidate cohort; removing target-availability leakage from each "
                "transition does not make the cohort a point-in-time live universe."
            ),
            (
                "SIC codes come from the current candidate-universe snapshot and are "
                "a coarse practitioner baseline, not a historically vintaged taxonomy."
            ),
            (
                "Six adjacent transitions are dependent; means and win counts are "
                "descriptive and no IID p-values are reported."
            ),
        ],
        "validations": {
            "selection_completed_before_target_load_each_transition": True,
            "all_selection_securities_are_queries": True,
            "query_excluded_from_each_peer_set": True,
            "five_unique_peers_per_query": True,
            "unevaluable_queries_dropped_without_reranking": True,
            "raw_feature_residuals_reconstruct_standardized_paths": True,
            "future_correlation_matrix_absent_from_inputs": True,
            "complete_selection_rows": len(query_rows),
            "expected_selection_rows": expected_rows,
            "all_evaluable_rows_have_finite_metrics": all(
                all(
                    np.isfinite(float(row[key]))
                    for key in (
                        "query_variance",
                        "basket_error_variance",
                        "query_basket_correlation",
                        "mean_signed_query_peer_correlation",
                    )
                )
                for row in query_rows
                if row["evaluable"]
            ),
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
            "input_sha256": {relative(path): sha256_file(path) for path in inputs},
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    if len(query_rows) != expected_rows:
        raise AssertionError("selection-row census is incomplete")
    if not census["validations"]["all_evaluable_rows_have_finite_metrics"]:
        raise AssertionError("non-finite metric in evaluable row")

    census_path = run_dir / "strict_application_census.json"
    write_json_new(census_path, census)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "run_directory": relative(run_dir),
        "artifacts": {
            census_path.name: sha256_file(census_path),
            per_query_path.name: sha256_file(per_query_path),
        },
        "input_count": len(inputs),
        "selection_uses_target_availability": False,
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
        print(
            f"  {row['method']}: corr="
            f"{row['macro_equal_transition_mean_query_basket_correlation']:.4f}, "
            f"pooled VR="
            f"{row['macro_equal_transition_pooled_variance_reduction']:.4f}, "
            f"peer corr="
            f"{row['macro_equal_transition_mean_signed_query_peer_sample_correlation']:.4f}, "
            f"coverage={row['macro_equal_transition_evaluation_coverage']:.4%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
