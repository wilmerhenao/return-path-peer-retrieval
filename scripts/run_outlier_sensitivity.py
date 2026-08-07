#!/usr/bin/env python3
"""Fully reranked raw-residual outlier sensitivity for the modern census.

The modern residual artifacts contain a small number of daily FF5 residuals
whose magnitudes are difficult to interpret as ordinary economic returns.  For
each W_t -> W_{t+1} transition, this audit starts from the paper's exact
feature-time universe and evaluates five independently rebuilt universes:

* no residual-magnitude screen;
* max |raw FF5 residual| <= 1.0 in W_t only;
* max |raw FF5 residual| <= 1.0 in both W_t and W_{t+1};
* the corresponding W_t-only and either-window screens at 0.5.

Each condition refits TF-IDF, recomputes ReturnPath and the frozen 80/20 rank
blend, reranks every query, and recomputes target-window correlations directly
from raw residual paths.  It does not merely delete affected query rows from
an existing result.  The target-window screen is explicitly retrospective: it
is a data-quality sensitivity, not an implementable ex-ante trading rule.  The
W_t-only condition isolates what an ex-ante magnitude check would do, although
the underlying benchmark still conditions eligibility on W_{t+1} return
availability and uses an ex-post candidate panel.

In addition to positive-relevance NDCG@5, the script evaluates a no-fit,
equal-notional top-five peer basket on W_{t+1}: mean query--basket correlation
and pooled variance reduction.  These are descriptive co-movement diagnostics,
not hedge, trading, or profitability claims.

Every invocation creates a new immutable result directory containing the full
census, a compact publication-oriented summary, a transition CSV, a manifest,
and SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
import sklearn


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import run_acceptance_robustness as core  # noqa: E402
from scripts import run_downstream_peer_hedging as downstream  # noqa: E402
from scripts.metrics import top_k_indices  # noqa: E402


SCHEMA_VERSION = "raw-residual-outlier-sensitivity-v1"
OUTPUT_ROOT = ROOT / "results" / "outlier_sensitivity"
TRANSITIONS = tuple((f"W{i}", f"W{i + 1}") for i in range(1, 7))
METHODS = ("ReturnPath", "TF-IDF", "FrozenBlend")
K = 5


@dataclass(frozen=True)
class Screen:
    screen_id: str
    threshold: float | None
    scope: str
    primary: bool = False

    def keep_mask(
        self,
        feature_max_abs: np.ndarray,
        target_max_abs: np.ndarray,
    ) -> np.ndarray:
        if self.scope == "none":
            return np.ones(feature_max_abs.size, dtype=bool)
        if self.threshold is None or self.threshold <= 0.0:
            raise AssertionError(f"{self.screen_id}: invalid threshold")
        feature_ok = feature_max_abs <= self.threshold
        if self.scope == "feature_only":
            return feature_ok
        if self.scope == "feature_and_target":
            return feature_ok & (target_max_abs <= self.threshold)
        raise AssertionError(f"{self.screen_id}: unknown scope {self.scope}")


SCREENS = (
    Screen("unscreened", None, "none"),
    Screen("feature_max_abs_le_1p0", 1.0, "feature_only"),
    Screen(
        "either_window_max_abs_le_1p0",
        1.0,
        "feature_and_target",
        primary=True,
    ),
    Screen("feature_max_abs_le_0p5", 0.5, "feature_only"),
    Screen("either_window_max_abs_le_0p5", 0.5, "feature_and_target"),
)


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
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def create_run_directory(output_root: Path, run_id: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    else:
        if re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None:
            raise ValueError("run-id may contain only letters, numbers, _, ., and -")
        name = run_id
    run_dir = output_root / name
    run_dir.mkdir(exist_ok=False)
    return run_dir


def load_frozen_configuration() -> tuple[dict[str, Any], float, dict[str, Any]]:
    tfidf_kwargs, alpha = downstream.frozen_selection()
    selection = json.loads(
        downstream.SELECTION_ARTIFACT.read_text(encoding="utf-8")
    )
    summary = selection["summary"]
    expected = {
        "ReturnPath": float(summary["modern_mean_return_path_ndcg"]),
        "TF-IDF": float(summary["modern_mean_tfidf_ndcg"]),
        "FrozenBlend": float(summary["modern_mean_blend_ndcg"]),
    }
    return tfidf_kwargs, alpha, expected


def load_base_transition(
    feature_name: str,
    target_name: str,
) -> dict[str, Any]:
    feature = downstream.load_window(feature_name)
    target = downstream.load_window(target_name)
    text_map = downstream.load_vintage_text(feature_name, feature.start)
    tickers, texts, feature_raw, target_raw, feature_standardized = (
        downstream.feature_time_arrays(feature, target, text_map)
    )
    target_idx = np.fromiter(
        (target.ticker_to_idx[ticker] for ticker in tickers), dtype=np.int64
    )
    target_standardized = np.asarray(
        target.standardized_residuals[target_idx], dtype=np.float32
    )
    if target_standardized.shape != target_raw.shape:
        raise AssertionError(f"{feature_name}->{target_name}: target shape mismatch")

    feature_max_abs = np.max(np.abs(feature_raw), axis=1)
    target_max_abs = np.max(np.abs(target_raw), axis=1)
    if not np.isfinite(feature_max_abs).all() or not np.isfinite(target_max_abs).all():
        raise AssertionError(f"{feature_name}->{target_name}: non-finite maxima")

    return {
        "feature": feature,
        "target": target,
        "tickers": tickers,
        "texts": texts,
        "feature_raw": feature_raw,
        "target_raw": target_raw,
        "feature_standardized": feature_standardized,
        "target_standardized": target_standardized,
        "feature_max_abs": feature_max_abs,
        "target_max_abs": target_max_abs,
    }


def exclusion_audit(base: dict[str, Any], screen: Screen) -> dict[str, Any]:
    feature_max_abs = base["feature_max_abs"]
    target_max_abs = base["target_max_abs"]
    keep = screen.keep_mask(feature_max_abs, target_max_abs)
    threshold = screen.threshold
    if threshold is None:
        feature_flag = np.zeros(keep.size, dtype=bool)
        target_flag = np.zeros(keep.size, dtype=bool)
    else:
        feature_flag = feature_max_abs > threshold
        target_flag = target_max_abs > threshold

    excluded = []
    for idx in np.flatnonzero(~keep):
        excluded.append(
            {
                "ticker": base["tickers"][int(idx)],
                "feature_max_abs_raw_residual": float(feature_max_abs[idx]),
                "target_max_abs_raw_residual": float(target_max_abs[idx]),
                "feature_exceeds_threshold": bool(feature_flag[idx]),
                "target_exceeds_threshold": bool(target_flag[idx]),
            }
        )
    return {
        "n_before": int(keep.size),
        "n_after": int(keep.sum()),
        "n_excluded": int((~keep).sum()),
        "excluded_share": float((~keep).mean()),
        "n_feature_exceeds_threshold": int(feature_flag.sum()),
        "n_target_exceeds_threshold": int(target_flag.sum()),
        "n_either_exceeds_threshold": int((feature_flag | target_flag).sum()),
        "n_both_exceed_threshold": int((feature_flag & target_flag).sum()),
        "excluded": excluded,
        "keep_mask": keep,
    }


def top_extremes(base: dict[str, Any], count: int = 12) -> list[dict[str, Any]]:
    either = np.maximum(base["feature_max_abs"], base["target_max_abs"])
    order = np.argsort(-either, kind="stable")[:count]
    return [
        {
            "ticker": base["tickers"][int(idx)],
            "feature_max_abs_raw_residual": float(base["feature_max_abs"][idx]),
            "target_max_abs_raw_residual": float(base["target_max_abs"][idx]),
            "either_window_max_abs_raw_residual": float(either[idx]),
        }
        for idx in order
    ]


def equal_notional_basket_metrics(
    scores: np.ndarray,
    realized: np.ndarray,
    target_raw: np.ndarray,
) -> dict[str, Any]:
    n = scores.shape[0]
    if scores.shape != realized.shape or target_raw.shape[0] != n:
        raise AssertionError("basket inputs have incompatible shapes")
    correlations: list[float] = []
    variance_reductions: list[float] = []
    target_variances: list[float] = []
    error_variances: list[float] = []
    selected_raw_correlations: list[float] = []
    selected_positive_relevance: list[float] = []

    for query in range(n):
        peers = top_k_indices(
            scores[query], K, excluded=query, query_seed=query
        )
        if peers.size != K or query in peers or len(set(peers.tolist())) != K:
            raise AssertionError(f"query {query}: invalid peer set")
        basket = np.mean(target_raw[peers], axis=0)
        query_path = target_raw[query]
        query_variance = float(np.var(query_path, ddof=0))
        basket_variance = float(np.var(basket, ddof=0))
        error_variance = float(np.var(query_path - basket, ddof=0))
        if query_variance <= 0.0 or basket_variance <= 0.0:
            raise AssertionError(f"query {query}: non-positive target variance")
        correlation = float(np.corrcoef(query_path, basket)[0, 1])
        if not math.isfinite(correlation):
            raise AssertionError(f"query {query}: non-finite basket correlation")
        correlations.append(correlation)
        target_variances.append(query_variance)
        error_variances.append(error_variance)
        variance_reductions.append(1.0 - error_variance / query_variance)
        selected = np.asarray(realized[query, peers], dtype=np.float64)
        selected_raw_correlations.extend(selected.tolist())
        selected_positive_relevance.extend(np.maximum(selected, 0.0).tolist())

    pooled_target_variance = float(np.sum(target_variances))
    pooled_error_variance = float(np.sum(error_variances))
    return {
        "definition": (
            "Each W_{t+1} basket is the arithmetic mean of the five peers' "
            "raw FF5 residuals; no weights or intercept are fit."
        ),
        "n_queries": n,
        "mean_query_basket_correlation": statistics.fmean(correlations),
        "median_query_basket_correlation": statistics.median(correlations),
        "population_std_query_basket_correlation": statistics.pstdev(correlations),
        "mean_selected_pairwise_target_correlation": statistics.fmean(
            selected_raw_correlations
        ),
        "mean_selected_positive_relevance": statistics.fmean(
            selected_positive_relevance
        ),
        "mean_equal_query_variance_reduction": statistics.fmean(
            variance_reductions
        ),
        "median_equal_query_variance_reduction": statistics.median(
            variance_reductions
        ),
        "share_queries_positive_variance_reduction": statistics.fmean(
            float(value > 0.0) for value in variance_reductions
        ),
        "pooled_target_variance_reduction": (
            1.0 - pooled_error_variance / pooled_target_variance
        ),
    }


def evaluate_screen(
    base: dict[str, Any],
    screen: Screen,
    tfidf_kwargs: dict[str, Any],
    blend_alpha: float,
) -> dict[str, Any]:
    exclusion = exclusion_audit(base, screen)
    keep = exclusion.pop("keep_mask")
    tickers = tuple(
        ticker for ticker, retained in zip(base["tickers"], keep, strict=True)
        if retained
    )
    texts = tuple(
        text for text, retained in zip(base["texts"], keep, strict=True)
        if retained
    )
    feature_standardized = np.asarray(
        base["feature_standardized"][keep], dtype=np.float32
    )
    target_standardized = np.asarray(
        base["target_standardized"][keep], dtype=np.float32
    )
    target_raw = np.asarray(base["target_raw"][keep], dtype=np.float64)
    if len(tickers) < 100 or len(tickers) != len(texts):
        raise AssertionError(f"{screen.screen_id}: invalid screened universe")

    # Recompute every matrix after screening.  In particular, target relevance
    # comes from the screened raw-residual paths; corr_target.npy is never read.
    realized = core.return_path_scores(target_standardized)
    direct_target_standardized = (
        (target_raw - target_raw.mean(axis=1, keepdims=True))
        / target_raw.std(axis=1, keepdims=True)
    )
    direct_target_correlation = core.return_path_scores(
        direct_target_standardized.astype(np.float32)
    )
    recomputation_max_abs_error = float(
        np.max(np.abs(realized - direct_target_correlation))
    )
    if recomputation_max_abs_error > 2e-5:
        raise AssertionError(
            f"{screen.screen_id}: target correlation recomputation mismatch "
            f"{recomputation_max_abs_error}"
        )

    return_scores = core.return_path_scores(feature_standardized)
    tfidf_scores, vocab_size = core.fit_tfidf_scores(texts, tfidf_kwargs)
    blend_scores = (
        blend_alpha * core.average_rank_matrix(return_scores)
        + (1.0 - blend_alpha) * core.average_rank_matrix(tfidf_scores)
    )
    score_matrices = {
        "ReturnPath": return_scores,
        "TF-IDF": tfidf_scores,
        "FrozenBlend": blend_scores,
    }

    method_results: dict[str, Any] = {}
    for method, scores in score_matrices.items():
        ndcg = core.evaluate_scores(
            scores,
            realized,
            k=K,
            oracle_check=(method == "ReturnPath"),
        )
        if ndcg["n_queries"] != len(tickers) or ndcg["n_valid"] != len(tickers):
            raise AssertionError(
                f"{screen.screen_id}/{method}: incomplete all-query census"
            )
        method_results[method] = {
            "ndcg_at_5": ndcg,
            "equal_notional_top5_basket": equal_notional_basket_metrics(
                scores, realized, target_raw
            ),
        }

    return {
        "screen_id": screen.screen_id,
        "threshold_max_abs_raw_residual": screen.threshold,
        "scope": screen.scope,
        "primary_retrospective_sensitivity": screen.primary,
        "ranking_rebuilt_after_screen": True,
        "tfidf_refit_after_screen": True,
        "ideal_lists_rebuilt_after_screen": True,
        "target_correlation_recomputed_from_residual_paths": True,
        "target_correlation_recomputation_max_abs_error": (
            recomputation_max_abs_error
        ),
        "n_universe": len(tickers),
        "n_feature_days": int(feature_standardized.shape[1]),
        "n_target_days": int(target_raw.shape[1]),
        "tfidf_vocab_size": vocab_size,
        "exclusions": exclusion,
        "methods": method_results,
    }


def method_macro(
    rows: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    ndcg = np.asarray(
        [row["methods"][method]["ndcg_at_5"]["mean_ndcg"] for row in rows],
        dtype=np.float64,
    )
    valid_n = np.asarray(
        [row["methods"][method]["ndcg_at_5"]["n_valid"] for row in rows],
        dtype=np.int64,
    )
    basket_corr = np.asarray(
        [
            row["methods"][method]["equal_notional_top5_basket"][
                "mean_query_basket_correlation"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    pooled_vr = np.asarray(
        [
            row["methods"][method]["equal_notional_top5_basket"][
                "pooled_target_variance_reduction"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    pair_corr = np.asarray(
        [
            row["methods"][method]["equal_notional_top5_basket"][
                "mean_selected_pairwise_target_correlation"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "n_transitions": len(rows),
        "total_queries_across_transitions": int(valid_n.sum()),
        "macro_mean_ndcg_at_5": float(ndcg.mean()),
        "query_weighted_mean_ndcg_at_5": float(
            np.sum(ndcg * valid_n) / valid_n.sum()
        ),
        "population_std_transition_mean_ndcg_at_5": float(ndcg.std(ddof=0)),
        "minimum_transition_mean_ndcg_at_5": float(ndcg.min()),
        "maximum_transition_mean_ndcg_at_5": float(ndcg.max()),
        "macro_mean_query_basket_correlation": float(basket_corr.mean()),
        "population_std_transition_mean_query_basket_correlation": float(
            basket_corr.std(ddof=0)
        ),
        "macro_mean_selected_pairwise_target_correlation": float(
            pair_corr.mean()
        ),
        "macro_mean_pooled_target_variance_reduction": float(pooled_vr.mean()),
        "population_std_pooled_target_variance_reduction": float(
            pooled_vr.std(ddof=0)
        ),
        "transitions_positive_pooled_target_variance_reduction": int(
            np.sum(pooled_vr > 0.0)
        ),
    }


def comparison(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    definitions = {
        "ndcg_at_5": lambda row, method: row["methods"][method]["ndcg_at_5"][
            "mean_ndcg"
        ],
        "query_basket_correlation": lambda row, method: row["methods"][method][
            "equal_notional_top5_basket"
        ]["mean_query_basket_correlation"],
        "pooled_target_variance_reduction": lambda row, method: row["methods"][
            method
        ]["equal_notional_top5_basket"]["pooled_target_variance_reduction"],
    }
    result: dict[str, Any] = {"left_method": left, "right_method": right}
    for metric, getter in definitions.items():
        deltas = np.asarray(
            [float(getter(row, left) - getter(row, right)) for row in rows],
            dtype=np.float64,
        )
        result[metric] = {
            "macro_mean_left_minus_right": float(deltas.mean()),
            "population_std_left_minus_right": float(deltas.std(ddof=0)),
            "left_wins": int(np.sum(deltas > 0.0)),
            "ties": int(np.sum(deltas == 0.0)),
            "right_wins": int(np.sum(deltas < 0.0)),
            "per_transition_left_minus_right": [float(value) for value in deltas],
        }
    return result


def summarize_screens(
    transition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for screen in SCREENS:
        rows = [
            row for row in transition_rows if row["screen_id"] == screen.screen_id
        ]
        if len(rows) != len(TRANSITIONS):
            raise AssertionError(f"{screen.screen_id}: incomplete transition set")
        before = np.asarray(
            [row["exclusions"]["n_before"] for row in rows], dtype=np.int64
        )
        after = np.asarray(
            [row["exclusions"]["n_after"] for row in rows], dtype=np.int64
        )
        summaries[screen.screen_id] = {
            "threshold_max_abs_raw_residual": screen.threshold,
            "scope": screen.scope,
            "primary_retrospective_sensitivity": screen.primary,
            "universe": {
                "minimum_n_before": int(before.min()),
                "maximum_n_before": int(before.max()),
                "minimum_n_after": int(after.min()),
                "maximum_n_after": int(after.max()),
                "total_transition_security_exclusions": int(
                    np.sum(before - after)
                ),
                "exclusion_share_across_transition_security_rows": float(
                    np.sum(before - after) / np.sum(before)
                ),
            },
            "methods": {
                method: method_macro(rows, method) for method in METHODS
            },
            "comparisons": [
                comparison(rows, "ReturnPath", "TF-IDF"),
                comparison(rows, "FrozenBlend", "ReturnPath"),
                comparison(rows, "FrozenBlend", "TF-IDF"),
            ],
        }

    baseline = summaries["unscreened"]
    for screen in SCREENS:
        if screen.screen_id == "unscreened":
            continue
        current = summaries[screen.screen_id]
        current["change_from_unscreened"] = {}
        for method in METHODS:
            current_method = current["methods"][method]
            baseline_method = baseline["methods"][method]
            current["change_from_unscreened"][method] = {
                "macro_mean_ndcg_at_5_delta": (
                    current_method["macro_mean_ndcg_at_5"]
                    - baseline_method["macro_mean_ndcg_at_5"]
                ),
                "macro_mean_query_basket_correlation_delta": (
                    current_method["macro_mean_query_basket_correlation"]
                    - baseline_method["macro_mean_query_basket_correlation"]
                ),
                "macro_mean_pooled_target_variance_reduction_delta": (
                    current_method["macro_mean_pooled_target_variance_reduction"]
                    - baseline_method["macro_mean_pooled_target_variance_reduction"]
                ),
            }
    return summaries


def publication_summary(screen_summaries: dict[str, Any]) -> dict[str, Any]:
    primary_id = next(screen.screen_id for screen in SCREENS if screen.primary)
    primary = screen_summaries[primary_id]
    baseline = screen_summaries["unscreened"]
    methods = primary["methods"]
    blend_vs_path = next(
        item
        for item in primary["comparisons"]
        if item["left_method"] == "FrozenBlend"
        and item["right_method"] == "ReturnPath"
    )
    blend_vs_text = next(
        item
        for item in primary["comparisons"]
        if item["left_method"] == "FrozenBlend"
        and item["right_method"] == "TF-IDF"
    )
    ndcg_sentence = (
        "After full reranking on the retrospective 1.0 either-window screen, "
        f"macro NDCG@5 is {methods['ReturnPath']['macro_mean_ndcg_at_5']:.3f}/"
        f"{methods['TF-IDF']['macro_mean_ndcg_at_5']:.3f}/"
        f"{methods['FrozenBlend']['macro_mean_ndcg_at_5']:.3f} for ReturnPath, "
        "TF-IDF, and the frozen blend; the blend exceeds ReturnPath in "
        f"{blend_vs_path['ndcg_at_5']['left_wins']}/6 transitions and TF-IDF "
        f"in {blend_vs_text['ndcg_at_5']['left_wins']}/6."
    )
    application_sentence = (
        "For the no-fit equal-notional peer basket under the same screen, mean "
        "next-window query--basket correlation is "
        f"{methods['ReturnPath']['macro_mean_query_basket_correlation']:.3f}/"
        f"{methods['TF-IDF']['macro_mean_query_basket_correlation']:.3f}/"
        f"{methods['FrozenBlend']['macro_mean_query_basket_correlation']:.3f}, "
        "while macro pooled variance reduction is "
        f"{methods['ReturnPath']['macro_mean_pooled_target_variance_reduction']:.3f}/"
        f"{methods['TF-IDF']['macro_mean_pooled_target_variance_reduction']:.3f}/"
        f"{methods['FrozenBlend']['macro_mean_pooled_target_variance_reduction']:.3f}; "
        "these are co-movement diagnostics, not hedge or profitability results."
    )
    return {
        "primary_screen": primary_id,
        "primary_screen_is_retrospective": True,
        "primary": primary,
        "unscreened_reference": baseline,
        "publication_ready_ndcg_sentence": ndcg_sentence,
        "publication_ready_application_sentence": application_sentence,
        "required_caveat": (
            "The either-window screen uses W_{t+1} residual magnitudes to define "
            "the evaluation universe and is therefore a retrospective data-quality "
            "sensitivity, not a live or leakage-free backtest."
        ),
    }


def provenance_files() -> list[Path]:
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_acceptance_robustness.py",
        ROOT / "scripts" / "run_downstream_peer_hedging.py",
        ROOT / "scripts" / "metrics.py",
        ROOT / "requirements.lock",
        downstream.SELECTION_ARTIFACT,
        downstream.SELECTION_MANIFEST,
    ]
    for index in range(1, 8):
        window = ROOT / "results" / "windows_v3" / f"W{index}"
        files.extend(
            [
                window / "raw_resid.npy",
                window / "ts_features.npy",
                window / "universe.parquet",
                window / "meta.json",
            ]
        )
        if index <= 6:
            files.append(
                ROOT
                / "results"
                / "vintage_descriptions"
                / f"W{index}_descriptions.parquet"
            )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing provenance inputs: {missing}")
    if any(path.name == "corr_target.npy" for path in files):
        raise AssertionError("stored target correlation entered provenance")
    return files


def write_transition_csv(
    path: Path, transition_rows: list[dict[str, Any]]
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    fields = [
        "transition",
        "screen_id",
        "threshold",
        "scope",
        "n_before",
        "n_after",
        "n_excluded",
        "method",
        "mean_ndcg_at_5",
        "mean_query_basket_correlation",
        "pooled_target_variance_reduction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in transition_rows:
            for method in METHODS:
                writer.writerow(
                    {
                        "transition": row["transition"],
                        "screen_id": row["screen_id"],
                        "threshold": row["threshold_max_abs_raw_residual"],
                        "scope": row["scope"],
                        "n_before": row["exclusions"]["n_before"],
                        "n_after": row["exclusions"]["n_after"],
                        "n_excluded": row["exclusions"]["n_excluded"],
                        "method": method,
                        "mean_ndcg_at_5": row["methods"][method]["ndcg_at_5"][
                            "mean_ndcg"
                        ],
                        "mean_query_basket_correlation": row["methods"][method][
                            "equal_notional_top5_basket"
                        ]["mean_query_basket_correlation"],
                        "pooled_target_variance_reduction": row["methods"][method][
                            "equal_notional_top5_basket"
                        ]["pooled_target_variance_reduction"],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    metric_validations = core.validation_oracles()
    tfidf_kwargs, blend_alpha, expected_baseline = load_frozen_configuration()
    transition_rows: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []

    print(f"Outlier sensitivity: {SCHEMA_VERSION}", flush=True)
    print(
        "Screens: " + ", ".join(screen.screen_id for screen in SCREENS),
        flush=True,
    )
    for feature_name, target_name in TRANSITIONS:
        transition_started = time.perf_counter()
        label = f"{feature_name}->{target_name}"
        base = load_base_transition(feature_name, target_name)
        inventories.append(
            {
                "transition": label,
                "n_base_universe": len(base["tickers"]),
                "feature_window_max_abs_raw_residual": float(
                    np.max(base["feature_max_abs"])
                ),
                "target_window_max_abs_raw_residual": float(
                    np.max(base["target_max_abs"])
                ),
                "largest_either_window_observations": top_extremes(base),
            }
        )
        for screen in SCREENS:
            result = evaluate_screen(
                base, screen, tfidf_kwargs, blend_alpha
            )
            row = {
                "transition": label,
                "feature_window": feature_name,
                "target_window": target_name,
                "feature_start": base["feature"].start.isoformat(),
                "feature_end": base["feature"].end.isoformat(),
                "target_start": base["target"].start.isoformat(),
                "target_end": base["target"].end.isoformat(),
                **result,
            }
            transition_rows.append(row)
            print(
                f"  {label} {screen.screen_id:34s} "
                f"N={row['n_universe']:4d} "
                f"RP/TF/BL={row['methods']['ReturnPath']['ndcg_at_5']['mean_ndcg']:.4f}/"
                f"{row['methods']['TF-IDF']['ndcg_at_5']['mean_ndcg']:.4f}/"
                f"{row['methods']['FrozenBlend']['ndcg_at_5']['mean_ndcg']:.4f}",
                flush=True,
            )
        print(
            f"    transition runtime={time.perf_counter() - transition_started:.1f}s",
            flush=True,
        )

    screen_summaries = summarize_screens(transition_rows)
    baseline_observed = {
        method: screen_summaries["unscreened"]["methods"][method][
            "macro_mean_ndcg_at_5"
        ]
        for method in METHODS
    }
    baseline_abs_errors = {
        method: abs(baseline_observed[method] - expected_baseline[method])
        for method in METHODS
    }
    if max(baseline_abs_errors.values()) > 2e-6:
        raise AssertionError(
            "unscreened NDCG does not reconcile to frozen modern artifact: "
            f"{baseline_abs_errors}"
        )

    provenance_inputs = provenance_files()
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = None

    runtime_seconds = time.perf_counter() - started
    generated = utc_now()
    census = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "experiment": "fully reranked raw-FF5-residual magnitude sensitivity",
        "configuration": {
            "modern_transitions": [f"{a}->{b}" for a, b in TRANSITIONS],
            "base_universe": (
                "lexicographically sorted intersection of W_t and W_{t+1} "
                "return availability and non-empty W_t vintage 10-K Item 1"
            ),
            "screens": [
                {
                    "screen_id": screen.screen_id,
                    "threshold_max_abs_raw_residual": screen.threshold,
                    "scope": screen.scope,
                    "exclude_rule": (
                        None
                        if screen.threshold is None
                        else (
                            "exclude security if max absolute daily raw FF5 "
                            f"residual exceeds {screen.threshold:g} in "
                            + (
                                "W_t"
                                if screen.scope == "feature_only"
                                else "either W_t or W_{t+1}"
                            )
                        )
                    ),
                    "primary_retrospective_sensitivity": screen.primary,
                }
                for screen in SCREENS
            ],
            "metric": {
                "name": "canonical positive-relevance NDCG@5",
                "implementation": "scripts.metrics.ndcg_at_k",
                "relevance": "max(0, recomputed target-window correlation)",
                "query_policy": "every security retained by each screen",
                "tie_handling": (
                    "canonical seeded jitter with TIE_RNG_BASE plus sorted "
                    "screened-universe query index"
                ),
            },
            "return_path": (
                "cosine similarity of W_t standardized raw FF5 residual paths"
            ),
            "tfidf": {
                "kwargs": tfidf_kwargs,
                "selection": "configuration b selected only on historical transitions",
                "refit_policy": "vocabulary and IDF refit for each screened corpus",
            },
            "frozen_blend": {
                "alpha_return_path": blend_alpha,
                "alpha_tfidf": 1.0 - blend_alpha,
                "definition": (
                    "alpha*per-query average-percentile-rank(ReturnPath) + "
                    "(1-alpha)*per-query average-percentile-rank(TF-IDF)"
                ),
                "selection": "alpha selected only on historical transitions",
            },
            "application_diagnostic": (
                "unfitted equal-notional arithmetic mean of five selected peer "
                "raw FF5 residual paths in W_{t+1}"
            ),
            "aggregation": (
                "six transitions receive equal weight in macro means; NDCG also "
                "reports query-weighted means; basket pooled variance reduction "
                "sums per-query target and error variances within transition, then "
                "macro-averages transition values"
            ),
        },
        "anti_leakage_and_interpretation": {
            "ranking_rebuilt_within_every_screen": True,
            "target_correlations_recomputed_from_raw_paths": True,
            "stored_corr_target_loaded": False,
            "target_return_availability_used_for_base_eligibility": True,
            "feature_only_screens_use_residual_values_from_W_t_only": True,
            "either_window_screens_use_W_t1_values_for_eligibility": True,
            "either_window_screen_status": (
                "retrospective data-quality sensitivity; not an implementable "
                "ex-ante screen and not a leakage-free live backtest"
            ),
            "benchmark_panel_status": (
                "ex-post survivor/news-visible cohort; feature-only screening does "
                "not repair point-in-time-universe or W_{t+1}-availability limitations"
            ),
            "application_status": (
                "co-movement diagnostic only; no hedge effectiveness, investment, "
                "trading, or profitability claim"
            ),
        },
        "data_quality_inventory": inventories,
        "per_transition": transition_rows,
        "screen_summaries": screen_summaries,
        "validations": {
            **metric_validations,
            "transition_count": len(TRANSITIONS),
            "screen_count": len(SCREENS),
            "all_transition_screen_cells_present": (
                len(transition_rows) == len(TRANSITIONS) * len(SCREENS)
            ),
            "all_eligible_securities_are_queries_after_each_screen": True,
            "query_excluded_from_every_retrieved_peer_set": True,
            "five_unique_peers_per_query": True,
            "tfidf_refit_after_every_screen": True,
            "target_relevance_recomputed_after_every_screen": True,
            "stored_target_correlation_absent_from_inputs": True,
            "unscreened_reconciliation": {
                "expected_from_frozen_modern_artifact": expected_baseline,
                "observed": baseline_observed,
                "absolute_error": baseline_abs_errors,
                "tolerance": 2e-6,
                "passes": True,
            },
        },
        "limitations": [
            (
                "The primary either-window filter is outcome-aware because it "
                "uses W_{t+1} residual magnitudes; it is suitable only as a "
                "retrospective data-quality sensitivity."
            ),
            (
                "A large residual is not proven to be erroneous.  The 1.0 and 0.5 "
                "cutoffs are transparent magnitude screens, not issuer-specific "
                "corporate-action adjudications."
            ),
            (
                "The base universe inherits the paper's ex-post candidate cohort "
                "and conditions on target-window return availability."
            ),
            (
                "Equal-notional basket variance reduction can be negative and is "
                "not evidence of a viable hedge; costs, constraints, and factor-leg "
                "implementation are absent."
            ),
            (
                "Adjacent modern transitions are dependent.  Results are descriptive; "
                "no IID p-values are reported."
            ),
        ],
        "provenance": {
            "project_git_commit": git_commit,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "polars": pl.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "input_sha256": {
                relative(path): sha256_file(path) for path in provenance_inputs
            },
        },
        "runtime_seconds": runtime_seconds,
    }
    compact = publication_summary(screen_summaries)
    compact.update(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": generated,
            "baseline_reconciliation": census["validations"][
                "unscreened_reconciliation"
            ],
            "runtime_seconds": runtime_seconds,
        }
    )

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = create_run_directory(output_root, args.run_id)
    census_path = run_dir / "outlier_sensitivity_census.json"
    compact_path = run_dir / "publication_summary.json"
    csv_path = run_dir / "per_transition.csv"
    write_json_new(census_path, census)
    write_json_new(compact_path, compact)
    write_transition_csv(csv_path, transition_rows)

    artifact_hashes = {
        census_path.name: sha256_file(census_path),
        compact_path.name: sha256_file(compact_path),
        csv_path.name: sha256_file(csv_path),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "configuration": census["configuration"],
        "primary_summary": compact,
        "artifact_sha256": artifact_hashes,
        "provenance": census["provenance"],
        "runtime_seconds": runtime_seconds,
    }
    manifest_path = run_dir / "artifact_manifest.json"
    write_json_new(manifest_path, manifest)
    all_hashes = {
        **artifact_hashes,
        manifest_path.name: sha256_file(manifest_path),
    }
    sums_path = run_dir / "SHA256SUMS.txt"
    if sums_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {sums_path}")
    sums_path.write_text(
        "".join(
            f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items())
        ),
        encoding="utf-8",
    )

    print("\n" + compact["publication_ready_ndcg_sentence"], flush=True)
    print(compact["publication_ready_application_sentence"], flush=True)
    print(compact["required_caveat"], flush=True)
    print(f"Runtime: {runtime_seconds:.1f}s", flush=True)
    print(f"Artifacts: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
