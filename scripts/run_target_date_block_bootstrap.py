#!/usr/bin/env python3
"""Conditional target-date block perturbation for the modern ICAIF results.

The feature-time ReturnPath, TF-IDF, and historically tuned blend rankings are
held fixed.  Within each of the six disjoint modern target windows, circular
blocks of dates are sampled and the *same* sampled dates are applied to every
security.  Target residual correlations and NDCG@5 are then recomputed.

This is deliberately reported as a dependence-aware sensitivity analysis, not
as an IID query/transition bootstrap, a p-value, or an unconditional confidence
interval.  It is conditional on the fitted target residual panels, fixed
feature rankings, selected universe, and observed target-window regimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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

import scripts.run_acceptance_robustness as core  # noqa: E402
from scripts.metrics import top_k_indices  # noqa: E402


SCHEMA_VERSION = "target-date-block-bootstrap-v1"
OUTPUT_ROOT = ROOT / "results" / "acceptance_target_date_uncertainty"
ROBUST_RUN = (
    ROOT
    / "results"
    / "acceptance_robustness"
    / "run_20260730T032543Z"
)
TFIDF_ARTIFACT = ROBUST_RUN / "01_tfidf_historical_tuning.json"
BLEND_ARTIFACT = (
    ROBUST_RUN / "02_blend_historical_tuning_modern_frozen.json"
)
K = 5
METHODS = ("return_path", "tfidf", "blend")
GAPS = ("return_path_minus_tfidf", "blend_minus_return_path")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def verify_checksum_manifest(directory: Path) -> None:
    manifest = directory / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, filename = line.split(maxsplit=1)
        candidate = directory / filename.strip()
        if not candidate.is_file():
            raise FileNotFoundError(f"manifest input is absent: {candidate}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise AssertionError(
                f"checksum mismatch for {relative(candidate)}: "
                f"{actual} != {expected}"
            )


def load_frozen_configuration() -> tuple[dict[str, Any], float, dict[str, Any]]:
    tfidf_payload = json.loads(TFIDF_ARTIFACT.read_text(encoding="utf-8"))
    blend_payload = json.loads(BLEND_ARTIFACT.read_text(encoding="utf-8"))
    tfidf_summary = tfidf_payload["summary"]
    blend_summary = blend_payload["summary"]

    if tfidf_summary["modern_data_used_for_selection"]:
        raise AssertionError("TF-IDF artifact reports modern selection leakage")
    if blend_summary["modern_data_used_for_selection"]:
        raise AssertionError("blend artifact reports modern selection leakage")

    selected_id = str(tfidf_summary["selected_config_id"])
    matches = [
        config
        for config in core.TFIDF_CONFIGS
        if str(config["config_id"]) == selected_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"frozen TF-IDF configuration {selected_id!r} is unavailable"
        )
    config = matches[0]
    safe_kwargs = core.json_safe(config["kwargs"])
    if safe_kwargs != tfidf_summary["selected_config_kwargs"]:
        raise AssertionError("frozen TF-IDF kwargs differ from the core grid")
    blend_config = blend_summary["tfidf_config_frozen_from_historical"]
    if (
        str(blend_config["config_id"]) != selected_id
        or blend_config["kwargs"] != safe_kwargs
    ):
        raise AssertionError("blend and TF-IDF artifacts disagree")

    alpha = float(blend_summary["selected_alpha_return_path"])
    if not 0.0 <= alpha <= 1.0:
        raise AssertionError(f"invalid frozen blend alpha: {alpha}")
    modern_rows = {
        str(row["transition"]): row
        for row in blend_payload["per_transition"]
        if row["phase"] == "modern_frozen_evaluation"
    }
    expected_labels = {
        f"{wt}->{wt1}" for wt, wt1 in core.MODERN_TRANSITIONS
    }
    if set(modern_rows) != expected_labels:
        raise AssertionError("frozen blend artifact has the wrong modern rows")
    return config, alpha, modern_rows


def fixed_retrieval_indices(scores: np.ndarray) -> np.ndarray:
    """Return the canonical fixed top-k list for every query."""
    n = scores.shape[0]
    if scores.shape != (n, n):
        raise AssertionError("score matrix must be square")
    return np.stack(
        [
            top_k_indices(
                scores[query_idx],
                K,
                excluded=query_idx,
                query_seed=query_idx,
            )
            for query_idx in range(n)
        ],
        axis=0,
    )


def census_ndcg_from_correlation(
    correlation: np.ndarray,
    retrieval: dict[str, np.ndarray],
) -> tuple[dict[str, float], int]:
    """Vectorized canonical positive-correlation NDCG for fixed rankings."""
    values = np.asarray(correlation, dtype=np.float64)
    n = values.shape[0]
    if values.shape != (n, n) or not np.isfinite(values).all():
        raise AssertionError("invalid target correlation matrix")

    rows = np.arange(n, dtype=np.int64)
    eligible = values.copy()
    eligible[rows, rows] = -np.inf
    ideal = np.partition(eligible, n - K, axis=1)[:, -K:]
    ideal.sort(axis=1)
    ideal = ideal[:, ::-1]
    discounts = 1.0 / np.log2(np.arange(2, K + 2, dtype=np.float64))
    ideal_dcg = (np.maximum(ideal, 0.0) * discounts).sum(axis=1)
    valid = ideal_dcg > 0.0
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise AssertionError("no query has positive ideal relevance")

    result: dict[str, float] = {}
    for method, indices in retrieval.items():
        if indices.shape != (n, K):
            raise AssertionError(f"{method}: invalid retrieval-index shape")
        retrieved = values[rows[:, None], indices]
        dcg = (np.maximum(retrieved, 0.0) * discounts).sum(axis=1)
        ndcg = np.clip(dcg[valid] / ideal_dcg[valid], 0.0, 1.0)
        result[method] = float(ndcg.mean())
    return result, n_valid


def sample_correlation(residuals: np.ndarray, date_indices: np.ndarray) -> np.ndarray:
    """Recompute the target sample correlation after a common date resample."""
    sampled = np.asarray(residuals[:, date_indices], dtype=np.float64)
    sampled -= sampled.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(sampled, axis=1, keepdims=True)
    if np.any(norms <= 1e-14):
        raise AssertionError("date resample produced a zero-variance security")
    standardized = sampled / norms
    correlation = standardized @ standardized.T
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def circular_block_indices(
    rng: np.random.Generator,
    n_dates: int,
    block_length: int,
) -> np.ndarray:
    """Sample enough circular moving blocks to reconstruct one target window."""
    n_blocks = math.ceil(n_dates / block_length)
    starts = rng.integers(0, n_dates, size=n_blocks)
    offsets = np.arange(block_length, dtype=np.int64)
    return ((starts[:, None] + offsets) % n_dates).reshape(-1)[:n_dates]


def gap_values(method_values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "return_path_minus_tfidf": (
            method_values["return_path"] - method_values["tfidf"]
        ),
        "blend_minus_return_path": (
            method_values["blend"] - method_values["return_path"]
        ),
    }


def summarize_replicates(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    lower, median, upper = np.quantile(array, [0.025, 0.5, 0.975])
    return {
        "conditional_replicate_mean": float(array.mean()),
        "conditional_replicate_std": float(array.std(ddof=0)),
        "conditional_percentile_interval_95_not_confidence_interval": [
            float(lower),
            float(upper),
        ],
        "conditional_median": float(median),
        "fraction_greater_than_zero": float(np.mean(array > 0.0)),
        "n_replicates": int(array.size),
    }


def input_paths() -> list[Path]:
    paths = {
        Path(__file__).resolve(),
        ROOT / "scripts" / "metrics.py",
        ROOT / "scripts" / "run_acceptance_robustness.py",
        TFIDF_ARTIFACT,
        BLEND_ARTIFACT,
        ROBUST_RUN / "artifact_manifest.json",
        ROBUST_RUN / "SHA256SUMS.txt",
    }
    for wt, wt1 in core.MODERN_TRANSITIONS:
        for name in (wt, wt1):
            directory = core.window_directory(name)
            paths.update(
                {
                    directory / "ts_features.npy",
                    directory / "corr_target.npy",
                    directory / "universe.parquet",
                    directory / "meta.json",
                }
            )
        target_directory = core.window_directory(wt1)
        paths.add(target_directory / "raw_resid.npy")
        paths.add(core.VINTAGE_DIR / f"{wt}_descriptions.parquet")
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing uncertainty-analysis inputs: {missing}")
    return sorted(paths, key=relative)


def run_analysis(
    *,
    replicates: int,
    block_lengths: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    verify_checksum_manifest(ROBUST_RUN)
    config, alpha, frozen_rows = load_frozen_configuration()
    transition_labels = [
        f"{wt}->{wt1}" for wt, wt1 in core.MODERN_TRANSITIONS
    ]
    n_transitions = len(transition_labels)

    observed_methods = {
        method: np.empty(n_transitions, dtype=np.float64)
        for method in METHODS
    }
    bootstrap_methods = {
        block_length: {
            method: np.empty(
                (n_transitions, replicates), dtype=np.float64
            )
            for method in METHODS
        }
        for block_length in block_lengths
    }
    unique_date_fractions = {
        block_length: np.empty(
            (n_transitions, replicates), dtype=np.float64
        )
        for block_length in block_lengths
    }
    transition_metadata: list[dict[str, Any]] = []

    for transition_index, (wt, wt1) in enumerate(core.MODERN_TRANSITIONS):
        transition_started = time.perf_counter()
        label = f"{wt}->{wt1}"
        frozen = core.build_frozen_transition(wt, wt1, config)
        td: core.TransitionData = frozen["transition_data"]
        rp_scores = np.asarray(frozen["rp_scores"], dtype=np.float32)
        tfidf_scores = np.asarray(frozen["tf_scores"], dtype=np.float32)
        blend_scores = (
            alpha * core.average_rank_matrix(rp_scores)
            + (1.0 - alpha) * core.average_rank_matrix(tfidf_scores)
        ).astype(np.float32)
        retrieval = {
            "return_path": fixed_retrieval_indices(rp_scores),
            "tfidf": fixed_retrieval_indices(tfidf_scores),
            "blend": fixed_retrieval_indices(blend_scores),
        }

        observed, observed_valid = census_ndcg_from_correlation(
            td.realized, retrieval
        )
        for method in METHODS:
            observed_methods[method][transition_index] = observed[method]

        frozen_row = frozen_rows[label]
        expected = {
            "return_path": float(frozen_row["return_path_ndcg"]),
            "tfidf": float(frozen_row["tfidf_ndcg"]),
            "blend": float(frozen_row["blend_ndcg"]),
        }
        max_artifact_error = max(
            abs(observed[method] - expected[method]) for method in METHODS
        )
        if max_artifact_error > 1e-9:
            raise AssertionError(
                f"{label}: point estimate does not reproduce frozen artifact "
                f"(max error {max_artifact_error})"
            )

        target = core.load_window(wt1)
        target_indices = np.fromiter(
            (target.ticker_to_idx[ticker] for ticker in td.tickers),
            dtype=np.int64,
        )
        raw_all = np.load(
            target.directory / "raw_resid.npy", mmap_mode="r"
        )
        residuals = np.asarray(raw_all[target_indices], dtype=np.float64)
        if residuals.shape[0] != td.n or not np.isfinite(residuals).all():
            raise AssertionError(f"{label}: invalid aligned target residuals")
        n_dates = residuals.shape[1]
        if max(block_lengths) >= n_dates:
            raise AssertionError(
                f"{label}: block length must be shorter than {n_dates} dates"
            )

        raw_point_corr = sample_correlation(
            residuals, np.arange(n_dates, dtype=np.int64)
        )
        raw_point, raw_point_valid = census_ndcg_from_correlation(
            raw_point_corr, retrieval
        )
        max_raw_reproduction_error = max(
            abs(raw_point[method] - observed[method]) for method in METHODS
        )
        if max_raw_reproduction_error > 1e-8:
            raise AssertionError(
                f"{label}: raw sample correlations do not reproduce "
                f"Ledoit-Wolf NDCG (max error {max_raw_reproduction_error})"
            )

        for block_length in block_lengths:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, block_length, transition_index]
                )
            )
            for replicate_index in range(replicates):
                date_indices = circular_block_indices(
                    rng, n_dates, block_length
                )
                unique_date_fractions[block_length][
                    transition_index, replicate_index
                ] = np.unique(date_indices).size / n_dates
                target_corr = sample_correlation(residuals, date_indices)
                values, _ = census_ndcg_from_correlation(
                    target_corr, retrieval
                )
                for method in METHODS:
                    bootstrap_methods[block_length][method][
                        transition_index, replicate_index
                    ] = values[method]

        transition_metadata.append(
            {
                "transition": label,
                "feature_window": wt,
                "target_window": wt1,
                "n_universe": td.n,
                "n_target_dates": n_dates,
                "n_valid_observed": observed_valid,
                "n_valid_raw_reproduction": raw_point_valid,
                "observed_ndcg": observed,
                "observed_gaps": {
                    "return_path_minus_tfidf": (
                        observed["return_path"] - observed["tfidf"]
                    ),
                    "blend_minus_return_path": (
                        observed["blend"] - observed["return_path"]
                    ),
                },
                "max_abs_error_vs_frozen_artifact": max_artifact_error,
                "max_abs_raw_sample_vs_ledoit_wolf_ndcg_error": (
                    max_raw_reproduction_error
                ),
                "runtime_seconds": time.perf_counter() - transition_started,
            }
        )
        print(
            f"{label}: N={td.n}, T={n_dates}, "
            f"RP-TF={observed['return_path'] - observed['tfidf']:+.4f}, "
            f"blend-RP={observed['blend'] - observed['return_path']:+.4f}",
            flush=True,
        )

    observed_gaps = gap_values(observed_methods)
    block_results: dict[str, Any] = {}
    for block_length in block_lengths:
        method_arrays = bootstrap_methods[block_length]
        gap_arrays = gap_values(method_arrays)
        macro_methods = {
            method: values.mean(axis=0)
            for method, values in method_arrays.items()
        }
        macro_gaps = gap_values(macro_methods)
        per_transition = []
        for transition_index, label in enumerate(transition_labels):
            per_transition.append(
                {
                    "transition": label,
                    "gap_summaries": {
                        gap: summarize_replicates(
                            gap_arrays[gap][transition_index]
                        )
                        for gap in GAPS
                    },
                    "replicate_ndcg": {
                        method: method_arrays[method][
                            transition_index
                        ].tolist()
                        for method in METHODS
                    },
                    "replicate_gaps": {
                        gap: gap_arrays[gap][transition_index].tolist()
                        for gap in GAPS
                    },
                }
            )
        block_results[str(block_length)] = {
            "block_length_trading_days": block_length,
            "blocks_per_transition_range": [
                min(
                    math.ceil(row["n_target_dates"] / block_length)
                    for row in transition_metadata
                ),
                max(
                    math.ceil(row["n_target_dates"] / block_length)
                    for row in transition_metadata
                ),
            ],
            "mean_unique_target_date_fraction": float(
                unique_date_fractions[block_length].mean()
            ),
            "macro_gap_summaries": {
                gap: summarize_replicates(macro_gaps[gap])
                for gap in GAPS
            },
            "fraction_all_six_transition_gaps_positive": {
                gap: float(np.mean(np.all(gap_arrays[gap] > 0.0, axis=0)))
                for gap in GAPS
            },
            "macro_replicate_ndcg": {
                method: macro_methods[method].tolist()
                for method in METHODS
            },
            "macro_replicate_gaps": {
                gap: macro_gaps[gap].tolist() for gap in GAPS
            },
            "per_transition": per_transition,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "description": (
            "Conditional common-date circular moving-block perturbation of "
            "the six modern target residual panels with feature rankings fixed"
        ),
        "configuration": {
            "replicates_per_block_length": replicates,
            "block_lengths_trading_days": list(block_lengths),
            "seed": seed,
            "primary_k": K,
            "methods": list(METHODS),
            "frozen_tfidf_config_id": config["config_id"],
            "frozen_tfidf_kwargs": core.json_safe(config["kwargs"]),
            "frozen_blend_alpha_return_path": alpha,
            "frozen_blend_alpha_tfidf": 1.0 - alpha,
            "feature_rankings_refit_in_replicates": False,
            "factor_regressions_refit_in_replicates": False,
            "cross_security_date_resampling": (
                "one common circular moving-block index vector per target "
                "window and replicate, applied to every security"
            ),
            "transition_resampling": (
                "independent deterministic RNG streams across the six "
                "disjoint modern target windows"
            ),
        },
        "interpretation_boundary": {
            "valid_label": (
                "conditional target-date dependence-aware sensitivity "
                "distribution"
            ),
            "not_valid_labels": [
                "IID query bootstrap",
                "IID transition bootstrap",
                "p-value",
                "unconditional confidence interval",
            ],
            "preserved_dependence": [
                "contemporaneous cross-security residual dependence",
                "within-block ordering of adjacent target dates",
                "all query dependence induced by a common target matrix",
            ],
            "conditioned_on": [
                "the fitted full-window FF5 residual panels",
                "the fixed feature-time universe",
                "historically selected TF-IDF and blend configurations",
                "the six observed modern target-window regimes",
            ],
            "not_captured": [
                "factor-estimation uncertainty",
                "feature-ranking estimation uncertainty",
                "universe-selection uncertainty",
                "filing-text uncertainty",
                "regime sampling beyond the six observed target windows",
                "nonstationarity invalidating local block resampling",
            ],
            "ledoit_wolf_note": (
                "With each security re-centered and standardized, Ledoit-Wolf "
                "multiplies off-diagonal sample correlations by one common "
                "positive factor; positive-relevance NDCG is invariant to "
                "that factor, so bootstrap targets use sample correlations."
            ),
        },
        "observed": {
            "per_transition": transition_metadata,
            "macro_mean_ndcg": {
                method: float(values.mean())
                for method, values in observed_methods.items()
            },
            "macro_mean_gaps": {
                gap: float(values.mean())
                for gap, values in observed_gaps.items()
            },
            "positive_transition_counts": {
                gap: int(np.sum(values > 0.0))
                for gap, values in observed_gaps.items()
            },
        },
        "block_bootstrap": block_results,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": polars.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replicates",
        type=int,
        default=1000,
        help="replicates per block length (default: paper protocol, 1000)",
    )
    parser.add_argument(
        "--block-lengths", type=int, nargs="+", default=[5, 10]
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()

    if args.replicates < 50:
        parser.error("at least 50 replicates are required")
    block_lengths = tuple(sorted(set(args.block_lengths)))
    if not block_lengths or any(length <= 1 for length in block_lengths):
        parser.error("block lengths must be unique integers greater than one")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not args.run_id.startswith("run_") or not all(
        char.isalnum() or char in "._-" for char in args.run_id
    ):
        parser.error("run id must be a safe run_* label")
    run_dir = OUTPUT_ROOT / args.run_id
    if run_dir.exists():
        parser.error(f"refusing to overwrite {run_dir}")

    started = time.perf_counter()
    payload = run_analysis(
        replicates=args.replicates,
        block_lengths=block_lengths,
        seed=args.seed,
    )
    paths = input_paths()
    payload["input_sha256"] = {
        relative(path): sha256_file(path) for path in paths
    }
    payload["runtime_seconds"] = time.perf_counter() - started

    run_dir.mkdir(parents=True)
    result_path = run_dir / "target_date_block_bootstrap.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "artifact_sha256": {
            result_path.name: sha256_file(result_path),
        },
        "source_sha256": {
            "scripts/run_target_date_block_bootstrap.py": sha256_file(
                Path(__file__).resolve()
            ),
            "scripts/run_acceptance_robustness.py": sha256_file(
                ROOT / "scripts" / "run_acceptance_robustness.py"
            ),
            "scripts/metrics.py": sha256_file(ROOT / "scripts" / "metrics.py"),
        },
        "notes": [
            "No feature ranking or hyperparameter is refit in a replicate.",
            "All securities share the same sampled target dates in a replicate.",
            "Percentile ranges are conditional sensitivity intervals, not confidence intervals.",
        ],
    }
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums_path = run_dir / "SHA256SUMS.txt"
    sums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (result_path, manifest_path)
        ),
        encoding="utf-8",
    )

    print(f"run_dir={relative(run_dir)}")
    for block_length in block_lengths:
        summaries = payload["block_bootstrap"][str(block_length)][
            "macro_gap_summaries"
        ]
        for gap in GAPS:
            summary = summaries[gap]
            interval = summary[
                "conditional_percentile_interval_95_not_confidence_interval"
            ]
            print(
                f"L={block_length} {gap}: "
                f"interval=[{interval[0]:+.4f},{interval[1]:+.4f}], "
                f"fraction>0={summary['fraction_greater_than_zero']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
