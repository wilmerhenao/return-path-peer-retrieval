"""Frozen-config full-19-transition feature-time census.

This follow-up does no tuning and does not modify manuscript or prior artifacts.
It verifies the already selected TF-IDF configuration from the immutable
historical-tuning run, then evaluates ReturnPath and TF-IDF on:

    H1->H2, ..., H12->H13, H13->W1, W1->W2, ..., W6->W7.

Every eligible security is a query.  Eligibility requires returns in feature
and target windows plus non-empty vintage 10-K Item 1 text strictly before the
feature-window start; target-window text coverage is never inspected.

Outputs are written to a new immutable directory beneath
``results/acceptance_full19_feature_time/`` with SHA-256 provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
import sklearn
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_acceptance_robustness as core  # noqa: E402


SCHEMA_VERSION = "acceptance-full19-feature-time-v1"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "acceptance_full19_feature_time"
FF5_PATH = PROJECT_ROOT / "results" / "source_cache" / "ff5_daily.parquet"
SELECTION_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "acceptance_robustness"
    / "run_20260730T032543Z"
    / "01_tfidf_historical_tuning.json"
)
WINDOWS = tuple(f"H{i}" for i in range(1, 14)) + tuple(
    f"W{i}" for i in range(1, 8)
)
TRANSITIONS = tuple(zip(WINDOWS[:-1], WINDOWS[1:]))
K = 5

FROZEN_TFIDF_CONFIG = {
    "config_id": "b",
    "kwargs": {
        "max_features": 50_000,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "min_df": 2,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    return core.json_safe(value)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def validate_frozen_selection() -> dict[str, Any]:
    if not SELECTION_SOURCE.exists():
        raise FileNotFoundError(
            "historical selection artifact missing: "
            f"{SELECTION_SOURCE}"
        )
    selection = json.loads(SELECTION_SOURCE.read_text())
    summary = selection["summary"]
    selected_id = summary["selected_config_id"]
    selected_kwargs = summary["selected_config_kwargs"]
    expected_kwargs = json_safe(FROZEN_TFIDF_CONFIG["kwargs"])
    if selected_id != FROZEN_TFIDF_CONFIG["config_id"]:
        raise AssertionError(
            f"historical selection is {selected_id}, expected config b"
        )
    if selected_kwargs != expected_kwargs:
        raise AssertionError(
            "frozen TF-IDF kwargs do not match historical selection artifact"
        )
    if summary.get("modern_data_used_for_selection") is not False:
        raise AssertionError("selection artifact does not certify historical-only tuning")
    return {
        "path": relative(SELECTION_SOURCE),
        "sha256": core.sha256_file(SELECTION_SOURCE),
        "selected_config_id": selected_id,
        "selected_config_kwargs": selected_kwargs,
        "selection_data": summary["selection_data"],
        "selection_objective": summary["selection_objective"],
        "modern_data_used_for_selection": False,
    }


def compute_regimes() -> dict[str, Any]:
    """Reproduce the paper's 20-window FF5 market-volatility terciles."""
    if not FF5_PATH.exists():
        raise FileNotFoundError(f"missing cached FF5 factors: {FF5_PATH}")
    ff5 = pl.read_parquet(FF5_PATH)
    required = {"date", "mkt_rf"}
    if not required.issubset(ff5.columns):
        raise AssertionError(f"FF5 cache missing columns: {required - set(ff5.columns)}")

    annualized_volatility: dict[str, float] = {}
    n_factor_days: dict[str, int] = {}
    for name in WINDOWS:
        window = core.load_window(name)
        values = ff5.filter(
            (pl.col("date") >= window.start)
            & (pl.col("date") <= window.end)
        )["mkt_rf"].to_numpy()
        if values.size < 5 or not np.isfinite(values).all():
            raise AssertionError(f"{name}: insufficient finite FF5 market observations")
        annualized_volatility[name] = float(
            np.std(values, ddof=0) * math.sqrt(252.0)
        )
        n_factor_days[name] = int(values.size)

    ordered_vols = np.asarray(
        [annualized_volatility[name] for name in WINDOWS], dtype=np.float64
    )
    threshold_33_3, threshold_66_7 = np.percentile(
        ordered_vols, [33.3, 66.7], method="linear"
    )
    regimes: dict[str, str] = {}
    for name in WINDOWS:
        value = annualized_volatility[name]
        if value <= threshold_33_3:
            regimes[name] = "LOW_VOL"
        elif value <= threshold_66_7:
            regimes[name] = "MED_VOL"
        else:
            regimes[name] = "HIGH_VOL"

    counts = {
        regime: sum(regimes[name] == regime for name in WINDOWS[:-1])
        for regime in ("HIGH_VOL", "MED_VOL", "LOW_VOL")
    }
    if counts != {"HIGH_VOL": 7, "MED_VOL": 6, "LOW_VOL": 6}:
        raise AssertionError(f"unexpected transition-regime counts: {counts}")

    return {
        "definition": (
            "Regime of each transition is determined by the feature window W_t. "
            "Annualized volatility is population std of daily FF5 mkt_rf times "
            "sqrt(252). LOW/MED/HIGH use the 33.3 and 66.7 percentiles across "
            "all 20 feature/target windows H1--W7."
        ),
        "factor_source": relative(FF5_PATH),
        "percentile_method": "numpy.percentile(method='linear')",
        "threshold_33_3": float(threshold_33_3),
        "threshold_66_7": float(threshold_66_7),
        "annualized_volatility": annualized_volatility,
        "n_factor_days": n_factor_days,
        "window_regimes": regimes,
        "transition_regime_counts": counts,
    }


def input_provenance() -> dict[str, Any]:
    files: list[Path] = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "run_acceptance_robustness.py",
        PROJECT_ROOT / "scripts" / "metrics.py",
        PROJECT_ROOT / "requirements.lock",
        FF5_PATH,
        SELECTION_SOURCE,
    ]
    for name in WINDOWS:
        directory = core.window_directory(name)
        files.extend(
            [
                directory / "ts_features.npy",
                directory / "corr_target.npy",
                directory / "universe.parquet",
                directory / "meta.json",
                core.VINTAGE_DIR / f"{name}_descriptions.parquet",
            ]
        )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"provenance inputs missing: {missing}")

    try:
        git_commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = None

    return {
        "schema_version": SCHEMA_VERSION,
        "project_git_commit": git_commit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sha256": {
            relative(path): core.sha256_file(path)
            for path in files
        },
    }


def coverage_counts(wt: str, wt1: str, eligible_n: int) -> dict[str, Any]:
    feature = core.load_window(wt)
    target = core.load_window(wt1)
    text_map = core.load_vintage_text_map(wt)
    return_intersection = set(feature.tickers) & set(target.tickers)
    feature_text_count = len(set(feature.tickers) & set(text_map))
    n_return_intersection = len(return_intersection)
    independently_computed_eligible = len(return_intersection & set(text_map))
    if independently_computed_eligible != eligible_n:
        raise AssertionError(
            f"{wt}->{wt1}: feature-time universe reconciliation failed"
        )
    return {
        "n_feature_return_universe": len(feature.tickers),
        "n_target_return_universe": len(target.tickers),
        "n_return_intersection": n_return_intersection,
        "n_feature_return_with_vintage_text": feature_text_count,
        "n_feature_time_eligible": eligible_n,
        "text_coverage_of_return_intersection": (
            eligible_n / n_return_intersection
        ),
        "text_coverage_of_feature_return_universe": (
            feature_text_count / len(feature.tickers)
        ),
    }


def method_aggregate(
    rows: list[dict[str, Any]],
    method_key: str,
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[method_key]["mean_ndcg"]) for row in rows],
        dtype=np.float64,
    )
    valid_counts = np.asarray(
        [int(row[method_key]["n_valid"]) for row in rows],
        dtype=np.int64,
    )
    weighted_mean = float(
        np.sum(values * valid_counts) / np.sum(valid_counts)
    )
    return {
        "macro_mean_ndcg": float(values.mean()),
        "population_std_across_transition_means": float(values.std(ddof=0)),
        "minimum_transition_ndcg": float(values.min()),
        "maximum_transition_ndcg": float(values.max()),
        "query_weighted_mean_ndcg": weighted_mean,
        "n_transitions": len(rows),
        "n_valid_queries_total": int(valid_counts.sum()),
    }


def comparison_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray(
        [float(row["return_path_minus_tfidf"]) for row in rows],
        dtype=np.float64,
    )
    wins = int(np.sum(deltas > 0.0))
    losses = int(np.sum(deltas < 0.0))
    ties = int(np.sum(deltas == 0.0))
    non_ties = wins + losses
    sign_p = (
        float(binomtest(wins, non_ties, p=0.5, alternative="greater").pvalue)
        if non_ties
        else 1.0
    )
    return {
        "return_path_wins": wins,
        "tfidf_wins": losses,
        "ties": ties,
        "mean_return_path_minus_tfidf": float(deltas.mean()),
        "population_std_return_path_minus_tfidf": float(deltas.std(ddof=0)),
        "minimum_return_path_minus_tfidf": float(deltas.min()),
        "maximum_return_path_minus_tfidf": float(deltas.max()),
        "one_sided_exact_binomial_sign_p": sign_p,
        "sign_test_note": (
            "Exact binomial calculation; adjacent-window dependence makes the "
            "independence interpretation anti-conservative."
        ),
    }


def summarize(
    rows: list[dict[str, Any]],
    regime_info: dict[str, Any],
) -> dict[str, Any]:
    regime_stats: dict[str, Any] = {}
    for regime in ("HIGH_VOL", "MED_VOL", "LOW_VOL"):
        subset = [row for row in rows if row["regime"] == regime]
        regime_stats[regime] = {
            "ReturnPath": method_aggregate(subset, "return_path"),
            "TF-IDF": method_aggregate(subset, "tfidf"),
            "comparison": comparison_aggregate(subset),
        }

    ns = np.asarray([row["n_feature_time_eligible"] for row in rows], dtype=int)
    return_coverages = np.asarray(
        [row["text_coverage_of_return_intersection"] for row in rows],
        dtype=np.float64,
    )
    feature_coverages = np.asarray(
        [row["text_coverage_of_feature_return_universe"] for row in rows],
        dtype=np.float64,
    )
    return {
        "overall": {
            "ReturnPath": method_aggregate(rows, "return_path"),
            "TF-IDF": method_aggregate(rows, "tfidf"),
            "comparison": comparison_aggregate(rows),
        },
        "by_regime": regime_stats,
        "universe_and_coverage": {
            "minimum_n": int(ns.min()),
            "maximum_n": int(ns.max()),
            "mean_n": float(ns.mean()),
            "minimum_text_coverage_of_return_intersection": float(
                return_coverages.min()
            ),
            "maximum_text_coverage_of_return_intersection": float(
                return_coverages.max()
            ),
            "mean_text_coverage_of_return_intersection": float(
                return_coverages.mean()
            ),
            "minimum_text_coverage_of_feature_return_universe": float(
                feature_coverages.min()
            ),
            "maximum_text_coverage_of_feature_return_universe": float(
                feature_coverages.max()
            ),
            "mean_text_coverage_of_feature_return_universe": float(
                feature_coverages.mean()
            ),
        },
        "regime_thresholds": {
            "threshold_33_3": regime_info["threshold_33_3"],
            "threshold_66_7": regime_info["threshold_66_7"],
        },
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    return parser.parse_args()


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    run_dir = create_run_directory(args.output_root, args.run_id)
    print(f"Full-19 feature-time census: {SCHEMA_VERSION}")
    print(f"Output directory: {run_dir}")

    metric_validations = core.validation_oracles()
    frozen_selection = validate_frozen_selection()
    regime_info = compute_regimes()
    provenance = input_provenance()
    rows: list[dict[str, Any]] = []

    for index, (wt, wt1) in enumerate(TRANSITIONS):
        transition_started = time.perf_counter()
        td = core.build_transition(wt, wt1, universe_policy="feature_time")
        coverage = coverage_counts(wt, wt1, td.n)
        rp_scores = core.return_path_scores(td.ts_feature)
        tf_scores, vocab_size = core.fit_tfidf_scores(
            td.texts, FROZEN_TFIDF_CONFIG["kwargs"]
        )
        rp_eval = core.evaluate_scores(
            rp_scores,
            td.realized,
            k=K,
            oracle_check=True,
        )
        tf_eval = core.evaluate_scores(
            tf_scores,
            td.realized,
            k=K,
            oracle_check=False,
        )
        if rp_eval["n_queries"] != td.n or tf_eval["n_queries"] != td.n:
            raise AssertionError(f"{td.label}: census query count mismatch")
        if rp_eval["n_valid"] != td.n or tf_eval["n_valid"] != td.n:
            raise AssertionError(f"{td.label}: undefined query in current artifacts")
        row = {
            "transition_index": index + 1,
            "transition": td.label,
            "feature_window": wt,
            "target_window": wt1,
            "bridge_transition": bool(wt == "H13" and wt1 == "W1"),
            "feature_start": str(core.load_window(wt).start),
            "feature_end": str(core.load_window(wt).end),
            "target_start": str(core.load_window(wt1).start),
            "target_end": str(core.load_window(wt1).end),
            "regime": regime_info["window_regimes"][wt],
            "feature_window_annualized_market_volatility": regime_info[
                "annualized_volatility"
            ][wt],
            **coverage,
            "vocab_size": vocab_size,
            "return_path": rp_eval,
            "tfidf": tf_eval,
            "return_path_minus_tfidf": (
                rp_eval["mean_ndcg"] - tf_eval["mean_ndcg"]
            ),
            "return_path_wins": bool(
                rp_eval["mean_ndcg"] > tf_eval["mean_ndcg"]
            ),
            "runtime_seconds": time.perf_counter() - transition_started,
        }
        rows.append(row)
        print(
            f"  {td.label:9s} regime={row['regime']:8s} N={td.n:4d} "
            f"coverage={100.0 * coverage['text_coverage_of_return_intersection']:.2f}% "
            f"RP={rp_eval['mean_ndcg']:.6f} TF={tf_eval['mean_ndcg']:.6f} "
            f"delta={row['return_path_minus_tfidf']:+.6f}",
            flush=True,
        )

    if len(rows) != 19:
        raise AssertionError(f"expected 19 transitions, evaluated {len(rows)}")
    summary = summarize(rows, regime_info)
    runtime_seconds = time.perf_counter() - started
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "configuration": {
            "metric": {
                "implementation": "scripts.metrics.ndcg_at_k",
                "name": "canonical positive-relevance NDCG@5",
                "k": K,
                "relevance": "max(0, realized target-window correlation)",
                "query_policy": "all eligible securities; no sampling",
                "tie_handling": (
                    "canonical deterministic seeded jitter with sorted ticker order"
                ),
            },
            "universe": (
                "returns in W_t and W_{t+1}; non-empty source=10K vintage "
                "Item 1 text filed strictly before W_t start only"
            ),
            "target_window_text": "never loaded for eligibility or features",
            "return_path": (
                "row-L2-normalized standardized FF5 residual-return path in W_t"
            ),
            "tfidf_frozen": {
                **FROZEN_TFIDF_CONFIG,
                "effective_sklearn_params": core.effective_tfidf_params(
                    FROZEN_TFIDF_CONFIG["kwargs"]
                ),
                "refit_policy": (
                    "vocabulary and IDF refit separately on each feature-time corpus; "
                    "hyperparameters frozen"
                ),
            },
            "transitions": [f"{wt}->{wt1}" for wt, wt1 in TRANSITIONS],
            "historical_selection_source": frozen_selection,
            "regime_definition": regime_info,
        },
        "per_transition": rows,
        "summary": summary,
        "runtime_seconds": runtime_seconds,
        "validations": {
            **metric_validations,
            "real_data_oracle_every_transition": "1.0",
            "all_transition_queries_valid": True,
            "frozen_historical_selection_verified": True,
            "transition_count": len(rows),
        },
        "provenance": provenance,
        "limitations": [
            (
                "Historical window artifacts use the repository's fixed "
                "current-survivor cohort rather than a point-in-time listing master."
            ),
            (
                "Regime labels are descriptive full-sample volatility terciles, "
                "not a deployable ex-ante regime classifier."
            ),
            (
                "Adjacent transitions share boundaries and are not independent; "
                "the exact sign-test calculation is therefore anti-conservative."
            ),
        ],
    }

    result_path = run_dir / "full19_feature_time_census.json"
    write_json_new(result_path, payload)
    result_hash = core.sha256_file(result_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "configuration": payload["configuration"],
        "summary": summary,
        "runtime_seconds": runtime_seconds,
        "artifact_sha256": {result_path.name: result_hash},
        "provenance": provenance,
    }
    manifest_path = run_dir / "artifact_manifest.json"
    write_json_new(manifest_path, manifest)
    manifest_hash = core.sha256_file(manifest_path)
    sums_path = run_dir / "SHA256SUMS.txt"
    if sums_path.exists():
        raise FileExistsError(f"refusing to overwrite {sums_path}")
    sums_path.write_text(
        f"{manifest_hash}  {manifest_path.name}\n"
        f"{result_hash}  {result_path.name}\n"
    )

    overall = summary["overall"]
    coverage_summary = summary["universe_and_coverage"]
    print("\nCompleted full-19 census with all assertions passing.")
    print(
        f"ReturnPath={overall['ReturnPath']['macro_mean_ndcg']:.6f}, "
        f"TF-IDF={overall['TF-IDF']['macro_mean_ndcg']:.6f}, "
        f"wins={overall['comparison']['return_path_wins']}/19"
    )
    print(
        f"N range={coverage_summary['minimum_n']}--"
        f"{coverage_summary['maximum_n']}; return-intersection text coverage="
        f"{100.0 * coverage_summary['minimum_text_coverage_of_return_intersection']:.2f}%--"
        f"{100.0 * coverage_summary['maximum_text_coverage_of_return_intersection']:.2f}%"
    )
    print(f"Runtime: {runtime_seconds:.1f}s")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
