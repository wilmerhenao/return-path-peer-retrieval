"""Check return-path robustness on frozen inputs.

Every eligible security is a query. Evaluation uses positive-correlation NDCG
from ``scripts.metrics.ndcg_at_k``, sorted tickers, and seeded tie handling.
Securities need returns in both W_t and W_{t+1}; filing text must be available
strictly before W_t starts. TF-IDF and blend settings are selected on H1--H13
transitions and held fixed for W1--W7.

Each run writes seven JSON files, a manifest, and SHA256SUMS.txt to a new
directory under ``results/acceptance_robustness/``. The JSON files record
settings, transition results, summaries, runtimes, checks, and input hashes.
Existing manuscript and result files are left intact.

Run from the project root:

    python3 scripts/run_acceptance_robustness.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
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
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metrics import TIE_RNG_BASE, ndcg_at_k  # noqa: E402


SCHEMA_VERSION = "acceptance-robustness-v1"
K_PRIMARY = 5
K_GRID = (1, 5, 10, 20)
HALF_LIVES = (20, 40)
ALPHA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MIN_UNIVERSE = 50

WINDOWS_MODERN_DIR = PROJECT_ROOT / "results" / "windows_v3"
WINDOWS_HISTORICAL_DIR = PROJECT_ROOT / "results" / "ea_historical" / "windows"
VINTAGE_DIR = PROJECT_ROOT / "results" / "vintage_descriptions"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "acceptance_robustness"

HISTORICAL_WINDOWS = tuple(f"H{i}" for i in range(1, 14))
MODERN_WINDOWS = tuple(f"W{i}" for i in range(1, 8))
HISTORICAL_TUNING_TRANSITIONS = tuple(
    zip(HISTORICAL_WINDOWS[:-1], HISTORICAL_WINDOWS[1:])
)
MODERN_TRANSITIONS = tuple(zip(MODERN_WINDOWS[:-1], MODERN_WINDOWS[1:]))
PREVIOUS_WINDOW = {
    "W1": "H13",
    "W2": "W1",
    "W3": "W2",
    "W4": "W3",
    "W5": "W4",
    "W6": "W5",
}

# Keep the six labels from scripts/gate_d_tfidf_sweep.py, including the
# duplicate settings for a and e.
TFIDF_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "config_id": "a",
        "kwargs": {
            "max_features": 20_000,
            "ngram_range": (1, 1),
            "stop_words": "english",
            "min_df": 2,
            "sublinear_tf": True,
        },
        "description": "20k unigram",
    },
    {
        "config_id": "b",
        "kwargs": {
            "max_features": 50_000,
            "ngram_range": (1, 2),
            "stop_words": "english",
            "min_df": 2,
            "sublinear_tf": True,
            "strip_accents": "unicode",
        },
        "description": "50k unigram+bigram with Unicode accent stripping",
    },
    {
        "config_id": "c",
        "kwargs": {
            "max_features": 50_000,
            "ngram_range": (1, 1),
            "stop_words": "english",
            "min_df": 2,
            "sublinear_tf": True,
        },
        "description": "50k unigram",
    },
    {
        "config_id": "d",
        "kwargs": {
            "max_features": None,
            "ngram_range": (1, 1),
            "stop_words": "english",
            "min_df": 2,
            "sublinear_tf": True,
        },
        "description": "uncapped unigram",
    },
    {
        "config_id": "e",
        "kwargs": {
            "max_features": 20_000,
            "ngram_range": (1, 1),
            "stop_words": "english",
            "min_df": 2,
            "sublinear_tf": True,
        },
        "description": "duplicate of config a retained from existing grid",
    },
    {
        "config_id": "f",
        "kwargs": {
            "max_features": 20_000,
            "ngram_range": (1, 2),
            "stop_words": "english",
            "min_df": 5,
            "sublinear_tf": True,
        },
        "description": "20k unigram+bigram with min_df=5",
    },
)

NON_PRIMARY_SHARE_RE = re.compile(r"-[A-Z]{1,2}$")


@dataclass(frozen=True)
class WindowData:
    name: str
    directory: Path
    tickers: tuple[str, ...]
    ticker_to_idx: dict[str, int]
    ts_features: np.ndarray
    corr_target: np.ndarray
    start: date
    end: date


@dataclass
class TransitionData:
    wt: str
    wt1: str
    tickers: tuple[str, ...]
    ts_feature: np.ndarray
    realized: np.ndarray
    texts: tuple[str, ...]
    filing_dates: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.wt}->{self.wt1}"

    @property
    def n(self) -> int:
        return len(self.tickers)


WINDOW_CACHE: dict[str, WindowData] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    absolute_path = Path(os.path.abspath(path))
    try:
        return absolute_path.relative_to(Path(os.path.abspath(PROJECT_ROOT))).as_posix()
    except ValueError:
        return f"external/{absolute_path.name}"


def window_directory(name: str) -> Path:
    if name.startswith("H"):
        return WINDOWS_HISTORICAL_DIR / name
    if name.startswith("W"):
        return WINDOWS_MODERN_DIR / name
    raise ValueError(f"unknown window: {name}")


def load_window(name: str) -> WindowData:
    cached = WINDOW_CACHE.get(name)
    if cached is not None:
        return cached

    directory = window_directory(name)
    required = (
        directory / "ts_features.npy",
        directory / "corr_target.npy",
        directory / "universe.parquet",
        directory / "meta.json",
    )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing window inputs for {name}: {missing}")

    ts_features = np.load(directory / "ts_features.npy", mmap_mode="r")
    corr_target = np.load(directory / "corr_target.npy", mmap_mode="r")
    tickers_list = pl.read_parquet(directory / "universe.parquet")[
        "ticker"
    ].to_list()
    tickers = tuple(str(t) for t in tickers_list)
    meta = json.loads((directory / "meta.json").read_text())

    if len(tickers) != len(set(tickers)):
        raise AssertionError(f"{name}: duplicate ticker in return universe")
    if ts_features.ndim != 2 or ts_features.shape[0] != len(tickers):
        raise AssertionError(f"{name}: ts_features shape/universe mismatch")
    if corr_target.shape != (len(tickers), len(tickers)):
        raise AssertionError(f"{name}: corr_target shape/universe mismatch")
    if not np.isfinite(ts_features).all() or not np.isfinite(corr_target).all():
        raise AssertionError(f"{name}: non-finite return artifact")

    result = WindowData(
        name=name,
        directory=directory,
        tickers=tickers,
        ticker_to_idx={ticker: i for i, ticker in enumerate(tickers)},
        ts_features=ts_features,
        corr_target=corr_target,
        start=date.fromisoformat(meta["start"]),
        end=date.fromisoformat(meta["end"]),
    )
    WINDOW_CACHE[name] = result
    return result


def load_vintage_text_map(name: str) -> dict[str, tuple[str, str]]:
    """Load 10-K text and check that each filing predates the feature window."""
    window = load_window(name)
    path = VINTAGE_DIR / f"{name}_descriptions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing vintage text: {path}")

    df = pl.read_parquet(path)
    required_columns = {"ticker", "source", "item1_text", "filing_date"}
    if not required_columns.issubset(df.columns):
        raise AssertionError(
            f"{name}: vintage file lacks {sorted(required_columns - set(df.columns))}"
        )
    df = df.filter(
        (pl.col("source") == "10K")
        & pl.col("item1_text").is_not_null()
        & (pl.col("item1_text").str.len_chars() > 0)
    )

    result: dict[str, tuple[str, str]] = {}
    for row in df.select(
        ["ticker", "item1_text", "filing_date"]
    ).iter_rows(named=True):
        ticker = str(row["ticker"])
        filing_date = str(row["filing_date"])
        if date.fromisoformat(filing_date) >= window.start:
            raise AssertionError(
                f"{name}: non-vintage filing {ticker} on {filing_date} "
                f"is not strictly before feature start {window.start}"
            )
        if ticker in result:
            raise AssertionError(f"{name}: duplicate vintage 10-K for {ticker}")
        result[ticker] = (str(row["item1_text"]), filing_date)
    return result


def build_transition(
    wt: str,
    wt1: str,
    *,
    universe_policy: str = "feature_time",
) -> TransitionData:
    """Align returns and vintage text for one transition."""
    if universe_policy not in {"feature_time", "symmetric"}:
        raise ValueError(f"unsupported universe policy: {universe_policy}")
    feature = load_window(wt)
    target = load_window(wt1)
    text_t = load_vintage_text_map(wt)

    eligible = set(feature.tickers) & set(target.tickers) & set(text_t)
    if universe_policy == "symmetric":
        # This comparison reproduces the older symmetric eligibility rule.
        # Only target-window text availability is used, never its content.
        text_t1 = load_vintage_text_map(wt1)
        eligible &= set(text_t1)

    tickers = tuple(sorted(eligible))
    if len(tickers) < MIN_UNIVERSE:
        raise AssertionError(
            f"{wt}->{wt1}: only {len(tickers)} securities under {universe_policy}"
        )

    idx_t = np.fromiter(
        (feature.ticker_to_idx[t] for t in tickers), dtype=np.int64
    )
    idx_t1 = np.fromiter(
        (target.ticker_to_idx[t] for t in tickers), dtype=np.int64
    )
    ts_feature = np.asarray(feature.ts_features[idx_t], dtype=np.float32)
    realized = np.asarray(
        target.corr_target[np.ix_(idx_t1, idx_t1)], dtype=np.float32
    )
    texts = tuple(text_t[t][0] for t in tickers)
    filing_dates = tuple(text_t[t][1] for t in tickers)

    if len(tickers) != len(texts) or not all(texts):
        raise AssertionError(f"{wt}->{wt1}: missing feature-time text")
    if max(date.fromisoformat(d) for d in filing_dates) >= feature.start:
        raise AssertionError(f"{wt}->{wt1}: feature-time vintage check failed")

    return TransitionData(
        wt=wt,
        wt1=wt1,
        tickers=tickers,
        ts_feature=ts_feature,
        realized=realized,
        texts=texts,
        filing_dates=filing_dates,
    )


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise AssertionError("zero-norm return-path row")
    return (values / norms).astype(np.float32)


def return_path_scores(ts_feature: np.ndarray) -> np.ndarray:
    embedding = l2_rows(ts_feature)
    result = (embedding @ embedding.T).astype(np.float32)
    validate_score_matrix(result, "ReturnPath")
    return result


def ewma_return_path_scores(
    ts_feature: np.ndarray, half_life: int
) -> np.ndarray:
    """Apply fixed square-root EWMA weights to the standardized return path.

    The residuals already come from an OLS fit with an intercept. Keep that fit
    and centering, apply the fixed recency weights, then L2-normalize each row.
    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    n_days = ts_feature.shape[1]
    ages = (n_days - 1) - np.arange(n_days, dtype=np.float64)
    weights = np.power(0.5, ages / float(half_life))
    weighted_path = np.asarray(ts_feature, dtype=np.float64) * np.sqrt(weights)
    embedding = l2_rows(weighted_path)
    result = (embedding @ embedding.T).astype(np.float32)
    validate_score_matrix(result, f"EWMA-{half_life}")
    return result


def effective_tfidf_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(**kwargs)
    return json_safe(vectorizer.get_params(deep=False))


def fit_tfidf_scores(
    texts: tuple[str, ...] | list[str],
    kwargs: dict[str, Any],
) -> tuple[np.ndarray, int]:
    vectorizer = TfidfVectorizer(**kwargs)
    matrix = vectorizer.fit_transform(texts)
    matrix = normalize(matrix, norm="l2")
    scores = np.asarray((matrix @ matrix.T).toarray(), dtype=np.float32)
    validate_score_matrix(scores, "TF-IDF")
    return scores, len(vectorizer.vocabulary_)


def validate_score_matrix(scores: np.ndarray, label: str) -> None:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise AssertionError(f"{label}: score matrix must be square")
    if not np.isfinite(scores).all():
        raise AssertionError(f"{label}: score matrix has non-finite values")
    if not np.allclose(scores, scores.T, atol=2e-5, rtol=2e-5):
        raise AssertionError(f"{label}: score matrix is not symmetric")


def evaluate_scores(
    scores: np.ndarray,
    realized: np.ndarray,
    *,
    k: int,
    oracle_check: bool = False,
) -> dict[str, Any]:
    """Score every row as a query using the shared NDCG implementation."""
    if scores.shape != realized.shape:
        raise AssertionError("score/relevance shape mismatch")
    n = scores.shape[0]
    values: list[float | None] = []
    for query_idx in range(n):
        values.append(
            ndcg_at_k(
                scores[query_idx],
                realized[query_idx],
                query_idx,
                k=k,
            )
        )
    valid = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if valid.size == 0:
        raise AssertionError("no valid census queries")
    if np.any((valid < 0.0) | (valid > 1.0)):
        raise AssertionError("NDCG outside [0,1]")

    result = {
        "n_queries": n,
        "n_valid": int(valid.size),
        "n_undefined_no_positive_idcg": int(n - valid.size),
        "mean_ndcg": float(valid.mean()),
        "population_std_ndcg": float(valid.std(ddof=0)),
        "min_ndcg": float(valid.min()),
        "max_ndcg": float(valid.max()),
    }

    if oracle_check:
        oracle_values = []
        for query_idx in range(n):
            value = ndcg_at_k(
                realized[query_idx],
                realized[query_idx],
                query_idx,
                k=k,
            )
            if value is not None:
                oracle_values.append(value)
        oracle = np.asarray(oracle_values, dtype=np.float64)
        if oracle.size == 0 or not np.allclose(oracle, 1.0, atol=1e-12):
            raise AssertionError(
                f"real-data oracle failed: mean={oracle.mean() if oracle.size else None}"
            )
        result["oracle_mean_ndcg"] = float(oracle.mean())
    return result


def average_rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Convert scores to per-query percentile ranks, averaging exact ties."""
    validate_score_matrix(scores, "rank input")
    n = scores.shape[0]
    work = np.asarray(scores, dtype=np.float64).copy()
    np.fill_diagonal(work, -np.inf)
    ranks = rankdata(work, axis=1, method="average")
    # The excluded query has rank 1; candidates occupy ranks [2,N].
    # Exact ties receive their average rank.
    ranks = (ranks - 1.0) / float(n - 1)
    np.fill_diagonal(ranks, 0.0)
    return ranks.astype(np.float32)


def mean(values: list[float]) -> float:
    if not values:
        raise AssertionError("cannot summarize empty values")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def population_std(values: list[float]) -> float:
    if not values:
        raise AssertionError("cannot summarize empty values")
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def validation_oracles() -> dict[str, Any]:
    """Check the metric and configuration grid before running experiments."""
    realized = np.array([1.0, 0.8, 0.4, 0.2, -0.5, -0.2], dtype=float)
    oracle = ndcg_at_k(realized, realized, 0, k=3)
    if oracle is None or not math.isclose(oracle, 1.0, abs_tol=1e-12):
        raise AssertionError(f"toy oracle NDCG failed: {oracle}")

    negative_first = np.array([99.0, -1.0, -2.0, 0.0, 0.0, 0.0])
    positive_relevance_check = ndcg_at_k(
        negative_first, realized, 0, k=2
    )
    if positive_relevance_check is None:
        raise AssertionError("positive-relevance toy check unexpectedly undefined")

    tied_scores = np.ones(10, dtype=float)
    tied_realized = np.linspace(-0.4, 0.5, 10)
    tie_1 = ndcg_at_k(tied_scores, tied_realized, 3, k=5)
    tie_2 = ndcg_at_k(tied_scores, tied_realized, 3, k=5)
    if tie_1 != tie_2:
        raise AssertionError("canonical tie handling is not deterministic")

    # The duplicate a/e settings must stay identical.
    config_a = next(c for c in TFIDF_CONFIGS if c["config_id"] == "a")
    config_e = next(c for c in TFIDF_CONFIGS if c["config_id"] == "e")
    if config_a["kwargs"] != config_e["kwargs"]:
        raise AssertionError("existing TF-IDF configs a and e should be identical")

    if not all(a.startswith("H") and b.startswith("H")
               for a, b in HISTORICAL_TUNING_TRANSITIONS):
        raise AssertionError("historical tuning set contains a modern window")
    if any(a.startswith("H") or b.startswith("H") for a, b in MODERN_TRANSITIONS):
        raise AssertionError("modern evaluation set contains a historical window")

    return {
        "canonical_metric_import": "scripts.metrics.ndcg_at_k",
        "positive_relevance": "max(0, realized correlation) in canonical scorer",
        "toy_oracle_ndcg": float(oracle),
        "toy_positive_relevance_ndcg": float(positive_relevance_check),
        "deterministic_tie_repeat_ndcg": float(tie_1),
        "canonical_tie_rng_base": TIE_RNG_BASE,
        "all_queries_policy": "range(N); no query sampling",
        "historical_modern_partition": "passed",
        "duplicate_config_a_e": "passed",
    }


def collect_input_provenance() -> dict[str, Any]:
    files: list[Path] = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "metrics.py",
        PROJECT_ROOT / "requirements.lock",
    ]
    for name in HISTORICAL_WINDOWS + MODERN_WINDOWS:
        directory = window_directory(name)
        files.extend(
            [
                directory / "ts_features.npy",
                directory / "corr_target.npy",
                directory / "universe.parquet",
                directory / "meta.json",
                VINTAGE_DIR / f"{name}_descriptions.parquet",
            ]
        )
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"provenance inputs missing: {missing}")

    try:
        git_commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
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
        "sha256": {relative(path): sha256_file(path) for path in files},
    }


def base_configuration() -> dict[str, Any]:
    return {
        "metric": {
            "name": "canonical positive-relevance NDCG",
            "implementation": "scripts.metrics.ndcg_at_k",
            "primary_k": K_PRIMARY,
            "relevance": "max(0, realized W_{t+1} Ledoit-Wolf residual correlation)",
            "query_exclusion": "query security excluded from retrieved and ideal lists",
            "tie_handling": (
                "canonical seeded jitter with seed TIE_RNG_BASE + sorted query index"
            ),
            "query_policy": "census: every eligible security",
        },
        "universe": {
            "current_protocol": (
                "sorted intersection of return availability in W_t and W_{t+1} "
                "and non-empty vintage 10-K Item 1 text in W_t only"
            ),
            "text_timing": "source=10K and filing_date strictly before W_t start",
            "target_window_text": "not used except to define the legacy symmetric universe in experiment 7",
        },
        "return_path": (
            "row-L2-normalized standardized FF5 residual-return path from W_t; "
            "cosine/inner-product score"
        ),
        "historical_tuning_transitions": [
            f"{a}->{b}" for a, b in HISTORICAL_TUNING_TRANSITIONS
        ],
        "modern_frozen_evaluation_transitions": [
            f"{a}->{b}" for a, b in MODERN_TRANSITIONS
        ],
    }


def make_bundle(
    experiment_id: str,
    configuration: dict[str, Any],
    per_transition: list[dict[str, Any]],
    summary: dict[str, Any],
    runtime_seconds: float,
    validations: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_utc": utc_now(),
        "configuration": json_safe(configuration),
        "per_transition": json_safe(per_transition),
        "summary": json_safe(summary),
        "runtime_seconds": float(runtime_seconds),
        "validations": json_safe(validations),
        "provenance": provenance,
    }


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_tfidf_historical_tuning(
    provenance: dict[str, Any],
    validations: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    config_by_id = {c["config_id"]: c for c in TFIDF_CONFIGS}

    print("\n[1/7] Historical-only TF-IDF configuration tuning", flush=True)
    for wt, wt1 in HISTORICAL_TUNING_TRANSITIONS:
        transition_start = time.perf_counter()
        td = build_transition(wt, wt1, universe_policy="feature_time")
        cache: dict[str, tuple[np.ndarray, int, float, dict[str, Any]]] = {}

        for config in TFIDF_CONFIGS:
            config_id = config["config_id"]
            key = json.dumps(json_safe(config["kwargs"]), sort_keys=True)
            fit_start = time.perf_counter()
            if key in cache:
                scores, vocab_size, _, cached_eval = cache[key]
                evaluation = dict(cached_eval)
                fit_runtime = time.perf_counter() - fit_start
                reused_from = next(
                    r["config_id"]
                    for r in rows
                    if r["transition"] == td.label
                    and config_by_id[r["config_id"]]["kwargs"] == config["kwargs"]
                )
            else:
                scores, vocab_size = fit_tfidf_scores(td.texts, config["kwargs"])
                evaluation = evaluate_scores(
                    scores, td.realized, k=K_PRIMARY, oracle_check=False
                )
                fit_runtime = time.perf_counter() - fit_start
                cache[key] = (scores, vocab_size, fit_runtime, dict(evaluation))
                reused_from = None

            rows.append(
                {
                    "transition": td.label,
                    "feature_window": wt,
                    "target_window": wt1,
                    "config_id": config_id,
                    "n_universe": td.n,
                    "vocab_size": vocab_size,
                    **evaluation,
                    "runtime_seconds": fit_runtime,
                    "reused_identical_scores_from": reused_from,
                }
            )

        # The duplicate a/e settings should produce identical results.
        a_row = next(
            r for r in rows
            if r["transition"] == td.label and r["config_id"] == "a"
        )
        e_row = next(
            r for r in rows
            if r["transition"] == td.label and r["config_id"] == "e"
        )
        if a_row["mean_ndcg"] != e_row["mean_ndcg"]:
            raise AssertionError(f"{td.label}: duplicate configs a/e diverged")
        print(
            f"  {td.label}: N={td.n}, "
            f"{time.perf_counter() - transition_start:.1f}s",
            flush=True,
        )

    aggregate_rows = []
    for config in TFIDF_CONFIGS:
        values = [
            float(r["mean_ndcg"])
            for r in rows
            if r["config_id"] == config["config_id"]
        ]
        aggregate_rows.append(
            {
                "config_id": config["config_id"],
                "macro_mean_ndcg": mean(values),
                "population_std_across_transitions": population_std(values),
                "n_historical_transitions": len(values),
            }
        )

    # Break selection ties by alphabetical config id.
    winner_row = sorted(
        aggregate_rows,
        key=lambda r: (-float(r["macro_mean_ndcg"]), str(r["config_id"])),
    )[0]
    winner = config_by_id[str(winner_row["config_id"])]
    elapsed = time.perf_counter() - started
    summary = {
        "selection_data": "H1->H2 through H12->H13 only",
        "selection_objective": "unweighted macro mean of transition-level census NDCG@5",
        "tie_policy": "alphabetically smallest config_id",
        "config_aggregates": aggregate_rows,
        "selected_config_id": winner["config_id"],
        "selected_config_kwargs": winner["kwargs"],
        "selected_macro_mean_ndcg": winner_row["macro_mean_ndcg"],
        "modern_data_used_for_selection": False,
    }
    configuration = {
        **base_configuration(),
        "grid": [
            {
                "config_id": c["config_id"],
                "description": c["description"],
                "specified_kwargs": c["kwargs"],
                "effective_sklearn_params": effective_tfidf_params(c["kwargs"]),
            }
            for c in TFIDF_CONFIGS
        ],
        "selection_rule": summary["selection_objective"],
    }
    bundle = make_bundle(
        "01_tfidf_historical_tuning",
        configuration,
        rows,
        summary,
        elapsed,
        validations,
        provenance,
    )
    print(
        f"  selected TF-IDF config {winner['config_id']} "
        f"(historical macro NDCG={winner_row['macro_mean_ndcg']:.6f})",
        flush=True,
    )
    return winner, bundle


def build_frozen_transition(
    wt: str,
    wt1: str,
    tfidf_config: dict[str, Any],
    *,
    oracle_check: bool = False,
) -> dict[str, Any]:
    td = build_transition(wt, wt1, universe_policy="feature_time")
    rp_scores = return_path_scores(td.ts_feature)
    tf_scores, vocab_size = fit_tfidf_scores(td.texts, tfidf_config["kwargs"])
    rp_eval = evaluate_scores(
        rp_scores, td.realized, k=K_PRIMARY, oracle_check=oracle_check
    )
    tf_eval = evaluate_scores(
        tf_scores, td.realized, k=K_PRIMARY, oracle_check=False
    )
    return {
        "transition_data": td,
        "rp_scores": rp_scores,
        "tf_scores": tf_scores,
        "vocab_size": vocab_size,
        "rp_eval": rp_eval,
        "tf_eval": tf_eval,
    }


def run_blend_tuning_and_modern(
    tfidf_config: dict[str, Any],
    provenance: dict[str, Any],
    validations: dict[str, Any],
) -> tuple[float, dict[str, dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    historical_rows: list[dict[str, Any]] = []

    print("\n[2/7] Historical-only blend-alpha tuning; frozen modern evaluation", flush=True)
    for wt, wt1 in HISTORICAL_TUNING_TRANSITIONS:
        transition_start = time.perf_counter()
        frozen = build_frozen_transition(wt, wt1, tfidf_config)
        td: TransitionData = frozen["transition_data"]
        rp_ranks = average_rank_matrix(frozen["rp_scores"])
        tf_ranks = average_rank_matrix(frozen["tf_scores"])
        for alpha in ALPHA_GRID:
            blend_scores = alpha * rp_ranks + (1.0 - alpha) * tf_ranks
            evaluation = evaluate_scores(
                blend_scores, td.realized, k=K_PRIMARY, oracle_check=False
            )
            historical_rows.append(
                {
                    "phase": "historical_tuning",
                    "transition": td.label,
                    "alpha_return_path": alpha,
                    "alpha_tfidf": 1.0 - alpha,
                    "n_universe": td.n,
                    "return_path_ndcg": frozen["rp_eval"]["mean_ndcg"],
                    "tfidf_ndcg": frozen["tf_eval"]["mean_ndcg"],
                    "blend_ndcg": evaluation["mean_ndcg"],
                    "n_queries": evaluation["n_queries"],
                    "n_valid": evaluation["n_valid"],
                }
            )
        print(
            f"  tune {td.label}: N={td.n}, "
            f"{time.perf_counter() - transition_start:.1f}s",
            flush=True,
        )

    alpha_aggregates = []
    for alpha in ALPHA_GRID:
        values = [
            float(r["blend_ndcg"])
            for r in historical_rows
            if r["alpha_return_path"] == alpha
        ]
        alpha_aggregates.append(
            {
                "alpha_return_path": alpha,
                "alpha_tfidf": 1.0 - alpha,
                "macro_mean_ndcg": mean(values),
                "population_std_across_transitions": population_std(values),
                "n_historical_transitions": len(values),
            }
        )

    # Break selection ties in favor of the larger return-path weight.
    selected = sorted(
        alpha_aggregates,
        key=lambda r: (
            -float(r["macro_mean_ndcg"]),
            -float(r["alpha_return_path"]),
        ),
    )[0]
    frozen_alpha = float(selected["alpha_return_path"])

    modern_cache: dict[str, dict[str, Any]] = {}
    modern_rows: list[dict[str, Any]] = []
    for index, (wt, wt1) in enumerate(MODERN_TRANSITIONS):
        transition_start = time.perf_counter()
        frozen = build_frozen_transition(
            wt, wt1, tfidf_config, oracle_check=(index == 0)
        )
        td: TransitionData = frozen["transition_data"]
        rp_ranks = average_rank_matrix(frozen["rp_scores"])
        tf_ranks = average_rank_matrix(frozen["tf_scores"])
        blend_scores = frozen_alpha * rp_ranks + (1.0 - frozen_alpha) * tf_ranks
        blend_eval = evaluate_scores(
            blend_scores, td.realized, k=K_PRIMARY, oracle_check=False
        )
        delta = float(blend_eval["mean_ndcg"] - frozen["rp_eval"]["mean_ndcg"])
        modern_rows.append(
            {
                "phase": "modern_frozen_evaluation",
                "transition": td.label,
                "alpha_return_path": frozen_alpha,
                "alpha_tfidf": 1.0 - frozen_alpha,
                "tfidf_config_id": tfidf_config["config_id"],
                "n_universe": td.n,
                "vocab_size": frozen["vocab_size"],
                "return_path_ndcg": frozen["rp_eval"]["mean_ndcg"],
                "tfidf_ndcg": frozen["tf_eval"]["mean_ndcg"],
                "blend_ndcg": blend_eval["mean_ndcg"],
                "blend_minus_return_path": delta,
                "improves_return_path": bool(delta > 0.0),
                "n_queries": blend_eval["n_queries"],
                "n_valid": blend_eval["n_valid"],
                "runtime_seconds": time.perf_counter() - transition_start,
            }
        )
        frozen["blend_scores"] = blend_scores.astype(np.float32)
        frozen["blend_eval"] = blend_eval
        modern_cache[td.label] = frozen
        print(
            f"  modern {td.label}: N={td.n}, RP={frozen['rp_eval']['mean_ndcg']:.4f}, "
            f"blend={blend_eval['mean_ndcg']:.4f}, delta={delta:+.4f}",
            flush=True,
        )

    modern_deltas = [float(r["blend_minus_return_path"]) for r in modern_rows]
    elapsed = time.perf_counter() - started
    summary = {
        "tfidf_config_frozen_from_historical": {
            "config_id": tfidf_config["config_id"],
            "kwargs": tfidf_config["kwargs"],
        },
        "alpha_selection_data": "H1->H2 through H12->H13 only",
        "alpha_selection_objective": (
            "unweighted macro mean of transition-level census NDCG@5"
        ),
        "alpha_tie_policy": "largest ReturnPath alpha",
        "alpha_aggregates": alpha_aggregates,
        "selected_alpha_return_path": frozen_alpha,
        "selected_alpha_tfidf": 1.0 - frozen_alpha,
        "selected_historical_macro_mean_ndcg": selected["macro_mean_ndcg"],
        "modern_data_used_for_selection": False,
        "modern_mean_return_path_ndcg": mean(
            [float(r["return_path_ndcg"]) for r in modern_rows]
        ),
        "modern_mean_tfidf_ndcg": mean(
            [float(r["tfidf_ndcg"]) for r in modern_rows]
        ),
        "modern_mean_blend_ndcg": mean(
            [float(r["blend_ndcg"]) for r in modern_rows]
        ),
        "modern_mean_blend_minus_return_path": mean(modern_deltas),
        "modern_transitions_improved": sum(
            bool(r["improves_return_path"]) for r in modern_rows
        ),
        "modern_transition_count": len(modern_rows),
    }
    configuration = {
        **base_configuration(),
        "tfidf_config": {
            "config_id": tfidf_config["config_id"],
            "specified_kwargs": tfidf_config["kwargs"],
            "effective_sklearn_params": effective_tfidf_params(
                tfidf_config["kwargs"]
            ),
        },
        "alpha_grid_return_path_weight": ALPHA_GRID,
        "rank_normalization": (
            "per-query scipy.stats.rankdata(method='average'); exact ties "
            "receive equal average ranks; self excluded before ranking"
        ),
        "blend": "alpha * rank(ReturnPath) + (1-alpha) * rank(TF-IDF)",
    }
    bundle = make_bundle(
        "02_blend_historical_tuning_modern_frozen",
        configuration,
        historical_rows + modern_rows,
        summary,
        elapsed,
        validations,
        provenance,
    )
    print(
        f"  selected alpha={frozen_alpha:.1f}; modern improvements "
        f"{summary['modern_transitions_improved']}/{len(modern_rows)}",
        flush=True,
    )
    return frozen_alpha, modern_cache, bundle


def run_k_sensitivity(
    tfidf_config: dict[str, Any],
    modern_cache: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    validations: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    print("\n[3/7] Modern census k sensitivity", flush=True)
    for wt, wt1 in MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        frozen = modern_cache[label]
        td: TransitionData = frozen["transition_data"]
        for k in K_GRID:
            for method, scores in (
                ("ReturnPath", frozen["rp_scores"]),
                ("TF-IDF", frozen["tf_scores"]),
            ):
                evaluation = evaluate_scores(
                    scores, td.realized, k=k, oracle_check=False
                )
                rows.append(
                    {
                        "transition": label,
                        "method": method,
                        "k": k,
                        "n_universe": td.n,
                        **evaluation,
                    }
                )
        print(f"  {label}: N={td.n}", flush=True)

    aggregates = []
    for k in K_GRID:
        for method in ("ReturnPath", "TF-IDF"):
            values = [
                float(r["mean_ndcg"])
                for r in rows
                if r["k"] == k and r["method"] == method
            ]
            aggregates.append(
                {
                    "method": method,
                    "k": k,
                    "macro_mean_ndcg": mean(values),
                    "population_std_across_transitions": population_std(values),
                }
            )
    return make_bundle(
        "03_modern_k_sensitivity",
        {
            **base_configuration(),
            "k_grid": K_GRID,
            "tfidf_config": {
                "config_id": tfidf_config["config_id"],
                "kwargs": tfidf_config["kwargs"],
            },
            "universe_alignment": (
                "ReturnPath and frozen TF-IDF use the identical feature-time "
                "universe within each transition"
            ),
        },
        rows,
        {"aggregates": aggregates},
        time.perf_counter() - started,
        validations,
        provenance,
    )


def run_ewma_baselines(
    modern_cache: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    validations: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    print("\n[4/7] Fixed EWMA return-path baselines", flush=True)
    for wt, wt1 in MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        frozen = modern_cache[label]
        td: TransitionData = frozen["transition_data"]
        row: dict[str, Any] = {
            "transition": label,
            "n_universe": td.n,
            "n_feature_days": td.ts_feature.shape[1],
            "return_path_ndcg": frozen["rp_eval"]["mean_ndcg"],
            "n_queries": frozen["rp_eval"]["n_queries"],
            "n_valid": frozen["rp_eval"]["n_valid"],
        }
        for half_life in HALF_LIVES:
            scores = ewma_return_path_scores(td.ts_feature, half_life)
            evaluation = evaluate_scores(
                scores, td.realized, k=K_PRIMARY, oracle_check=False
            )
            row[f"ewma_hl{half_life}_ndcg"] = evaluation["mean_ndcg"]
            row[f"ewma_hl{half_life}_minus_return_path"] = (
                evaluation["mean_ndcg"] - frozen["rp_eval"]["mean_ndcg"]
            )
        rows.append(row)
        print(
            f"  {label}: RP={row['return_path_ndcg']:.4f}, "
            f"HL20={row['ewma_hl20_ndcg']:.4f}, "
            f"HL40={row['ewma_hl40_ndcg']:.4f}",
            flush=True,
        )

    summary: dict[str, Any] = {
        "mean_return_path_ndcg": mean(
            [float(r["return_path_ndcg"]) for r in rows]
        )
    }
    for half_life in HALF_LIVES:
        summary[f"mean_ewma_hl{half_life}_ndcg"] = mean(
            [float(r[f"ewma_hl{half_life}_ndcg"]) for r in rows]
        )
        summary[f"mean_ewma_hl{half_life}_minus_return_path"] = mean(
            [
                float(r[f"ewma_hl{half_life}_minus_return_path"])
                for r in rows
            ]
        )
        summary[f"transitions_ewma_hl{half_life}_improved"] = sum(
            float(r[f"ewma_hl{half_life}_minus_return_path"]) > 0.0
            for r in rows
        )

    return make_bundle(
        "04_fixed_ewma_return_path",
        {
            **base_configuration(),
            "half_lives_trading_days": HALF_LIVES,
            "weight_definition": "w_age = 0.5 ** (age / half_life)",
            "embedding_definition": (
                "standardized residual path multiplied elementwise by sqrt(w), "
                "then row-L2 normalized; no additional weighted re-centering"
            ),
            "universe_alignment": (
                "unweighted and EWMA paths use identical feature-time universes"
            ),
        },
        rows,
        summary,
        time.perf_counter() - started,
        validations,
        provenance,
    )


def run_two_window_concat(
    modern_cache: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    validations: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    limitations: list[str] = []
    print("\n[5/7] Two-window concatenated return path", flush=True)
    for wt, wt1 in MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        previous_name = PREVIOUS_WINDOW.get(wt)
        if previous_name is None:
            limitations.append(f"{label}: no declared previous window")
            continue
        previous = load_window(previous_name)
        frozen = modern_cache[label]
        td: TransitionData = frozen["transition_data"]
        eligible = tuple(sorted(set(td.tickers) & set(previous.tickers)))
        if len(eligible) < MIN_UNIVERSE:
            limitations.append(
                f"{label}: only {len(eligible)} tickers with previous-window returns"
            )
            continue

        base_idx = {t: i for i, t in enumerate(td.tickers)}
        idx_base = np.fromiter((base_idx[t] for t in eligible), dtype=np.int64)
        idx_prev = np.fromiter(
            (previous.ticker_to_idx[t] for t in eligible), dtype=np.int64
        )
        current_ts = td.ts_feature[idx_base]
        previous_ts = np.asarray(previous.ts_features[idx_prev], dtype=np.float32)
        realized = td.realized[np.ix_(idx_base, idx_base)]

        one_scores = return_path_scores(current_ts)
        concatenated = np.concatenate((previous_ts, current_ts), axis=1)
        two_scores = return_path_scores(concatenated)
        one_eval = evaluate_scores(
            one_scores, realized, k=K_PRIMARY, oracle_check=False
        )
        two_eval = evaluate_scores(
            two_scores, realized, k=K_PRIMARY, oracle_check=False
        )
        delta = float(two_eval["mean_ndcg"] - one_eval["mean_ndcg"])
        rows.append(
            {
                "transition": label,
                "previous_feature_window": previous_name,
                "n_universe": len(eligible),
                "n_previous_days": previous_ts.shape[1],
                "n_current_days": current_ts.shape[1],
                "one_window_ndcg": one_eval["mean_ndcg"],
                "two_window_concat_ndcg": two_eval["mean_ndcg"],
                "two_minus_one": delta,
                "two_window_improves": bool(delta > 0.0),
                "n_queries": one_eval["n_queries"],
                "n_valid_one": one_eval["n_valid"],
                "n_valid_two": two_eval["n_valid"],
            }
        )
        print(
            f"  {label}: prev={previous_name}, N={len(eligible)}, "
            f"one={one_eval['mean_ndcg']:.4f}, two={two_eval['mean_ndcg']:.4f}",
            flush=True,
        )

    if not rows:
        raise AssertionError("no two-window comparisons were feasible")
    deltas = [float(r["two_minus_one"]) for r in rows]
    return make_bundle(
        "05_two_window_concatenated_return_path",
        {
            **base_configuration(),
            "previous_window_mapping": PREVIOUS_WINDOW,
            "one_window": "row-L2-normalized W_t standardized residual path",
            "two_window": (
                "concatenate previous-window and W_t standardized residual paths "
                "in chronological order, then row-L2 normalize"
            ),
            "universe_alignment": (
                "one- and two-window paths use the identical subset with returns "
                "in previous, feature, and target windows plus W_t vintage text"
            ),
        },
        rows,
        {
            "n_transitions_evaluated": len(rows),
            "n_transitions_improved": sum(
                bool(r["two_window_improves"]) for r in rows
            ),
            "mean_two_minus_one": mean(deltas),
            "mean_one_window_ndcg": mean(
                [float(r["one_window_ndcg"]) for r in rows]
            ),
            "mean_two_window_ndcg": mean(
                [float(r["two_window_concat_ndcg"]) for r in rows]
            ),
            "limitations": limitations,
        },
        time.perf_counter() - started,
        validations,
        provenance,
    )


def is_non_primary_share_heuristic(ticker: str) -> bool:
    return NON_PRIMARY_SHARE_RE.search(ticker) is not None


def run_common_share_exclusion(
    tfidf_config: dict[str, Any],
    modern_cache: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    validations: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    print("\n[6/7] Common-share ticker-heuristic exclusion", flush=True)
    all_flagged: set[str] = set()
    for wt, wt1 in MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        frozen = modern_cache[label]
        td: TransitionData = frozen["transition_data"]
        keep = tuple(t for t in td.tickers if not is_non_primary_share_heuristic(t))
        flagged = tuple(t for t in td.tickers if is_non_primary_share_heuristic(t))
        all_flagged.update(flagged)
        if len(keep) < MIN_UNIVERSE:
            raise AssertionError(f"{label}: common-share restriction too small")

        base_idx = {t: i for i, t in enumerate(td.tickers)}
        idx = np.fromiter((base_idx[t] for t in keep), dtype=np.int64)
        restricted_ts = td.ts_feature[idx]
        restricted_realized = td.realized[np.ix_(idx, idx)]
        text_map = load_vintage_text_map(wt)
        restricted_texts = tuple(text_map[t][0] for t in keep)

        rp_scores = return_path_scores(restricted_ts)
        tf_scores, vocab_size = fit_tfidf_scores(
            restricted_texts, tfidf_config["kwargs"]
        )
        rp_eval = evaluate_scores(
            rp_scores, restricted_realized, k=K_PRIMARY, oracle_check=False
        )
        tf_eval = evaluate_scores(
            tf_scores, restricted_realized, k=K_PRIMARY, oracle_check=False
        )
        rows.append(
            {
                "transition": label,
                "n_before": td.n,
                "n_flagged_excluded": len(flagged),
                "n_after": len(keep),
                "return_path_before_ndcg": frozen["rp_eval"]["mean_ndcg"],
                "return_path_after_ndcg": rp_eval["mean_ndcg"],
                "return_path_after_minus_before": (
                    rp_eval["mean_ndcg"] - frozen["rp_eval"]["mean_ndcg"]
                ),
                "tfidf_before_ndcg": frozen["tf_eval"]["mean_ndcg"],
                "tfidf_after_ndcg": tf_eval["mean_ndcg"],
                "tfidf_after_minus_before": (
                    tf_eval["mean_ndcg"] - frozen["tf_eval"]["mean_ndcg"]
                ),
                "tier_return_path_above_tfidf_after": bool(
                    rp_eval["mean_ndcg"] > tf_eval["mean_ndcg"]
                ),
                "tfidf_restricted_vocab_size": vocab_size,
                "n_queries_before": frozen["rp_eval"]["n_queries"],
                "n_queries_after": rp_eval["n_queries"],
                "n_valid_return_path_after": rp_eval["n_valid"],
                "n_valid_tfidf_after": tf_eval["n_valid"],
            }
        )
        print(
            f"  {label}: {td.n}->{len(keep)}, "
            f"RP={rp_eval['mean_ndcg']:.4f}, TF={tf_eval['mean_ndcg']:.4f}",
            flush=True,
        )

    return make_bundle(
        "06_common_share_heuristic_exclusion",
        {
            **base_configuration(),
            "heuristic": {
                "regex": NON_PRIMARY_SHARE_RE.pattern,
                "meaning": (
                    "exclude ticker symbols ending in dash plus one or two "
                    "letters; vendor-syntax heuristic for preferred/non-primary "
                    "share classes, not a definitive security-type master"
                ),
            },
            "tfidf_config": {
                "config_id": tfidf_config["config_id"],
                "kwargs": tfidf_config["kwargs"],
            },
            "tfidf_refit": "refit on each restricted feature-window corpus",
        },
        rows,
        {
            "distinct_flagged_tickers_across_modern_feature_time_universes": len(
                all_flagged
            ),
            "flagged_ticker_examples": sorted(all_flagged)[:30],
            "return_path_above_tfidf_after_in_transitions": sum(
                bool(r["tier_return_path_above_tfidf_after"]) for r in rows
            ),
            "transition_count": len(rows),
            "mean_return_path_after_minus_before": mean(
                [float(r["return_path_after_minus_before"]) for r in rows]
            ),
            "mean_tfidf_after_minus_before": mean(
                [float(r["tfidf_after_minus_before"]) for r in rows]
            ),
        },
        time.perf_counter() - started,
        validations,
        provenance,
    )


def run_universe_reconciliation(
    tfidf_config: dict[str, Any],
    modern_cache: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    validations: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    print("\n[7/7] Symmetric-vs-feature-time universe reconciliation", flush=True)
    for wt, wt1 in MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        feature_frozen = modern_cache[label]
        feature_td: TransitionData = feature_frozen["transition_data"]
        symmetric_td = build_transition(wt, wt1, universe_policy="symmetric")
        if not set(symmetric_td.tickers).issubset(feature_td.tickers):
            raise AssertionError(f"{label}: symmetric universe is not a subset")

        symmetric_rp_scores = return_path_scores(symmetric_td.ts_feature)
        symmetric_tf_scores, symmetric_vocab = fit_tfidf_scores(
            symmetric_td.texts, tfidf_config["kwargs"]
        )
        symmetric_rp_eval = evaluate_scores(
            symmetric_rp_scores,
            symmetric_td.realized,
            k=K_PRIMARY,
            oracle_check=False,
        )
        symmetric_tf_eval = evaluate_scores(
            symmetric_tf_scores,
            symmetric_td.realized,
            k=K_PRIMARY,
            oracle_check=False,
        )
        rows.append(
            {
                "transition": label,
                "n_return_intersection": len(
                    set(load_window(wt).tickers) & set(load_window(wt1).tickers)
                ),
                "n_feature_time": feature_td.n,
                "n_symmetric": symmetric_td.n,
                "n_removed_by_target_text_coverage": feature_td.n - symmetric_td.n,
                "return_path_feature_time_ndcg": feature_frozen["rp_eval"][
                    "mean_ndcg"
                ],
                "return_path_symmetric_ndcg": symmetric_rp_eval["mean_ndcg"],
                "return_path_feature_time_minus_symmetric": (
                    feature_frozen["rp_eval"]["mean_ndcg"]
                    - symmetric_rp_eval["mean_ndcg"]
                ),
                "tfidf_feature_time_ndcg": feature_frozen["tf_eval"]["mean_ndcg"],
                "tfidf_symmetric_ndcg": symmetric_tf_eval["mean_ndcg"],
                "tfidf_feature_time_minus_symmetric": (
                    feature_frozen["tf_eval"]["mean_ndcg"]
                    - symmetric_tf_eval["mean_ndcg"]
                ),
                "feature_time_vocab_size": feature_frozen["vocab_size"],
                "symmetric_vocab_size": symmetric_vocab,
                "n_queries_feature_time": feature_frozen["rp_eval"]["n_queries"],
                "n_queries_symmetric": symmetric_rp_eval["n_queries"],
            }
        )
        print(
            f"  {label}: feature N={feature_td.n}, symmetric N={symmetric_td.n}, "
            f"RP {feature_frozen['rp_eval']['mean_ndcg']:.4f}/"
            f"{symmetric_rp_eval['mean_ndcg']:.4f}",
            flush=True,
        )

    return make_bundle(
        "07_symmetric_vs_feature_time_reconciliation",
        {
            **base_configuration(),
            "feature_time_universe": (
                "returns in W_t and W_{t+1}, vintage 10-K text in W_t only"
            ),
            "symmetric_universe": (
                "returns in W_t and W_{t+1}, vintage 10-K coverage in both "
                "W_t and W_{t+1}; W_{t+1} text content is never used"
            ),
            "tfidf_config": {
                "config_id": tfidf_config["config_id"],
                "kwargs": tfidf_config["kwargs"],
            },
            "tfidf_refit": "separately refit on each compared universe",
        },
        rows,
        {
            "transition_count": len(rows),
            "mean_n_feature_time": mean(
                [float(r["n_feature_time"]) for r in rows]
            ),
            "mean_n_symmetric": mean([float(r["n_symmetric"]) for r in rows]),
            "mean_n_added_by_feature_time": mean(
                [float(r["n_removed_by_target_text_coverage"]) for r in rows]
            ),
            "mean_return_path_feature_time_minus_symmetric": mean(
                [
                    float(r["return_path_feature_time_minus_symmetric"])
                    for r in rows
                ]
            ),
            "mean_tfidf_feature_time_minus_symmetric": mean(
                [
                    float(r["tfidf_feature_time_minus_symmetric"])
                    for r in rows
                ]
            ),
        },
        time.perf_counter() - started,
        validations,
        provenance,
    )


def create_run_directory(output_root: Path, requested_run_id: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if requested_run_id is None:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    else:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", requested_run_id):
            raise ValueError("run-id may contain only letters, numbers, _, ., and -")
        run_id = requested_run_id
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=None,
        help="optional immutable run-directory name; defaults to UTC timestamp",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="output root (default: results/acceptance_robustness)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()
    run_dir = create_run_directory(args.output_root, args.run_id)
    print(f"Return-path robustness checks: {SCHEMA_VERSION}")
    print(f"Output directory: {run_dir}")

    validations = validation_oracles()
    print("Initial metric/oracle assertions: PASS", flush=True)
    provenance_start = time.perf_counter()
    provenance = collect_input_provenance()
    provenance_seconds = time.perf_counter() - provenance_start
    print(
        f"SHA-256 provenance collected in {provenance_seconds:.1f}s",
        flush=True,
    )

    selected_tfidf, bundle_1 = run_tfidf_historical_tuning(
        provenance, validations
    )
    frozen_alpha, modern_cache, bundle_2 = run_blend_tuning_and_modern(
        selected_tfidf, provenance, validations
    )
    bundle_3 = run_k_sensitivity(
        selected_tfidf, modern_cache, provenance, validations
    )
    bundle_4 = run_ewma_baselines(modern_cache, provenance, validations)
    bundle_5 = run_two_window_concat(modern_cache, provenance, validations)
    bundle_6 = run_common_share_exclusion(
        selected_tfidf, modern_cache, provenance, validations
    )
    bundle_7 = run_universe_reconciliation(
        selected_tfidf, modern_cache, provenance, validations
    )

    bundles = {
        "01_tfidf_historical_tuning.json": bundle_1,
        "02_blend_historical_tuning_modern_frozen.json": bundle_2,
        "03_modern_k_sensitivity.json": bundle_3,
        "04_fixed_ewma_return_path.json": bundle_4,
        "05_two_window_concatenated_return_path.json": bundle_5,
        "06_common_share_heuristic_exclusion.json": bundle_6,
        "07_symmetric_vs_feature_time_reconciliation.json": bundle_7,
    }
    for filename, bundle in bundles.items():
        write_json_new(run_dir / filename, bundle)

    artifact_hashes = {
        filename: sha256_file(run_dir / filename) for filename in bundles
    }
    total_seconds = time.perf_counter() - overall_start
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "run_directory": relative(run_dir),
        "configuration": {
            **base_configuration(),
            "selected_tfidf_config_id": selected_tfidf["config_id"],
            "selected_tfidf_kwargs": selected_tfidf["kwargs"],
            "selected_alpha_return_path": frozen_alpha,
            "selected_alpha_tfidf": 1.0 - frozen_alpha,
        },
        "experiment_summaries": {
            filename: bundle["summary"] for filename, bundle in bundles.items()
        },
        "runtime": {
            "total_seconds": total_seconds,
            "provenance_seconds": provenance_seconds,
            "per_experiment_seconds": {
                filename: bundle["runtime_seconds"]
                for filename, bundle in bundles.items()
            },
        },
        "validations": validations,
        "artifact_sha256": artifact_hashes,
        "provenance": provenance,
        "design_limitations": [
            (
                "Historical return artifacts use the repository's fixed "
                "current-survivor cohort rather than a point-in-time listing master."
            ),
            (
                "The share-class screen is a vendor ticker-syntax heuristic, "
                "not authoritative security-type metadata."
            ),
            (
                "EWMA uses the stored, already-standardized residual path and "
                "does not refit FF5 or perform weighted re-centering."
            ),
            (
                "Two-window concatenation conditions on availability in an "
                "additional previous window, so its identical comparison subset "
                "is smaller than the one-window headline universe."
            ),
            (
                "The symmetric-universe reconciliation deliberately inspects "
                "target-window text coverage only to reconstruct the legacy "
                "universe; no target-window text content enters any feature."
            ),
        ],
    }
    manifest_path = run_dir / "artifact_manifest.json"
    write_json_new(manifest_path, manifest)

    all_hashes = {
        **artifact_hashes,
        manifest_path.name: sha256_file(manifest_path),
    }
    sums_path = run_dir / "SHA256SUMS.txt"
    if sums_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {sums_path}")
    sums_path.write_text(
        "".join(f"{digest}  {filename}\n"
                for filename, digest in sorted(all_hashes.items()))
    )

    print("\nAll experiments completed and assertions passed.")
    print(
        f"Selected TF-IDF={selected_tfidf['config_id']}; "
        f"selected ReturnPath alpha={frozen_alpha:.1f}"
    )
    print(f"Total runtime: {total_seconds:.1f}s")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
