#!/usr/bin/env python3
"""Reproduce the contemporaneous-text diagnostic reported in Section 5.

Research question
-----------------
Does filing text fail because it cannot forecast next-window residual
co-movement, or because the frozen representations do not closely describe
even contemporaneous co-movement?  The diagnostic holds the primary
feature-time security universe and retrieval metric fixed, but evaluates W_t
text similarity against the W_t residual-correlation matrix.

Inputs
------
For each W1--W6 feature window, the script reads the frozen return-window
universe and Ledoit--Wolf correlation matrix, the vintage Item 1 text filed
strictly before the window starts, and the exact short-context embedding cache
used by the walk-forward experiment.  Eligibility additionally requires
return availability in W_{t+1}, exactly as in the primary feature-time census.
The historically selected TF-IDF configuration is loaded from the frozen
robustness result rather than restated independently.

Protocol
--------
Every eligible security is a query.  Rankings use cosine similarity and are
scored by the canonical positive-correlation NDCG@5 implementation.  The
same-window target is never used to fit, tune, or alter a representation.  The
return-path identity check must attain NDCG@5=1 for every defined query.

Outputs
-------
The script creates a new, immutable ``results/same_window_text_diagnostic``
run directory containing the result JSON, an artifact manifest, and SHA-256
checksums.  It refuses to overwrite an existing run.

Interpretation limits
---------------------
This is a descriptive diagnostic, not a deployable forecast or an independent
pair-level inference exercise.  Its universe is held equal to the walk-forward
benchmark so that contemporaneous and next-window means differ only in the
relevance matrix, not in coverage or query sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import scripts.run_acceptance_robustness as core  # noqa: E402
from scripts.run_adversarial_followup import (  # noqa: E402
    CACHE_DIR,
    DENSE_CACHE_FILES,
)


SCHEMA_VERSION = "same-window-text-diagnostic-v1"
OUTPUT_ROOT = ROOT / "results" / "same_window_text_diagnostic"
DEFAULT_WF_ROBUSTNESS = (
    ROOT
    / "results"
    / "acceptance_robustness"
    / "run_20260730T032543Z"
    / "02_blend_historical_tuning_modern_frozen.json"
)
DEFAULT_WF_DENSE = (
    ROOT
    / "results"
    / "acceptance_followup"
    / "run_20260730T065913Z"
    / "adversarial_followup_census.json"
)
METHODS = ("TF-IDF", "Granite-r2 (512 tok.)", "MiniLM")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    """Represent project files without recording a workstation path."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_tfidf_config(authority: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the historical-only TF-IDF selection authority."""
    summary = authority["summary"]
    config = summary["tfidf_config_frozen_from_historical"]
    if config["config_id"] != "b":
        raise AssertionError("frozen TF-IDF authority does not select config b")
    if summary["modern_data_used_for_selection"] is not False:
        raise AssertionError("TF-IDF authority reports modern-data selection")
    kwargs = dict(config["kwargs"])
    if isinstance(kwargs.get("ngram_range"), list):
        kwargs["ngram_range"] = tuple(kwargs["ngram_range"])
    return {"config_id": config["config_id"], "kwargs": kwargs}


def same_window_target(
    feature: core.WindowData,
    tickers: tuple[str, ...],
) -> np.ndarray:
    """Subset C_t to the already frozen feature-time transition universe."""
    feature_idx = np.fromiter(
        (feature.ticker_to_idx[ticker] for ticker in tickers),
        dtype=np.int64,
    )
    return np.asarray(
        feature.corr_target[np.ix_(feature_idx, feature_idx)],
        dtype=np.float32,
    )


def dense_scores(
    window: str,
    method: str,
    feature: core.WindowData,
    tickers: tuple[str, ...],
) -> np.ndarray:
    """Subset the frozen W_t embedding cache and return cosine scores."""
    cache_path = CACHE_DIR / DENSE_CACHE_FILES[window][method]
    embeddings = np.load(cache_path, mmap_mode="r")
    if embeddings.shape[0] != len(feature.tickers):
        raise AssertionError(f"{cache_path.name}: cache/universe mismatch")
    indices = np.fromiter(
        (feature.ticker_to_idx[ticker] for ticker in tickers),
        dtype=np.int64,
    )
    return core.return_path_scores(
        np.asarray(embeddings[indices], dtype=np.float32)
    )


def walk_forward_means(
    robustness: dict[str, Any],
    dense: dict[str, Any],
) -> dict[str, float]:
    """Extract comparator means from the active frozen authorities."""
    return {
        "TF-IDF": float(robustness["summary"]["modern_mean_tfidf_ndcg"]),
        "Granite-r2 (512 tok.)": float(
            dense["dense_feature_time_census"]["summary"]["aggregates"]
            ["Granite-r2 (512 tok.)"]["macro_mean_ndcg"]
        ),
        "MiniLM": float(
            dense["dense_feature_time_census"]["summary"]["aggregates"]
            ["MiniLM"]["macro_mean_ndcg"]
        ),
    }


def input_paths(
    wf_robustness_path: Path,
    wf_dense_path: Path,
) -> list[Path]:
    """List every scientific input whose identity anchors this rerun."""
    paths: set[Path] = {
        wf_robustness_path,
        wf_dense_path,
        ROOT / "scripts" / "metrics.py",
        ROOT / "scripts" / "run_acceptance_robustness.py",
        Path(__file__).resolve(),
    }
    for wt, wt1 in core.MODERN_TRANSITIONS:
        for window_name in (wt, wt1):
            directory = core.window_directory(window_name)
            paths.add(directory / "universe.parquet")
            paths.add(directory / "meta.json")
        feature_directory = core.window_directory(wt)
        paths.add(feature_directory / "corr_target.npy")
        paths.add(feature_directory / "ts_features.npy")
        paths.add(core.VINTAGE_DIR / f"{wt}_descriptions.parquet")
        for method in ("Granite-r2 (512 tok.)", "MiniLM"):
            paths.add(CACHE_DIR / DENSE_CACHE_FILES[wt][method])
    missing = [path for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing same-window inputs: "
            + ", ".join(str(path) for path in missing)
        )
    return sorted(paths, key=logical_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the matched contemporaneous-text NDCG@5 diagnostic."
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
        help="immutable output label beginning with run_",
    )
    parser.add_argument(
        "--wf-robustness",
        type=Path,
        default=DEFAULT_WF_ROBUSTNESS,
        help="frozen historical-selection and walk-forward TF-IDF result",
    )
    parser.add_argument(
        "--wf-dense",
        type=Path,
        default=DEFAULT_WF_DENSE,
        help="frozen feature-time dense-encoder result",
    )
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not all(
        character.isalnum() or character in "._-" for character in args.run_id
    ):
        parser.error("--run-id must be a safe label beginning with run_")

    wf_robustness_path = args.wf_robustness.resolve()
    wf_dense_path = args.wf_dense.resolve()
    run_dir = OUTPUT_ROOT / args.run_id
    if run_dir.exists():
        parser.error(f"refusing to overwrite {run_dir}")

    started = time.perf_counter()
    wf_robustness = load_json(wf_robustness_path)
    wf_dense = load_json(wf_dense_path)
    tfidf = selected_tfidf_config(wf_robustness)
    wf_means = walk_forward_means(wf_robustness, wf_dense)

    rows: list[dict[str, Any]] = []
    oracle_means: list[float] = []
    for wt, wt1 in core.MODERN_TRANSITIONS:
        transition = core.build_transition(
            wt, wt1, universe_policy="feature_time"
        )
        feature = core.load_window(wt)
        realized_same_window = same_window_target(feature, transition.tickers)

        # The identity check is a protocol invariant: C_t is an affine
        # off-diagonal transformation of the standardized-path correlation.
        return_evaluation = core.evaluate_scores(
            core.return_path_scores(transition.ts_feature),
            realized_same_window,
            k=core.K_PRIMARY,
            oracle_check=True,
        )
        oracle_means.append(float(return_evaluation["mean_ndcg"]))

        tfidf_matrix, vocabulary_size = core.fit_tfidf_scores(
            transition.texts, tfidf["kwargs"]
        )
        scores_by_method = {
            "TF-IDF": tfidf_matrix,
            "Granite-r2 (512 tok.)": dense_scores(
                wt,
                "Granite-r2 (512 tok.)",
                feature,
                transition.tickers,
            ),
            "MiniLM": dense_scores(
                wt, "MiniLM", feature, transition.tickers
            ),
        }
        for method, scores in scores_by_method.items():
            evaluation = core.evaluate_scores(
                scores,
                realized_same_window,
                k=core.K_PRIMARY,
            )
            rows.append(
                {
                    "transition": transition.label,
                    "feature_window": wt,
                    "target_window": wt,
                    "method": method,
                    "n_universe": transition.n,
                    "n_queries": evaluation["n_queries"],
                    "n_valid": evaluation["n_valid"],
                    "mean_ndcg": evaluation["mean_ndcg"],
                    "population_std_across_queries": evaluation[
                        "population_std_ndcg"
                    ],
                    "tfidf_vocabulary_size": (
                        vocabulary_size if method == "TF-IDF" else None
                    ),
                }
            )
        print(f"{transition.label}: N={transition.n}", flush=True)

    summary: dict[str, Any] = {}
    for method in METHODS:
        values = [
            float(row["mean_ndcg"])
            for row in rows
            if row["method"] == method
        ]
        same_window_mean = statistics.fmean(values)
        summary[method] = {
            "same_window_macro_mean_ndcg": same_window_mean,
            "same_window_population_std_across_transitions": statistics.pstdev(
                values
            ),
            "walk_forward_macro_mean_ndcg": wf_means[method],
            "same_window_minus_walk_forward": (
                same_window_mean - wf_means[method]
            ),
            "n_transitions": len(values),
        }

    if not np.allclose(oracle_means, 1.0, atol=1e-12):
        raise AssertionError(
            f"same-window return-path identity failed: {oracle_means}"
        )
    paths = input_paths(wf_robustness_path, wf_dense_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": (
            "How well do frozen W_t text representations retrieve W_t "
            "residual-correlation peers on the primary feature-time universe?"
        ),
        "configuration": {
            "metric": "canonical positive-correlation NDCG@5",
            "query_policy": "every eligible security",
            "universe": (
                "return availability in W_t and W_{t+1}, plus non-empty "
                "vintage Item 1 text filed strictly before W_t"
            ),
            "relevance": "W_t Ledoit-Wolf FF5 residual correlation",
            "tfidf": tfidf,
            "dense_caches": "exact frozen W_t caches used in walk-forward evaluation",
            "configuration_selection_uses_modern_targets": False,
            "inference": "descriptive finite-panel census; no IID inference",
        },
        "per_transition": rows,
        "summary": summary,
        "validations": {
            "same_security_universe_as_walk_forward": True,
            "all_eligible_securities_are_queries": all(
                row["n_queries"] == row["n_universe"] for row in rows
            ),
            "target_window_text_not_loaded": True,
            "return_path_same_window_identity": True,
            "return_path_identity_means": oracle_means,
        },
        "walk_forward_authorities": {
            "tfidf": logical_path(wf_robustness_path),
            "dense": logical_path(wf_dense_path),
        },
        "input_sha256": {
            logical_path(path): sha256_file(path) for path in paths
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
    result_path = run_dir / "same_window_text_diagnostic.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "artifact_sha256": {result_path.name: sha256_file(result_path)},
        "source_sha256": {
            "scripts/run_same_window_text_diagnostic.py": sha256_file(
                Path(__file__).resolve()
            ),
            "scripts/run_acceptance_robustness.py": sha256_file(
                ROOT / "scripts" / "run_acceptance_robustness.py"
            ),
            "scripts/metrics.py": sha256_file(ROOT / "scripts" / "metrics.py"),
        },
        "notes": [
            "The relevance matrix changes from W_{t+1} to W_t; the feature-time universe does not.",
            "No contemporaneous or future target selected a representation or hyperparameter.",
            "The diagnostic is descriptive and does not support IID pair-level inference.",
        ],
    }
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = (result_path, manifest_path)
    (run_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in checksums
        ),
        encoding="utf-8",
    )

    print(f"run_dir={logical_path(run_dir)}")
    for method in METHODS:
        row = summary[method]
        print(
            f"{method}: SW={row['same_window_macro_mean_ndcg']:.6f}, "
            f"WF={row['walk_forward_macro_mean_ndcg']:.6f}, "
            f"delta={row['same_window_minus_walk_forward']:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
