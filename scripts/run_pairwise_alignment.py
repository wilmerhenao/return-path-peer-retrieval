#!/usr/bin/env python3
"""Compare W1 text and return-path similarity with W1 and W2 residual correlation.

Use the W1->W2 comparison universe: returns in both windows and nonempty
Item 1 text filed strictly before W1 starts. Inputs are the W1 standardized
paths, W1/W2 Ledoit--Wolf matrices, frozen dense-encoder caches, and the
historically selected TF-IDF settings from the frozen robustness result.

Compute Spearman's rho over every unordered pair, excluding self-pairs.
Pairs are not subsampled. They share securities and dates, so no IID p-value
or confidence interval is reported. These coefficients describe one fixed
sample and measure all-pair alignment, rather than top-k retrieval quality.

Each run writes result JSON, a manifest, and SHA-256 checksums to a new
directory under ``results/pairwise_alignment``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_acceptance_robustness as core  # noqa: E402


SCHEMA_VERSION = "pairwise-alignment-v3"
OUTPUT_ROOT = ROOT / "results" / "pairwise_alignment"
DEFAULT_SELECTION = (
    ROOT
    / "results"
    / "acceptance_robustness"
    / "run_20260730T032543Z"
    / "01_tfidf_historical_tuning.json"
)
DENSE_CACHE_DIR = (
    ROOT / "results" / "vintage_all_encoders_v3" / "embedding_cache"
)
DENSE_CACHE_FILES = {
    "BGE": "W1_bge_c0bc3c577ef1ba41.npy",
    "Granite-r2 (512 tok.)": "W1_granite_r2_f3bc40b4e05dc9dd.npy",
    "MiniLM": "W1_minilm_33d5a9064bb324bb.npy",
    "SEC-BERT": "W1_sec_bert_a12ac7836dd58089.npy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_tfidf_config(path: Path) -> dict[str, Any]:
    """Read the selected setting instead of duplicating its parameters."""
    authority = json.loads(path.read_text(encoding="utf-8"))
    summary = authority["summary"]
    if summary["selected_config_id"] != "b":
        raise AssertionError("historical selection authority no longer selects b")
    if summary["modern_data_used_for_selection"] is not False:
        raise AssertionError("historical selection authority used modern data")
    selected = next(
        row
        for row in authority["configuration"]["grid"]
        if row["config_id"] == summary["selected_config_id"]
    )
    kwargs = dict(selected["specified_kwargs"])
    if isinstance(kwargs.get("ngram_range"), list):
        kwargs["ngram_range"] = tuple(kwargs["ngram_range"])
    return {"config_id": selected["config_id"], "kwargs": kwargs}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full W1->W2 pairwise Spearman alignment census."
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
        help="immutable output label beginning with run_",
    )
    parser.add_argument(
        "--tfidf-selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="historical TF-IDF selection authority",
    )
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not all(
        character.isalnum() or character in "._-" for character in args.run_id
    ):
        parser.error("--run-id must be a safe label beginning with run_")
    run_dir = OUTPUT_ROOT / args.run_id
    if run_dir.exists():
        parser.error(f"refusing to overwrite {run_dir}")

    started = time.perf_counter()
    selection_path = args.tfidf_selection.resolve()
    dense_paths = {
        label: DENSE_CACHE_DIR / filename
        for label, filename in DENSE_CACHE_FILES.items()
    }
    missing_dense = [path for path in dense_paths.values() if not path.is_file()]
    if missing_dense:
        raise FileNotFoundError(
            "missing frozen dense-embedding caches: "
            + ", ".join(str(path) for path in missing_dense)
        )
    config = load_tfidf_config(selection_path)
    transition = core.build_transition(
        "W1", "W2", universe_policy="feature_time"
    )
    tfidf_scores, vocabulary_size = core.fit_tfidf_scores(
        transition.texts, config["kwargs"]
    )
    return_scores = core.return_path_scores(transition.ts_feature)
    upper = np.triu_indices(transition.n, k=1)
    realized_pairs = np.asarray(transition.realized[upper], dtype=np.float64)
    tfidf_pairs = np.asarray(tfidf_scores[upper], dtype=np.float64)
    return_pairs = np.asarray(return_scores[upper], dtype=np.float64)
    expected_pairs = transition.n * (transition.n - 1) // 2
    if realized_pairs.size != expected_pairs:
        raise AssertionError("upper-triangle pair count is inconsistent")

    tfidf_rho = float(spearmanr(tfidf_pairs, realized_pairs).statistic)
    return_rho = float(spearmanr(return_pairs, realized_pairs).statistic)
    if not np.isfinite([tfidf_rho, return_rho]).all():
        raise AssertionError("non-finite Spearman coefficient")

    # Dense caches follow the full W1 return universe. Reindex them to the
    # smaller return/text universe used by this transition.
    feature = core.load_window("W1")
    dense_indices = np.fromiter(
        (feature.ticker_to_idx[ticker] for ticker in transition.tickers),
        dtype=np.int64,
    )
    same_window_target = np.asarray(
        feature.corr_target[np.ix_(dense_indices, dense_indices)],
        dtype=np.float32,
    )
    same_window_pairs = np.asarray(
        same_window_target[upper], dtype=np.float64
    )
    if same_window_pairs.size != expected_pairs:
        raise AssertionError("same-window upper-triangle count is inconsistent")
    tfidf_same_window_rho = float(
        spearmanr(tfidf_pairs, same_window_pairs).statistic
    )
    return_same_window_rho = float(
        spearmanr(return_pairs, same_window_pairs).statistic
    )
    if not np.isfinite([tfidf_same_window_rho, return_same_window_rho]).all():
        raise AssertionError("non-finite same-window Spearman coefficient")

    dense_results: dict[str, dict[str, float | int | str]] = {}
    for label, path in dense_paths.items():
        cached = np.load(path, mmap_mode="r")
        if cached.ndim != 2 or cached.shape[0] != len(feature.tickers):
            raise AssertionError(f"{label}: dense cache/universe mismatch")
        embedding = core.l2_rows(np.asarray(cached[dense_indices]))
        dense_pairs = np.asarray(
            (embedding @ embedding.T)[upper], dtype=np.float64
        )
        same_window_rho = float(
            spearmanr(dense_pairs, same_window_pairs).statistic
        )
        next_window_rho = float(
            spearmanr(dense_pairs, realized_pairs).statistic
        )
        if not np.isfinite([same_window_rho, next_window_rho]).all():
            raise AssertionError(f"{label}: non-finite Spearman coefficient")
        dense_results[label] = {
            "spearman_rho": next_window_rho,
            "same_window_spearman_rho": same_window_rho,
            "embedding_dimension": int(cached.shape[1]),
            "cache_file": logical_path(path),
        }

    input_paths = [
        selection_path,
        ROOT / "scripts" / "metrics.py",
        ROOT / "scripts" / "run_acceptance_robustness.py",
        Path(__file__).resolve(),
        core.window_directory("W1") / "corr_target.npy",
        core.window_directory("W1") / "ts_features.npy",
        core.window_directory("W1") / "universe.parquet",
        core.window_directory("W1") / "meta.json",
        core.window_directory("W2") / "corr_target.npy",
        core.window_directory("W2") / "universe.parquet",
        core.window_directory("W2") / "meta.json",
        core.VINTAGE_DIR / "W1_descriptions.parquet",
        *dense_paths.values(),
    ]
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing pairwise-alignment inputs: "
            + ", ".join(str(path) for path in missing)
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": (
            "Spearman rank correlation across all unordered W1 feature-time "
            "security pairs between each feature similarity and W1 or W2 "
            "Ledoit-Wolf FF5 residual correlation"
        ),
        "transition": transition.label,
        "n_securities": transition.n,
        "n_unordered_pairs": expected_pairs,
        "tfidf": {
            "spearman_rho": tfidf_rho,
            "same_window_spearman_rho": tfidf_same_window_rho,
            "vocabulary_size": vocabulary_size,
            "configuration": config,
        },
        "dense_text_encoders": dense_results,
        "return_path": {
            "spearman_rho": return_rho,
            "same_window_spearman_rho": return_same_window_rho,
        },
        "protocol": {
            "universe": (
                "return availability in W1 and W2 plus non-empty vintage "
                "Item 1 text filed strictly before W1"
            ),
            "pair_policy": "complete upper triangle; diagonal excluded",
            "pair_subsampling_for_statistic": False,
            "targets": (
                "W1 same-window and W2 next-window Ledoit-Wolf FF5 residual "
                "correlation"
            ),
            "inference": (
                "descriptive finite-panel coefficients only; no IID p-values "
                "or confidence intervals"
            ),
        },
        "validations": {
            "feature_time_universe": True,
            "target_window_text_not_loaded": True,
            "complete_pair_census": True,
            "pair_count_identity": expected_pairs == realized_pairs.size,
            "same_window_pair_count_identity": (
                expected_pairs == same_window_pairs.size
            ),
            "dense_cache_indexed_to_full_feature_universe": True,
            "modern_target_used_for_configuration_selection": False,
        },
        "input_sha256": {
            logical_path(path): sha256_file(path)
            for path in sorted(input_paths, key=logical_path)
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
    result_path = run_dir / "pairwise_alignment.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "artifact_sha256": {result_path.name: sha256_file(result_path)},
        "source_sha256": {
            "scripts/run_pairwise_alignment.py": sha256_file(
                Path(__file__).resolve()
            ),
            "scripts/run_acceptance_robustness.py": sha256_file(
                ROOT / "scripts" / "run_acceptance_robustness.py"
            ),
            "scripts/metrics.py": sha256_file(ROOT / "scripts" / "metrics.py"),
        },
        "notes": [
            "The statistic uses all unordered pairs; plotting subsamples are irrelevant.",
            "Pair dependence rules out IID pair-level inference.",
        ],
    }
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (result_path, manifest_path)
        ),
        encoding="utf-8",
    )

    print(f"run_dir={logical_path(run_dir)}")
    print(
        f"N={transition.n}; pairs={expected_pairs}; "
        f"rho_tfidf={tfidf_rho:.9f}; rho_return_path={return_rho:.9f}"
    )
    print(
        "dense_rho="
        + ", ".join(
            f"{label}:same={result['same_window_spearman_rho']:.9f},"
            f"next={result['spearman_rho']:.9f}"
            for label, result in dense_results.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
