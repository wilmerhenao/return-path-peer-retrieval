#!/usr/bin/env python3
"""Run four follow-up checks on frozen local inputs:

* a stale-return control using W_{t-1} paths against W_{t+1} relevance;
* a one-to-four-window history comparison on the same modern subset;
* a two-window return-only extension over every feasible transition; and
* feature-time rescoring of the cached short-context dense encoders.

Each run writes to a new directory and records input and output hashes.
Modern targets are not used to choose settings or history length.
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


SCHEMA_VERSION = "adversarial-followup-v1"
OUTPUT_ROOT = ROOT / "results" / "acceptance_followup"
SEQUENCE = tuple(f"H{i}" for i in range(1, 14)) + tuple(
    f"W{i}" for i in range(1, 8)
)
MEMORY_LENGTHS = (1, 2, 3, 4)

# Frozen caches from scripts/exp_vintage_all_encoders.py; each array follows
# the row order in its corresponding results/windows_v3/W*/universe file.
DENSE_CACHE_FILES: dict[str, dict[str, str]] = {
    "W1": {
        "Granite-r2 (512 tok.)": "W1_granite_r2_f3bc40b4e05dc9dd.npy",
        "SEC-BERT": "W1_sec_bert_a12ac7836dd58089.npy",
        "BGE": "W1_bge_c0bc3c577ef1ba41.npy",
        "MiniLM": "W1_minilm_33d5a9064bb324bb.npy",
    },
    "W2": {
        "Granite-r2 (512 tok.)": "W2_granite_r2_13684f8bfc9d4c9a.npy",
        "SEC-BERT": "W2_sec_bert_11ba0e4bd5725862.npy",
        "BGE": "W2_bge_2d82c56bd2a1d408.npy",
        "MiniLM": "W2_minilm_6c68160f396b3781.npy",
    },
    "W3": {
        "Granite-r2 (512 tok.)": "W3_granite_r2_46543edfba303fe1.npy",
        "SEC-BERT": "W3_sec_bert_4adc10a019d20bb7.npy",
        "BGE": "W3_bge_501abdc95ff2e6cd.npy",
        "MiniLM": "W3_minilm_5fec9e5b792667c2.npy",
    },
    "W4": {
        "Granite-r2 (512 tok.)": "W4_granite_r2_c3c8cb65c2ce4fb7.npy",
        "SEC-BERT": "W4_sec_bert_60339555ec805452.npy",
        "BGE": "W4_bge_905d87fb5b2ddce9.npy",
        "MiniLM": "W4_minilm_e12b374c9db817ca.npy",
    },
    "W5": {
        "Granite-r2 (512 tok.)": "W5_granite_r2_e81ecc7164c7d83f.npy",
        "SEC-BERT": "W5_sec_bert_cd65a775fc582889.npy",
        "BGE": "W5_bge_fa897237006be5eb.npy",
        "MiniLM": "W5_minilm_c39868aee0b37e87.npy",
    },
    "W6": {
        "Granite-r2 (512 tok.)": "W6_granite_r2_9ad31c70715b35cf.npy",
        "SEC-BERT": "W6_sec_bert_036429c86df834ca.npy",
        "BGE": "W6_bge_0bae34ef3e69154a.npy",
        "MiniLM": "W6_minilm_30ebe185aca39ce6.npy",
    },
}
CACHE_DIR = (
    ROOT / "results" / "vintage_all_encoders_v3" / "embedding_cache"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "macro_mean_ndcg": statistics.fmean(values),
        "population_std_across_transition_means": statistics.pstdev(values),
    }


def selected_tfidf_config() -> dict[str, Any]:
    matches = [
        config
        for config in core.TFIDF_CONFIGS
        if config["config_id"] == "b"
    ]
    if len(matches) != 1:
        raise AssertionError("historically selected TF-IDF config b is absent")
    return matches[0]


def subset_realized(
    transition: core.TransitionData,
    tickers: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    base_idx = {ticker: i for i, ticker in enumerate(transition.tickers)}
    indices = np.fromiter(
        (base_idx[ticker] for ticker in tickers), dtype=np.int64
    )
    realized = np.asarray(
        transition.realized[np.ix_(indices, indices)], dtype=np.float32
    )
    return indices, realized


def stale_return_control(
    modern: dict[str, core.TransitionData],
    tfidf_config: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for wt, wt1 in core.MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        td = modern[label]
        previous_name = core.PREVIOUS_WINDOW[wt]
        previous = core.load_window(previous_name)
        tickers = tuple(sorted(set(td.tickers) & set(previous.tickers)))
        base_idx, realized = subset_realized(td, tickers)
        previous_idx = np.fromiter(
            (previous.ticker_to_idx[ticker] for ticker in tickers),
            dtype=np.int64,
        )

        stale_path = np.asarray(
            previous.ts_features[previous_idx], dtype=np.float32
        )
        current_path = np.asarray(td.ts_feature[base_idx], dtype=np.float32)
        texts = tuple(td.texts[int(index)] for index in base_idx)
        tfidf_scores, vocab_size = core.fit_tfidf_scores(
            texts, tfidf_config["kwargs"]
        )

        stale_eval = core.evaluate_scores(
            core.return_path_scores(stale_path),
            realized,
            k=core.K_PRIMARY,
        )
        current_eval = core.evaluate_scores(
            core.return_path_scores(current_path),
            realized,
            k=core.K_PRIMARY,
        )
        tfidf_eval = core.evaluate_scores(
            tfidf_scores,
            realized,
            k=core.K_PRIMARY,
        )
        rows.append(
            {
                "transition": label,
                "stale_feature_window": previous_name,
                "current_feature_window": wt,
                "target_window": wt1,
                "n_universe": len(tickers),
                "stale_return_path_ndcg": stale_eval["mean_ndcg"],
                "current_return_path_ndcg": current_eval["mean_ndcg"],
                "tfidf_ndcg": tfidf_eval["mean_ndcg"],
                "stale_minus_tfidf": (
                    stale_eval["mean_ndcg"] - tfidf_eval["mean_ndcg"]
                ),
                "current_minus_stale": (
                    current_eval["mean_ndcg"] - stale_eval["mean_ndcg"]
                ),
                "tfidf_vocab_size": vocab_size,
                "n_queries": stale_eval["n_queries"],
            }
        )

    summary: dict[str, Any] = {
        "n_transitions": len(rows),
        "n_universe_min": min(row["n_universe"] for row in rows),
        "n_universe_max": max(row["n_universe"] for row in rows),
        "stale_return_path": mean_sd(
            [float(row["stale_return_path_ndcg"]) for row in rows]
        ),
        "current_return_path": mean_sd(
            [float(row["current_return_path_ndcg"]) for row in rows]
        ),
        "tfidf": mean_sd([float(row["tfidf_ndcg"]) for row in rows]),
        "stale_return_path_wins_over_tfidf": sum(
            row["stale_return_path_ndcg"] > row["tfidf_ndcg"]
            for row in rows
        ),
        "current_return_path_wins_over_stale": sum(
            row["current_return_path_ndcg"]
            > row["stale_return_path_ndcg"]
            for row in rows
        ),
    }
    if summary["stale_return_path_wins_over_tfidf"] != 6:
        raise AssertionError("stale return path does not beat TF-IDF in 6/6")
    return {"per_transition": rows, "summary": summary}


def modern_memory_curve(
    modern: dict[str, core.TransitionData],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for wt, wt1 in core.MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        td = modern[label]
        position = SEQUENCE.index(wt)
        history_names = SEQUENCE[position - 3 : position + 1]
        if len(history_names) != 4:
            raise AssertionError(f"{label}: four history windows unavailable")
        histories = [core.load_window(name) for name in history_names]
        eligible = set(td.tickers)
        for window in histories:
            eligible &= set(window.tickers)
        tickers = tuple(sorted(eligible))
        _, realized = subset_realized(td, tickers)

        ndcg_by_memory: dict[str, float] = {}
        dimensions_by_memory: dict[str, int] = {}
        for memory in MEMORY_LENGTHS:
            parts = []
            for window in histories[-memory:]:
                index = np.fromiter(
                    (window.ticker_to_idx[ticker] for ticker in tickers),
                    dtype=np.int64,
                )
                parts.append(
                    np.asarray(window.ts_features[index], dtype=np.float32)
                )
            concatenated = np.concatenate(parts, axis=1)
            evaluation = core.evaluate_scores(
                core.return_path_scores(concatenated),
                realized,
                k=core.K_PRIMARY,
            )
            ndcg_by_memory[str(memory)] = evaluation["mean_ndcg"]
            dimensions_by_memory[str(memory)] = concatenated.shape[1]

        rows.append(
            {
                "transition": label,
                "history_windows_oldest_to_current": list(history_names),
                "target_window": wt1,
                "n_universe": len(tickers),
                "ndcg_by_memory_windows": ndcg_by_memory,
                "dimensions_by_memory_windows": dimensions_by_memory,
                "n_queries": len(tickers),
            }
        )

    aggregates = []
    for memory in MEMORY_LENGTHS:
        values = [
            float(row["ndcg_by_memory_windows"][str(memory)])
            for row in rows
        ]
        aggregate = {
            "memory_windows": memory,
            **mean_sd(values),
        }
        if memory > 1:
            aggregate["wins_over_previous_memory"] = sum(
                row["ndcg_by_memory_windows"][str(memory)]
                > row["ndcg_by_memory_windows"][str(memory - 1)]
                for row in rows
            )
            if aggregate["wins_over_previous_memory"] != 6:
                raise AssertionError(
                    f"{memory}-window memory does not improve all transitions"
                )
        aggregates.append(aggregate)

    return {
        "per_transition": rows,
        "summary": {
            "n_transitions": len(rows),
            "n_universe_min": min(row["n_universe"] for row in rows),
            "n_universe_max": max(row["n_universe"] for row in rows),
            "identical_subset_across_memory_lengths_within_transition": True,
            "aggregates": aggregates,
        },
    }


def two_window_full_extension() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    # H1->H2 has no previous return window, leaving 18 feasible transitions.
    for position in range(1, len(SEQUENCE) - 1):
        previous_name = SEQUENCE[position - 1]
        wt = SEQUENCE[position]
        wt1 = SEQUENCE[position + 1]
        previous = core.load_window(previous_name)
        feature = core.load_window(wt)
        target = core.load_window(wt1)
        tickers = tuple(
            sorted(
                set(previous.tickers)
                & set(feature.tickers)
                & set(target.tickers)
            )
        )
        previous_idx = np.fromiter(
            (previous.ticker_to_idx[ticker] for ticker in tickers),
            dtype=np.int64,
        )
        feature_idx = np.fromiter(
            (feature.ticker_to_idx[ticker] for ticker in tickers),
            dtype=np.int64,
        )
        target_idx = np.fromiter(
            (target.ticker_to_idx[ticker] for ticker in tickers),
            dtype=np.int64,
        )
        previous_path = np.asarray(
            previous.ts_features[previous_idx], dtype=np.float32
        )
        current_path = np.asarray(
            feature.ts_features[feature_idx], dtype=np.float32
        )
        realized = np.asarray(
            target.corr_target[np.ix_(target_idx, target_idx)],
            dtype=np.float32,
        )
        one_eval = core.evaluate_scores(
            core.return_path_scores(current_path),
            realized,
            k=core.K_PRIMARY,
        )
        two_eval = core.evaluate_scores(
            core.return_path_scores(
                np.concatenate((previous_path, current_path), axis=1)
            ),
            realized,
            k=core.K_PRIMARY,
        )
        rows.append(
            {
                "transition": f"{wt}->{wt1}",
                "previous_feature_window": previous_name,
                "n_universe": len(tickers),
                "one_window_ndcg": one_eval["mean_ndcg"],
                "two_window_ndcg": two_eval["mean_ndcg"],
                "two_minus_one": (
                    two_eval["mean_ndcg"] - one_eval["mean_ndcg"]
                ),
                "n_queries": len(tickers),
            }
        )

    summary = {
        "n_feasible_transitions": len(rows),
        "omitted_transition": "H1->H2 (no preceding return window)",
        "n_universe_min": min(row["n_universe"] for row in rows),
        "n_universe_max": max(row["n_universe"] for row in rows),
        "one_window": mean_sd(
            [float(row["one_window_ndcg"]) for row in rows]
        ),
        "two_window": mean_sd(
            [float(row["two_window_ndcg"]) for row in rows]
        ),
        "two_window_wins": sum(row["two_minus_one"] > 0.0 for row in rows),
    }
    if summary["n_feasible_transitions"] != 18:
        raise AssertionError("unexpected feasible-transition count")
    if summary["two_window_wins"] != 18:
        raise AssertionError("two-window path does not improve 18/18")
    return {"per_transition": rows, "summary": summary}


def dense_feature_time_census(
    modern: dict[str, core.TransitionData],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    methods = tuple(DENSE_CACHE_FILES["W1"])
    for wt, wt1 in core.MODERN_TRANSITIONS:
        label = f"{wt}->{wt1}"
        td = modern[label]
        feature = core.load_window(wt)
        feature_idx = np.fromiter(
            (feature.ticker_to_idx[ticker] for ticker in td.tickers),
            dtype=np.int64,
        )
        scores_by_method: dict[str, float] = {}
        for method in methods:
            path = CACHE_DIR / DENSE_CACHE_FILES[wt][method]
            embeddings = np.load(path, mmap_mode="r")
            if embeddings.shape[0] != len(feature.tickers):
                raise AssertionError(f"{path.name}: cache/universe mismatch")
            subset = np.asarray(embeddings[feature_idx], dtype=np.float32)
            evaluation = core.evaluate_scores(
                core.return_path_scores(subset),
                td.realized,
                k=core.K_PRIMARY,
            )
            scores_by_method[method] = evaluation["mean_ndcg"]
        rows.append(
            {
                "transition": label,
                "n_universe": td.n,
                "ndcg_by_method": scores_by_method,
                "n_queries": td.n,
            }
        )

    aggregates = {}
    for method in methods:
        aggregates[method] = mean_sd(
            [float(row["ndcg_by_method"][method]) for row in rows]
        )
    return {
        "per_transition": rows,
        "summary": {
            "n_transitions": len(rows),
            "n_universe_min": min(row["n_universe"] for row in rows),
            "n_universe_max": max(row["n_universe"] for row in rows),
            "aggregates": aggregates,
        },
    }


def input_paths() -> list[Path]:
    paths = {
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_acceptance_robustness.py",
        ROOT / "scripts" / "metrics.py",
        ROOT / "scripts" / "exp_vintage_all_encoders.py",
        ROOT / "src" / "s01_universe.py",
        ROOT / "src" / "config.py",
        ROOT / "data" / "universe.parquet",
    }
    for name in SEQUENCE:
        directory = core.window_directory(name)
        paths.update(
            {
                directory / "ts_features.npy",
                directory / "corr_target.npy",
                directory / "universe.parquet",
                directory / "meta.json",
            }
        )
    for wt, _ in core.MODERN_TRANSITIONS:
        paths.add(core.VINTAGE_DIR / f"{wt}_descriptions.parquet")
        for filename in DENSE_CACHE_FILES[wt].values():
            paths.add(CACHE_DIR / filename)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing follow-up inputs: {missing}")
    return sorted(paths, key=lambda path: logical_relative(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    if not args.run_id.startswith("run_") or not all(
        char.isalnum() or char in "._-" for char in args.run_id
    ):
        parser.error("run id must be a safe run_* label")
    run_dir = OUTPUT_ROOT / args.run_id
    if run_dir.exists():
        parser.error(f"refusing to overwrite {run_dir}")

    started = time.perf_counter()
    tfidf_config = selected_tfidf_config()
    modern = {
        f"{wt}->{wt1}": core.build_transition(
            wt, wt1, universe_policy="feature_time"
        )
        for wt, wt1 in core.MODERN_TRANSITIONS
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Follow-up comparisons: stale returns, return history, "
            "historical extension, and feature-time dense rescoring"
        ),
        "configuration": {
            "metric": (
                "canonical positive-correlation NDCG@5; every eligible "
                "security is a query"
            ),
            "inference": (
                "descriptive finite-panel census; no IID query or transition "
                "inference"
            ),
            "tfidf_config": tfidf_config,
            "modern_data_used_for_configuration_selection": False,
            "stale_control": (
                "W_{t-1} return path and W_t vintage TF-IDF evaluated against "
                "W_{t+1} on an identical matched subset"
            ),
            "memory_curve": (
                "predeclared M=1,2,3,4 consecutive separately standardized "
                "paths, concatenated and row-L2 normalized; all M evaluated "
                "on the M=4 availability subset"
            ),
            "two_window_extension": (
                "return-only intersection of previous, feature, and target "
                "windows for all 18 feasible transitions"
            ),
            "dense_rescoring": (
                "frozen cached embeddings rescored on primary feature-time "
                "eligibility; no target-window text coverage"
            ),
        },
        "stale_return_control": stale_return_control(modern, tfidf_config),
        "modern_memory_curve": modern_memory_curve(modern),
        "two_window_full_extension": two_window_full_extension(),
        "dense_feature_time_census": dense_feature_time_census(modern),
        "universe_construction_disclosure": {
            "candidate_universe_n": 2200,
            "candidate_file_frozen_date": "2026-04-12",
            "construction": (
                "current SEC company metadata matched to GDELT organization "
                "coverage, including a 2025-2026 news-coverage filter"
            ),
            "point_in_time_security_master": False,
            "interpretation": (
                "historical analyses are conditional on an ex-post, "
                "current-survivor/news-visible cohort; bias direction is not "
                "identified by this census"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": polars.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    paths = input_paths()
    payload["input_sha256"] = {
        logical_relative(path): sha256_file(path) for path in paths
    }
    payload["runtime_seconds"] = time.perf_counter() - started

    run_dir.mkdir(parents=True)
    result_path = run_dir / "adversarial_followup_census.json"
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
            "scripts/run_adversarial_followup.py": sha256_file(
                Path(__file__).resolve()
            ),
            "scripts/run_acceptance_robustness.py": sha256_file(
                ROOT / "scripts" / "run_acceptance_robustness.py"
            ),
            "scripts/metrics.py": sha256_file(ROOT / "scripts" / "metrics.py"),
        },
        "notes": [
            "No modern target selected TF-IDF configuration or memory length.",
            "H1->H2 has no preceding path and is absent from the two-window extension.",
            "The fixed candidate universe is not a point-in-time security master.",
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

    print(f"run_dir={logical_relative(run_dir)}")
    print(
        "stale_vs_tfidf="
        f"{payload['stale_return_control']['summary']['stale_return_path_wins_over_tfidf']}/6"
    )
    print(
        "memory_increment_wins="
        + ",".join(
            str(row.get("wins_over_previous_memory", "NA"))
            for row in payload["modern_memory_curve"]["summary"]["aggregates"]
        )
    )
    print(
        "two_window_extension_wins="
        f"{payload['two_window_full_extension']['summary']['two_window_wins']}/18"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
