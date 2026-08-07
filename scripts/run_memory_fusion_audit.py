#!/usr/bin/env python3
"""Audit whether vintage filing text complements longer return-path memory.

The primary paper selects a ReturnPath/TF-IDF rank-fusion weight using twelve
pre-modern transitions and freezes alpha=0.8 before the six modern transitions.
This audit applies that same frozen weight to M=1,...,4 concatenated return
paths on one identical four-window-availability universe per transition.  It
also reports a pre-modern, memory-specific alpha diagnostic; modern targets
never select a weight or memory length.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

import scripts.run_acceptance_robustness as core  # noqa: E402


SCHEMA_VERSION = "memory-fusion-audit-v1"
OUTPUT_ROOT = ROOT / "results" / "memory_fusion_audit"
SEQUENCE = tuple(f"H{i}" for i in range(1, 14)) + tuple(
    f"W{i}" for i in range(1, 8)
)
MEMORY_LENGTHS = (1, 2, 3, 4)
FROZEN_ALPHA = 0.8
ALPHA_GRID = tuple(float(value) for value in core.ALPHA_GRID)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def selected_tfidf_config() -> dict[str, Any]:
    matches = [
        config for config in core.TFIDF_CONFIGS if config["config_id"] == "b"
    ]
    if len(matches) != 1:
        raise AssertionError("historically selected TF-IDF config b is absent")
    return matches[0]


def histories_for(feature_window: str, memory: int) -> tuple[str, ...]:
    position = SEQUENCE.index(feature_window)
    names = SEQUENCE[position - memory + 1 : position + 1]
    if len(names) != memory:
        return ()
    return tuple(names)


def build_matched_scores(
    wt: str,
    wt1: str,
    memory_lengths: tuple[int, ...],
    tfidf_config: dict[str, Any],
) -> dict[str, Any]:
    """Build every requested memory score on the longest-memory subset."""
    longest = max(memory_lengths)
    history_names = histories_for(wt, longest)
    if not history_names:
        raise ValueError(f"{wt}->{wt1}: {longest} histories unavailable")

    td = core.build_transition(wt, wt1, universe_policy="feature_time")
    histories = [core.load_window(name) for name in history_names]
    eligible = set(td.tickers)
    for window in histories:
        eligible &= set(window.tickers)
    tickers = tuple(sorted(eligible))
    if len(tickers) < core.MIN_UNIVERSE:
        raise AssertionError(f"{wt}->{wt1}: matched universe is too small")

    td_index = {ticker: idx for idx, ticker in enumerate(td.tickers)}
    feature_indices = np.fromiter(
        (td_index[ticker] for ticker in tickers), dtype=np.int64
    )
    realized = np.asarray(
        td.realized[np.ix_(feature_indices, feature_indices)], dtype=np.float32
    )
    texts = [td.texts[idx] for idx in feature_indices]
    tfidf_scores, vocab_size = core.fit_tfidf_scores(
        texts, tfidf_config["kwargs"]
    )
    tfidf_ranks = core.average_rank_matrix(tfidf_scores)

    score_by_memory: dict[int, np.ndarray] = {}
    dimension_by_memory: dict[int, int] = {}
    for memory in memory_lengths:
        parts: list[np.ndarray] = []
        for window in histories[-memory:]:
            indices = np.fromiter(
                (window.ticker_to_idx[ticker] for ticker in tickers),
                dtype=np.int64,
            )
            parts.append(np.asarray(window.ts_features[indices], dtype=np.float32))
        concatenated = np.concatenate(parts, axis=1)
        score_by_memory[memory] = core.return_path_scores(concatenated)
        dimension_by_memory[memory] = int(concatenated.shape[1])

    return {
        "transition": f"{wt}->{wt1}",
        "history_windows_oldest_to_current": list(history_names),
        "tickers": tickers,
        "realized": realized,
        "tfidf_scores": tfidf_scores,
        "tfidf_ranks": tfidf_ranks,
        "return_scores": score_by_memory,
        "dimensions": dimension_by_memory,
        "vocab_size": vocab_size,
    }


def modern_frozen_audit(tfidf_config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for wt, wt1 in core.MODERN_TRANSITIONS:
        started = time.perf_counter()
        bundle = build_matched_scores(
            wt, wt1, MEMORY_LENGTHS, tfidf_config
        )
        for memory in MEMORY_LENGTHS:
            return_scores = bundle["return_scores"][memory]
            return_ranks = core.average_rank_matrix(return_scores)
            blend_scores = (
                FROZEN_ALPHA * return_ranks
                + (1.0 - FROZEN_ALPHA) * bundle["tfidf_ranks"]
            )
            return_eval = core.evaluate_scores(
                return_scores, bundle["realized"], k=core.K_PRIMARY
            )
            blend_eval = core.evaluate_scores(
                blend_scores, bundle["realized"], k=core.K_PRIMARY
            )
            rows.append(
                {
                    "transition": bundle["transition"],
                    "memory_windows": memory,
                    "history_windows_oldest_to_current": (
                        bundle["history_windows_oldest_to_current"][-memory:]
                    ),
                    "n_universe": len(bundle["tickers"]),
                    "n_queries": return_eval["n_queries"],
                    "path_dimension": bundle["dimensions"][memory],
                    "tfidf_vocab_size": bundle["vocab_size"],
                    "alpha_return_path": FROZEN_ALPHA,
                    "alpha_tfidf": 1.0 - FROZEN_ALPHA,
                    "return_path_ndcg": return_eval["mean_ndcg"],
                    "frozen_blend_ndcg": blend_eval["mean_ndcg"],
                    "blend_minus_return_path": (
                        blend_eval["mean_ndcg"] - return_eval["mean_ndcg"]
                    ),
                    "runtime_seconds": time.perf_counter() - started,
                }
            )

    aggregates: list[dict[str, Any]] = []
    for memory in MEMORY_LENGTHS:
        selected = [row for row in rows if row["memory_windows"] == memory]
        path_values = [float(row["return_path_ndcg"]) for row in selected]
        blend_values = [float(row["frozen_blend_ndcg"]) for row in selected]
        deltas = [float(row["blend_minus_return_path"]) for row in selected]
        aggregates.append(
            {
                "memory_windows": memory,
                "return_path_macro_mean_ndcg": fmean(path_values),
                "return_path_population_std": pstdev(path_values),
                "frozen_blend_macro_mean_ndcg": fmean(blend_values),
                "frozen_blend_population_std": pstdev(blend_values),
                "macro_mean_blend_minus_return_path": fmean(deltas),
                "blend_wins_over_return_path": sum(delta > 0.0 for delta in deltas),
                "n_transitions": len(selected),
                "n_universe_min": min(int(row["n_universe"]) for row in selected),
                "n_universe_max": max(int(row["n_universe"]) for row in selected),
            }
        )
    return {"per_transition": rows, "summary": {"aggregates": aggregates}}


def historical_memory_specific_alpha(
    tfidf_config: dict[str, Any],
) -> dict[str, Any]:
    # Use only the nine transitions with four available feature paths, and use
    # the identical four-window-availability universe for every M.  This makes
    # the memory-specific alpha comparison apples-to-apples.
    transition_rows: list[dict[str, Any]] = []
    for wt, wt1 in core.HISTORICAL_TUNING_TRANSITIONS:
        if not histories_for(wt, max(MEMORY_LENGTHS)):
            continue
        bundle = build_matched_scores(
            wt, wt1, MEMORY_LENGTHS, tfidf_config
        )
        for memory in MEMORY_LENGTHS:
            return_ranks = core.average_rank_matrix(
                bundle["return_scores"][memory]
            )
            for alpha in ALPHA_GRID:
                blend_scores = (
                    alpha * return_ranks
                    + (1.0 - alpha) * bundle["tfidf_ranks"]
                )
                evaluation = core.evaluate_scores(
                    blend_scores, bundle["realized"], k=core.K_PRIMARY
                )
                transition_rows.append(
                    {
                        "transition": bundle["transition"],
                        "memory_windows": memory,
                        "alpha_return_path": alpha,
                        "alpha_tfidf": 1.0 - alpha,
                        "n_universe": len(bundle["tickers"]),
                        "blend_ndcg": evaluation["mean_ndcg"],
                    }
                )

    results: list[dict[str, Any]] = []
    for memory in MEMORY_LENGTHS:
        memory_rows = [
            row for row in transition_rows if row["memory_windows"] == memory
        ]
        aggregates: list[dict[str, Any]] = []
        for alpha in ALPHA_GRID:
            values = [
                float(row["blend_ndcg"])
                for row in memory_rows
                if row["alpha_return_path"] == alpha
            ]
            aggregates.append(
                {
                    "alpha_return_path": alpha,
                    "alpha_tfidf": 1.0 - alpha,
                    "macro_mean_ndcg": fmean(values),
                    "population_std_across_transitions": pstdev(values),
                    "n_historical_transitions": len(values),
                }
            )
        selected = sorted(
            aggregates,
            key=lambda row: (
                -float(row["macro_mean_ndcg"]),
                -float(row["alpha_return_path"]),
            ),
        )[0]
        transitions = sorted(
            {row["transition"] for row in memory_rows},
            key=lambda label: SEQUENCE.index(label.split("->")[0]),
        )
        results.append(
            {
                "memory_windows": memory,
                "first_feasible_transition": transitions[0],
                "last_feasible_transition": transitions[-1],
                "alpha_aggregates": aggregates,
                "selected_alpha_return_path": selected["alpha_return_path"],
                "selected_alpha_tfidf": selected["alpha_tfidf"],
                "selected_macro_mean_ndcg": selected["macro_mean_ndcg"],
                "n_historical_transitions": selected["n_historical_transitions"],
                "n_universe_min": min(int(row["n_universe"]) for row in memory_rows),
                "n_universe_max": max(int(row["n_universe"]) for row in memory_rows),
            }
        )
    return {
        "selection_objective": (
            "unweighted macro mean of transition-level census NDCG@5"
        ),
        "tie_policy": "largest ReturnPath alpha",
        "modern_targets_used": False,
        "common_transition_set": "H4->H5 through H12->H13",
        "common_four_window_availability_universe": True,
        "by_memory": results,
    }


def input_paths() -> list[Path]:
    paths = [
        ROOT / "scripts" / "run_memory_fusion_audit.py",
        ROOT / "scripts" / "run_acceptance_robustness.py",
        ROOT / "scripts" / "metrics.py",
        ROOT
        / "results"
        / "acceptance_robustness"
        / "run_20260730T032543Z"
        / "02_blend_historical_tuning_modern_frozen.json",
    ]
    for window in SEQUENCE:
        directory = core.window_directory(window)
        paths.extend(
            (
                directory / "ts_features.npy",
                directory / "corr_target.npy",
                directory / "universe.parquet",
                directory / "meta.json",
            )
        )
        if window not in {"H13", "W7"}:
            paths.append(core.VINTAGE_DIR / f"{window}_descriptions.parquet")
    return sorted(set(path for path in paths if path.exists()))


def main() -> int:
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = selected_tfidf_config()

    modern = modern_frozen_audit(config)
    historical = historical_memory_specific_alpha(config)
    summary_by_memory = {
        int(row["memory_windows"]): row
        for row in modern["summary"]["aggregates"]
    }
    if summary_by_memory[2]["blend_wins_over_return_path"] != 6:
        raise AssertionError("frozen blend does not improve M=2 in all transitions")
    if summary_by_memory[4]["blend_wins_over_return_path"] != 0:
        raise AssertionError("frozen blend unexpectedly improves M=4")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Matched-universe audit of frozen return/text rank fusion as "
            "return-path memory grows"
        ),
        "configuration": {
            "memory_windows": list(MEMORY_LENGTHS),
            "modern_frozen_alpha_return_path": FROZEN_ALPHA,
            "modern_frozen_alpha_tfidf": 1.0 - FROZEN_ALPHA,
            "alpha_grid_return_path": list(ALPHA_GRID),
            "tfidf_config": {
                "config_id": config["config_id"],
                "kwargs": config["kwargs"],
            },
            "rank_normalization": "per-query average percentile ranks",
            "universe": (
                "identical within transition for M=1,...,4: feature-time text "
                "eligibility plus four consecutive feature paths and target returns"
            ),
            "metric": "canonical positive-correlation census NDCG@5",
        },
        "anti_leakage": {
            "modern_targets_used_for_alpha_or_memory_selection": False,
            "frozen_alpha_source": (
                "H1->H2 through H12->H13 one-window historical selection"
            ),
            "modern_memory_lengths": "descriptive predeclared sensitivities",
        },
        "modern_frozen_alpha": modern,
        "historical_memory_specific_alpha": historical,
        "validations": {
            "modern_transition_count": len(core.MODERN_TRANSITIONS),
            "identical_modern_subset_across_memory_lengths": True,
            "m2_frozen_blend_wins": summary_by_memory[2][
                "blend_wins_over_return_path"
            ],
            "m4_frozen_blend_wins": summary_by_memory[4][
                "blend_wins_over_return_path"
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
        "input_sha256": {
            relative(path): sha256_file(path) for path in input_paths()
        },
    }
    artifact_path = run_dir / "memory_fusion_audit.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "artifact_sha256": {artifact_path.name: sha256_file(artifact_path)},
        "source_sha256": {
            "scripts/run_memory_fusion_audit.py": sha256_file(
                ROOT / "scripts" / "run_memory_fusion_audit.py"
            )
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums_path = run_dir / "SHA256SUMS.txt"
    sums_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in (artifact_path, manifest_path)
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"run_dir={run_dir}")
    for row in modern["summary"]["aggregates"]:
        print(
            f"M={row['memory_windows']}: path={row['return_path_macro_mean_ndcg']:.6f}, "
            f"blend={row['frozen_blend_macro_mean_ndcg']:.6f}, "
            f"delta={row['macro_mean_blend_minus_return_path']:+.6f}, "
            f"wins={row['blend_wins_over_return_path']}/6"
        )
    for row in historical["by_memory"]:
        print(
            f"historical M={row['memory_windows']}: "
            f"selected_alpha={row['selected_alpha_return_path']:.1f}, "
            f"n={row['n_historical_transitions']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
