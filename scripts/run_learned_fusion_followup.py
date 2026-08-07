#!/usr/bin/env python3
"""Train a historical-only nonlinear peer-retrieval fusion baseline.

The classifier observes return-path and TF-IDF similarity/rank features.  Its
binary target is membership among a query's five most positively correlated
peers in the next window.  Training uses H1->H2 through H12->H13 only.  The
model and all configuration choices are then frozen before W1->W2 through
W6->W7 evaluation.

This is a comparator, not a replacement primary method.  It creates a new
immutable result directory with exact input and source hashes.
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
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_acceptance_robustness as core  # noqa: E402


SCHEMA_VERSION = "learned-fusion-followup-v1"
OUTPUT_ROOT = ROOT / "results" / "learned_fusion_followup"
SEED = 20260730
TOP_RELEVANT = 5
HARD_PER_METHOD = 30
RANDOM_CANDIDATES = 30
PREDICT_QUERY_CHUNK = 128
MODEL_PARAMETERS: dict[str, Any] = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "random_state": SEED,
}


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
        config
        for config in core.TFIDF_CONFIGS
        if config["config_id"] == "b"
    ]
    if len(matches) != 1:
        raise AssertionError("historically selected TF-IDF config b is absent")
    return matches[0]


def pair_features(
    rp_scores: np.ndarray,
    tf_scores: np.ndarray,
    rp_ranks: np.ndarray,
    tf_ranks: np.ndarray,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    rp_rank = rp_ranks[query_indices, candidate_indices].astype(np.float32)
    tf_rank = tf_ranks[query_indices, candidate_indices].astype(np.float32)
    rp_raw = rp_scores[query_indices, candidate_indices].astype(np.float32)
    tf_raw = tf_scores[query_indices, candidate_indices].astype(np.float32)
    return np.column_stack(
        (
            rp_rank,
            tf_rank,
            rp_raw,
            tf_raw,
            np.minimum(rp_rank, tf_rank),
            np.abs(rp_rank - tf_rank),
        )
    ).astype(np.float32)


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k >= values.size:
        return np.argsort(-values)
    selected = np.argpartition(values, -k)[-k:]
    return selected[np.argsort(-values[selected], kind="stable")]


def build_training_rows(
    tfidf_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(SEED)
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []

    for wt, wt1 in core.HISTORICAL_TUNING_TRANSITIONS:
        started = time.perf_counter()
        td = core.build_transition(wt, wt1, universe_policy="feature_time")
        rp_scores = core.return_path_scores(td.ts_feature)
        tf_scores, vocab_size = core.fit_tfidf_scores(
            td.texts, tfidf_config["kwargs"]
        )
        rp_ranks = core.average_rank_matrix(rp_scores)
        tf_ranks = core.average_rank_matrix(tf_scores)
        n = td.n
        transition_features: list[np.ndarray] = []
        transition_labels: list[np.ndarray] = []

        for query in range(n):
            relevance = np.maximum(
                np.asarray(td.realized[query], dtype=np.float64), 0.0
            )
            relevance[query] = -np.inf
            positive = top_indices(relevance, TOP_RELEVANT)
            if np.any(relevance[positive] <= 0.0):
                raise AssertionError(
                    f"{td.label}: query {query} lacks five positive peers"
                )

            rp_work = np.asarray(rp_ranks[query], dtype=np.float64).copy()
            tf_work = np.asarray(tf_ranks[query], dtype=np.float64).copy()
            rp_work[query] = -np.inf
            tf_work[query] = -np.inf
            hard_rp = top_indices(rp_work, HARD_PER_METHOD)
            hard_tf = top_indices(tf_work, HARD_PER_METHOD)

            excluded = np.zeros(n, dtype=bool)
            excluded[query] = True
            excluded[positive] = True
            random_pool = np.flatnonzero(~excluded)
            random_draw = rng.choice(
                random_pool,
                size=min(RANDOM_CANDIDATES, random_pool.size),
                replace=False,
            )
            candidates = np.unique(
                np.concatenate((positive, hard_rp, hard_tf, random_draw))
            )
            queries = np.full(candidates.size, query, dtype=np.int64)
            labels = np.isin(candidates, positive).astype(np.uint8)
            transition_features.append(
                pair_features(
                    rp_scores,
                    tf_scores,
                    rp_ranks,
                    tf_ranks,
                    queries,
                    candidates,
                )
            )
            transition_labels.append(labels)

        x_transition = np.concatenate(transition_features, axis=0)
        y_transition = np.concatenate(transition_labels, axis=0)
        feature_blocks.append(x_transition)
        label_blocks.append(y_transition)
        audit_rows.append(
            {
                "transition": td.label,
                "n_universe": n,
                "n_training_pairs": int(y_transition.size),
                "n_positive_pairs": int(y_transition.sum()),
                "tfidf_vocab_size": vocab_size,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"train rows {td.label}: N={n}, pairs={y_transition.size:,}",
            flush=True,
        )

    return (
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(label_blocks, axis=0),
        audit_rows,
    )


def predict_score_matrix(
    model: HistGradientBoostingClassifier,
    rp_scores: np.ndarray,
    tf_scores: np.ndarray,
    rp_ranks: np.ndarray,
    tf_ranks: np.ndarray,
) -> np.ndarray:
    n = rp_scores.shape[0]
    result = np.empty((n, n), dtype=np.float32)
    candidates = np.arange(n, dtype=np.int64)
    for start in range(0, n, PREDICT_QUERY_CHUNK):
        stop = min(start + PREDICT_QUERY_CHUNK, n)
        queries = np.repeat(
            np.arange(start, stop, dtype=np.int64), n
        )
        tiled_candidates = np.tile(candidates, stop - start)
        features = pair_features(
            rp_scores,
            tf_scores,
            rp_ranks,
            tf_ranks,
            queries,
            tiled_candidates,
        )
        probabilities = model.predict_proba(features)[:, 1]
        result[start:stop] = probabilities.reshape(stop - start, n)
    np.fill_diagonal(result, -np.inf)
    return result


def evaluate_modern(
    model: HistGradientBoostingClassifier,
    tfidf_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wt, wt1 in core.MODERN_TRANSITIONS:
        started = time.perf_counter()
        td = core.build_transition(wt, wt1, universe_policy="feature_time")
        rp_scores = core.return_path_scores(td.ts_feature)
        tf_scores, vocab_size = core.fit_tfidf_scores(
            td.texts, tfidf_config["kwargs"]
        )
        rp_ranks = core.average_rank_matrix(rp_scores)
        tf_ranks = core.average_rank_matrix(tf_scores)
        blend_scores = 0.8 * rp_ranks + 0.2 * tf_ranks
        learned_scores = predict_score_matrix(
            model, rp_scores, tf_scores, rp_ranks, tf_ranks
        )
        learned_eval = core.evaluate_scores(
            learned_scores, td.realized, k=core.K_PRIMARY
        )
        rp_eval = core.evaluate_scores(
            rp_scores, td.realized, k=core.K_PRIMARY
        )
        tf_eval = core.evaluate_scores(
            tf_scores, td.realized, k=core.K_PRIMARY
        )
        blend_eval = core.evaluate_scores(
            blend_scores, td.realized, k=core.K_PRIMARY
        )
        rows.append(
            {
                "transition": td.label,
                "n_universe": td.n,
                "n_queries": learned_eval["n_queries"],
                "tfidf_vocab_size": vocab_size,
                "return_path_ndcg": rp_eval["mean_ndcg"],
                "tfidf_ndcg": tf_eval["mean_ndcg"],
                "frozen_rank_blend_ndcg": blend_eval["mean_ndcg"],
                "historically_trained_nonlinear_fusion_ndcg": (
                    learned_eval["mean_ndcg"]
                ),
                "learned_minus_return_path": (
                    learned_eval["mean_ndcg"] - rp_eval["mean_ndcg"]
                ),
                "learned_minus_frozen_rank_blend": (
                    learned_eval["mean_ndcg"] - blend_eval["mean_ndcg"]
                ),
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"modern {td.label}: learned={learned_eval['mean_ndcg']:.4f}, "
            f"blend={blend_eval['mean_ndcg']:.4f}",
            flush=True,
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_fields = {
        "return_path": "return_path_ndcg",
        "tfidf": "tfidf_ndcg",
        "frozen_rank_blend": "frozen_rank_blend_ndcg",
        "historically_trained_nonlinear_fusion": (
            "historically_trained_nonlinear_fusion_ndcg"
        ),
    }
    aggregates: dict[str, Any] = {}
    for method, field in method_fields.items():
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        aggregates[method] = {
            "macro_mean_ndcg": float(values.mean()),
            "population_std_across_transition_means": float(values.std()),
        }
    return {
        "n_modern_transitions": len(rows),
        "aggregates": aggregates,
        "learned_wins_over_return_path": sum(
            row["learned_minus_return_path"] > 0.0 for row in rows
        ),
        "learned_wins_over_frozen_rank_blend": sum(
            row["learned_minus_frozen_rank_blend"] > 0.0 for row in rows
        ),
        "modern_data_used_for_training_or_configuration": False,
    }


def input_paths() -> list[Path]:
    result = [
        ROOT / "scripts" / "metrics.py",
        ROOT / "scripts" / "run_acceptance_robustness.py",
        ROOT / "scripts" / "run_learned_fusion_followup.py",
    ]
    windows = set()
    for left, right in (
        tuple(core.HISTORICAL_TUNING_TRANSITIONS)
        + tuple(core.MODERN_TRANSITIONS)
    ):
        windows.update((left, right))
    for window in sorted(windows):
        directory = core.window_directory(window)
        result.extend(
            (
                directory / "ts_features.npy",
                directory / "corr_target.npy",
                directory / "universe.parquet",
                directory / "meta.json",
            )
        )
        if window != "W7" and window != "H13":
            result.append(
                core.VINTAGE_DIR / f"{window}_descriptions.parquet"
            )
    return sorted(set(result))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir()

    started = time.perf_counter()
    tfidf_config = selected_tfidf_config()
    x_train, y_train, training_rows = build_training_rows(tfidf_config)
    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    model.fit(x_train, y_train)
    modern_rows = evaluate_modern(model, tfidf_config)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Historical-only nonlinear fusion comparator for modern "
            "return-defined peer retrieval"
        ),
        "configuration": {
            "training_transitions": [
                f"{left}->{right}"
                for left, right in core.HISTORICAL_TUNING_TRANSITIONS
            ],
            "evaluation_transitions": [
                f"{left}->{right}" for left, right in core.MODERN_TRANSITIONS
            ],
            "features": [
                "return-path percentile rank",
                "TF-IDF percentile rank",
                "return-path cosine similarity",
                "TF-IDF cosine similarity",
                "minimum modality rank",
                "absolute rank disagreement",
            ],
            "historical_label": (
                "membership among each query's five highest positive "
                "next-window residual-correlation peers"
            ),
            "candidate_sampling_per_historical_query": {
                "positive": TOP_RELEVANT,
                "hard_by_return_path": HARD_PER_METHOD,
                "hard_by_tfidf": HARD_PER_METHOD,
                "uniform_random": RANDOM_CANDIDATES,
                "deduplication": "set union",
            },
            "model": (
                "sklearn.ensemble.HistGradientBoostingClassifier"
            ),
            "model_parameters": MODEL_PARAMETERS,
            "tfidf_config": {
                "config_id": tfidf_config["config_id"],
                "kwargs": tfidf_config["kwargs"],
            },
            "comparison_blend": (
                "historically selected fixed 0.8 return-path / 0.2 TF-IDF "
                "average-percentile-rank blend"
            ),
            "modern_target_used_for_training_or_configuration": False,
        },
        "training_audit": {
            "n_rows": int(y_train.size),
            "n_positive": int(y_train.sum()),
            "n_features": int(x_train.shape[1]),
            "per_transition": training_rows,
        },
        "modern_evaluation": {
            "per_transition": modern_rows,
            "summary": summarize(modern_rows),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
        "input_sha256": {
            relative(path): sha256_file(path) for path in input_paths()
        },
    }
    artifact_path = run_dir / "learned_fusion_followup.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "artifact_sha256": {
            artifact_path.name: sha256_file(artifact_path)
        },
        "source_sha256": {
            "scripts/run_learned_fusion_followup.py": sha256_file(
                ROOT / "scripts" / "run_learned_fusion_followup.py"
            ),
            "scripts/run_acceptance_robustness.py": sha256_file(
                ROOT / "scripts" / "run_acceptance_robustness.py"
            ),
            "scripts/metrics.py": sha256_file(
                ROOT / "scripts" / "metrics.py"
            ),
        },
        "notes": [
            "No modern target enters training, feature choice, or configuration.",
            "This comparator is evaluated on the identical primary feature-time universes.",
        ],
    }
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = {
        artifact_path.name: sha256_file(artifact_path),
        manifest_path.name: sha256_file(manifest_path),
    }
    (run_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(checksums.items())
        ),
        encoding="utf-8",
    )
    print(f"saved immutable run: {run_dir}")
    print(json.dumps(artifact["modern_evaluation"]["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
