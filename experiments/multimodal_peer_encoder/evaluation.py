"""Pure evaluation utilities for the locked multimodal peer experiment.

This module deliberately performs no file access and imports no experiment
runner.  It accepts already aligned arrays, constructs the two frozen
comparators, and delegates every query-level NDCG calculation to the paper's
canonical :func:`scripts.metrics.ndcg_at_k` implementation.

The learned retriever and cosine baselines produce symmetric score matrices.
The frozen rank blend is query-specific and need not be symmetric because a
candidate's percentile rank can differ across query rows.  Consequently,
``census_ndcg_at_5`` requires symmetry by default and callers must explicitly
opt out only for such a row-ranked comparator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts.metrics import ndcg_at_k


K_PRIMARY = 5
FROZEN_RETURN_WEIGHT = 0.8
FROZEN_TFIDF_WEIGHT = 0.2
SYMMETRY_ATOL = 2e-5
SYMMETRY_RTOL = 2e-5


def _as_finite_matrix(values: np.ndarray, label: str) -> np.ndarray:
    """Return a float64 two-dimensional array after basic validation."""

    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a numeric matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must contain only finite values")
    return matrix


def validate_score_matrix(
    scores: np.ndarray,
    *,
    label: str = "scores",
    require_symmetric: bool = True,
) -> np.ndarray:
    """Validate and return a finite square score matrix.

    A finite diagonal is required even though the canonical retrieval metric
    excludes the query security.  Keeping it finite prevents a training-time
    ``-inf`` self mask from leaking into census evaluation or provenance.
    """

    matrix = _as_finite_matrix(scores, label)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be square")
    if require_symmetric and not np.allclose(
        matrix,
        matrix.T,
        atol=SYMMETRY_ATOL,
        rtol=SYMMETRY_RTOL,
    ):
        raise ValueError(f"{label} must be symmetric")
    return matrix


def row_l2_cosine(features: np.ndarray) -> np.ndarray:
    """Return all-pairs cosine similarity after row-wise L2 normalization."""

    matrix = _as_finite_matrix(features, "features")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("features must not contain a zero-norm row")

    # Match the paper's active ReturnPath implementation: normalize in float64,
    # store the embedding as float32, and form the similarity matrix in float32.
    normalized = (matrix / norms).astype(np.float32)
    scores = (normalized @ normalized.T).astype(np.float32)
    validate_score_matrix(scores, label="cosine scores")
    return scores


def average_percentile_rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Compute deterministic per-query average percentile ranks.

    The query security receives rank zero and is absent from the denominator.
    Eligible candidates receive ranks in ``[1 / (N - 1), 1]``; larger scores
    receive larger ranks.  Exact ties receive their average rank.  This matches
    the frozen ranking convention used by the active paper authority without
    relying on a second ranking implementation.
    """

    matrix = validate_score_matrix(scores, label="rank scores")
    n_items = matrix.shape[0]
    if n_items < 2:
        raise ValueError("rank scores must contain at least two securities")

    percentile_ranks = np.zeros((n_items, n_items), dtype=np.float32)
    all_indices = np.arange(n_items)
    denominator = float(n_items - 1)

    for query in range(n_items):
        eligible = all_indices != query
        eligible_indices = all_indices[eligible]
        values = matrix[query, eligible]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]

        sorted_ranks = np.empty(n_items - 1, dtype=np.float64)
        start = 0
        while start < sorted_values.size:
            stop = start + 1
            while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
                stop += 1
            # Eligible ranks are one-based.  The average of ranks
            # (start + 1), ..., stop is ((start + 1) + stop) / 2.
            sorted_ranks[start:stop] = ((start + 1) + stop) / 2.0
            start = stop

        row_ranks = np.empty(n_items - 1, dtype=np.float64)
        row_ranks[order] = sorted_ranks / denominator
        percentile_ranks[query, eligible_indices] = row_ranks.astype(np.float32)

    np.fill_diagonal(percentile_ranks, 0.0)
    if not np.isfinite(percentile_ranks).all():
        raise AssertionError("rank construction produced a non-finite value")
    return percentile_ranks


def frozen_rank_blend(
    return_scores: np.ndarray,
    tfidf_scores: np.ndarray,
) -> np.ndarray:
    """Return the historically frozen 80/20 ReturnPath/TF-IDF rank blend."""

    return_matrix = validate_score_matrix(return_scores, label="return scores")
    tfidf_matrix = validate_score_matrix(tfidf_scores, label="TF-IDF scores")
    if return_matrix.shape != tfidf_matrix.shape:
        raise ValueError("return and TF-IDF score matrices must have equal shape")

    return_ranks = average_percentile_rank_matrix(return_matrix)
    tfidf_ranks = average_percentile_rank_matrix(tfidf_matrix)
    blended = (
        FROZEN_RETURN_WEIGHT * return_ranks
        + FROZEN_TFIDF_WEIGHT * tfidf_ranks
    ).astype(np.float32)
    if not np.isfinite(blended).all():
        raise AssertionError("frozen rank blend produced a non-finite value")
    return blended


def census_ndcg_at_5(
    scores: np.ndarray,
    realized_correlation: np.ndarray,
    *,
    require_symmetric_scores: bool = True,
) -> dict[str, Any]:
    """Evaluate every security with canonical positive-correlation NDCG@5.

    ``require_symmetric_scores`` should remain true for learned scores and
    cosine baselines.  It may be set to false for the frozen row-percentile
    blend, whose query-specific rank transformation is not generally
    symmetric.
    """

    score_matrix = validate_score_matrix(
        scores,
        label="scores",
        require_symmetric=require_symmetric_scores,
    )
    realized = validate_score_matrix(
        realized_correlation,
        label="realized correlation",
        require_symmetric=True,
    )
    if score_matrix.shape != realized.shape:
        raise ValueError("scores and realized correlation must have equal shape")
    if score_matrix.shape[0] < 2:
        raise ValueError("census evaluation requires at least two securities")

    query_values: list[float | None] = []
    for query in range(score_matrix.shape[0]):
        query_values.append(
            ndcg_at_k(
                score_matrix[query],
                realized[query],
                query,
                k=K_PRIMARY,
            )
        )

    valid = np.asarray(
        [value for value in query_values if value is not None],
        dtype=np.float64,
    )
    if valid.size == 0:
        raise ValueError("no census query has positive eligible relevance")
    if np.any((valid < 0.0) | (valid > 1.0)):
        raise AssertionError("canonical NDCG returned a value outside [0, 1]")

    return {
        "n_queries": score_matrix.shape[0],
        "n_valid": int(valid.size),
        "n_undefined_no_positive_idcg": int(score_matrix.shape[0] - valid.size),
        "mean_ndcg": float(valid.mean()),
        "population_std_ndcg": float(valid.std(ddof=0)),
        "min_ndcg": float(valid.min()),
        "max_ndcg": float(valid.max()),
        "query_ndcg": tuple(query_values),
    }


def aggregate_transition_ndcg(
    transition_results: Sequence[Mapping[str, Any]],
    *,
    transition_key: str = "transition",
    metric_key: str = "mean_ndcg",
) -> dict[str, Any]:
    """Aggregate transition means with one equal weight per transition."""

    if not transition_results:
        raise ValueError("transition_results must not be empty")

    labels: list[str] = []
    values: list[float] = []
    for row in transition_results:
        if transition_key not in row or metric_key not in row:
            raise ValueError(
                f"each transition row must contain {transition_key!r} and {metric_key!r}"
            )
        label = str(row[transition_key])
        if not label:
            raise ValueError("transition labels must not be empty")
        try:
            value = float(row[metric_key])
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{metric_key} must be numeric") from exc
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{metric_key} must be finite and lie in [0, 1]")
        labels.append(label)
        values.append(value)

    if len(labels) != len(set(labels)):
        raise ValueError("transition labels must be unique")
    array = np.asarray(values, dtype=np.float64)
    return {
        "n_transitions": len(labels),
        "transitions": tuple(labels),
        "macro_mean_ndcg": float(array.mean()),
        "population_std_across_transition_means": float(array.std(ddof=0)),
        "min_transition_mean_ndcg": float(array.min()),
        "max_transition_mean_ndcg": float(array.max()),
    }
