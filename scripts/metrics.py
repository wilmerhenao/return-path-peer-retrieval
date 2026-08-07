"""Canonical retrieval metrics for the ICAIF 2026 experiments.

All active paper experiments must import NDCG from this module.  Relevance is
defined as positive realized correlation: both the retrieved and ideal lists
therefore use ``max(0, relevance)``.  Query members are excluded from both
lists, and score ties are broken deterministically.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


TIE_RNG_BASE = 314159


def dcg(relevances: np.ndarray) -> float:
    """Return discounted cumulative gain for an ordered relevance vector."""
    rel = np.asarray(relevances, dtype=float)
    if rel.ndim != 1:
        raise ValueError("relevances must be one-dimensional")
    discounts = np.log2(np.arange(2, rel.size + 2, dtype=float))
    return float(np.sum(rel / discounts))


def _query_array(query_indices: int | Iterable[int]) -> np.ndarray:
    if np.isscalar(query_indices):
        result = np.array([int(query_indices)], dtype=int)
    else:
        result = np.asarray(list(query_indices), dtype=int)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("query_indices must contain at least one index")
    return np.unique(result)


def top_k_indices(
    scores: np.ndarray,
    k: int,
    *,
    excluded: int | Iterable[int] = (),
    query_seed: int = 0,
) -> np.ndarray:
    """Return deterministic top-k indices after excluding query members."""
    values = np.asarray(scores, dtype=float).copy()
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    excluded_idx = (
        np.array([], dtype=int)
        if isinstance(excluded, tuple) and not excluded
        else _query_array(excluded)
    )
    if np.any((excluded_idx < 0) | (excluded_idx >= values.size)):
        raise IndexError("excluded query index is outside the score vector")
    values[excluded_idx] = -np.inf
    eligible_count = values.size - excluded_idx.size
    k_eff = min(int(k), eligible_count)
    if k_eff <= 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(TIE_RNG_BASE + int(query_seed))
    jitter = rng.uniform(0.0, 1e-12, size=values.size)
    jitter[excluded_idx] = 0.0
    return np.argsort(-(values + jitter), kind="stable")[:k_eff]


def ndcg_at_k(
    scores: np.ndarray,
    realized_relevance: np.ndarray,
    query_indices: int | Iterable[int],
    *,
    k: int = 5,
    query_seed: int | None = None,
) -> float | None:
    """Compute positive-correlation NDCG@k.

    Negative realized correlations have zero relevance in both DCG and IDCG.
    Returns ``None`` when no eligible company has positive relevance.
    """
    values = np.asarray(scores, dtype=float)
    realized = np.asarray(realized_relevance, dtype=float)
    if values.ndim != 1 or realized.ndim != 1 or values.size != realized.size:
        raise ValueError("scores and realized_relevance must be equal-length vectors")
    if k <= 0:
        raise ValueError("k must be positive")

    query = _query_array(query_indices)
    if np.any((query < 0) | (query >= values.size)):
        raise IndexError("query index is outside the score vector")

    eligible_mask = np.ones(values.size, dtype=bool)
    eligible_mask[query] = False
    eligible_relevance = np.maximum(realized[eligible_mask], 0.0)
    ideal = np.sort(eligible_relevance)[::-1][: min(k, eligible_relevance.size)]
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0.0:
        return None

    seed = int(query[0]) if query_seed is None else int(query_seed)
    retrieved = top_k_indices(values, k, excluded=query, query_seed=seed)
    retrieved_relevance = np.maximum(realized[retrieved], 0.0)
    score = dcg(retrieved_relevance) / ideal_dcg
    # Numerical noise must not create a score outside the metric's range.
    return float(np.clip(score, 0.0, 1.0))

