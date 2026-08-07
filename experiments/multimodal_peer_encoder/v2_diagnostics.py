"""Fixed interpretability and application diagnostics for v2.

All functions operate on already frozen out-of-fold scores and historical
arrays.  They do not select a model and do not perform file access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from experiments.multimodal_peer_encoder.v2_model import (
    AdditiveBoostingHead,
    CenteredCrossModalHead,
    build_pair_features,
    rank_band_candidate_indices,
    raw_candidate_order,
)
from experiments.multimodal_peer_encoder.v2_training import (
    FitResult,
    FitSpec,
    TransitionLike,
)
from scripts.metrics import top_k_indices


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic distribution must be nonempty and finite")
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def average_distributions(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty diagnostic collection")
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("mean", "standard_deviation", "p10", "p50", "p90")
    }


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("Pearson inputs must have equal nontrivial length")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Pearson inputs must be finite")
    if x.std(ddof=0) == 0.0 or y.std(ddof=0) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _tensors(
    fit: FitResult,
    transition: TransitionLike,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.tensor(transition.raw_scores, dtype=torch.float32)
    text = torch.tensor(transition.text_scores, dtype=torch.float32)
    zloglen = torch.as_tensor(
        fit.scaler.transform(transition.log_lengths), dtype=torch.float32
    )
    return raw, text, zloglen


def _window_context(transition: TransitionLike, config: Mapping[str, Any]) -> float:
    reporting = config["reporting"]
    return (
        float(transition.window_length) - float(reporting["context_window_center"])
    ) / float(reporting["context_window_scale"])


def _top_five(scores: np.ndarray) -> list[np.ndarray]:
    matrix = np.asarray(scores, dtype=np.float64)
    return [
        top_k_indices(matrix[q], 5, excluded=q, query_seed=q)
        for q in range(matrix.shape[0])
    ]


def diagnose_head_transition(
    fit: FitResult,
    spec: FitSpec,
    transition: TransitionLike,
    config: Mapping[str, Any],
    *,
    frozen_scores: np.ndarray,
    semantic_ablation_ndcg_change: float,
) -> dict[str, Any]:
    """Describe one frozen model on one held-out transition."""

    raw, text, zloglen = _tensors(fit, transition)
    n_items = raw.shape[0]
    chunk_size = int(config["optimization"]["query_chunk_size"])
    full_order = raw_candidate_order(raw)
    top_five = _top_five(frozen_scores)

    primary_values: list[np.ndarray] = []
    derivative_values: list[np.ndarray] = []
    query_context: list[np.ndarray] = []
    candidate_context: list[np.ndarray] = []
    top_five_gamma: list[np.ndarray] = []

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        queries = torch.arange(start, stop, dtype=torch.long)
        text_leaf = text.detach().clone().requires_grad_(
            isinstance(fit.head, AdditiveBoostingHead)
        )
        features = build_pair_features(
            raw,
            text_leaf,
            zloglen,
            _window_context(transition, config),
            query_indices=queries,
        )
        if spec.application == "rank_band":
            active = rank_band_candidate_indices(full_order[start:stop])
        else:
            mask = torch.ones((stop - start, n_items), dtype=torch.bool)
            mask.scatter_(1, queries.unsqueeze(1), False)
            active = None

        if isinstance(fit.head, AdditiveBoostingHead):
            correction = fit.head.effective_correction(features)
            selected = (
                torch.gather(correction, 1, active)
                if active is not None
                else correction[mask]
            )
            gradient = torch.autograd.grad(selected.sum(), text_leaf)[0]
            gradient_rows = gradient.index_select(0, queries)
            selected_gradient = (
                torch.gather(gradient_rows, 1, active)
                if active is not None
                else gradient_rows[mask]
            )
            primary_values.append(selected.detach().cpu().numpy().reshape(-1))
            derivative_values.append(
                selected_gradient.detach().cpu().numpy().reshape(-1)
            )
        elif isinstance(fit.head, CenteredCrossModalHead):
            with torch.no_grad():
                gamma = fit.head.gamma(features)
                if active is not None:
                    selected_gamma = torch.gather(gamma, 1, active)
                    query_z = zloglen.index_select(0, queries).unsqueeze(1).expand_as(
                        selected_gamma
                    )
                    candidate_z = zloglen[active]
                else:
                    selected_gamma = gamma[mask]
                    query_grid = zloglen.index_select(0, queries).unsqueeze(1).expand_as(
                        gamma
                    )
                    candidate_grid = zloglen.unsqueeze(0).expand_as(gamma)
                    query_z = query_grid[mask]
                    candidate_z = candidate_grid[mask]
                primary_values.append(
                    selected_gamma.detach().cpu().numpy().reshape(-1)
                )
                query_context.append(query_z.detach().cpu().numpy().reshape(-1))
                candidate_context.append(
                    candidate_z.detach().cpu().numpy().reshape(-1)
                )
                for local, query in enumerate(range(start, stop)):
                    indices = torch.as_tensor(top_five[query], dtype=torch.long)
                    top_five_gamma.append(
                        gamma[local, indices].detach().cpu().numpy()
                    )
        else:
            raise TypeError("unsupported v2 head")

    values = np.concatenate(primary_values)
    result: dict[str, Any] = {
        "transition": transition.transition,
        "semantic_ablation_ndcg_change": float(semantic_ablation_ndcg_change),
    }
    if isinstance(fit.head, AdditiveBoostingHead):
        derivatives = np.concatenate(derivative_values)
        result.update(
            {
                "active_pair_correction": distribution(values),
                "max_abs_effective_correction": float(np.max(np.abs(values))),
                "positive_correction_fraction": float(np.mean(values > 0.0)),
                "score_derivative_cos_text": float(np.mean(np.abs(derivatives))),
            }
        )
    else:
        query_values = np.concatenate(query_context)
        candidate_values = np.concatenate(candidate_context)
        result.update(
            {
                "active_pair_gamma": distribution(values),
                "top_five_gamma": distribution(np.concatenate(top_five_gamma)),
                "associations": {
                    "zloglen_query": safe_pearson(values, query_values),
                    "zloglen_candidate": safe_pearson(values, candidate_values),
                },
                "fraction_gamma_below_zero": float(np.mean(values < 0.0)),
                "fraction_abs_gamma_near_zero": float(
                    np.mean(np.abs(values) < 0.01)
                ),
                "fraction_gamma_above_half": float(np.mean(values > 0.5)),
            }
        )
    return result


def aggregate_model_diagnostics(
    fit: FitResult,
    spec: FitSpec,
    fold: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Give two held-out transitions equal weight in one model record."""

    if len(rows) != 2:
        raise ValueError("each gate model must have exactly two diagnostic rows")
    base: dict[str, Any] = {
        "fold": int(fold),
        "seed": int(spec.seed),
        "target_softness": spec.target_softness,
        "application": spec.application,
        "learning_rate": float(spec.learning_rate),
        "semantic_ablation_ndcg_change": float(
            np.mean([float(row["semantic_ablation_ndcg_change"]) for row in rows])
        ),
    }
    if isinstance(fit.head, AdditiveBoostingHead):
        base.update(
            {
                "lambda": float(fit.head.lambda_.detach()),
                "active_pair_correction": average_distributions(
                    [row["active_pair_correction"] for row in rows]
                ),
                "max_abs_effective_correction": float(
                    np.mean([row["max_abs_effective_correction"] for row in rows])
                ),
                "positive_correction_fraction": float(
                    np.mean([row["positive_correction_fraction"] for row in rows])
                ),
                "score_derivative_cos_text": float(
                    np.mean([row["score_derivative_cos_text"] for row in rows])
                ),
            }
        )
    elif isinstance(fit.head, CenteredCrossModalHead):
        association: dict[str, float | None] = {}
        for key in ("zloglen_query", "zloglen_candidate"):
            values = [row["associations"][key] for row in rows]
            finite = [float(value) for value in values if value is not None]
            association[key] = float(np.mean(finite)) if finite else None
        base.update(
            {
                "w": [float(value) for value in fit.head.w.detach()],
                "b": float(fit.head.b.detach()),
                "active_pair_gamma": average_distributions(
                    [row["active_pair_gamma"] for row in rows]
                ),
                "top_five_gamma": average_distributions(
                    [row["top_five_gamma"] for row in rows]
                ),
                "associations": association,
                "fraction_gamma_below_zero": float(
                    np.mean([row["fraction_gamma_below_zero"] for row in rows])
                ),
                "fraction_abs_gamma_near_zero": float(
                    np.mean([row["fraction_abs_gamma_near_zero"] for row in rows])
                ),
                "fraction_gamma_above_half": float(
                    np.mean([row["fraction_gamma_above_half"] for row in rows])
                ),
            }
        )
    else:
        raise TypeError("unsupported v2 head")
    return base


def retrieval_diagnostic(
    *,
    transition: str,
    method: str,
    scores: np.ndarray,
    return_scores: np.ndarray,
    blend_scores: np.ndarray,
    lw_target: np.ndarray,
) -> dict[str, Any]:
    """Compute the fixed top-five and signed query-to-peer summaries."""

    model_top = _top_five(scores)
    return_top = _top_five(return_scores)
    blend_top = _top_five(blend_scores)
    raw_order = [
        top_k_indices(return_scores[q], return_scores.shape[0] - 1, excluded=q, query_seed=q)
        for q in range(return_scores.shape[0])
    ]
    overlaps_return: list[float] = []
    overlaps_blend: list[float] = []
    differs: list[bool] = []
    promoted_ranks: list[int] = []
    signed_means: list[float] = []
    for query, selected in enumerate(model_top):
        selected_set = set(int(value) for value in selected)
        return_set = set(int(value) for value in return_top[query])
        blend_set = set(int(value) for value in blend_top[query])
        overlaps_return.append(len(selected_set & return_set) / 5.0)
        overlaps_blend.append(len(selected_set & blend_set) / 5.0)
        differs.append(not np.array_equal(selected, return_top[query]))
        positions = {int(candidate): rank + 1 for rank, candidate in enumerate(raw_order[query])}
        promoted_ranks.extend(
            positions[candidate] for candidate in selected_set - return_set
        )
        signed_means.append(float(np.mean(lw_target[query, selected])))

    if promoted_ranks:
        promoted = np.asarray(promoted_ranks, dtype=np.float64)
        promoted_summary: dict[str, float | None] = {
            "p10": float(np.percentile(promoted, 10)),
            "median": float(np.percentile(promoted, 50)),
            "p90": float(np.percentile(promoted, 90)),
            "fraction_6_10": float(np.mean((promoted >= 6) & (promoted <= 10))),
            "fraction_11_20": float(np.mean((promoted >= 11) & (promoted <= 20))),
            "fraction_21_50": float(np.mean((promoted >= 21) & (promoted <= 50))),
            "fraction_above_50": float(np.mean(promoted > 50)),
        }
    else:
        promoted_summary = {
            "p10": None,
            "median": None,
            "p90": None,
            "fraction_6_10": 0.0,
            "fraction_11_20": 0.0,
            "fraction_21_50": 0.0,
            "fraction_above_50": 0.0,
        }
    signed = np.asarray(signed_means, dtype=np.float64)
    return {
        "transition": transition,
        "method": method,
        "top5_overlap_return_only": float(np.mean(overlaps_return)),
        "top5_overlap_frozen_blend": float(np.mean(overlaps_blend)),
        "fraction_queries_different_from_return": float(np.mean(differs)),
        "promoted_raw_rank": promoted_summary,
        "application_proxy": {
            "mean_signed_query_to_peer_correlation": float(signed.mean()),
            "fraction_queries_positive": float(np.mean(signed > 0.0)),
        },
    }


def average_retrieval_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average seed-specific scalar diagnostics for one method/transition."""

    if not rows:
        raise ValueError("cannot aggregate no retrieval diagnostics")
    first = rows[0]
    promoted_keys = (
        "p10",
        "median",
        "p90",
        "fraction_6_10",
        "fraction_11_20",
        "fraction_21_50",
        "fraction_above_50",
    )
    promoted: dict[str, float | None] = {}
    for key in promoted_keys:
        values = [row["promoted_raw_rank"][key] for row in rows]
        finite = [float(value) for value in values if value is not None]
        promoted[key] = float(np.mean(finite)) if finite else None
    return {
        "transition": first["transition"],
        "method": first["method"],
        "top5_overlap_return_only": float(
            np.mean([row["top5_overlap_return_only"] for row in rows])
        ),
        "top5_overlap_frozen_blend": float(
            np.mean([row["top5_overlap_frozen_blend"] for row in rows])
        ),
        "fraction_queries_different_from_return": float(
            np.mean([row["fraction_queries_different_from_return"] for row in rows])
        ),
        "promoted_raw_rank": promoted,
        "application_proxy": {
            "mean_signed_query_to_peer_correlation": float(
                np.mean(
                    [
                        row["application_proxy"][
                            "mean_signed_query_to_peer_correlation"
                        ]
                        for row in rows
                    ]
                )
            ),
            "fraction_queries_positive": float(
                np.mean(
                    [
                        row["application_proxy"]["fraction_queries_positive"]
                        for row in rows
                    ]
                )
            ),
        },
    }
