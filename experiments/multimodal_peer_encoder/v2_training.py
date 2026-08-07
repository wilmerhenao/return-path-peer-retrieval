"""Pure fitting and scoring utilities for the sealed v2 historical run.

This module performs no file discovery and knows no modern path.  It accepts
already aligned historical transitions from :mod:`v2_data`, fits one of the
two tiny baseline-anchored heads, and returns dense query-conditional scores.
Every optimizer update averages the census query losses from one transition,
even when query chunks are used to bound memory.
"""

from __future__ import annotations

import platform
import random
import resource
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch import Tensor, nn

from experiments.multimodal_peer_encoder.evaluation import (
    census_ndcg_at_5,
    frozen_rank_blend,
)
from experiments.multimodal_peer_encoder.model import listwise_softmax_loss
from experiments.multimodal_peer_encoder.v2_model import (
    AdditiveBoostingHead,
    CenteredCrossModalHead,
    apply_application_policy,
    build_pair_features,
    rank_band_candidate_indices,
    raw_candidate_order,
)


Architecture = Literal["H-A", "H-B"]
TargetSoftness = Literal["raw_scale", "lw"]
Application = Literal["global", "rank_band"]


TFIDF_CONFIG_B: dict[str, Any] = {
    "max_features": 50_000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "min_df": 2,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}


class TransitionLike(Protocol):
    """Structural data contract consumed by the pure trainer."""

    transition: str
    tickers: tuple[str, ...]
    raw_scores: np.ndarray
    text_scores: np.ndarray
    raw_target: np.ndarray
    lw_target: np.ndarray
    texts: tuple[str, ...]
    log_lengths: np.ndarray
    window_length: int


@dataclass(frozen=True)
class FitSpec:
    architecture: Architecture
    target_softness: TargetSoftness
    application: Application
    learning_rate: float
    seed: int

    @property
    def label(self) -> str:
        return (
            f"{self.architecture}|{self.target_softness}|{self.application}|"
            f"lr={self.learning_rate:g}"
        )


@dataclass(frozen=True)
class LengthScaler:
    mean: float
    population_std: float
    n_records: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("log filing lengths must be one-dimensional and finite")
        result = (array - self.mean) / self.population_std
        return result.astype(np.float32)


@dataclass
class FitResult:
    head: nn.Module
    scaler: LengthScaler
    epoch_mean_losses: tuple[float, ...]
    final_transition_losses: dict[str, float]
    wall_seconds: float
    peak_rss_mb: float
    trainable_parameters: int


@dataclass(frozen=True)
class ScoreResult:
    scores: np.ndarray
    ndcg_at_5: float
    wall_seconds: float


def fit_length_scaler(transitions: Sequence[TransitionLike]) -> LengthScaler:
    if not transitions:
        raise ValueError("at least one training transition is required")
    pooled = np.concatenate(
        [np.asarray(item.log_lengths, dtype=np.float64) for item in transitions]
    )
    if pooled.ndim != 1 or pooled.size == 0 or not np.isfinite(pooled).all():
        raise ValueError("training filing lengths are empty or non-finite")
    population_std = float(pooled.std(ddof=0))
    if population_std <= 0.0:
        raise ValueError("training filing-length standard deviation must be positive")
    return LengthScaler(
        mean=float(pooled.mean()),
        population_std=population_std,
        n_records=int(pooled.size),
    )


def make_head(architecture: Architecture) -> nn.Module:
    if architecture == "H-A":
        return AdditiveBoostingHead()
    if architecture == "H-B":
        return CenteredCrossModalHead()
    raise ValueError("architecture must be 'H-A' or 'H-B'")


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters() if value.requires_grad)


def _window_context(transition: TransitionLike, config: dict[str, Any]) -> float:
    reporting = config["reporting"]
    return (
        float(transition.window_length) - float(reporting["context_window_center"])
    ) / float(reporting["context_window_scale"])


def _target(transition: TransitionLike, target_softness: TargetSoftness) -> np.ndarray:
    if target_softness == "raw_scale":
        result = transition.raw_target
    elif target_softness == "lw":
        result = transition.lw_target
    else:
        raise ValueError("target softness must be 'raw_scale' or 'lw'")
    array = np.asarray(result, dtype=np.float32)
    if array.shape != transition.raw_scores.shape or not np.isfinite(array).all():
        raise ValueError(f"{transition.transition}: invalid training target")
    return array


def _as_feature_tensors(
    transition: TransitionLike,
    scaler: LengthScaler,
) -> tuple[Tensor, Tensor, Tensor]:
    raw = torch.tensor(transition.raw_scores, dtype=torch.float32)
    text = torch.tensor(transition.text_scores, dtype=torch.float32)
    zloglen = torch.as_tensor(
        scaler.transform(transition.log_lengths), dtype=torch.float32
    )
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or text.shape != raw.shape:
        raise ValueError(f"{transition.transition}: invalid feature score shape")
    if len(transition.tickers) != raw.shape[0]:
        raise ValueError(f"{transition.transition}: ticker/score shape mismatch")
    return raw, text, zloglen


def _peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return raw / divisor


def fit_head(
    transitions: Sequence[TransitionLike],
    spec: FitSpec,
    config: dict[str, Any],
    *,
    progress_prefix: str = "fit",
) -> FitResult:
    """Fit one fresh head for exactly the declared forty epochs."""

    if not transitions:
        raise ValueError("cannot fit without transitions")
    optimization = config["optimization"]
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    torch.set_num_threads(int(optimization["torch_num_threads"]))
    torch.use_deterministic_algorithms(bool(optimization["deterministic_algorithms"]))

    head = make_head(spec.architecture).to(dtype=torch.float32, device="cpu")
    scaler = fit_length_scaler(transitions)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(spec.learning_rate),
        weight_decay=float(optimization["weight_decay"]),
    )
    chunk_size = int(optimization["query_chunk_size"])
    if chunk_size < 1:
        raise ValueError("query chunk size must be positive")

    prepared: list[dict[str, Any]] = []
    for transition in transitions:
        raw, text, zloglen = _as_feature_tensors(transition, scaler)
        n_items = raw.shape[0]
        queries = torch.arange(n_items, dtype=torch.long)
        order = (
            raw_candidate_order(raw, queries)
            if spec.application == "rank_band"
            else None
        )
        prepared.append(
            {
                "transition": transition,
                "raw": raw,
                "text": text,
                "zloglen": zloglen,
                "target": torch.as_tensor(
                    np.maximum(_target(transition, spec.target_softness), 0.0),
                    dtype=torch.float32,
                ),
                "queries": queries,
                "raw_order": order,
                "window_context": _window_context(transition, config),
            }
        )

    order_rng = random.Random(spec.seed)
    n_epochs = int(optimization["epochs"])
    epoch_means: list[float] = []
    final_transition_losses: dict[str, float] = {}
    started = time.perf_counter()
    for epoch_index in range(n_epochs):
        transition_order = list(range(len(prepared)))
        order_rng.shuffle(transition_order)
        transition_losses: list[float] = []
        for transition_index in transition_order:
            item = prepared[transition_index]
            n_queries = int(item["queries"].numel())
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for start in range(0, n_queries, chunk_size):
                stop = min(start + chunk_size, n_queries)
                query_indices = item["queries"][start:stop]
                if spec.application == "rank_band":
                    candidate_indices = rank_band_candidate_indices(
                        item["raw_order"][start:stop]
                    )
                    features = build_pair_features(
                        item["raw"],
                        item["text"],
                        item["zloglen"],
                        item["window_context"],
                        query_indices=query_indices,
                        candidate_indices=candidate_indices,
                    )
                    scores = head.score(features)
                    relevance = torch.gather(
                        item["target"].index_select(0, query_indices),
                        1,
                        candidate_indices,
                    )
                    candidate_mask = None
                else:
                    features = build_pair_features(
                        item["raw"],
                        item["text"],
                        item["zloglen"],
                        item["window_context"],
                        query_indices=query_indices,
                    )
                    scores = head.score(features)
                    relevance = item["target"].index_select(0, query_indices)
                    candidate_mask = torch.ones_like(scores, dtype=torch.bool)
                    candidate_mask.scatter_(1, query_indices.unsqueeze(1), False)

                chunk_loss_sum = listwise_softmax_loss(
                    scores,
                    relevance,
                    candidate_mask=candidate_mask,
                    temperature=float(optimization["score_temperature"]),
                    target_temperature=float(optimization["target_temperature"]),
                    reduction="sum",
                )
                normalized = chunk_loss_sum / float(n_queries)
                normalized.backward()
                loss_sum += float(chunk_loss_sum.detach())

            torch.nn.utils.clip_grad_norm_(
                head.parameters(), float(optimization["gradient_norm_clip"])
            )
            optimizer.step()
            transition_loss = loss_sum / float(n_queries)
            transition_losses.append(transition_loss)
            final_transition_losses[item["transition"].transition] = transition_loss

        epoch_mean = float(np.mean(transition_losses))
        epoch_means.append(epoch_mean)
        completed = epoch_index + 1
        if completed == 1 or completed % 10 == 0:
            print(
                f"[{progress_prefix}] {spec.label} seed={spec.seed} "
                f"epoch={completed}/{n_epochs} mean_loss={epoch_mean:.8f}",
                flush=True,
            )

    if len(epoch_means) != n_epochs or not np.isfinite(epoch_means).all():
        raise RuntimeError("fit did not produce the declared finite epoch trace")
    return FitResult(
        head=head,
        scaler=scaler,
        epoch_mean_losses=tuple(epoch_means),
        final_transition_losses=final_transition_losses,
        wall_seconds=float(time.perf_counter() - started),
        peak_rss_mb=_peak_rss_mb(),
        trainable_parameters=count_trainable_parameters(head),
    )


def _semantic_mean_matrix(text_scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(text_scores, dtype=np.float32)
    n_items = matrix.shape[0]
    if matrix.shape != (n_items, n_items) or n_items < 2:
        raise ValueError("text score matrix must be nontrivial and square")
    mask = ~np.eye(n_items, dtype=bool)
    means = (matrix * mask).sum(axis=1) / float(n_items - 1)
    return np.broadcast_to(means[:, None], matrix.shape).copy().astype(np.float32)


def score_matrix(
    fit: FitResult,
    spec: FitSpec,
    transition: TransitionLike,
    config: dict[str, Any],
    *,
    semantic_ablation: bool = False,
) -> tuple[np.ndarray, float]:
    """Score one transition with the frozen fit and application policy."""

    raw, text, zloglen = _as_feature_tensors(transition, fit.scaler)
    if semantic_ablation:
        text = torch.as_tensor(_semantic_mean_matrix(text.numpy()), dtype=torch.float32)
    n_items = raw.shape[0]
    chunk_size = int(config["optimization"]["query_chunk_size"])
    blocks: list[Tensor] = []
    fit.head.eval()
    started = time.perf_counter()
    with torch.no_grad():
        full_order = (
            raw_candidate_order(raw)
            if spec.application == "rank_band"
            else None
        )
        for start in range(0, n_items, chunk_size):
            stop = min(start + chunk_size, n_items)
            queries = torch.arange(start, stop, dtype=torch.long)
            features = build_pair_features(
                raw,
                text,
                zloglen,
                _window_context(transition, config),
                query_indices=queries,
            )
            corrected = fit.head.score(features)
            applied = apply_application_policy(
                raw,
                corrected,
                queries,
                policy="rank-band" if spec.application == "rank_band" else "global",
                raw_order=None if full_order is None else full_order[start:stop],
            )
            blocks.append(applied.cpu())
    result = torch.cat(blocks, dim=0).numpy()
    if result.shape != (n_items, n_items) or not np.isfinite(result).all():
        raise RuntimeError("scoring produced an invalid census matrix")
    return result, float(time.perf_counter() - started)


def evaluate_fit(
    fit: FitResult,
    spec: FitSpec,
    transition: TransitionLike,
    config: dict[str, Any],
    *,
    semantic_ablation: bool = False,
) -> ScoreResult:
    scores, wall_seconds = score_matrix(
        fit,
        spec,
        transition,
        config,
        semantic_ablation=semantic_ablation,
    )
    metric = census_ndcg_at_5(
        scores,
        transition.lw_target,
        require_symmetric_scores=False,
    )
    return ScoreResult(
        scores=scores,
        ndcg_at_5=float(metric["mean_ndcg"]),
        wall_seconds=wall_seconds,
    )


def frozen_baselines(transition: TransitionLike) -> dict[str, Any]:
    """Recompute return, config-B TF-IDF, and the frozen rank blend."""

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG_B)
    sparse = normalize(vectorizer.fit_transform(transition.texts), norm="l2")
    tfidf_scores = (sparse @ sparse.T).toarray().astype(np.float32)
    blend_scores = frozen_rank_blend(transition.raw_scores, tfidf_scores)
    return_result = census_ndcg_at_5(transition.raw_scores, transition.lw_target)
    text_result = census_ndcg_at_5(tfidf_scores, transition.lw_target)
    blend_result = census_ndcg_at_5(
        blend_scores,
        transition.lw_target,
        require_symmetric_scores=False,
    )
    return {
        "transition": transition.transition,
        "n": len(transition.tickers),
        "return_scores": np.asarray(transition.raw_scores, dtype=np.float32),
        "tfidf_scores": tfidf_scores,
        "blend_scores": blend_scores,
        "return_only_ndcg_at_5": float(return_result["mean_ndcg"]),
        "tfidf_only_ndcg_at_5": float(text_result["mean_ndcg"]),
        "frozen_blend_ndcg_at_5": float(blend_result["mean_ndcg"]),
    }
