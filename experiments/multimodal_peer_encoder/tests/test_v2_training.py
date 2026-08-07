"""Synthetic tests for v2 fitting, scoring, and diagnostics."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from experiments.multimodal_peer_encoder.run_v2_historical import load_config
from experiments.multimodal_peer_encoder.v2_diagnostics import (
    aggregate_model_diagnostics,
    diagnose_head_transition,
    retrieval_diagnostic,
)
from experiments.multimodal_peer_encoder.v2_training import (
    FitSpec,
    evaluate_fit,
    fit_head,
    frozen_baselines,
)


@dataclass(frozen=True)
class SyntheticTransition:
    transition: str
    tickers: tuple[str, ...]
    raw_scores: np.ndarray
    text_scores: np.ndarray
    raw_target: np.ndarray
    lw_target: np.ndarray
    texts: tuple[str, ...]
    log_lengths: np.ndarray
    window_length: int


def _transition(index: int, n_items: int = 56) -> SyntheticTransition:
    generator = np.random.default_rng(9000 + index)
    raw_features = generator.normal(size=(n_items, 9)).astype(np.float32)
    text_features = generator.normal(size=(n_items, 7)).astype(np.float32)

    def cosine(values):
        normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
        return (normalized @ normalized.T).astype(np.float32)

    raw = cosine(raw_features)
    text = cosine(text_features)
    target = (0.7 * raw + 0.3 * text).astype(np.float32)
    np.fill_diagonal(target, 1.0)
    lw = target.copy()
    tickers = tuple(f"T{position:03d}" for position in range(n_items))
    texts = tuple(
        "business segment market product " + ("x" * (position + index + 5))
        for position in range(n_items)
    )
    return SyntheticTransition(
        transition=f"H{index}->H{index + 1}",
        tickers=tickers,
        raw_scores=raw,
        text_scores=text,
        raw_target=target,
        lw_target=lw,
        texts=texts,
        log_lengths=np.log1p([len(value) for value in texts]).astype(np.float32),
        window_length=80 + index,
    )


def _fast_config():
    config = copy.deepcopy(load_config())
    config["optimization"]["epochs"] = 2
    config["optimization"]["query_chunk_size"] = 5
    config["optimization"]["torch_num_threads"] = 1
    return config


def test_both_heads_fit_and_score_under_both_candidate_policies():
    config = _fast_config()
    transitions = tuple(_transition(index) for index in range(1, 4))
    for architecture, application in (("H-A", "global"), ("H-B", "rank_band")):
        spec = FitSpec(architecture, "lw", application, 0.003, 11)
        fit = fit_head(transitions[:2], spec, config, progress_prefix="test")
        evaluated = evaluate_fit(fit, spec, transitions[2], config)
        assert len(fit.epoch_mean_losses) == 2
        assert np.isfinite(fit.epoch_mean_losses).all()
        assert evaluated.scores.shape == transitions[2].raw_scores.shape
        assert 0.0 <= evaluated.ndcg_at_5 <= 1.0
        assert fit.trainable_parameters == (114 if architecture == "H-A" else 4)


def test_baselines_and_fixed_diagnostics_are_finite():
    config = _fast_config()
    transitions = tuple(_transition(index) for index in range(1, 4))
    baseline = frozen_baselines(transitions[2])
    assert set(
        (
            "return_only_ndcg_at_5",
            "tfidf_only_ndcg_at_5",
            "frozen_blend_ndcg_at_5",
        )
    ) <= set(baseline)

    for architecture in ("H-A", "H-B"):
        spec = FitSpec(architecture, "raw_scale", "global", 0.003, 29)
        fit = fit_head(transitions[:2], spec, config, progress_prefix="test")
        full = evaluate_fit(fit, spec, transitions[2], config)
        ablated = evaluate_fit(
            fit, spec, transitions[2], config, semantic_ablation=True
        )
        row = diagnose_head_transition(
            fit,
            spec,
            transitions[2],
            config,
            frozen_scores=full.scores,
            semantic_ablation_ndcg_change=full.ndcg_at_5 - ablated.ndcg_at_5,
        )
        aggregate = aggregate_model_diagnostics(fit, spec, 1, [row, row])
        assert aggregate["fold"] == 1
        retrieval = retrieval_diagnostic(
            transition=transitions[2].transition,
            method=architecture,
            scores=full.scores,
            return_scores=baseline["return_scores"],
            blend_scores=baseline["blend_scores"],
            lw_target=transitions[2].lw_target,
        )
        assert 0.0 <= retrieval["top5_overlap_return_only"] <= 1.0


def test_query_chunking_preserves_loss_and_update_for_both_policies():
    transitions = tuple(_transition(index) for index in range(1, 3))
    for architecture in ("H-A", "H-B"):
        for application in ("global", "rank_band"):
            full = _fast_config()
            full["optimization"]["query_chunk_size"] = 10_000
            chunked = copy.deepcopy(full)
            chunked["optimization"]["query_chunk_size"] = 7
            spec = FitSpec(architecture, "lw", application, 0.003, 11)
            full_fit = fit_head(transitions, spec, full, progress_prefix="chunk-full")
            chunked_fit = fit_head(
                transitions, spec, chunked, progress_prefix="chunk-small"
            )
            np.testing.assert_allclose(
                full_fit.epoch_mean_losses,
                chunked_fit.epoch_mean_losses,
                rtol=2e-7,
                atol=2e-7,
            )
            for name, full_value in full_fit.head.state_dict().items():
                np.testing.assert_allclose(
                    full_value.detach().numpy(),
                    chunked_fit.head.state_dict()[name].detach().numpy(),
                    rtol=3e-6,
                    atol=3e-7,
                    err_msg=f"chunk-dependent parameter: {name}",
                )


def test_different_top_five_order_counts_as_a_different_list():
    n_items = 6
    return_scores = np.broadcast_to(
        np.arange(n_items, dtype=np.float32), (n_items, n_items)
    ).copy()
    model_scores = -return_scores
    target = np.ones((n_items, n_items), dtype=np.float32)
    diagnostic = retrieval_diagnostic(
        transition="fixture",
        method="reordered",
        scores=model_scores,
        return_scores=return_scores,
        blend_scores=return_scores,
        lw_target=target,
    )
    assert diagnostic["top5_overlap_return_only"] == 1.0
    assert diagnostic["fraction_queries_different_from_return"] == 1.0
