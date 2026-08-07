"""Synthetic tests for the sealed v2 baseline-anchored heads."""

from __future__ import annotations

import numpy as np
import torch

from experiments.multimodal_peer_encoder.v2_model import (
    AdditiveBoostingHead,
    CenteredCrossModalHead,
    active_rank_band_mask,
    apply_application_policy,
    build_pair_features,
    encode_integer_rank_scores,
    gather_active_rank_band,
    rank_band_candidate_indices,
    raw_candidate_order,
    splice_rank_band_order,
)
from scripts.metrics import top_k_indices


def _fixture(n_items: int = 8):
    generator = torch.Generator().manual_seed(20260802)
    raw_vectors = torch.randn(n_items, 7, generator=generator)
    text_vectors = torch.randn(n_items, 5, generator=generator)
    raw_vectors = torch.nn.functional.normalize(raw_vectors, dim=1)
    text_vectors = torch.nn.functional.normalize(text_vectors, dim=1)
    cos_raw = raw_vectors @ raw_vectors.T
    cos_text = text_vectors @ text_vectors.T
    zloglen = torch.linspace(-1.5, 1.5, n_items, dtype=torch.float32)
    return cos_raw, cos_text, zloglen


def test_pair_features_are_exact_and_query_chunk_compatible():
    cos_raw, cos_text, zloglen = _fixture()
    full = build_pair_features(cos_raw, cos_text, zloglen, -0.2)
    queries = torch.tensor([1, 5])
    chunk = build_pair_features(
        cos_raw,
        cos_text,
        zloglen,
        -0.2,
        query_indices=queries,
    )

    assert full.phi.shape == (8, 8, 5)
    assert full.psi.shape == (8, 8, 3)
    torch.testing.assert_close(chunk.cos_raw, full.cos_raw[queries])
    torch.testing.assert_close(chunk.cos_text, full.cos_text[queries])
    torch.testing.assert_close(chunk.phi, full.phi[queries])
    torch.testing.assert_close(chunk.psi, full.psi[queries])
    torch.testing.assert_close(chunk.phi[..., 0], chunk.cos_raw)
    torch.testing.assert_close(chunk.phi[..., 1], chunk.cos_text)
    torch.testing.assert_close(
        chunk.phi[..., 2], zloglen[queries].unsqueeze(1).expand(2, 8)
    )
    torch.testing.assert_close(
        chunk.phi[..., 3], zloglen.unsqueeze(0).expand(2, 8)
    )
    torch.testing.assert_close(chunk.phi[..., 4], torch.full((2, 8), -0.2))

    query_specific = torch.tensor([[0, 3, 7], [2, 4, 6]])
    band = build_pair_features(
        cos_raw,
        cos_text,
        zloglen,
        -0.2,
        query_indices=queries,
        candidate_indices=query_specific,
    )
    assert band.shape == (2, 3)
    torch.testing.assert_close(
        band.cos_raw, cos_raw[queries.unsqueeze(1), query_specific]
    )


def test_both_heads_equal_raw_scores_bitwise_at_initialization_for_chunks():
    torch.manual_seed(11)
    cos_raw, cos_text, zloglen = _fixture()
    queries = torch.tensor([0, 2, 6])
    features = build_pair_features(
        cos_raw,
        cos_text,
        zloglen,
        0.1,
        query_indices=queries,
    )

    additive = AdditiveBoostingHead()
    centered = CenteredCrossModalHead()
    assert torch.equal(additive.score(features), features.cos_raw)
    assert torch.equal(centered.gamma(features), torch.zeros_like(features.cos_raw))
    assert torch.equal(centered.score(features), features.cos_raw)


def test_nonzero_heads_score_query_chunks_identically_to_full_candidate_matrix():
    torch.manual_seed(29)
    cos_raw, cos_text, zloglen = _fixture()
    full_features = build_pair_features(cos_raw, cos_text, zloglen, 0.1)
    queries = torch.tensor([0, 2, 6])
    chunk_features = build_pair_features(
        cos_raw,
        cos_text,
        zloglen,
        0.1,
        query_indices=queries,
    )
    additive = AdditiveBoostingHead()
    centered = CenteredCrossModalHead()
    with torch.no_grad():
        additive.lambda_.fill_(0.35)
        centered.w.copy_(torch.tensor([0.3, -0.2, 0.4]))
        centered.b.fill_(-3.6)

    torch.testing.assert_close(
        additive.score(chunk_features), additive.score(full_features)[queries]
    )
    torch.testing.assert_close(
        centered.score(chunk_features), centered.score(full_features)[queries]
    )


def test_raw_order_delegates_exactly_to_canonical_query_seeded_tie_breaking():
    cos_raw = torch.tensor(
        [
            [1.0, 0.5, 0.5, 0.2, 0.2],
            [0.5, 1.0, 0.4, 0.4, 0.1],
            [0.5, 0.4, 1.0, 0.3, 0.3],
            [0.2, 0.4, 0.3, 1.0, 0.6],
            [0.2, 0.1, 0.3, 0.6, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = torch.tensor([0, 3])
    observed = raw_candidate_order(cos_raw, queries)
    expected = torch.tensor(
        [
            top_k_indices(
                cos_raw[q].numpy(), 4, excluded=q, query_seed=q
            ).tolist()
            for q in queries.tolist()
        ]
    )
    assert torch.equal(observed.cpu(), expected)


def test_rank_band_splice_preserves_boundaries_ties_and_integer_encoding():
    n_items = 8
    query = torch.tensor([0])
    raw_order = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    corrected = torch.zeros((1, n_items), dtype=torch.float32)
    # The active band is raw positions 3--6.  Candidates 3 and 4 tie, so their
    # earlier raw order must survive after candidate 5 is promoted.
    corrected[0, 3] = 0.7
    corrected[0, 4] = 0.7
    corrected[0, 5] = 0.9
    corrected[0, 6] = 0.1

    active = rank_band_candidate_indices(raw_order, start=3, end=6)
    assert torch.equal(active, torch.tensor([[3, 4, 5, 6]]))
    final_order = splice_rank_band_order(
        raw_order, corrected, query, start=3, end=6
    )
    assert torch.equal(final_order, torch.tensor([[1, 2, 5, 3, 4, 6, 7]]))

    encoded = encode_integer_rank_scores(final_order, query)
    assert encoded.dtype == torch.int64
    assert torch.equal(
        encoded,
        torch.tensor([[-8, 7, 6, 4, 3, 5, 2, 1]], dtype=torch.int64),
    )

    cos_raw = torch.eye(n_items, dtype=torch.float32)
    policy_scores = apply_application_policy(
        cos_raw,
        corrected,
        query,
        policy="rank-band",
        raw_order=raw_order,
        start=3,
        end=6,
    )
    assert torch.equal(policy_scores, encoded)
    assert torch.equal(
        apply_application_policy(
            cos_raw, corrected, query, policy="global"
        ),
        corrected,
    )


def test_active_rank_band_gather_is_differentiable_only_for_fixed_band():
    raw_order = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    query = torch.tensor([0])
    mask = active_rank_band_mask(
        raw_order, query, start=3, end=6
    )
    assert torch.equal(
        mask,
        torch.tensor(
            [[False, False, False, True, True, True, True, False]]
        ),
    )

    values = torch.arange(8, dtype=torch.float32).unsqueeze(0).requires_grad_()
    gathered = gather_active_rank_band(values, raw_order, start=3, end=6)
    gathered.square().sum().backward()
    expected_gradient = torch.zeros_like(values)
    expected_gradient[mask] = 2.0 * values.detach()[mask]
    torch.testing.assert_close(values.grad, expected_gradient)


def test_additive_head_first_step_moves_lambda_and_second_step_reaches_mlp():
    torch.manual_seed(11)
    cos_raw, cos_text, zloglen = _fixture(n_items=6)
    features = build_pair_features(cos_raw, cos_text, zloglen, 0.0)
    target = 0.6 * cos_text - 0.25 * cos_raw
    head = AdditiveBoostingHead()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.2)
    initial_scores = head.score(features).detach().clone()

    optimizer.zero_grad(set_to_none=True)
    first_loss = (head.score(features) - target).square().mean()
    first_loss.backward()
    assert head.lambda_.grad is not None
    assert torch.isfinite(head.lambda_.grad)
    assert head.lambda_.grad.abs().item() > 0.0
    first_step_mlp_grads = [parameter.grad for parameter in head.mlp.parameters()]
    assert all(gradient is not None for gradient in first_step_mlp_grads)
    assert all(torch.count_nonzero(gradient).item() == 0 for gradient in first_step_mlp_grads)
    optimizer.step()
    assert not torch.equal(head.score(features).detach(), initial_scores)

    optimizer.zero_grad(set_to_none=True)
    second_loss = (head.score(features) - target).square().mean()
    second_loss.backward()
    for parameter in head.mlp.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert sum(
        torch.count_nonzero(parameter.grad).item()
        for parameter in head.mlp.parameters()
    ) > 0


def test_centered_head_one_step_has_nonzero_w_and_b_gradients_and_changes_scores():
    cos_raw, cos_text, zloglen = _fixture(n_items=6)
    features = build_pair_features(cos_raw, cos_text, zloglen, 0.2)
    target = 0.75 * cos_text + 0.1 * cos_raw
    head = CenteredCrossModalHead()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.3)
    initial_scores = head.score(features).detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = (head.score(features) - target).square().mean()
    loss.backward()
    for parameter in (head.w, head.b):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm().item() > 0.0
    optimizer.step()

    assert not torch.equal(head.score(features).detach(), initial_scores)
    assert head.gamma(features).min().item() > -0.0181
    assert head.gamma(features).max().item() < 0.9821


def test_invalid_or_incomplete_candidate_order_is_rejected():
    duplicate = torch.tensor([[1, 2, 2, 4]])
    with np.testing.assert_raises_regex(ValueError, "exactly once"):
        encode_integer_rank_scores(duplicate, torch.tensor([0]), n_candidates=5)
