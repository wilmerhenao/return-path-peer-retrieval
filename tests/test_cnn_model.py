"""Unit tests for the compact multimodal peer encoder."""

import torch

from experiments.multimodal_peer_encoder.model import (
    ResidualGatedMultimodalRetriever,
    count_trainable_parameters,
    listwise_softmax_loss,
)


def _batch(batch_size: int = 5, time_steps: int = 11):
    generator = torch.Generator().manual_seed(20260802)
    returns = torch.randn(batch_size, time_steps, generator=generator)
    text = torch.randn(batch_size, 768, generator=generator)
    lengths = torch.tensor([time_steps - i for i in range(batch_size)])
    return returns, text, lengths


def test_output_shapes_normalization_and_gate_ranges():
    model = ResidualGatedMultimodalRetriever()
    returns, text, lengths = _batch()

    output = model(returns, text, lengths=lengths)

    assert output.return_embedding.shape == (5, 64)
    assert output.text_embedding.shape == (5, 64)
    assert output.fused_embedding.shape == (5, 64)
    assert output.raw_return_embedding.shape == (5, 11)
    assert output.return_gate.shape == (5, 1)
    for embedding in (
        output.return_embedding,
        output.text_embedding,
        output.fused_embedding,
        output.raw_return_embedding,
    ):
        torch.testing.assert_close(
            torch.linalg.vector_norm(embedding, dim=1), torch.ones(5), atol=1e-5, rtol=1e-5
        )
    assert torch.all((output.return_gate >= 0.0) & (output.return_gate <= 1.0))
    assert 0.0 <= model.raw_skip_weight.item() <= 1.0


def test_masked_padding_values_and_extra_padding_do_not_change_scores():
    torch.manual_seed(11)
    model = ResidualGatedMultimodalRetriever().eval()
    returns = torch.randn(4, 7)
    text = torch.randn(4, 768)
    lengths = torch.tensor([7, 6, 4, 2])

    baseline = model.full_score_matrix(
        returns, text, lengths=lengths, exclude_self=False
    )
    padded = torch.cat((returns, torch.randn(4, 5) * 1000.0), dim=1)
    padded_scores = model.full_score_matrix(
        padded, text, lengths=lengths, exclude_self=False
    )

    # Values already inside the padded region should also be irrelevant.
    internal_padding_changed = padded.clone()
    position = torch.arange(padded.shape[1]).unsqueeze(0)
    invalid = position >= lengths.unsqueeze(1)
    internal_padding_changed[invalid] = torch.randn(invalid.sum()) * 1000.0
    changed_scores = model.full_score_matrix(
        internal_padding_changed, text, lengths=lengths, exclude_self=False
    )

    torch.testing.assert_close(padded_scores, baseline, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(changed_scores, baseline, atol=1e-6, rtol=1e-6)


def test_full_and_query_specific_lists_mask_self_matches():
    model = ResidualGatedMultimodalRetriever()
    returns, text, lengths = _batch(batch_size=5, time_steps=9)

    full = model.full_score_matrix(returns, text, lengths=lengths)
    assert torch.all(torch.isneginf(torch.diagonal(full)))
    assert torch.all(torch.isfinite(full[~torch.eye(5, dtype=torch.bool)]))

    queries = torch.tensor([0, 3])
    candidates = torch.tensor([[0, 1, 4], [2, 3, 1]])
    scores = model.listwise_scores(
        returns,
        text,
        query_indices=queries,
        candidate_indices=candidates,
        lengths=lengths,
    )
    assert scores.shape == (2, 3)
    assert torch.isneginf(scores[0, 0])
    assert torch.isneginf(scores[1, 1])
    assert torch.all(torch.isfinite(scores[torch.tensor([[False, True, True], [True, False, True]])]))


def test_listwise_loss_propagates_finite_gradients_through_all_components():
    torch.manual_seed(23)
    model = ResidualGatedMultimodalRetriever()
    returns, text, lengths = _batch(batch_size=5, time_steps=8)
    scores = model.full_score_matrix(returns, text, lengths=lengths)
    relevance = torch.randn_like(scores)
    relevance.fill_diagonal_(-torch.inf)

    loss = listwise_softmax_loss(scores, relevance)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.raw_skip_logit.grad is not None
    assert torch.isfinite(model.raw_skip_logit.grad)
    for parameter in model.return_tower.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))
    for parameter in model.text_projection.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))
    for parameter in model.modality_gate.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_forward_is_deterministic_and_frozen_text_input_is_detached():
    torch.manual_seed(31)
    model = ResidualGatedMultimodalRetriever().eval()
    returns, text, lengths = _batch(batch_size=4, time_steps=8)
    text.requires_grad_(True)

    first = model(returns, text, lengths=lengths)
    second = model(returns, text, lengths=lengths)

    torch.testing.assert_close(first.fused_embedding, second.fused_embedding)
    first.fused_embedding.sum().backward()
    assert text.grad is None
    assert model.text_projection[0].weight.grad is not None


def test_width_64_model_stays_below_parameter_budget():
    model = ResidualGatedMultimodalRetriever(
        embedding_dim=64,
        return_channels=64,
    )
    parameter_count = count_trainable_parameters(model)

    assert 60_000 < parameter_count < 100_000
