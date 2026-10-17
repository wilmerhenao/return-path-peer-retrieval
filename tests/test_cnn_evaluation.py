"""Tests for the isolated multimodal evaluation utilities."""

import numpy as np
import pytest

from experiments.multimodal_peer_encoder.evaluation import (
    FROZEN_RETURN_WEIGHT,
    FROZEN_TFIDF_WEIGHT,
    aggregate_transition_ndcg,
    average_percentile_rank_matrix,
    census_ndcg_at_5,
    frozen_rank_blend,
    row_l2_cosine,
    validate_score_matrix,
)


def test_cosine_oracle_has_unit_diagonal_and_perfect_census_ndcg():
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.6, 0.0, 0.8],
            [0.0, 0.6, 0.8],
        ]
    )
    oracle = row_l2_cosine(features)
    result = census_ndcg_at_5(oracle, oracle)

    np.testing.assert_allclose(np.diag(oracle), 1.0, atol=1e-6)
    np.testing.assert_allclose(oracle, oracle.T, atol=1e-7)
    assert result["n_queries"] == 4
    assert result["n_valid"] == 4
    assert result["mean_ndcg"] == pytest.approx(1.0)
    assert all(value == pytest.approx(1.0) for value in result["query_ndcg"])


def test_average_percentile_ranks_exclude_self_and_average_exact_ties():
    scores = np.array(
        [
            [99.0, 1.0, 1.0, 3.0],
            [1.0, 99.0, 2.0, 2.0],
            [1.0, 2.0, 99.0, 0.0],
            [3.0, 2.0, 0.0, 99.0],
        ]
    )
    ranks = average_percentile_rank_matrix(scores)

    np.testing.assert_array_equal(np.diag(ranks), np.zeros(4))
    np.testing.assert_allclose(ranks[0], [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(ranks[1], [1.0 / 3.0, 0.0, 5.0 / 6.0, 5.0 / 6.0])
    # Changing only the self score cannot change any eligible rank.
    changed_self = scores.copy()
    np.fill_diagonal(changed_self, -12345.0)
    np.testing.assert_array_equal(
        average_percentile_rank_matrix(changed_self),
        ranks,
    )


def test_frozen_blend_is_exactly_eighty_twenty_rank_formula():
    return_scores = np.array(
        [
            [1.0, 0.9, 0.4, 0.2],
            [0.9, 1.0, 0.3, 0.1],
            [0.4, 0.3, 1.0, 0.8],
            [0.2, 0.1, 0.8, 1.0],
        ]
    )
    tfidf_scores = np.array(
        [
            [1.0, 0.2, 0.7, 0.6],
            [0.2, 1.0, 0.8, 0.5],
            [0.7, 0.8, 1.0, 0.1],
            [0.6, 0.5, 0.1, 1.0],
        ]
    )
    expected = (
        FROZEN_RETURN_WEIGHT * average_percentile_rank_matrix(return_scores)
        + FROZEN_TFIDF_WEIGHT * average_percentile_rank_matrix(tfidf_scores)
    )

    np.testing.assert_array_equal(
        frozen_rank_blend(return_scores, tfidf_scores),
        expected.astype(np.float32),
    )


def test_transition_aggregation_weights_transition_means_equally():
    result = aggregate_transition_ndcg(
        [
            {"transition": "H1->H2", "mean_ndcg": 0.2, "n_queries": 10},
            {"transition": "H2->H3", "mean_ndcg": 0.8, "n_queries": 10_000},
        ]
    )

    assert result["n_transitions"] == 2
    assert result["macro_mean_ndcg"] == pytest.approx(0.5)
    assert result["population_std_across_transition_means"] == pytest.approx(0.3)
    assert result["transitions"] == ("H1->H2", "H2->H3")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: row_l2_cosine(np.array([[0.0, 0.0], [1.0, 0.0]])), "zero-norm"),
        (lambda: validate_score_matrix(np.ones((2, 3))), "square"),
        (
            lambda: census_ndcg_at_5(
                np.array([[1.0, 0.7], [0.2, 1.0]]),
                np.array([[1.0, 0.5], [0.5, 1.0]]),
            ),
            "symmetric",
        ),
        (
            lambda: census_ndcg_at_5(
                np.array([[np.inf, 0.5], [0.5, 1.0]]),
                np.array([[1.0, 0.5], [0.5, 1.0]]),
            ),
            "finite",
        ),
        (
            lambda: frozen_rank_blend(np.eye(3), np.eye(4)),
            "equal shape",
        ),
        (
            lambda: aggregate_transition_ndcg(
                [
                    {"transition": "H1->H2", "mean_ndcg": 0.2},
                    {"transition": "H1->H2", "mean_ndcg": 0.3},
                ]
            ),
            "unique",
        ),
    ],
)
def test_validation_errors(call, match):
    with pytest.raises((TypeError, ValueError), match=match):
        call()


def test_rank_blend_can_be_evaluated_only_with_explicit_asymmetry_opt_out():
    realized = np.array(
        [
            [1.0, 0.7, 0.3, 0.1],
            [0.7, 1.0, 0.2, 0.4],
            [0.3, 0.2, 1.0, 0.8],
            [0.1, 0.4, 0.8, 1.0],
        ]
    )
    # A finite query-specific score matrix is valid for a row-rank baseline,
    # but learned score evaluation retains the stricter symmetric default.
    row_rank_scores = np.array(
        [
            [0.0, 1.0, 0.7, 0.3],
            [0.4, 0.0, 1.0, 0.7],
            [0.7, 0.4, 0.0, 1.0],
            [1.0, 0.7, 0.4, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="symmetric"):
        census_ndcg_at_5(row_rank_scores, realized)

    result = census_ndcg_at_5(
        row_rank_scores,
        realized,
        require_symmetric_scores=False,
    )
    assert result["n_valid"] == 4
