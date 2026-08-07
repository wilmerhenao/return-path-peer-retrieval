import unittest

import numpy as np

from scripts.metrics import ndcg_at_k, top_k_indices


class NdcgTests(unittest.TestCase):
    def test_oracle_is_one_and_excludes_query(self):
        realized = np.array([1.0, 0.9, 0.7, 0.3, -0.2, 0.1])
        value = ndcg_at_k(realized, realized, 0, k=3)
        self.assertAlmostEqual(value, 1.0)

    def test_negative_retrieved_relevance_is_clipped(self):
        realized = np.array([1.0, 0.8, 0.4, -0.9, -0.2])
        scores = np.array([99.0, 0.1, 0.0, 2.0, 1.0])
        value = ndcg_at_k(scores, realized, 0, k=2)
        self.assertEqual(value, 0.0)

    def test_no_positive_eligible_relevance_returns_none(self):
        realized = np.array([1.0, 0.0, -0.4])
        self.assertIsNone(ndcg_at_k(realized, realized, 0, k=2))

    def test_group_query_is_excluded_from_ideal_and_retrieved(self):
        realized = np.array([1.0, 0.95, 0.8, 0.6, 0.2])
        scores = realized.copy()
        value = ndcg_at_k(scores, realized, [0, 1], k=2)
        self.assertAlmostEqual(value, 1.0)

    def test_ties_are_deterministic(self):
        scores = np.ones(12)
        first = top_k_indices(scores, 5, excluded=3, query_seed=7)
        second = top_k_indices(scores, 5, excluded=3, query_seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertNotIn(3, first)


if __name__ == "__main__":
    unittest.main()
