import unittest

import numpy as np
from sklearn.covariance import LedoitWolf


class ReturnPathGeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        factors = rng.normal(size=(3, 40))
        loadings = rng.normal(size=(12, 3))
        raw = loadings @ factors + 0.35 * rng.normal(size=(12, 40))
        self.z = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(
            axis=1, keepdims=True
        )

    def test_scaled_inner_product_equals_cosine_and_sample_correlation(self):
        z = self.z
        t = z.shape[1]
        scaled_ip = z @ z.T / t
        cosine = (z / np.linalg.norm(z, axis=1, keepdims=True)) @ (
            z / np.linalg.norm(z, axis=1, keepdims=True)
        ).T
        sample_corr = np.corrcoef(z)
        np.testing.assert_allclose(scaled_ip, cosine, atol=1e-12)
        np.testing.assert_allclose(scaled_ip, sample_corr, atol=1e-12)

    def test_ledoit_wolf_preserves_off_diagonal_ranking(self):
        z = self.z
        sample_corr = z @ z.T / z.shape[1]
        estimator = LedoitWolf().fit(z.T)
        covariance = estimator.covariance_
        scale = np.sqrt(np.diag(covariance))
        lw_corr = covariance / np.outer(scale, scale)

        self.assertLess(estimator.shrinkage_, 1.0)
        upper = np.triu_indices(len(z), k=1)
        expected = (1.0 - estimator.shrinkage_) * sample_corr[upper]
        np.testing.assert_allclose(lw_corr[upper], expected, atol=1e-12)
        np.testing.assert_array_equal(
            np.argsort(sample_corr[upper]),
            np.argsort(lw_corr[upper]),
        )


if __name__ == "__main__":
    unittest.main()
