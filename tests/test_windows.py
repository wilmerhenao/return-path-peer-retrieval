"""Tests for the return-window construction contracts used in the paper."""

import unittest
from datetime import date

import numpy as np
import polars as pl

from scripts.build_windows import (
    WINDOWS,
    compute_ff5_residuals,
    simple_returns,
    validate_windows,
)


class WindowTests(unittest.TestCase):
    @staticmethod
    def _factor_frame(dates):
        return pl.DataFrame(
            {
                "date": dates,
                "mkt_rf": [0.0] * len(dates),
                "smb": [0.0] * len(dates),
                "hml": [0.0] * len(dates),
                "rmw": [0.0] * len(dates),
                "cma": [0.0] * len(dates),
                "rf": [0.0] * len(dates),
            }
        )

    def test_canonical_windows_are_strictly_disjoint(self):
        validate_windows(WINDOWS)

    def test_overlap_fails(self):
        with self.assertRaises(ValueError):
            validate_windows(
                [
                    ("A", date(2025, 1, 1), date(2025, 1, 31)),
                    ("B", date(2025, 1, 31), date(2025, 2, 28)),
                ]
            )

    def test_price_changes_are_simple_not_log_returns(self):
        prices = pl.DataFrame(
            {
                "date": [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)],
                "ABC": [100.0, 110.0, 99.0],
            }
        )
        returns = simple_returns(prices)
        self.assertEqual(returns["date"].to_list(), prices["date"].to_list()[1:])
        self.assertAlmostEqual(returns["ABC"][0], 0.10)
        self.assertAlmostEqual(returns["ABC"][1], -0.10)

    def test_residual_panel_excludes_incomplete_tickers(self):
        dates = [date(2025, 1, day) for day in range(2, 12)]
        returns = pl.DataFrame(
            {
                "date": dates,
                "AAA": [
                    0.01, -0.02, 0.03, 0.00, -0.01,
                    0.02, 0.04, -0.03, 0.01, 0.02,
                ],
                "BBB": [
                    -0.01, 0.01, 0.02, -0.02, 0.03,
                    0.01, -0.01, 0.04, 0.00, -0.03,
                ],
                "PARTIAL": [
                    0.01, -0.01, None, 0.02, 0.01,
                    0.00, -0.02, 0.03, 0.01, 0.02,
                ],
            }
        )
        result = compute_ff5_residuals(
            returns,
            self._factor_frame(dates),
            dates[0],
            dates[-1],
            min_days=5,
        )
        standardized, raw, tickers, _, correlation, counts = result

        self.assertEqual(tickers, ["AAA", "BBB"])
        self.assertEqual(standardized.shape, (2, len(dates)))
        self.assertEqual(raw.shape, (2, len(dates)))
        self.assertEqual(correlation.shape, (2, 2))
        np.testing.assert_array_equal(counts, [len(dates), len(dates)])
        self.assertTrue(np.isfinite(standardized).all())

    def test_empty_residual_panel_preserves_six_value_contract(self):
        dates = [date(2025, 1, day) for day in range(2, 7)]
        returns = pl.DataFrame({"date": dates})
        standardized, raw, tickers, shrinkage, correlation, counts = (
            compute_ff5_residuals(
                returns,
                self._factor_frame(dates),
                dates[0],
                dates[-1],
                min_days=2,
            )
        )

        self.assertEqual(standardized.shape, (0, len(dates)))
        self.assertEqual(raw.shape, (0, len(dates)))
        self.assertEqual(tickers, [])
        self.assertEqual(shrinkage, 0.0)
        self.assertEqual(correlation.shape, (0, 0))
        self.assertEqual(counts.shape, (0,))

    def test_non_complete_missing_policy_is_rejected(self):
        dates = [date(2025, 1, day) for day in range(2, 7)]
        with self.assertRaisesRegex(ValueError, "complete-case"):
            compute_ff5_residuals(
                pl.DataFrame({"date": dates}),
                self._factor_frame(dates),
                dates[0],
                dates[-1],
                min_days=2,
                missing_policy="observed-mean",
            )


if __name__ == "__main__":
    unittest.main()
