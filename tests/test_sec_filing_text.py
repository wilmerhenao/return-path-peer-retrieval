"""Tests for deterministic SEC filing acquisition without network calls."""

from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import polars as pl

from scripts.build_windows import HISTORICAL_WINDOWS, MODERN_WINDOWS
from scripts.fetch_vintage_10k import WINDOW_STARTS, write_window
from scripts.sec_filing_text import MAX_ITEM1_CHARS, SecClient, extract_item1


class JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FlakyFilingClient:
    def __init__(self):
        self.calls = 0

    def most_recent_annual_filing(self, cik, cutoff):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("simulated transient failure")
        return None


class ItemOneExtractionTests(unittest.TestCase):
    def test_filing_cutoffs_match_return_window_starts(self):
        expected = {
            name: start
            for name, start, _ in HISTORICAL_WINDOWS + MODERN_WINDOWS
        }
        self.assertEqual(WINDOW_STARTS, expected)

    def test_table_of_contents_entry_is_not_selected(self):
        body = "The company designs instruments for research laboratories. " * 30
        filing = (
            "<html><body>Item 1. Business 3 Item 1A. Risk Factors 9 "
            f"Item 1. Business {body} Item 1A. Risk Factors"
            "</body></html>"
        )
        result = extract_item1(filing)
        self.assertIsNotNone(result)
        self.assertIn("designs instruments", result)
        self.assertGreater(len(result), 500)

    def test_collection_cap_is_applied(self):
        filing = "Item 1. Business " + ("operations " * 2000) + " Item 1A"
        result = extract_item1(filing)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), MAX_ITEM1_CHARS)

    def test_missing_item_returns_none(self):
        self.assertIsNone(extract_item1("Annual report without a business header."))

    def test_company_submission_metadata_is_reused_across_cutoffs(self):
        client = SecClient("Artifact audit anonymous@example.com")
        calls = []
        payloads = {
            "https://data.sec.gov/submissions/CIK0000000123.json": {
                "filings": {
                    "recent": {
                        "form": ["10-K", "10-K/A"],
                        "filingDate": ["2024-02-01", "2023-02-01"],
                        "accessionNumber": ["0001-24-000001", "0001-23-000001"],
                        "primaryDocument": ["annual.htm", "amendment.htm"],
                    },
                    "files": [{"name": "CIK0000000123-submissions-001.json"}],
                }
            },
            "https://data.sec.gov/submissions/CIK0000000123-submissions-001.json": {
                "form": ["10-K"],
                "filingDate": ["2020-02-03"],
                "accessionNumber": ["0001-20-000001"],
                "primaryDocument": ["annual2020.htm"],
            },
        }

        def fake_get(url):
            calls.append(url)
            return JsonResponse(payloads[url])

        client.get = fake_get
        recent = client.most_recent_annual_filing(123, date(2025, 1, 1))
        older = client.most_recent_annual_filing(123, date(2021, 1, 1))
        self.assertEqual(recent.filing_date, date(2024, 2, 1))
        self.assertEqual(older.filing_date, date(2020, 2, 3))
        self.assertEqual(len(calls), 2)

    def test_transient_failure_is_retried_on_resume(self):
        client = FlakyFilingClient()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "client": client,
                "tickers": ["ABC"],
                "cik_map": {"ABC": 123},
                "window": "W1",
                "output_dir": root,
                "item1_cache_dir": root / "cache",
                "batch_size": 1,
                "limit": None,
            }
            with redirect_stdout(StringIO()):
                write_window(**kwargs)
            first = pl.read_parquet(root / "W1_descriptions.parquet")
            self.assertTrue(first["status"][0].startswith("request_error:"))

            with redirect_stdout(StringIO()):
                write_window(**kwargs)
            resumed = pl.read_parquet(root / "W1_descriptions.parquet")
            self.assertEqual(resumed.height, 1)
            self.assertEqual(resumed["status"][0], "annual_filing_not_found")
            self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
