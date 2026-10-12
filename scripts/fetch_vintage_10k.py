#!/usr/bin/env python3
"""Fetch vintage 10-K Item 1 text for all paper windows.

For each candidate security and feature-window start, the script selects the
most recent supported annual filing submitted strictly before that date. It
then extracts Item 1, retains at most 8,000 characters, and writes both the text
and metadata containing CIK, accession number, canonical SEC URL,
filing date, length, and text hash.

Completed ticker rows survive a restart. Failed requests are retried, extracted
sections are cached by accession, and progress is saved atomically after each
batch. Use a real, monitored contact address in ``--sec-user-agent`` or
``SEC_USER_AGENT``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sec_filing_text import (  # noqa: E402
    MAX_ITEM1_CHARS,
    SecClient,
    extract_item1,
    text_sha256,
)


WINDOW_STARTS = {
    "H1": date(2020, 2, 3),
    "H2": date(2020, 5, 1),
    "H3": date(2020, 9, 1),
    "H4": date(2021, 1, 4),
    "H5": date(2021, 5, 3),
    "H6": date(2021, 9, 1),
    "H7": date(2022, 1, 3),
    "H8": date(2022, 5, 2),
    "H9": date(2022, 9, 1),
    "H10": date(2023, 1, 3),
    "H11": date(2023, 5, 1),
    "H12": date(2023, 9, 1),
    "H13": date(2024, 1, 2),
    "W1": date(2024, 5, 1),
    "W2": date(2024, 8, 21),
    "W3": date(2024, 11, 21),
    "W4": date(2025, 2, 3),
    "W5": date(2025, 5, 1),
    "W6": date(2025, 8, 1),
    "W7": date(2025, 11, 3),
}

OUTPUT_SCHEMA = {
    "ticker": pl.String,
    "cik": pl.Int64,
    "window": pl.String,
    "filing_date": pl.String,
    "accession_number": pl.String,
    "filing_url": pl.String,
    "item1_text": pl.String,
    "item1_len": pl.Int64,
    "item1_sha256": pl.String,
    "source": pl.String,
    "status": pl.String,
}


def normalize_ticker(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def load_universe(path: Path) -> list[str]:
    frame = pl.read_parquet(path)
    if "ticker" not in frame.columns:
        raise ValueError(f"{path}: ticker column is required")
    tickers = [normalize_ticker(str(value)) for value in frame["ticker"].to_list()]
    if any(not value for value in tickers) or len(tickers) != len(set(tickers)):
        raise ValueError("universe tickers must be non-empty and unique")
    return tickers


def empty_record(ticker: str, window: str, status: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "cik": None,
        "window": window,
        "filing_date": None,
        "accession_number": None,
        "filing_url": None,
        "item1_text": "",
        "item1_len": 0,
        "item1_sha256": None,
        "source": "missing",
        "status": status,
    }


def process_ticker(
    client: SecClient,
    ticker: str,
    cik_map: dict[str, int],
    window: str,
    cutoff: date,
    item1_cache_dir: Path,
) -> dict[str, Any]:
    cik = cik_map.get(ticker)
    if cik is None:
        return empty_record(ticker, window, "ticker_not_in_sec_mapping")
    record = empty_record(ticker, window, "annual_filing_not_found")
    record["cik"] = cik
    reference = client.most_recent_annual_filing(cik, cutoff)
    if reference is None:
        return record
    record.update(
        {
            "filing_date": reference.filing_date.isoformat(),
            "accession_number": reference.accession_number,
            "filing_url": reference.filing_url,
            "status": "item1_not_extracted",
        }
    )
    cache_key = reference.accession_without_dashes
    text_cache = item1_cache_dir / f"{cache_key}.txt"
    missing_cache = item1_cache_dir / f"{cache_key}.missing"
    if text_cache.is_file():
        item1 = text_cache.read_text(encoding="utf-8")
    elif missing_cache.is_file():
        item1 = None
    else:
        item1 = extract_item1(client.filing_text(reference))
        item1_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = text_cache if item1 is not None else missing_cache
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        temporary.write_text(item1 or "", encoding="utf-8")
        temporary.replace(cache_path)
    if item1 is None:
        return record
    record.update(
        {
            "item1_text": item1,
            "item1_len": len(item1),
            "item1_sha256": text_sha256(item1),
            "source": "10K",
            "status": "ok",
        }
    )
    return record


def atomic_write(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def write_window(
    client: SecClient,
    tickers: list[str],
    cik_map: dict[str, int],
    window: str,
    output_dir: Path,
    item1_cache_dir: Path,
    *,
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    output_path = output_dir / f"{window}_descriptions.parquet"
    existing = pl.DataFrame(schema=OUTPUT_SCHEMA)
    if output_path.is_file():
        saved = pl.read_parquet(output_path)
        missing_columns = set(OUTPUT_SCHEMA) - set(saved.columns)
        if missing_columns:
            raise ValueError(
                f"{output_path}: existing file is not a fresh-driver output; "
                f"missing {sorted(missing_columns)}. Use a separate output directory."
            )
        saved = saved.select(list(OUTPUT_SCHEMA)).cast(OUTPUT_SCHEMA)
        # Drop rows with request errors so the next run retries them.
        existing = saved.filter(
            ~pl.col("status").str.starts_with("request_error:")
        )
    done = set(existing.get_column("ticker").to_list()) if existing.height else set()
    todo = [ticker for ticker in tickers if ticker not in done]
    if limit is not None:
        todo = todo[:limit]
    print(
        f"{window}: cutoff={WINDOW_STARTS[window]}, total={len(tickers)}, "
        f"resumed={len(done)}, remaining={len(todo)}"
    )

    buffer: list[dict[str, Any]] = []
    for index, ticker in enumerate(todo, start=1):
        try:
            record = process_ticker(
                client,
                ticker,
                cik_map,
                window,
                WINDOW_STARTS[window],
                item1_cache_dir,
            )
        except Exception as exc:
            record = empty_record(ticker, window, f"request_error:{type(exc).__name__}")
        buffer.append(record)
        print(
            f"  {index:4d}/{len(todo):4d} {ticker:<10} "
            f"{record['status']:<28} chars={record['item1_len']:4d}"
        )
        if len(buffer) >= batch_size or index == len(todo):
            new = pl.DataFrame(buffer, schema=OUTPUT_SCHEMA, strict=False)
            existing = pl.concat([existing, new], how="vertical")
            atomic_write(existing, output_path)
            buffer.clear()

    if existing.height:
        eligible = existing.filter(pl.col("source") == "10K")
        coverage = eligible.height / existing.height
    else:
        eligible = existing
        coverage = 0.0
    return {
        "window": window,
        "cutoff": WINDOW_STARTS[window].isoformat(),
        "records": existing.height,
        "eligible_10k": eligible.height,
        "coverage": coverage,
        "output": output_path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch vintage 10-K Item 1 text directly from SEC EDGAR."
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "data" / "universe.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "vintage_descriptions_rebuilt",
        help="fresh output tree; defaults away from the frozen submitted corpus",
    )
    parser.add_argument(
        "--item1-cache-dir",
        type=Path,
        default=ROOT / "results" / "source_cache" / "sec_item1_v1",
        help="restartable accession-level extracted-text cache",
    )
    parser.add_argument(
        "--windows",
        default=",".join(WINDOW_STARTS),
        help="comma-separated names from H1,...,H13,W1,...,W7",
    )
    parser.add_argument(
        "--sec-user-agent",
        default=os.environ.get("SEC_USER_AGENT"),
        help="research project and monitored email; may use SEC_USER_AGENT",
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=2.0,
        help="local SEC request ceiling; must not exceed 10",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional per-window smoke-test limit",
    )
    args = parser.parse_args()
    if not args.sec_user_agent:
        parser.error("--sec-user-agent or SEC_USER_AGENT is required")
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("batch size and limit must be positive")
    windows = tuple(value.strip() for value in args.windows.split(",") if value.strip())
    unknown = sorted(set(windows) - set(WINDOW_STARTS))
    if not windows or unknown:
        parser.error(f"invalid window list; unknown={unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tickers = load_universe(args.universe)
    client = SecClient(
        args.sec_user_agent,
        max_requests_per_second=args.max_rps,
    )
    cik_map = client.ticker_cik_map()
    summaries = [
        write_window(
            client,
            tickers,
            cik_map,
            window,
            args.output_dir,
            args.item1_cache_dir,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        for window in windows
    ]
    summary_path = args.output_dir / "vintage_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "vintage-item1-acquisition-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "SEC EDGAR submissions and archives endpoints",
                "supported_forms": ["10-K", "10-K405", "10-KSB", "10-KT"],
                "amendments_included": False,
                "item1_character_cap": MAX_ITEM1_CHARS,
                "windows": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
