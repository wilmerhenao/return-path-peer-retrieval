#!/usr/bin/env python3
"""Validate the private-input layout before an expensive reproduction run.

The validator is read-only. It checks path presence, table schemas, array
shapes, ticker order, and filing-time constraints, then reports every problem
in one pass. A missing input is distinct from an invalid input: exit status 2
means data must be acquired or supplied, while exit status 1 means a present
artifact violates the paper's contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_adversarial_followup import (  # noqa: E402
    CACHE_DIR,
    DENSE_CACHE_FILES,
)


MODERN = tuple(f"W{i}" for i in range(1, 8))
HISTORICAL = tuple(f"H{i}" for i in range(1, 14))
MODEL_DIRECTORIES = (
    "all-MiniLM-L12-v2",
    "bge-base-en-v1.5",
    "sec-bert-base",
    "granite-embedding-english-r2",
)


class ContractReport:
    def __init__(self) -> None:
        self.checked: list[str] = []
        self.missing: list[str] = []
        self.invalid: list[str] = []

    def require_file(self, path: Path) -> bool:
        relative = relative_path(path)
        if not path.is_file():
            self.missing.append(relative)
            return False
        self.checked.append(relative)
        return True

    def require_directory(self, path: Path) -> bool:
        relative = relative_path(path)
        if not path.is_dir():
            self.missing.append(relative + "/")
            return False
        self.checked.append(relative + "/")
        return True

    def fail(self, path: Path, message: str) -> None:
        self.invalid.append(f"{relative_path(path)}: {message}")

    def as_dict(self, tier: str) -> dict[str, Any]:
        status = "invalid" if self.invalid else "missing" if self.missing else "pass"
        return {
            "status": status,
            "tier": tier,
            "n_checked": len(self.checked),
            "checked": sorted(set(self.checked)),
            "missing": sorted(set(self.missing)),
            "invalid": sorted(set(self.invalid)),
        }


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def validate_universe(report: ContractReport, *, require_sic: bool) -> None:
    path = ROOT / "data" / "universe.parquet"
    if not report.require_file(path):
        return
    try:
        frame = pl.read_parquet(path)
        required = {"ticker"} | ({"sic"} if require_sic else set())
        if not required.issubset(frame.columns):
            report.fail(path, f"missing columns {sorted(required - set(frame.columns))}")
        elif frame["ticker"].null_count() or frame["ticker"].n_unique() != frame.height:
            report.fail(path, "ticker must be non-null and unique")
    except Exception as exc:
        report.fail(path, f"cannot read parquet: {exc}")


def validate_window(
    report: ContractReport,
    directory: Path,
    name: str,
    *,
    require_raw: bool,
) -> tuple[list[str], date] | None:
    window_dir = directory / name
    required = [
        window_dir / "meta.json",
        window_dir / "universe.parquet",
        window_dir / "ts_features.npy",
        window_dir / "corr_target.npy",
    ]
    if require_raw:
        required.append(window_dir / "raw_resid.npy")
    if not all(report.require_file(path) for path in required):
        return None
    try:
        meta = json.loads((window_dir / "meta.json").read_text(encoding="utf-8"))
        tickers = (
            pl.read_parquet(window_dir / "universe.parquet")
            .get_column("ticker")
            .cast(pl.String)
            .to_list()
        )
        ts = np.load(window_dir / "ts_features.npy", mmap_mode="r")
        correlation = np.load(window_dir / "corr_target.npy", mmap_mode="r")
        n, t = ts.shape
        if len(tickers) != n or len(tickers) != len(set(tickers)):
            report.fail(window_dir, "universe row count or uniqueness mismatch")
        if correlation.shape != (n, n):
            report.fail(window_dir, "correlation matrix shape mismatch")
        if int(meta["N"]) != n or int(meta["T"]) != t:
            report.fail(window_dir, "metadata dimensions do not match arrays")
        if require_raw:
            raw = np.load(window_dir / "raw_resid.npy", mmap_mode="r")
            if raw.shape != (n, t):
                report.fail(window_dir, "raw residual shape mismatch")
        return tickers, date.fromisoformat(meta["start"])
    except Exception as exc:
        report.fail(window_dir, f"cannot validate window: {exc}")
        return None


def validate_vintage(
    report: ContractReport,
    name: str,
    start: date,
    vintage_dir: Path,
) -> None:
    path = vintage_dir / f"{name}_descriptions.parquet"
    if not report.require_file(path):
        return
    try:
        frame = pl.read_parquet(path)
        required = {
            "ticker",
            "cik",
            "window",
            "filing_date",
            "item1_text",
            "item1_len",
            "source",
        }
        if not required.issubset(frame.columns):
            report.fail(path, f"missing columns {sorted(required - set(frame.columns))}")
            return
        eligible = frame.filter(
            (pl.col("source") == "10K")
            & pl.col("item1_text").is_not_null()
            & (pl.col("item1_text").str.len_chars() > 0)
        )
        duplicates = eligible.group_by("ticker").len().filter(pl.col("len") > 1)
        if duplicates.height:
            report.fail(path, "duplicate eligible ticker rows")
        for filing in eligible.get_column("filing_date").to_list():
            if date.fromisoformat(str(filing)) >= start:
                report.fail(path, f"filing {filing} is not strictly before {start}")
                break
    except Exception as exc:
        report.fail(path, f"cannot validate vintage table: {exc}")


def validate_dense_cache(
    report: ContractReport,
    window_rows: dict[str, tuple[list[str], date]],
) -> None:
    for window, methods in DENSE_CACHE_FILES.items():
        if window not in window_rows:
            continue
        n = len(window_rows[window][0])
        for filename in methods.values():
            path = CACHE_DIR / filename
            if not report.require_file(path):
                continue
            try:
                values = np.load(path, mmap_mode="r")
                if values.ndim != 2 or values.shape[0] != n:
                    report.fail(path, f"expected {n} rows, found shape {values.shape}")
            except Exception as exc:
                report.fail(path, f"cannot read embedding cache: {exc}")


def validate_models(report: ContractReport) -> None:
    models_dir = ROOT / "models"
    for name in MODEL_DIRECTORIES:
        directory = models_dir / name
        if not report.require_directory(directory):
            continue
        if not any(
            (directory / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        ):
            report.fail(directory, "no recognized weight file")
        for filename in ("config.json", "tokenizer_config.json"):
            if not (directory / filename).is_file():
                report.fail(directory, f"missing {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate reproduction inputs.")
    parser.add_argument(
        "--tier",
        choices=("windows", "primary", "dense", "granite", "full"),
        default="primary",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--vintage-dir",
        type=Path,
        default=ROOT / "results" / "vintage_descriptions",
        help="prepared H*/W* filing-parquet directory",
    )
    args = parser.parse_args()

    report = ContractReport()
    require_sic = args.tier == "full"
    validate_universe(report, require_sic=require_sic)
    modern_rows: dict[str, tuple[list[str], date]] = {}
    modern_dir = ROOT / "results" / "windows_v3"
    require_modern_raw = args.tier == "full"
    for window in MODERN:
        validated = validate_window(
            report,
            modern_dir,
            window,
            require_raw=require_modern_raw,
        )
        if validated is not None:
            modern_rows[window] = validated

    if args.tier in {"primary", "dense", "full"}:
        historical_dir = ROOT / "results" / "ea_historical" / "windows"
        historical_rows: dict[str, tuple[list[str], date]] = {}
        for window in HISTORICAL:
            validated = validate_window(
                report,
                historical_dir,
                window,
                require_raw=False,
            )
            if validated is not None:
                historical_rows[window] = validated
        for window, (_, start) in {**historical_rows, **modern_rows}.items():
            if window != "W7":
                validate_vintage(report, window, start, args.vintage_dir)

    if args.tier in {"dense", "full"}:
        validate_models(report)
        validate_dense_cache(report, modern_rows)
    elif args.tier == "granite":
        granite = ROOT / "models" / "granite-embedding-english-r2"
        report.require_directory(granite)
        for window, (_, start) in modern_rows.items():
            validate_vintage(report, window, start, args.vintage_dir)

    payload = report.as_dict(args.tier)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Data contract [{args.tier}]: {payload['status'].upper()}")
        print(f"  checked: {payload['n_checked']}")
        if payload["missing"]:
            print("  missing:")
            for value in payload["missing"]:
                print(f"    - {value}")
        if payload["invalid"]:
            print("  invalid:")
            for value in payload["invalid"]:
                print(f"    - {value}")
    if report.invalid:
        return 1
    if report.missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
