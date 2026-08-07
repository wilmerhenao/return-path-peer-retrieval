#!/usr/bin/env python3
"""Create a compact, verifiable summary of frozen return-window artifacts.

The large residual and correlation arrays need not be redistributed for an
artifact reviewer to verify their declared dates, dimensions, estimator, and
input identities. This script reads each saved ``meta.json``, checks it against
the adjacent array and universe shapes, and records SHA-256 hashes for later
identity checks. It does not recompute any statistical result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results" / "ea_historical" / "windows"
DEFAULT_OUTPUT = ROOT / "results" / "historical_windows_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_window(directory: Path, name: str) -> dict[str, Any]:
    window_dir = directory / name
    paths = {
        "meta.json": window_dir / "meta.json",
        "universe.parquet": window_dir / "universe.parquet",
        "ts_features.npy": window_dir / "ts_features.npy",
        "corr_target.npy": window_dir / "corr_target.npy",
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: missing {missing}")

    meta = json.loads(paths["meta.json"].read_text(encoding="utf-8"))
    tickers = (
        pl.read_parquet(paths["universe.parquet"])
        .get_column("ticker")
        .cast(pl.String)
        .to_list()
    )
    ts = np.load(paths["ts_features.npy"], mmap_mode="r")
    correlation = np.load(paths["corr_target.npy"], mmap_mode="r")
    n, t = ts.shape
    raw_path = window_dir / "raw_resid.npy"
    if raw_path.is_file():
        raw = np.load(raw_path, mmap_mode="r")
        if raw.shape != (n, t):
            raise ValueError(f"{name}: raw/standardized residual shape mismatch")
        paths["raw_resid.npy"] = raw_path
    if correlation.shape != (n, n):
        raise ValueError(f"{name}: correlation shape mismatch")
    if len(tickers) != n or len(tickers) != len(set(tickers)):
        raise ValueError(f"{name}: universe order or row count mismatch")
    if int(meta["N"]) != n or int(meta["T"]) != t:
        raise ValueError(f"{name}: metadata dimensions do not match arrays")
    if meta.get("missing_policy") not in {None, "complete"}:
        raise ValueError(f"{name}: expected complete-case missing-data policy")

    return {
        "window": name,
        "start": meta["start"],
        "end": meta["end"],
        "N": n,
        "T": t,
        "lw_shrinkage": float(meta["lw_shrinkage"]),
        "return_convention": meta.get("return_convention"),
        "factor_specification": meta.get("factor_specification"),
        "missing_policy_recorded_in_meta": meta.get("missing_policy"),
        "raw_residual_path_released": raw_path.is_file(),
        "artifact_sha256": {
            label: sha256_file(path) for label, path in paths.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize and hash saved return-window artifacts."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--windows",
        default=",".join(f"H{i}" for i in range(1, 14)),
        help="comma-separated saved window names",
    )
    args = parser.parse_args()
    names = tuple(value.strip() for value in args.windows.split(",") if value.strip())
    if not names or len(names) != len(set(names)):
        parser.error("--windows must contain unique non-empty names")

    rows = [summarize_window(args.input_dir, name) for name in names]
    for left, right in zip(rows, rows[1:]):
        if left["end"] >= right["start"]:
            raise ValueError(
                f"windows overlap or touch: {left['window']} and {right['window']}"
            )
    payload = {
        "schema_version": "window-metadata-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": args.input_dir.resolve().relative_to(ROOT.resolve()).as_posix(),
        "windows": rows,
        "summary": {
            "n_windows": len(rows),
            "N_min": min(row["N"] for row in rows),
            "N_max": max(row["N"] for row in rows),
            "T_min": min(row["T"] for row in rows),
            "T_max": max(row["T"] for row in rows),
            "lw_shrinkage_min": min(row["lw_shrinkage"] for row in rows),
            "lw_shrinkage_max": max(row["lw_shrinkage"] for row in rows),
            "all_recorded_missing_policies_are_complete": all(
                row["missing_policy_recorded_in_meta"] in {None, "complete"}
                for row in rows
            ),
            "windows_without_recorded_missing_policy": [
                row["window"]
                for row in rows
                if row["missing_policy_recorded_in_meta"] is None
            ],
        },
        "source_sha256": {
            "scripts/summarize_window_metadata.py": sha256_file(
                Path(__file__).resolve()
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
