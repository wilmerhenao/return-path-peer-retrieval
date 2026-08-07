"""Construct the modern or historical return windows used in the paper.

The script fetches adjusted prices for a caller-supplied ticker panel, computes
Fama--French five-factor residuals on a common complete-case date panel, builds
Ledoit--Wolf correlation targets, and saves per-window artifacts.

Output layout:
  results/windows_v3/       (default for ``--window-set modern``)
    W{1-7}/
      ts_features.npy       (N_w, T_w) — standardized FF5 residuals
      raw_resid.npy         (N_w, T_w) — unstandardized FF5 residuals
      corr_target.npy       (N_w, N_w) — LW-shrunk correlation matrix
      universe.parquet      (N_w, 2)   — ticker/company_idx subset
      meta.json             — date range, N, shrinkage intensity, provenance

Usage:
    cd paper_icaif2026
    python3 scripts/build_windows.py --universe data/universe.parquet \
      --output-dir results/windows_v3 --missing-policy complete

Historical tuning windows use the same estimator and data policy:

    python3 scripts/build_windows.py --window-set historical \
      --universe data/universe.parquet \
      --output-dir results/ea_historical/windows
"""
import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR     = PROJECT_ROOT / "data"
WINDOWS_DIR  = PROJECT_ROOT / "results" / "windows_v3"
WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

# ── Window definitions ────────────────────────────────────────────────────────
# Each date interval is inclusive.  Adjacent windows must be strictly disjoint;
# validate_windows() enforces this before any data are downloaded or written.
MODERN_WINDOWS = [
    ("W1", date(2024, 5,  1), date(2024, 8,  20)),   # May–Aug 2024
    ("W2", date(2024, 8, 21), date(2024, 11, 20)),   # Aug–Nov 2024
    ("W3", date(2024, 11,21), date(2025, 1,  31)),   # Nov 2024–Jan 2025
    ("W4", date(2025, 2,  3), date(2025, 4,  30)),   # Feb–Apr 2025
    ("W5", date(2025, 5,  1), date(2025, 7,  31)),   # May–Jul 2025
    ("W6", date(2025, 8,  1), date(2025, 10, 31)),   # Aug–Oct 2025
    ("W7", date(2025, 11, 3), date(2026, 1,  28)),   # Nov 2025–Jan 2026
]

HISTORICAL_WINDOWS = [
    ("H1", date(2020, 2, 3), date(2020, 4, 30)),
    ("H2", date(2020, 5, 1), date(2020, 8, 31)),
    ("H3", date(2020, 9, 1), date(2020, 12, 31)),
    ("H4", date(2021, 1, 4), date(2021, 4, 30)),
    ("H5", date(2021, 5, 3), date(2021, 8, 31)),
    ("H6", date(2021, 9, 1), date(2021, 12, 31)),
    ("H7", date(2022, 1, 3), date(2022, 4, 29)),
    ("H8", date(2022, 5, 2), date(2022, 8, 31)),
    ("H9", date(2022, 9, 1), date(2022, 12, 30)),
    ("H10", date(2023, 1, 3), date(2023, 4, 28)),
    ("H11", date(2023, 5, 1), date(2023, 8, 31)),
    ("H12", date(2023, 9, 1), date(2023, 12, 29)),
    ("H13", date(2024, 1, 2), date(2024, 4, 30)),
]

# Backward-compatible import used by the unit tests and older local callers.
WINDOWS = MODERN_WINDOWS

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MIN_VALID  = 45      # minimum trading days in a window to include a ticker
BATCH_SIZE = 50      # conservative batch size to avoid provider throttling
MAX_FETCH_ATTEMPTS = 3


def validate_windows(windows=WINDOWS) -> None:
    """Fail fast if window names repeat, dates invert, or windows overlap."""
    names = [name for name, _, _ in windows]
    if len(names) != len(set(names)):
        raise ValueError("window names must be unique")
    for name, start, end in windows:
        if start > end:
            raise ValueError(f"{name} starts after it ends: {start} > {end}")
    for (left, _, left_end), (right, right_start, _) in zip(windows, windows[1:]):
        if left_end >= right_start:
            raise ValueError(
                f"windows overlap or touch: {left} ends {left_end}, "
                f"{right} starts {right_start}"
            )


def fetch_ff5(cache_path: Path | None = None, refresh: bool = False) -> pl.DataFrame:
    """Download FF5 daily factors from Ken French's library using curl."""
    if cache_path and cache_path.exists() and not refresh:
        df = pl.read_parquet(cache_path)
        print(f"  FF5 cache: {len(df)} trading days from {cache_path}")
        return df
    print("  Fetching FF5 factors from Dartmouth...")
    archive = subprocess.check_output(
        ["curl", "--fail", "--silent", "--show-error", "--location", FF5_URL]
    )
    zf = zipfile.ZipFile(io.BytesIO(archive))
    fname = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    raw = zf.read(fname).decode("latin-1")

    # Parse: skip header lines until we hit the data block (lines starting with date)
    lines = []
    in_data = False
    for line in raw.splitlines():
        line = line.strip()
        if not in_data:
            if line and line[0].isdigit() and len(line.split(",")) >= 7:
                in_data = True
            else:
                continue
        if not line or not line[0].isdigit():
            break
        lines.append(line)

    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            d_str = parts[0]
            d = date(int(d_str[:4]), int(d_str[4:6]), int(d_str[6:8]))
            factors = [float(p) / 100.0 for p in parts[1:7]]
            rows.append((d, *factors))
        except (ValueError, IndexError):
            continue

    df = pl.DataFrame(
        rows,
        schema=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
        orient="row",
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
    print(f"  FF5: {len(df)} trading days, {df['date'].min()} to {df['date'].max()}")
    return df


def fetch_prices(
    tickers: list,
    start: date,
    end: date,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Fetch adjusted close prices in restartable, rate-limit-safe batches."""
    import pandas as pd
    import yfinance as yf
    from curl_cffi.requests import Session as CurlSession

    session = CurlSession(impersonate="chrome")
    start_str = (start - timedelta(days=10)).isoformat()
    end_str   = (end   + timedelta(days=5)).isoformat()

    all_prices = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        cache_path = (
            cache_dir / f"prices_{start}_{end}_batch_{i // BATCH_SIZE:04d}.parquet"
            if cache_dir
            else None
        )
        if cache_path and cache_path.exists() and not refresh:
            cached = pl.read_parquet(cache_path).to_pandas().set_index("date")
            for tkr in batch:
                if tkr in cached.columns:
                    col = cached[tkr].dropna()
                    if len(col) >= MIN_VALID:
                        all_prices[tkr] = col
            print(f"    batch {i // BATCH_SIZE + 1}: cache hit")
            continue

        t0 = time.time()
        prices = None
        for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
            try:
                raw = yf.download(
                    batch,
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    progress=False,
                    session=session,
                    threads=4,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    prices = raw["Close"]
                else:
                    prices = raw[["Close"]].rename(columns={"Close": batch[0]})
                if prices is not None and not prices.dropna(axis=1, how="all").empty:
                    break
            except Exception as exc:
                print(
                    f"    [WARN] batch {i // BATCH_SIZE + 1}, "
                    f"attempt {attempt} failed: {exc}"
                )
            prices = None
            if attempt < MAX_FETCH_ATTEMPTS:
                time.sleep(5 * attempt)

        if prices is None:
            print(f"    [WARN] batch {i // BATCH_SIZE + 1} exhausted retries")
            continue

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached = prices.copy()
            cached.index = pd.to_datetime(cached.index).date
            cached.index.name = "date"
            pl.from_pandas(cached.reset_index()).write_parquet(cache_path)

        for tkr in batch:
            if tkr in prices.columns:
                col = prices[tkr].dropna()
                if len(col) >= MIN_VALID:
                    all_prices[tkr] = col
        elapsed = time.time() - t0
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)

    print(f"  Prices fetched: {len(all_prices)} tickers with ≥{MIN_VALID} days")

    # Merge into a single Polars DataFrame
    import pandas as pd
    dfs = []
    for tkr, series in all_prices.items():
        df_t = pd.DataFrame({"date": series.index, tkr: series.values})
        dfs.append(df_t)
    if not dfs:
        return pl.DataFrame()

    merged = dfs[0]
    for df_t in dfs[1:]:
        merged = merged.merge(df_t, on="date", how="outer")
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged = merged.sort_values("date")
    return pl.from_pandas(merged)


def simple_returns(prices_df: pl.DataFrame) -> pl.DataFrame:
    """Compute daily simple returns from adjusted close prices."""
    tickers = [c for c in prices_df.columns if c != "date"]
    result = prices_df.select("date")
    for tkr in tickers:
        col = prices_df[tkr].cast(pl.Float64)
        simple_ret = col / col.shift(1) - 1.0
        result = result.with_columns(simple_ret.alias(tkr))
    return result.slice(1)  # drop first row (NaN)


def compute_ff5_residuals(
    returns_df: pl.DataFrame,
    ff5_df: pl.DataFrame,
    start: date,
    end: date,
    min_days: int = MIN_VALID,
    missing_policy: str = "complete",
) -> tuple[np.ndarray, np.ndarray, list, float, np.ndarray, np.ndarray]:
    """Regress each ticker's returns on FF5 factors within ``[start, end]``.

    The dependent variable is the stock's simple excess return, R_i - RF.
    The paper protocol retains only tickers observed on every factor date, so
    all returned residual series share the same complete date panel.

    Returns standardized residuals, raw excess-return residuals, tickers,
    shrinkage intensity, the shrunk correlation matrix, and per-ticker
    observation counts.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.covariance import LedoitWolf

    # Filter to window
    ret_window = returns_df.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    ff5_window = ff5_df.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )

    # Align dates
    common_dates = set(ret_window["date"].to_list()) & set(ff5_window["date"].to_list())
    common_dates = sorted(common_dates)
    T = len(common_dates)

    ret_window = ret_window.filter(pl.col("date").is_in(common_dates)).sort("date")
    ff5_window = ff5_window.filter(pl.col("date").is_in(common_dates)).sort("date")

    F = ff5_window.select(["mkt_rf", "smb", "hml", "rmw", "cma"]).to_numpy()
    rf = ff5_window["rf"].to_numpy()
    factor_valid = np.isfinite(F).all(axis=1) & np.isfinite(rf)
    if missing_policy != "complete":
        raise ValueError("the paper protocol supports only complete-case windows")

    tickers = [c for c in ret_window.columns if c != "date"]
    valid_tickers = []
    residuals_list = []
    raw_residuals_list = []
    observation_counts = []

    for tkr in tickers:
        col = ret_window[tkr].to_numpy()
        valid = np.isfinite(col) & factor_valid
        if valid.sum() < min_days:
            continue
        if valid.sum() != T:
            continue
        excess_return = col[valid] - rf[valid]
        reg = LinearRegression(fit_intercept=True)
        reg.fit(F[valid], excess_return)
        observed_resid = excess_return - reg.predict(F[valid])
        std = observed_resid.std()
        if std < 1e-10:
            continue

        raw_resid = observed_resid.astype(float, copy=True)
        resid_std = (observed_resid - observed_resid.mean()) / std
        valid_tickers.append(tkr)
        residuals_list.append(resid_std)
        raw_residuals_list.append(raw_resid)
        observation_counts.append(int(valid.sum()))

    if not valid_tickers:
        empty = np.zeros((0, T), dtype=float)
        empty_corr = np.zeros((0, 0), dtype=float)
        observation_counts = np.zeros(0, dtype=int)
        return empty, empty.copy(), [], 0.0, empty_corr, observation_counts

    resid_matrix = np.vstack(residuals_list)  # (N, T)
    raw_resid_matrix = np.vstack(raw_residuals_list)

    # LW shrinkage
    lw = LedoitWolf()
    lw.fit(resid_matrix.T)  # (T, N)
    shrinkage = lw.shrinkage_
    cov_lw = lw.covariance_

    # Convert to correlation
    std_diag = np.sqrt(np.diag(cov_lw))
    std_outer = np.outer(std_diag, std_diag)
    corr_lw = cov_lw / np.maximum(std_outer, 1e-12)
    np.fill_diagonal(corr_lw, 1.0)
    corr_lw = np.clip(corr_lw, -1.0, 1.0)

    return (
        resid_matrix,
        raw_resid_matrix,
        valid_tickers,
        float(shrinkage),
        corr_lw,
        np.asarray(observation_counts, dtype=int),
    )


def build_window(
    name: str,
    start: date,
    end: date,
    prices_df: pl.DataFrame,
    ff5_df: pl.DataFrame,
    universe_tickers: list,
    force: bool = False,
    windows_dir: Path = WINDOWS_DIR,
    missing_policy: str = "complete",
) -> dict | None:
    """Build and save one complete-case residual window and its LW target."""
    if missing_policy != "complete":
        raise ValueError("the paper protocol supports only complete-case windows")
    out_dir = windows_dir / name
    meta_path = out_dir / "meta.json"

    if meta_path.exists() and not force:
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  {name}: already built ({meta['N']} tickers, {start}–{end}) — skipping")
        return meta

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Building {name}: {start} → {end}")

    # Filter prices to universe tickers
    prices_cols = ["date"] + [t for t in universe_tickers if t in prices_df.columns]
    prices_sub = prices_df.select(prices_cols)

    (
        resid_matrix,
        raw_resid_matrix,
        valid_tickers,
        shrinkage,
        corr_lw,
        observation_counts,
    ) = compute_ff5_residuals(
        prices_sub, ff5_df, start, end, missing_policy=missing_policy
    )
    N = len(valid_tickers)
    T = resid_matrix.shape[1] if N > 0 else 0
    print(f"    N={N} valid tickers, T={T} days, LW shrinkage={shrinkage:.3f}")

    if N < 100:
        print(f"    [WARN] Only {N} tickers — window may be too sparse")

    # Save
    np.save(out_dir / "ts_features.npy", resid_matrix.astype(np.float32))
    np.save(out_dir / "raw_resid.npy", raw_resid_matrix.astype(np.float32))
    np.save(out_dir / "corr_target.npy", corr_lw.astype(np.float32))

    # Universe subset
    univ_sub = pl.DataFrame({
        "ticker": valid_tickers,
        "company_idx": list(range(N)),
    })
    univ_sub.write_parquet(out_dir / "universe.parquet")

    meta = {
        "window": name,
        "start": str(start),
        "end": str(end),
        "N": N,
        "T": T,
        "lw_shrinkage": shrinkage,
        "valid_tickers": valid_tickers,
        "ts_features_shape": list(resid_matrix.shape),
        "corr_target_shape": list(corr_lw.shape),
        "return_convention": "daily simple adjusted-close excess return",
        "factor_specification": "FF5 with intercept; dependent variable R_i - RF",
        "missing_policy": missing_policy,
        "min_observations": int(observation_counts.min()) if N else 0,
        "median_observations": float(np.median(observation_counts)) if N else 0,
        "max_observations": int(observation_counts.max()) if N else 0,
        "corr_range": [float(corr_lw[~np.eye(N, dtype=bool)].min()),
                       float(corr_lw[~np.eye(N, dtype=bool)].max())],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    Saved: ts_features={resid_matrix.shape}, "
          f"corr [{meta['corr_range'][0]:.3f}, {meta['corr_range'][1]:.3f}]")
    return meta


def verify_window_meta(meta: dict) -> None:
    """Fail if a saved window contradicts the requested construction policy."""
    n = int(meta.get("N", 0))
    t = int(meta.get("T", 0))
    if n < 100 or t < MIN_VALID:
        raise ValueError(f"implausible saved window dimensions: N={n}, T={t}")
    if meta.get("missing_policy") != "complete":
        raise ValueError("saved window does not use the complete-case paper protocol")
    if int(meta.get("min_observations", -1)) != t:
        raise ValueError(
            "complete-case window contains a security with fewer than T observations"
        )
    print(f"  {meta.get('window', 'window')} validation: N={n}, T={t}, policy OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rebuild windows even if already present")
    parser.add_argument("--universe", default=None,
                        help="Path to universe parquet (default: data/universe.parquet)")
    parser.add_argument("--output-dir", default=None,
                        help=(
                            "Output directory. Defaults to results/windows_v3 "
                            "for modern or results/ea_historical/windows for historical."
                        ))
    parser.add_argument(
        "--window-set",
        choices=("modern", "historical"),
        default="modern",
        help="Date grid to construct; both use the identical estimator.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["complete"],
        default="complete",
        help="Missing-return policy; the paper protocol uses complete cases only.",
    )
    parser.add_argument(
        "--source-cache",
        default="results/source_cache",
        help="Restartable source-data cache directory.",
    )
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Ignore cached source data and download it again.",
    )
    args = parser.parse_args()
    selected_windows = (
        MODERN_WINDOWS if args.window_set == "modern" else HISTORICAL_WINDOWS
    )
    validate_windows(selected_windows)

    # Allow overriding paths
    universe_path = Path(args.universe) if args.universe else DATA_DIR / "universe.parquet"
    default_output = (
        WINDOWS_DIR
        if args.window_set == "modern"
        else PROJECT_ROOT / "results" / "ea_historical" / "windows"
    )
    windows_dir = Path(args.output_dir) if args.output_dir else default_output
    source_cache = Path(args.source_cache)
    windows_dir.mkdir(parents=True, exist_ok=True)

    # Load universe
    universe = pl.read_parquet(universe_path)
    universe_tickers = universe["ticker"].to_list()
    print(f"Universe: {len(universe_tickers)} tickers from {universe_path.name}")

    # Determine date range needed
    all_start = min(w[1] for w in selected_windows)
    all_end = max(w[2] for w in selected_windows)
    print(f"Date range needed: {all_start} → {all_end}")

    # Fetch FF5
    ff5_df = fetch_ff5(
        source_cache / "ff5_daily.parquet",
        refresh=args.refresh_source,
    )

    # Fetch prices — one big pull to avoid redundant downloads
    calendar_days = (all_end - all_start).days + 1
    print(
        f"\nFetching prices for {len(universe_tickers)} tickers "
        f"({all_start} → {all_end}, {calendar_days} calendar days)..."
    )
    prices_raw = fetch_prices(
        universe_tickers,
        all_start,
        all_end,
        cache_dir=source_cache,
        refresh=args.refresh_source,
    )
    if prices_raw.is_empty():
        print("ERROR: No prices fetched. Check network access.")
        sys.exit(1)

    # Compute simple returns to match the simple-return FF5 factors.
    print("Computing simple returns...")
    returns_df = simple_returns(prices_raw)
    print(f"  Returns shape: {returns_df.shape[0]} days × {returns_df.shape[1]-1} tickers")

    # Build each window
    all_meta = {}
    for name, start, end in selected_windows:
        meta = build_window(name, start, end, returns_df, ff5_df,
                            universe_tickers, force=args.force,
                            windows_dir=windows_dir,
                            missing_policy=args.missing_policy)
        if meta:
            all_meta[name] = meta

    for meta in all_meta.values():
        verify_window_meta(meta)

    # Summary
    print("\n" + "=" * 60)
    print("WINDOWS BUILT")
    print("=" * 60)
    print(f"  {'Window':<6}  {'Start':<12}  {'End':<12}  {'N':>5}  {'T':>4}  {'LW':>6}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*5}  {'-'*4}  {'-'*6}")
    for name, start, end in selected_windows:
        m = all_meta.get(name, {})
        print(f"  {name:<6}  {str(start):<12}  {str(end):<12}  "
              f"{m.get('N', 0):>5}  {m.get('T', 0):>4}  {m.get('lw_shrinkage', 0):>6.3f}")

    # Save a portable summary inside the selected output tree.  Per-window
    # metadata contain dates, dimensions, estimator details, and data policy;
    # no repository history or workstation path is required to interpret it.
    summary_path = windows_dir / "windows_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "valid_tickers"}
                   for k, v in all_meta.items()}, f, indent=2)
    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    main()
