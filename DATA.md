# Data Sources and Schemas

This document describes the inputs expected by the active scripts. It also
states which inputs are omitted from the anonymous archive.

## Availability summary

| Input | Public source | Included in the archive | Needed for full recomputation |
|---|---|---:|---:|
| Frozen result JSON and CSV files | This artifact | Yes | No, for Tier 1 |
| 2,200-name candidate universe | Derived from SEC and GDELT | No | Yes |
| Adjusted price histories | Yahoo Finance through yfinance | No | Yes |
| Daily Fama-French five factors | Kenneth French Data Library | No raw cache | Yes |
| W1-W7 derived return arrays | Derived | No | Yes |
| H1-H13 derived return arrays | Derived | No | Yes |
| Vintage 10-K Item 1 corpus | SEC EDGAR | No | Yes |
| Short-context embedding cache | Derived from public models | No | Needed for dense diagnostics |
| Model weights and tokenizer files | Hugging Face repositories | No | Needed for model reruns |

The result artifacts contain hashes of the omitted prepared files consumed by
the active scripts. Those hashes permit identity checks when a restricted input
bundle is supplied. They do not reconstruct the file contents, and they are not
a complete manifest of the upstream price and filing downloads.

## Candidate universe

The submitted universe contains 2,200 tickers. It was frozen on April 12, 2026
after ex-post matching of current SEC company metadata to GDELT organization
coverage, including a 2025-2026 news-visibility filter. It is not a historical
security master.

The local file is:

```text
data/universe.parquet
```

The actual schema used in the run is:

| Column | Type | Meaning |
|---|---|---|
| `ticker` | string | Unique market-data symbol |
| `sec_current_name` | string | Current SEC company name used in matching |
| `org_norm` | string | Normalized organization name used for GDELT matching |
| `sic` | string | SIC code; the first two characters define SIC-2 |
| `sic_title` | string | SIC description |
| `total_articles` | unsigned integer | GDELT organization coverage count |
| `articles_2025_2026` | unsigned integer | Coverage used by the recent-news filter |

`scripts/build_windows.py` requires a unique `ticker` column. The SIC baseline
in `scripts/run_strict_feature_time_application_audit.py` also requires a
nonempty `sic` value whose string representation has at least two characters.

Official source locations are:

- SEC ticker, CIK, and company-name associations:
  `https://www.sec.gov/files/company_tickers.json`
- SEC ticker, CIK, and exchange associations used by the released client:
  `https://www.sec.gov/files/company_tickers_exchange.json`
- SEC documentation for programmatic EDGAR access:
  `https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data`
- GDELT GKG file index:
  `https://data.gdeltproject.org/gkg/index.html`
- GDELT GKG 2.1 codebook:
  `https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf`

The exact SEC-to-GDELT matching and filtering workflow is not included in the
anonymous artifact. A user starting from the current public endpoints should
not expect to recover the exact submitted 2,200-name file.

## Return and factor data

### Official and upstream locations

The modern window builder downloads:

- Adjusted price histories through yfinance 1.2.0:
  `https://pypi.org/project/yfinance/1.2.0/`
- Daily Fama-French five factors from the Kenneth French Data Library:
  `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip`
- Data-library description page:
  `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`

yfinance is a third-party client, not an official Yahoo product. The PyPI link
identifies the client version used in the run; it is not an independent source
for the underlying price history.

`scripts/build_windows.py` calls yfinance with `auto_adjust=True`. It computes
daily simple returns, subtracts the daily risk-free rate, and fits an FF5
regression with an intercept separately for each security and window. The
paper protocol retains only securities observed on every factor date in a
window. Missing returns are not set to zero.

The builder keeps restartable downloads under:

```text
results/source_cache/ff5_daily.parquet
results/source_cache/prices_*.parquet
```

These files are not released. Yahoo Finance histories may be revised. Running
the builder today is a fresh-data replication, not a bitwise reconstruction of
the submitted window arrays.

### Modern window layout

The prepared modern inputs occupy `results/windows_v3/W1` through `W7`. Each
directory contains:

| File | Shape or schema | Meaning |
|---|---|---|
| `ts_features.npy` | `(N, T)` float32 | Row-standardized FF5 residual paths |
| `raw_resid.npy` | `(N, T)` float32 | Unstandardized FF5 residual paths |
| `corr_target.npy` | `(N, N)` float32 | Ledoit-Wolf residual-correlation matrix |
| `universe.parquet` | `ticker`, `company_idx` | Row order for all arrays |
| `meta.json` | JSON | Dates, dimensions, shrinkage, counts, and policy |

The submitted windows span May 1, 2024 through January 28, 2026. They contain
1,921 to 1,995 securities and 47 to 77 complete trading dates per window.

### Historical window layout

The historical inputs occupy:

```text
results/ea_historical/windows/H1/
...
results/ea_historical/windows/H13/
```

The submitted historical directories contain `ts_features.npy`,
`corr_target.npy`, `universe.parquet`, and `meta.json`. They do not contain
`raw_resid.npy`, and their older metadata do not carry the modern
`missing_policy` and observation-count fields. The active scripts consume these
arrays for historical-only TF-IDF and fusion selection and for the 19-transition
extension.

`scripts/build_windows.py --window-set historical` defines the historical date
grid and current complete-case estimator. Its output uses the modern five-file
layout. Because the exact historical source cache is not released, running that
command on fresh data is a source-level replication, not a bitwise rebuild of
the submitted H1-H13 arrays.

`results/historical_windows_summary.json` records the dates, shapes, shrinkage
values, and SHA-256 hashes of the four submitted files in every H directory. It
also records that the older metadata omit an explicit missing-policy field.
Use those hashes to check a supplied historical bundle before rerunning the
selection experiments.

## Vintage SEC filing corpus

Each feature window has one file:

```text
results/vintage_descriptions/H1_descriptions.parquet
...
results/vintage_descriptions/H13_descriptions.parquet
results/vintage_descriptions/W1_descriptions.parquet
...
results/vintage_descriptions/W7_descriptions.parquet
```

The submitted files use a seven-column legacy schema. Fresh files written by
`scripts/fetch_vintage_10k.py` default to the separate
`results/vintage_descriptions_rebuilt/` tree and add accession and extraction
metadata:

| Column | Submitted files | Fresh driver output | Meaning |
|---|---:|---:|---|
| `ticker` | Yes | Yes | Security symbol |
| `cik` | Yes | Yes | Nullable SEC Central Index Key |
| `window` | Yes | Yes | Feature-window label |
| `filing_date` | Yes | Yes | Nullable ISO filing date used for the vintage check |
| `accession_number` | No | Yes | SEC accession number |
| `filing_url` | No | Yes | Canonical SEC primary-document URL |
| `item1_text` | Yes | Yes | Stored Item 1 excerpt; empty or null when unavailable |
| `item1_len` | Yes | Yes | Stored character count |
| `item1_sha256` | No | Yes | SHA-256 of the stored excerpt |
| `source` | Yes | Yes | `10K` for an eligible filing; otherwise a missing or failure label |
| `status` | No | Yes | Detailed acquisition or extraction status |

The primary scripts keep rows satisfying all of these conditions:

- `source == "10K"`
- `item1_text` is non-null and nonempty
- `filing_date` is strictly earlier than the feature-window start
- no duplicate eligible ticker exists within a window

The collection process capped stored excerpts at 8,000 characters. In the
modern W1-W6 corpus, 85.9 percent of eligible excerpts reach that cap. The
8,192-token Granite run processes each full stored excerpt, not necessarily the
complete Item 1 section.

Official SEC acquisition locations are:

- Per-company submissions API:
  `https://data.sec.gov/submissions/CIK##########.json`
- Bulk submissions archive:
  `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip`
- Filing archives, addressed by CIK and accession number:
  `https://www.sec.gov/Archives/edgar/data/`
- SEC API documentation:
  `https://www.sec.gov/search-filings/edgar-application-programming-interfaces`

Automated requests must follow the SEC access policy and identify the caller as
required by the SEC. `scripts/sec_filing_text.py` supplies a rate-limited SEC
client, annual-filing selection, and a tested Item 1 extraction heuristic.
`scripts/fetch_vintage_10k.py` applies it to all twenty feature windows, writes
progress atomically after each batch, reuses each company's submission metadata
across cutoffs, caches extracted sections by accession, skips completed rows,
and retries transient request failures. It selects the most recent supported
annual filing strictly before each window start, excludes amendments, and caps
Item 1 text at 8,000 characters. The restart cache is under
`results/source_cache/sec_item1_v1/` and is not released.

The driver begins from the frozen ticker list in `data/universe.parquet` and
downloads a current SEC ticker/CIK map. It does not rebuild the candidate panel,
its SEC-to-GDELT selection, or a historical ticker/CIK mapping. The submitted
legacy parquets also lack accession numbers and text hashes. A fresh SEC
acquisition is therefore a source-level replication and is not guaranteed to
be bitwise identical to the submitted text corpus.

## Dense model inputs

The local model directories expected by the short-context scripts are:

```text
models/all-MiniLM-L12-v2
models/bge-base-en-v1.5
models/sec-bert-base
models/granite-embedding-english-r2
```

Repository locations and the submitted weight-file hashes are:

| Model | Repository | Weight file | Submitted SHA-256 |
|---|---|---|---|
| MiniLM | `https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2` | `pytorch_model.bin` | `54609dea3ff88f3167f049eeadbfe780b1173a3117bfac862134ebcd8ce33661` |
| BGE | `https://huggingface.co/BAAI/bge-base-en-v1.5` | `model.safetensors` | `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7` |
| SEC-BERT | `https://huggingface.co/nlpaueb/sec-bert-base` | `pytorch_model.bin` | `cbc4a6a0a79d20a52e2b5599455fb65e10c9b1a115d895bd38c181e05948051b` |
| Granite-r2 | `https://huggingface.co/ibm-granite/granite-embedding-english-r2` | `model.safetensors` | `23bd9625b32680f180703408ed9d8d7b3a9c1efe4c950111e3ae9d9695b21b29` |

The Granite model-tree hash recorded by the long-context run is:

```text
198d7df8b72a9fd77f9823e5e278f60df77c4577c08dd54e6df9888ade0c6d99
```

The release does not record retrievable Hugging Face commit revisions for all
four snapshots. A current download must be checked against the hashes above.
Matching only a weight file does not prove that tokenizer and pooling files are
identical. This is a known provenance limit for a model rerun.

The short-context cache is stored under:

```text
results/vintage_all_encoders_v3/embedding_cache/
```

Each active file is a float32 NumPy array with shape `(N_w, D_m)`. Rows follow
the ticker order in `results/windows_v3/<window>/universe.parquet`. MiniLM uses
`D_m=384`; BGE, SEC-BERT, and Granite-r2 use `D_m=768`. The active comparison
uses W1-W6 feature-window caches. Its script names the exact files, and its
result JSON records their hashes, so unrelated legacy cache files are ignored.

The cache directory occupies about 143 MB.
`scripts/build_dense_embedding_cache.py --device cpu` rebuilds it when the
filing corpora and model snapshots are present. The active dense comparison is
the feature-time census in
`results/acceptance_followup/run_20260730T065913Z/`.

## Restricted-input placement

A supplied restricted bundle must preserve this relative layout:

```text
data/universe.parquet
results/windows_v3/W1/...W7/...
results/ea_historical/windows/H1/...H13/...
results/vintage_descriptions/H1_descriptions.parquet
results/vintage_descriptions/W7_descriptions.parquet
results/vintage_all_encoders_v3/embedding_cache/*.npy
models/<four-model-directories>/...
```

Before running any experiment, compare these files with the `input_sha256`
maps in the frozen result JSON. A mismatched hash means the rerun is using a
different input, even if its schema and row count agree.
