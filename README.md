# Return-path peer retrieval

[![DOI: 10.5281/zenodo.23342563](assets/zenodo-doi.svg)](https://doi.org/10.5281/zenodo.23342563)

Research code accompanying **When Text Complements the Return Path: A Strong Baseline for Behavioral Equity-Peer Retrieval**, accepted at ICAIF 2026.

**Wilmer E. Henao** · Advanced Engineering, Invesco · New York, USA

wilmer.henao@invesco.com

The paper studies companies that move together after Fama-French five-factor adjustment. It compares recent residual-return paths with vintage Form 10-K Item 1 text and evaluates retrieved peers using next-period positive residual correlations. Return paths are a strong baseline. Filing text improves retrieval when return history is short; its contribution diminishes with longer histories.

The study spans **February 2020 through January 2026**, including the **COVID-19 crash and the 2022 rate shock**. Its 20 return windows form 19 consecutive window-to-window comparisons: thirteen historical windows cover February 2020 to April 2024, followed by seven modern windows from May 2024 to January 2026. The retrospective candidate panel contains 2,200 securities; the eligible sample varies with each comparison's return-history and filing requirements.

## Results

The table below reports the primary comparison on the **six modern transitions from 2024 to 2026**, with 1,614 to 1,631 eligible companies per transition. NDCG@5 measures retrieval quality among each query's first five peers, using positive next-period residual correlation as relevance. Values are mean NDCG@5 ± population standard deviation across the six transition means.

| Representation | NDCG@5 |
| --- | ---: |
| One-period return path | 0.336 ± 0.007 |
| Vintage Item 1 TF-IDF | 0.300 ± 0.010 |
| Best dense comparator: Granite-r2, full stored excerpt | 0.279 ± 0.006 |
| Frozen 80% return / 20% TF-IDF rank blend | 0.373 ± 0.008 |

The TF-IDF configuration and blend weight were selected on twelve earlier transitions and held fixed for modern evaluation. The blend improves on the one-period path in all six tests. With four periods of return history, adding text gives no improvement; that comparison uses a separate common set of eligible companies.

The historical extension compares return paths with TF-IDF across the full 19-transition panel, including the twelve transitions used for configuration selection. The return path leads in 18 of 19 transitions, with mean NDCG@5 of 0.364, versus 0.318 for TF-IDF. These full-panel means summarize different periods and eligible samples from the modern comparison above.

The study evaluates behavioral peers in a retrospective U.S. cohort. It does not evaluate valuation peers or trading performance. “Full stored excerpt” refers to the collected, capped filing text rather than the complete Item 1 section.

## Code

| Component | Implementation |
| --- | --- |
| Positive-correlation NDCG, query exclusion and deterministic ties | [metrics.py](scripts/metrics.py) |
| Complete-case windows, Fama-French five-factor residuals and correlation targets | [build_windows.py](scripts/build_windows.py) |
| Pre-window Item 1 collection and extraction | [fetch_vintage_10k.py](scripts/fetch_vintage_10k.py), [sec_filing_text.py](scripts/sec_filing_text.py) |
| Frozen short-context embeddings: MiniLM at 128 tokens; BGE, SEC-BERT and Granite-r2 at 512 | [build_dense_embedding_cache.py](scripts/build_dense_embedding_cache.py) |
| Historical TF-IDF and rank-blend selection; modern evaluation | [run_acceptance_robustness.py](scripts/run_acceptance_robustness.py) |
| Stale returns, return-history length and short-context dense comparisons | [run_adversarial_followup.py](scripts/run_adversarial_followup.py) |
| Return-path, spectral-dot, SIC-2 and random baselines | [run_full_return_census.py](scripts/run_full_return_census.py) |
| Contemporaneous text and W1/W2 pairwise alignment | [run_same_window_text_diagnostic.py](scripts/run_same_window_text_diagnostic.py), [run_pairwise_alignment.py](scripts/run_pairwise_alignment.py) |
| Geometry, metrics, window construction, filing extraction and input provenance | [tests/](tests/) |

The repository contains source code and tests. It does not distribute the manuscript, market data, filing corpus, model weights, embedding caches or saved results.

Several reported analyses have no runner here: full-excerpt Granite encoding, the fixed-list application audit, target-date resampling, the full 19-transition census, matched-history text fusion and its weight selection, learned HGB fusion, and the multimodal CNN-text encoder. The included history comparison evaluates return paths alone. Reproducing those analyses requires their original inputs and additional code.

## Installation

The recorded CPU experiments used Python 3.13.2. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m unittest discover -s tests -v
```

Model weights are obtained separately. Tests use synthetic and local fixtures; empirical runs require the inputs described below.

### Return-path example

Each row below represents one company and each column a common date. The example verifies the same-period correlation identity and retrieves two peers without including the query company.

```python
import numpy as np
from scripts.metrics import top_k_indices
from scripts.run_acceptance_robustness import return_path_scores

rng = np.random.default_rng(42)
residuals = rng.normal(size=(6, 8))
paths = (residuals - residuals.mean(axis=1, keepdims=True)) / residuals.std(
    axis=1, keepdims=True, ddof=0
)
scores = return_path_scores(paths)
np.testing.assert_allclose(scores, np.corrcoef(residuals), atol=1e-6)

# Rank two peers for company 0, excluding the company itself.
peers = top_k_indices(scores[0], k=2, excluded=0, query_seed=0)
print(peers)
```

A unit-normalized standardized residual path has an inner product equal to sample residual correlation. The paper tests next-period retrieval separately, by evaluating a feature-period ranking against later residual correlations.

## Empirical inputs

The benchmark runners expect the paper's directory structure, ticker ordering and named caches.

| Input | Contents |
| --- | --- |
| `data/universe.parquet` | Unique `ticker` values; `sic` for the SIC-2 baseline |
| `results/windows_v3/W1/` through `W7/` | Modern residual paths, correlation targets, ticker order and metadata |
| `results/ea_historical/windows/H1/` through `H13/` | Historical configuration-selection windows |
| `results/vintage_descriptions/*_descriptions.parquet` | Vintage `ticker`, `source`, `item1_text` and `filing_date` records |
| Local model directories and embedding caches | Frozen dense-encoder inputs |
| Saved selection and walk-forward result summaries | Selected settings and reference means used by the same-window and pairwise diagnostics |

The 2,200-security candidate panel was assembled retrospectively from SEC metadata and GDELT coverage. That universe-construction workflow is not included. Filing collection caps excerpts at 8,000 characters. Fresh price downloads and filing collection can differ from the original inputs because of vendor revisions and changes in source coverage.

Each script describes its options through `--help`, for example:

```bash
python scripts/build_windows.py --help
python scripts/fetch_vintage_10k.py --help
python scripts/build_dense_embedding_cache.py --help
python scripts/run_acceptance_robustness.py --help
python scripts/run_adversarial_followup.py --help
python scripts/run_full_return_census.py --help
python scripts/run_same_window_text_diagnostic.py --help
python scripts/run_pairwise_alignment.py --help
```

The SEC collector requires a descriptive user agent through `--sec-user-agent` or `SEC_USER_AGENT`. Analysis runners record input hashes and write to new output directories. The follow-up runner also checks expected outcomes for the original benchmark sample. Embedding-cache reuse requires matching inputs, model files and device settings; `--force` rebuilds an existing cache.

## Software citation

Software citation metadata is available in [CITATION.cff](CITATION.cff).

## License

The software is licensed under [Apache-2.0](LICENSE). Market data, filing text and model weights retain their respective terms and are not distributed here.
