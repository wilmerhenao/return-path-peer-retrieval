# Return-path peer retrieval

Research code for **When Text Complements the Return Path: A Strong Baseline for Behavioral Equity-Peer Retrieval**, accepted at ICAIF 2026, paper 820.

**Wilmer E. Henao** · Advanced Engineering, Invesco · New York, USA

Contact: wilmer.henao@invesco.com

Repository: [wilmerhenao/return-path-peer-retrieval](https://github.com/wilmerhenao/return-path-peer-retrieval). Public release is planned by publication.

The paper asks which representations retrieve firms that will co-move with a query firm in the next period after Fama–French five-factor adjustment. It compares recent residual-return paths with vintage Form 10-K Item 1 text represented by TF-IDF and frozen dense encoders. A return path is a strong baseline; filing text adds value when return history is short, and its incremental benefit diminishes with longer histories.

This repository contains code and tests for part of the study. The table below shows what is included. The manuscript, market data, filing corpora, model weights, embedding caches, and saved results are not included.

## Results reported in the paper

These are mean NDCG@5 values over the six 2024–2026 transitions on the primary vintage-text comparison universe. They summarize the paper; the inputs and result files needed to reproduce these values are not included here.

| Representation | Mean NDCG@5 |
| --- | ---: |
| One-period return path | 0.336 |
| Vintage Item 1 TF-IDF | 0.300 |
| Best dense comparator: Granite-r2, full stored excerpt | 0.279 |
| Historically selected 80% return / 20% TF-IDF percentile-rank blend | 0.373 |

The blend weight and TF-IDF configuration were selected on twelve earlier transitions, then held fixed for the six modern evaluations. The blend improves on the one-period path in all six. With four periods of return history, adding text gives no improvement. That history comparison uses a different common set of eligible firms, so its means are not directly comparable with the table above.

“Full stored excerpt” means the capped text collected by the study, not necessarily all of Item 1. The included dense-cache builder implements short-context encodings and does not reproduce the full-excerpt Granite result. The tested setting is a retrospective U.S. cohort; these findings do not establish valuation peers, trading performance, or retrieval for companies without observed return histories.

## Paper-to-code map

| Component | Included implementation |
| --- | --- |
| Positive-correlation NDCG, self-exclusion, deterministic ties | [scripts/metrics.py](scripts/metrics.py) |
| Complete-case return windows, FF5 residuals, standardized paths and correlation targets | [scripts/build_windows.py](scripts/build_windows.py) |
| Pre-window Item 1 collection and extraction | [scripts/fetch_vintage_10k.py](scripts/fetch_vintage_10k.py), [scripts/sec_filing_text.py](scripts/sec_filing_text.py) |
| Frozen short-context text embeddings | [scripts/build_dense_embedding_cache.py](scripts/build_dense_embedding_cache.py): MiniLM uses 128 tokens; the other released model configurations use 512 |
| Historical TF-IDF/rank-blend selection and modern evaluation | [scripts/run_acceptance_robustness.py](scripts/run_acceptance_robustness.py) |
| Stale returns, path-only history length, and short-context dense follow-up | [scripts/run_adversarial_followup.py](scripts/run_adversarial_followup.py) |
| Return-path, spectral, SIC-2 and random baseline census | [scripts/run_full_return_census.py](scripts/run_full_return_census.py) |
| Contemporaneous text and W1/W2 pairwise alignment diagnostics | [scripts/run_same_window_text_diagnostic.py](scripts/run_same_window_text_diagnostic.py), [scripts/run_pairwise_alignment.py](scripts/run_pairwise_alignment.py) |
| Geometry, metric, window and filing-extraction checks | [tests/](tests/) |

Some analyses in the paper have no runner in this release: full-excerpt Granite encoding, the fixed-list application audit, target-date resampling, the full 19-transition census, the matched-history path/blend comparison and blend-weight selection for each history length, HGB learned fusion, and the multimodal CNN–text encoder. The included history curve uses return paths alone. These scripts cannot reproduce the paper's matched-history fusion table or the other analyses listed here.

## Installation

Python 3.13.2 was used for the recorded CPU experiments. From this checkout's root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m unittest discover -s tests -v
```

Install the dependencies from `requirements.lock`. Model weights must be obtained separately. The tests use synthetic and local fixtures to check the code; they do not reproduce the paper's empirical tables.

### A small return-path example

This demonstrates the geometry and query exclusion using synthetic residuals. Each row is one company and each column is a common date. The example does not download data or demonstrate next-period predictive performance.

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

The equality is a same-period identity: a unit-normalized standardized residual path has an inner product equal to sample residual correlation. Prediction is tested separately by evaluating a feature-period ranking against a later period's positive residual correlations.

## Inputs required for empirical runs

| Input | Required role |
| --- | --- |
| `data/universe.parquet` | Unique `ticker` values; `sic` is needed for the SIC-2 baseline |
| `results/windows_v3/W1/` through `W7/` | Modern residual paths, target matrices, ticker order and metadata |
| `results/ea_historical/windows/H1/` through `H13/` | Historical configuration-selection inputs |
| `results/vintage_descriptions/*_descriptions.parquet` | Vintage `ticker`, `source`, `item1_text`, and `filing_date` records |
| Local model directories and embedding caches | Inputs for the dense-encoder comparisons |
| Saved historical-selection results and named caches | Dependencies of the follow-up, same-window, and pairwise scripts |

The 2,200-name candidate panel was assembled retrospectively using SEC metadata and GDELT coverage. This repository does not contain that SEC-to-GDELT universe-construction workflow. Filing excerpts have an 8,000-character collection cap. The window builder can retrieve fresh Yahoo-derived histories, but vendor revisions and changed source coverage prevent a guarantee of exact reconstruction of the submission's inputs.

Inspect available options before running a stage:

```bash
python scripts/build_windows.py --help
python scripts/fetch_vintage_10k.py --help
python scripts/build_dense_embedding_cache.py --help
python scripts/run_acceptance_robustness.py --help
python scripts/run_full_return_census.py --help
python scripts/run_same_window_text_diagnostic.py --help
python scripts/run_pairwise_alignment.py --help
```

Several scripts use fixed project-relative paths or specific cache identifiers. The same-window and pairwise diagnostics also need saved historical-selection results that are absent from this release. Use `--help` to check their options and required inputs.

`run_adversarial_followup.py` hashes three legacy source files to record which code was used: `scripts/exp_vintage_all_encoders.py`, `src/s01_universe.py`, and `src/config.py`. These files are absent from this release, and the runner raises `FileNotFoundError` without them. Data inputs alone are not enough to run it. Its source shows the calculations, but the checkout lacks those dependencies.

The SEC acquisition driver requires a descriptive user agent supplied through its documented option or environment variable.

Use a new output directory for each run and record the inputs and settings. The paper's numbers come from its saved results. New downloads may produce different values.

## Citation

If you use this code, cite the software and the accompanying paper. [CITATION.cff](CITATION.cff) contains the software metadata for GitHub's **Cite this repository** panel. No software DOI has been assigned yet.

The paper citation below is provisional. Add the final proceedings metadata and paper DOI when assigned by ACM.

```bibtex
@misc{henao2026textreturnpath,
  author = {Henao, Wilmer E.},
  title = {When Text Complements the Return Path: A Strong Baseline for Behavioral Equity-Peer Retrieval},
  year = {2026},
  note = {Accepted for ICAIF 2026},
  url = {https://github.com/wilmerhenao/return-path-peer-retrieval}
}
```

Record the software version or commit used for your run. Once a release is archived on Zenodo, cite that release's DOI. The software DOI and the ACM paper DOI identify different works. The archive will contain the released code, not the omitted data or results.

## Archiving a release

[The release guide](docs/ARCHIVING.md) explains how to archive a reviewed version on Zenodo and obtain a software DOI. This can be done when the repository is ready for public release. If the camera-ready paper needs the DOI sooner, a manual Zenodo draft can reserve one before publication.

## License

The software in this code-only repository is licensed under [Apache-2.0](LICENSE). Market data, filing text, model weights, and other upstream materials retain their own terms and are not distributed here.
