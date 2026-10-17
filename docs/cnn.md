# CNN V1 historical continuation experiment

This is the CNN–text diagnostic reported in the paper. It combines a temporal
CNN over standardized residual-return paths, a learned projection of frozen
512-token Granite-r2 embeddings, and a direct return-cosine skip. The comparator
is the fixed 80% return / 20% TF-IDF percentile-rank blend on identical eligible
queries and candidates. The CNN and comparator use different text representations.

The [versioned result summary](../paper_results/cnn_historical_v1.json) records
the original result artifact's hash, both width candidates, seed results, continuation
slots, and gate decision. Width 32 was selected: mean NDCG@5 was approximately
0.377 versus 0.409 for the blend, with zero wins in six slots. The gate failed;
the original run stopped without final checkpoints or any CNN evaluation on
modern targets. These are historical results, not modern scores.

## Scope and unchanged settings

The public runner preserves V1's architecture, loss, training, historical
selection, comparator, and failure gate. It ends at the historical continuation
decision even if a new run passes. The original private final-fit/modern-lock
infrastructure and modern evaluator are not part of this release. There is no
modern-evaluation command. V2 is a separate experiment and is not included.

The original [configuration](../experiments/multimodal_peer_encoder/config/historical_protocol.json)
is retained verbatim, including its unused descriptions of later stages. Those
descriptions do not make later stages executable in the public runner. Source
and execution hashes from this portable extraction differ from the original
local runner; they must not be described as the original run's hashes.

The fixed settings are widths 32/64, seeds 11/29/47, 40 epochs, AdamW learning
rate 0.001 and weight decay 0.0001, float32 CPU execution, eight Torch threads,
gradient clipping at 1.0, and score/target temperatures of 0.1. Training uses
graded positive-correlation relevance with listwise softmax cross-entropy,
not binary top-five labels or direct NDCG optimization. Each epoch makes one
update per training transition; no validation-based checkpoint selection or
early stopping is used.

| Fold | Training transitions | Continuation slots |
| --- | --- | --- |
| 1 | H1→H2 through H6→H7 | H7→H8, H8→H9 |
| 2 | H1→H2 through H8→H9 | H9→H10, H10→H11 |
| 3 | H1→H2 through H10→H11 | H11→H12, H12→H13 |

Three seed scores are averaged within each continuation slot. The selected
width must improve the equal-slot mean by at least 0.005 and win at least four
of the six slots. There is no fallback to the other width after a gate failure.
The slots are held out from their corresponding CNN fold fits, but all six lie
inside the twelve transitions used earlier to select the comparator's TF-IDF
configuration and 80/20 weight. They are not untouched validation for the blend.
The six modern transitions W1→W2 through W6→W7 were never evaluated by this CNN.
The research team already knew the blend's modern results when V1 was proposed.

## Inputs

Run from the repository root. The release does not include returns, filing text,
Granite weights, or embedding caches. Supply the original inputs to replay the
published numbers; differently acquired inputs support a new run, not an exact
reproduction claim. There are no downloads or data reads when requesting help.

The historical loader uses only H1–H13:

- Each historical window directory has `universe.parquet` with `ticker` and
  `company_idx`, and `meta.json` with `window`, `start`, and `end`. The feature
  windows H1–H12 also need `ts_features.npy` (security × trading date). Target
  windows H2–H13 need `corr_target.npy` (security × security), containing the
  next-window Ledoit–Wolf residual correlations. Array rows follow `company_idx`.
- The vintage directory has `H1_descriptions.parquet` through
  `H12_descriptions.parquet`, with `ticker`, `window`, `source`, `item1_text`,
  and ISO `filing_date`. Only nonempty `source=10K` text filed strictly before
  the feature-window start is eligible.
- The cache directory has `cache_manifest.json`, `H*_eligible_tickers.json`, and
  `H*_granite_512.npy` for H1–H12. Each embedding row has 768 float32 coordinates.
  Cache hashes bind the matrices, ticker order, exact filing inputs, model
  snapshot, and inference settings. Feature and target securities are aligned
  independently, then intersected and sorted; every eligible security is a query.

## Commands

Install the root `requirements.lock` in a separate environment. The original
CPU run used Python 3.13.2, NumPy 2.4.2, Polars 1.39.3, SciPy 1.17.1,
scikit-learn 1.8.0, Torch 2.11.0, sentence-transformers 5.3.0, and Transformers
5.5.0. The cache builder enforces the four embedding-package pins and the
original Granite model-tree/weight hashes; it explicitly upcasts the frozen
bfloat16 checkpoint to float32 before CPU encoding. It does not silently accept
another checkpoint or regenerate a mismatched cache.

```sh
python -m experiments.multimodal_peer_encoder.build_historical_embeddings --help
python -m experiments.multimodal_peer_encoder.train_historical --help

python -m experiments.multimodal_peer_encoder.build_historical_embeddings \
  --historical-root /path/to/historical/windows \
  --vintage-root /path/to/vintage_descriptions \
  --model-dir /path/to/granite-embedding-english-r2 \
  --output-dir /path/to/new/historical_text_cache

python -m experiments.multimodal_peer_encoder.train_historical \
  --historical-root /path/to/historical/windows \
  --vintage-root /path/to/vintage_descriptions \
  --cache-dir /path/to/historical_text_cache \
  --output-dir /path/to/new/cnn_v1_run
```

The runner requires a new output directory. It writes `historical_selection.json`
and, if the gate fails, `STOP`. A passing public replay reports
`HISTORICAL_GATE_PASSED` and still performs no final fits or modern evaluation.
Inputs and aligned tensors are fingerprinted before and after selection.

Install `requirements-dev.txt` and run focused tests with
`python -m pytest tests/test_cnn_*.py`. Tests use synthetic
inputs and encoding stubs; they check model geometry, graded loss, ranking ties,
vintage timing, ticker alignment, cache identity, fixed settings, and both gate
outcomes without fitting the paper experiment or accessing private data.
