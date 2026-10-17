# Frozen-blend follow-ups

These two runners add the previously selected **80% return / 20% TF-IDF**
percentile-rank blend to comparisons that originally reported only the separate
signals. Neither runner selects a new configuration or weight.

- `scripts/run_full19_frozen_blend.py` reuses the twelve historical selection
  scores and six modern scores, computes only H13→W1, and summarizes the 19
  transitions by the original volatility labels. It separately reports the
  **12 selection transitions** and **7 transitions outside selection** (the
  bridge plus six modern transitions). The 19-transition and regime aggregates
  include selection observations and are descriptive.
- `scripts/run_frozen_blend_depth_ticker.py` evaluates k=1/5/10/20 on the six
  original modern universes, then applies the original ticker-syntax exclusion,
  refits TF-IDF with unchanged settings and reranks on the restricted universes.
  It does not evaluate new coverage, recency-weighting or other variants.

Both use configuration b: 50,000 unigram/bigram features, `min_df=2`, English
stop words, Unicode accent stripping and sublinear term frequency. Vocabulary
and IDF are refitted on each relevant feature corpus. Average percentile ranks,
self exclusion, seeded metric ties, relevance and aggregation are unchanged.

## Public authority extracts

`evidence/frozen_blend/authorities/` contains five small extracts used by the
runners, plus their `SHA256SUMS.txt`:

| Original authority | Fields retained for replay |
| --- | --- |
| `01_tfidf_historical_tuning.json` | Selected settings and selection protocol |
| `02_blend_historical_tuning_modern_frozen.json` | Fixed-weight transition scores, counts and protocol |
| `03_modern_k_sensitivity.json` | Return-path/TF-IDF transition scores at each depth |
| `06_common_share_heuristic_exclusion.json` | Exclusion rule and restricted comparator scores/counts |
| `full19_feature_time_census.json` | Transition comparator scores/counts and original volatility labels |

All retain the original input SHA-256 inventory. There are no raw prices,
residual arrays, filing texts, embeddings, ticker lists or per-query outcomes.
The extract's checksum identifies the public file. The separately recorded
`source_artifact_sha256` identifies the original complete result, **not** the
extract. The extraction is a documented field selection, not a claim that the
two files have identical bytes. The helper pins the five original identities
and verifies each public file against the shipped checksum manifest.

## Required prepared inputs

The public metadata supports auditing results; recomputation still requires the
original frozen input tree, supplied separately:

```text
results/ea_historical/windows/H1/ ... H13/
results/windows_v3/W1/ ... W7/
results/vintage_descriptions/H1_descriptions.parquet ... W7_descriptions.parquet
results/source_cache/ff5_daily.parquet   # full19 audit provenance
```

Each return-window directory needs `universe.parquet`, `meta.json`,
`ts_features.npy` and `corr_target.npy`. The runners check the complete archived
input inventory, including historical inputs underlying reused scores; the depth
runner therefore also requires historical files for verification. The full19
runner retains the original volatility labels rather than recomputing terciles.
New downloads with different hashes are not accepted as an exact replay.

After installing `requirements.lock`, run from this repository:

```bash
python scripts/run_full19_frozen_blend.py --help
python scripts/run_frozen_blend_depth_ticker.py --help

python scripts/run_full19_frozen_blend.py --input-root /path/to/frozen-input-tree
python scripts/run_frozen_blend_depth_ticker.py --input-root /path/to/frozen-input-tree
```

`--authority-root` selects another copy of the five extracts and checksum
manifest; `--output-root` selects an output parent. Each run creates a fresh
directory and refuses to overwrite one. Private absolute input paths are not
written to generated provenance. `--help` does not access experiment inputs.

## Source identity and replay checks

The public core has documentation and provenance fixes made after the original
experiments. Its bytes therefore differ from the original recorded source.
`scripts/frozen_blend_inputs.py` explicitly records and checks both the archived
and reviewed execution hashes. The metrics differences are comments/docstrings;
core differences concern documentation, provenance collection and external-path
labels. Numerical functions and settings are unchanged. An unreviewed code hash
fails. The old full19 producer's identity is retained as historical provenance;
that producer is not executed by this follow-up and need not be supplied.

Every numerical input hash must still match exactly. Original result identities
are checked separately from curated-file hashes. Lockfile identity differences
and actual runtime library versions are recorded rather than described as an
identical environment. Exact numerical replay remains mandatory:

- Full19: all 18 reused baseline pairs agree across authorities; both bridge
  comparator scores reproduce exactly on the same queries and candidates.
- Depth/ticker: all 18 primary k=5 scores, 48 depth comparator scores and 12
  restricted comparator scores reproduce exactly before new outputs are saved.

All checks are conditional on the original retrospective panel. They do not
remove survivorship, target-availability selection, shared-window dependence or
the limitations of the ticker heuristic. No new inferential tests are added.
