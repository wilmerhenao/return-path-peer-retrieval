# Current Vintage-Text Results

Updated 2026-08-02. These results supersede the earlier symmetric-eligibility,
sampled-query vintage report.

## Protocol

- Every eligible security is a query; there is no query sample.
- A transition requires returns in W_t and W_{t+1} and a nonempty source=10K
  Item 1 filing strictly before the start of W_t.
- Target-window text coverage and content are not used for eligibility,
  features, or tuning.
- TF-IDF vocabulary and IDF are refitted on each feature-time corpus. Its
  hyperparameters are selected on H1->H2 through H12->H13 only.
- The rank-fusion weight is selected on those same twelve historical
  transitions and frozen before the six modern transitions.
- Relevance is max(0, realized W_{t+1} Ledoit--Wolf residual correlation);
  the canonical implementation is `scripts/metrics.py`.

## Historical-only selection

The winning TF-IDF configuration is `b`: 50,000 unigram-and-bigram features,
`min_df=2`, English stop words, Unicode accent stripping, and sublinear term
frequency. Its historical macro NDCG@5 is 0.328525.

For per-query average-percentile-rank fusion, the selected weight is 0.8 on the
return path and 0.2 on TF-IDF. Its historical macro NDCG@5 is 0.415529.
Both selection artifacts record `modern_data_used_for_selection=false`.

## Frozen modern evaluation

| Transition | N | Return path | TF-IDF | Frozen 80/20 blend | Blend gain |
|---|---:|---:|---:|---:|---:|
| W1->W2 | 1,614 | 0.3373 | 0.2864 | 0.3653 | +0.0279 |
| W2->W3 | 1,614 | 0.3251 | 0.3045 | 0.3625 | +0.0375 |
| W3->W4 | 1,617 | 0.3324 | 0.3063 | 0.3823 | +0.0499 |
| W4->W5 | 1,617 | 0.3478 | 0.2950 | 0.3790 | +0.0312 |
| W5->W6 | 1,631 | 0.3331 | 0.2926 | 0.3693 | +0.0362 |
| W6->W7 | 1,630 | 0.3380 | 0.3159 | 0.3793 | +0.0413 |
| **Mean +/- population SD** | **1,614--1,631** | **0.336 +/- 0.007** | **0.300 +/- 0.010** | **0.373 +/- 0.008** | **+0.037 mean** |

The return path exceeds standalone TF-IDF in 6/6 transitions. The frozen blend
improves the return path in 6/6, with gains from +0.028 to +0.050.

## Timing and global alignment

Holding feature-time universes and representations fixed, contemporaneous-text
NDCG@5 is 0.297 +/- 0.007 for TF-IDF, 0.257 +/- 0.006 for 512-token
Granite-r2, and 0.213 +/- 0.004 for MiniLM. The matching walk-forward values
are 0.300 +/- 0.010, 0.259 +/- 0.006, and 0.215 +/- 0.006. The largest
unrounded gap is 0.0035, so this diagnostic does not show a material timing
advantage for text.

On the 1,614-name W1 feature-time universe, the complete upper triangle has
1,301,691 unordered pairs. Pairwise Spearman alignment for BGE, 512-token
Granite-r2, MiniLM, and SEC-BERT ranges from 0.029 to 0.046 against W1 residual
correlation and from 0.027 to 0.049 against W2. The corresponding TF-IDF values
are 0.036 and 0.035; the return-path values are 1.000 and 0.103. These global
descriptive coefficients answer a different question from NDCG@5, which
measures the top of each query-specific ranking.

## Dense encoders

All dense encoders use the primary feature-time universe, N=1,614--1,631.
Granite is shown both at the original 512-token cap and over the full stored
excerpt.

| Encoder | WF NDCG@5 |
|---|---:|
| Granite-r2, full stored excerpt (8,192-token limit) | 0.279 +/- 0.006 |
| Granite-r2, 512 tokens | 0.259 +/- 0.006 |
| SEC-BERT, 512 tokens | 0.256 +/- 0.009 |
| BGE, 512 tokens | 0.256 +/- 0.007 |
| MiniLM, 128 tokens | 0.215 +/- 0.006 |

The Granite tokenizer audit contains 11,261 window-ticker observations. Median
length is 1,511 tokens, 94.627% exceed 512, maximum length is 2,449, and none
exceeds 8,192. The 8,000-character collection cap is reached by 85.872% of
observations. Processing each full stored excerpt improves Granite in all six
transitions, but this is not a complete-Item-1 test.

## Robustness

- At k=1/5/10/20, return-path macro NDCG is
  0.356/0.336/0.329/0.326 versus 0.317/0.300/0.288/0.280 for TF-IDF.
  The return path leads in all six transitions at every depth.
- Fixed 20- and 40-day half-life weightings attain 0.321 and 0.331, versus
  0.336 unweighted.
- On the fixed four-window-availability subset (N=1,556--1,628), concatenating
  M=1/2/3/4 separately standardized consecutive paths attains
  0.336/0.391/0.414/0.430. Every increment improves all six transitions. These
  are descriptive sensitivities, not modern-target selections.
- On that identical subset, the frozen 80/20 blend attains
  0.373/0.403/0.416/0.425. Its gains over the path are
  +0.038/+0.012/+0.003/-0.005, with 6/6, 6/6, 5/6, and 0/6 wins. On the common
  nine pre-modern four-window-eligible transitions, memory-specific selection
  chooses alpha 0.8 for M=1--2 and pure returns for M=3--4.
- In the feature-time frozen-list audit, lists freeze before target
  availability and incomplete query--list pairs are dropped without reranking.
  Mean query--basket residual correlations are
  0.208/0.186/0.248/0.118/0.007 for return path, TF-IDF, blend, SIC-2, and
  random; the corresponding positive-list shares are
  74.1%/77.9%/79.7%/71.6%/50.9%. The blend leads every method on mean
  correlation and positive-list share in 6/6. The 99.2%--99.6% figure is
  target-window evaluability conditional on joint return/text eligibility, not
  deployment-wide coverage. The basket statistic reuses the target residual
  paths and is not evidence of analyst utility, hedge effectiveness, or
  profitability.
- In the fully reranked residual-magnitude audit, excluding names whose maximum
  absolute raw FF5 residual exceeds 1.0 in either window removes 148 of 9,723
  transition-security rows (1.52%; N=1,587--1,604). Return-path/TF-IDF/blend
  NDCG@5 is 0.340/0.305/0.378; path-over-text and blend-over-path remain 6/6.
  The 0.5 either-window screen and both feature-only screens preserve those
  orderings. Either-window screening is retrospective because it inspects the
  target window; large residuals are not automatically data errors.
- A historical-only HistGradientBoosting pairwise fusion diagnostic attains
  0.308 +/- 0.009 and trails the one-window path and fixed blend in 6/6.
- Across the 18 transitions with a preceding path, two-window concatenation
  attains 0.421 +/- 0.028 versus 0.368 +/- 0.030 for one window and improves
  18/18.
- On an identical N=1,584--1,630 subset, a stale W_{t-1} path attains
  0.328 +/- 0.009 versus 0.301 +/- 0.010 for current W_t TF-IDF and leads 6/6.
  The matched current path attains 0.335 +/- 0.007.
- Excluding 118 tickers flagged by a vendor-suffix heuristic yields
  N=1,496--1,513 and 0.324 +/- 0.007 for the return path versus
  0.301 +/- 0.011 for TF-IDF, again 6/6. This is not an authoritative
  common-share classification.
- Replacing legacy symmetric eligibility with feature-time eligibility adds
  0--3 names and changes both modern macro means by less than 0.001.
- In 1,000 common-date circular block perturbations of each target window at
  both 5- and 10-day block lengths, conditional 95% envelopes are
  [0.031, 0.038] for return path minus TF-IDF and [0.029, 0.036] for blend
  minus return path. Every transition gap is positive in every replicate.
  These are conditional target-date sensitivities, not confidence intervals.

## Full 19-transition feature-time census

| Regime | Transitions | Return path | TF-IDF | Return-path wins |
|---|---:|---:|---:|---:|
| All | 19 | 0.364 +/- 0.033 | 0.318 +/- 0.022 | 18/19 |
| High volatility | 7 | 0.381 +/- 0.038 | 0.333 +/- 0.019 | 6/7 |
| Medium volatility | 6 | 0.358 +/- 0.030 | 0.317 +/- 0.022 | 6/6 |
| Low volatility | 6 | 0.348 +/- 0.017 | 0.302 +/- 0.012 | 6/6 |

The only exception is H1->H2: return path 0.308 versus TF-IDF 0.348. Historical
feature-time transition universes contain 1,264--1,547 securities; the modern
range is 1,614--1,631. Vintage-text coverage of the feature-window return
universe is
89.5%--93.8% historically and 82.3%--84.1% across W1--W6.
Odd-indexed transitions, which do not directly reuse adjacent windows within
their subset, retain 9/10 wins; even-indexed transitions retain 9/9.

These are finite-panel descriptive results. Adjacent transitions share a
window, and query outcomes share dates, candidates, and target matrices; no IID
query-bootstrap interval or transition sign-test inference is claimed.

## Immutable artifact paths

```text
results/acceptance_robustness/run_20260730T032543Z/
results/acceptance_followup/run_20260730T065913Z/
results/acceptance_full19_feature_time/run_20260730T032944Z/
results/acceptance_target_date_uncertainty/run_20260731T014000Z/
results/memory_fusion_audit/run_20260731T131959Z/
results/strict_feature_time_application/run_20260801T233929Z/
results/outlier_sensitivity/run_20260801T234540Z/
results/learned_fusion_followup/run_20260731T012556Z/
results/granite_vintage_token_audit/run_20260731T132708Z/
results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/
results/full_return_census/run_20260802T_full_return_authority_v2/
results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/
results/pairwise_alignment/run_20260806T_pairwise_authority_v3/
```

Run `shasum -a 256 -c SHA256SUMS.txt` inside each directory before using the
artifacts. The active full-return table is in the timestamped full-return
census above. The older `full_universe_ndcg_corrected.csv` is retained only as
a superseded audit record and is not a current number authority.

The 2,200-name candidate file was frozen on 2026-04-12 after ex-post matching of
current SEC metadata to GDELT coverage, including a 2025--2026 news filter. It
is not a point-in-time security master; historical results are conditional on
this survivor/news-visible cohort. Raw prices, residual matrices, filing text,
and embedding caches are not redistributed, so the frozen hashes define the
submitted artifact boundary.

The primary modern retrieval universe requires target-window return
availability and remains a retrospective benchmark. The strict application
audit removes that transition-level selection leakage by freezing rankings on
the feature-time-only universe and reporting subsequent attrition without
reranking; it still inherits the ex-post candidate cohort.
