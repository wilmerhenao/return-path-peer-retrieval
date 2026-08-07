# Active Manuscript Number Provenance

Updated 2026-08-07. This file records the evidence used by the current
feature-time, all-query manuscript. Earlier sampled-query, symmetric-eligibility,
query-bootstrap, half-sample, and sign-test claims are superseded.

## Authoritative artifacts

| Scope | Artifact |
|---|---|
| W1--W7 construction | `results/windows_v3/windows_summary.json` |
| H1--H13 construction metadata | `results/historical_windows_summary.json` |
| Full-return census | `results/full_return_census/run_20260802T_full_return_authority_v2/full_return_census.json` |
| Historical TF-IDF selection | `results/acceptance_robustness/run_20260730T032543Z/01_tfidf_historical_tuning.json` |
| Frozen fusion, modern feature-time census | `results/acceptance_robustness/run_20260730T032543Z/02_blend_historical_tuning_modern_frozen.json` |
| Retrieval-depth sensitivity | `results/acceptance_robustness/run_20260730T032543Z/03_modern_k_sensitivity.json` |
| Fixed recency weighting | `results/acceptance_robustness/run_20260730T032543Z/04_fixed_ewma_return_path.json` |
| Two-window concatenation | `results/acceptance_robustness/run_20260730T032543Z/05_two_window_concatenated_return_path.json` |
| Ticker-syntax exclusion | `results/acceptance_robustness/run_20260730T032543Z/06_common_share_heuristic_exclusion.json` |
| Feature-time versus legacy symmetric eligibility | `results/acceptance_robustness/run_20260730T032543Z/07_symmetric_vs_feature_time_reconciliation.json` |
| Stale control, path memory, 18-transition memory extension, and feature-time dense rescoring | `results/acceptance_followup/run_20260730T065913Z/adversarial_followup_census.json` |
| Full 19-transition feature-time census | `results/acceptance_full19_feature_time/run_20260730T032944Z/full19_feature_time_census.json` |
| Conditional target-date block perturbation | `results/acceptance_target_date_uncertainty/run_20260731T014000Z/target_date_block_bootstrap.json` |
| Matched memory--fusion audit | `results/memory_fusion_audit/run_20260731T131959Z/memory_fusion_audit.json` |
| Feature-time frozen-list audit | `results/strict_feature_time_application/run_20260801T233929Z/strict_application_census.json`; per-query positive-list shares from `strict_application_per_query.csv.gz` |
| Fully reranked residual-magnitude sensitivity | `results/outlier_sensitivity/run_20260801T234540Z/outlier_sensitivity_census.json` |
| Historical-only nonlinear fusion | `results/learned_fusion_followup/run_20260731T012556Z/learned_fusion_followup.json` |
| Gated two-tower historical continuation | `experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v1/historical_selection.json` |
| Granite tokenizer and collection-cap audit | `results/granite_vintage_token_audit/run_20260731T132708Z/summary.json` |
| Full-stored-excerpt Granite census | `results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/granite_longctx_census.json` |
| Same-window text timing diagnostic | `results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/same_window_text_diagnostic.json` |
| Complete pairwise alignment diagnostic | `results/pairwise_alignment/run_20260806T_pairwise_authority_v3/pairwise_alignment.json` |

Every timestamped result run listed above contains `SHA256SUMS.txt`; its
checksums must pass before the numbers below are used. The sealed V1
historical-selection record is instead covered by the release-level
`ARTIFACT_MANIFEST.json`.

## Primary modern results

All values are unweighted means and population standard deviations across the
six observed W1->W2 through W6->W7 transition means. Every eligible security is
a query.

| Claim | Manuscript value | Artifact field or derivation | Status |
|---|---:|---|---|
| Return-window N | 1,921--1,995 | min/max `W1.N` ... `W7.N` | Verified |
| Feature-time vintage N | 1,614--1,631 | min/max modern `n_universe` | Verified |
| Historical return-window N and T | N=1,389--1,666; T=62--86 | min/max `historical_windows_summary.json` | Verified |
| Full-return query N | 1,919--1,979 | min/max `full_return_census.json` ReturnPath rows | Verified |
| Return path, feature-time | 0.336 +/- 0.007 | modern `return_path_ndcg` | Verified |
| TF-IDF, feature-time | 0.300 +/- 0.010 | modern `tfidf_ndcg` | Verified |
| Frozen 80/20 blend | 0.373 +/- 0.008 | modern `blend_ndcg` | Verified |
| Blend minus return path | +0.037 mean; +0.028 to +0.050 | `blend_minus_return_path` | Verified |
| Historical-only nonlinear fusion | 0.308 +/- 0.009; trails path and fixed blend 6/6 | learned-fusion `modern_evaluation.summary` | Verified |
| Gated two-tower historical continuation | 0.377 vs 0.409 for the frozen blend; 0/6 wins; per-transition seed SD 0.000528--0.001074 | selected-width rows in `historical_selection.json` | Verified |
| Feature-time frozen-list query--basket correlation | return 0.208; TF-IDF 0.186; blend 0.248; SIC-2 0.118; random 0.007; blend leads every method 6/6 | frozen-list audit `macro_summary` and `per_transition` | Verified |
| Positive frozen-list share | return 74.1%; TF-IDF 77.9%; blend 79.7%; SIC-2 71.6%; random 50.9%; equal-transition macro mean among evaluable lists | `strict_application_per_query.csv.gz`, derived as mean over transitions of `query_basket_correlation > 0` | Verified |
| Conditional target-window evaluability | 99.2%--99.6%; incomplete frozen lists dropped without reranking | frozen-list audit `macro_summary`; conditional on joint feature-time return/text eligibility | Verified |
| Retrospective 1.0 residual-magnitude screen | 148/9,723 rows excluded (1.52%); N=1,587--1,604; path/TF-IDF/blend NDCG@5 0.340/0.305/0.378; path and blend retain 6/6 ordering | outlier-sensitivity `screen_summaries.either_window_max_abs_le_1p0` | Verified |
| Other residual-magnitude screens | 0.5 either-window and 1.0/0.5 feature-only screens retain path-over-TF-IDF and blend-over-path ordering in 6/6 | outlier-sensitivity `screen_summaries` | Verified |
| Full-return path | 0.339 +/- 0.007 | `full_return_census.json`, `ReturnPath` | Verified |
| Full-return spectral dot | 0.336 +/- 0.006 | `full_return_census.json`, `Spectral-dot-128` | Verified |
| Full-return SIC-2 | 0.213 +/- 0.009 | `full_return_census.json`, `SIC-2` | Verified |
| Full-return random | 0.118 +/- 0.006 | `full_return_census.json`, `Random` | Verified |

TF-IDF configuration `b` (50,000 unigram-and-bigram features, `min_df=2`,
English stop words, Unicode accent stripping, and sublinear term frequency)
maximizes the macro historical NDCG@5 over H1->H2 through H12->H13 at
0.328525. The same historical transitions select return weight 0.8 and text
weight 0.2. The selection artifacts explicitly record that modern data were not
used.

## Dense encoders

Every dense encoder is scored on the same primary feature-time universe,
N=1,614--1,631. Granite-r2 was additionally rerun with an effective 8,192-token
limit. Values are census means and population standard deviations across
transitions.

| Encoder | NDCG@5 |
|---|---:|
| Granite-r2, full stored excerpt (8,192-token limit) | 0.279 +/- 0.006 |
| Granite-r2, 512-token cap | 0.259 +/- 0.006 |
| SEC-BERT | 0.256 +/- 0.009 |
| BGE | 0.256 +/- 0.007 |
| MiniLM | 0.215 +/- 0.006 |

The Granite tokenizer audit covers 11,261 window-ticker excerpts from W1--W6.
Median length is 1,511 Granite tokens, 94.627% exceed 512 tokens, maximum length
is 2,449, and none exceeds 8,192. The collection cap is reached by 9,670
observations (85.872%), so this is not a complete-Item-1 experiment. Processing
the full stored excerpt improves Granite over the 512-token result in all six
transitions, but Granite remains below both TF-IDF and the return path.

## Timing and global-alignment diagnostics

The same-window diagnostic keeps the primary feature-time universes and frozen
representations but scores text similarity against contemporaneous rather than
next-window correlations. TF-IDF, 512-token Granite-r2, and MiniLM attain
0.297 +/- 0.007, 0.257 +/- 0.006, and 0.213 +/- 0.004. Their corresponding
walk-forward values are 0.300 +/- 0.010, 0.259 +/- 0.006, and
0.215 +/- 0.006. The largest unrounded absolute gap is 0.0035.

The global-alignment diagnostic is a complete upper-triangle census on the
1,614-name W1 feature-time universe. Across 1,301,691 unordered pairs,
Spearman correlation for the four dense filing encoders ranges from 0.029 to
0.046 against W1 residual correlation and from 0.027 to 0.049 against W2.
The corresponding TF-IDF coefficients are 0.036 and 0.035. The return-path
coefficients are 1.000 in W1 by construction and 0.103 in W2. These are
descriptive coefficients. No pair-level IID p-value is reported because pair
observations share securities and target paths.

## Robustness and extension

| Claim | Manuscript value | Artifact |
|---|---:|---|
| Return path vs TF-IDF at k={1,5,10,20} | 0.356/0.336/0.329/0.326 vs 0.317/0.300/0.288/0.280; 6/6 wins at each depth | `03_modern_k_sensitivity.json` |
| Fixed half-life 20/40 | 0.321/0.331 vs 0.336 unweighted | `04_fixed_ewma_return_path.json` |
| Modern memory curve, fixed M=4-availability subset | M=1/2/3/4: 0.336/0.391/0.414/0.430; N=1,556--1,628; each increment improves 6/6 | `adversarial_followup_census.json` |
| Frozen 80/20 fusion across memory | 0.373/0.403/0.416/0.425; gains +0.038/+0.012/+0.003/-0.005; wins 6/6, 6/6, 5/6, 0/6 | `memory_fusion_audit.json` |
| Stale-return control | stale path 0.328 +/- 0.009 vs current TF-IDF 0.301 +/- 0.010; N=1,584--1,630; 6/6 | same |
| Two-window full extension | 0.421 +/- 0.028 vs one-window 0.368 +/- 0.030; 18/18 improvements | same |
| Ticker-syntax exclusion | 118 flagged; N=1,496--1,513; 0.324 +/- 0.007 vs 0.301 +/- 0.011; 6/6 | `06_common_share_heuristic_exclusion.json` |
| Feature-time versus symmetric eligibility | 0--3 added; both macro means change by less than 0.001 | `07_symmetric_vs_feature_time_reconciliation.json` |
| Conditional target-date sensitivity | 1,000 common-date circular block replicates at 5/10 days; RP minus TF-IDF 95% envelope [0.031, 0.038]; blend minus RP [0.029, 0.036]; every transition gap positive in every replicate | `target_date_block_bootstrap.json` |
| Full extension | Return path 0.364 +/- 0.033; TF-IDF 0.318 +/- 0.022; 18/19 wins | `full19_feature_time_census.json` |
| Alternating transition subsets | 9/10 odd-index wins; 9/9 even-index wins | same |
| High volatility, 7 transitions | 0.381 +/- 0.038 vs 0.333 +/- 0.019 | same |
| Medium volatility, 6 transitions | 0.358 +/- 0.030 vs 0.317 +/- 0.022 | same |
| Low volatility, 6 transitions | 0.348 +/- 0.017 vs 0.302 +/- 0.012 | same |
| Extension feature-time N | 1,264--1,547 historically | same, historical rows |
| Vintage-text coverage | 89.5%--93.8% historically; 82.3%--84.1% modern | same, `text_coverage_of_feature_return_universe` |

H1->H2 is the sole transition on which TF-IDF leads: 0.348 versus 0.308,
a difference of 0.039. Win counts are reported descriptively because adjacent
transitions share windows and query outcomes share dates, candidates, and target
matrices. No query-level confidence interval or IID transition p-value is an
active manuscript claim.

The candidate file was frozen on 2026-04-12 after ex-post matching of current
SEC metadata to GDELT organization coverage with a 2025--2026 news filter. It
is not a point-in-time security master. Historical claims are conditional on
this survivor/news-visible cohort, and the direction of selection bias is not
identified by the released census.

The primary modern NDCG census intersects feature-window names with
target-window return availability and is therefore a retrospective benchmark.
The feature-time frozen-list audit separately freezes every peer list on the
joint return-and-vintage-text universe before loading target returns, then drops
incomplete query--list pairs without reranking. Its 99.2%--99.6% figure is
target-window evaluability conditional on that selection universe, not
deployment-wide coverage. The selection universe itself covers 82.3%--84.1%
of the complete-return universe. The audit removes target-availability leakage
from this translation, but it neither cures the ex-post candidate-panel
construction nor measures an independent analyst decision or economic outcome.

The residual-magnitude audit independently refits TF-IDF, rebuilds ideal lists,
and fully reranks every query under each screen. Its either-window screens use
target-window magnitudes and are therefore retrospective data-quality
sensitivities, not implementable filters or evidence that every excluded return
is erroneous.
