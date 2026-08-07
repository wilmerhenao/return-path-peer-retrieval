# Code and Output Map

This map distinguishes active manuscript authorities from preparation tools,
supporting diagnostics, and legacy scripts.

## Verification and release

| Script | Purpose | Inputs | Output |
|---|---|---|---|
| `scripts/check_numbers.py` | Check active manuscript and abstract claims against frozen artifacts | Released paper and result authorities | Console pass or failure |
| `scripts/submission_preflight.py` | Check PDF format, anonymity, fonts, references, and numbers | Compiled paper, Poppler tools, frozen results | Console pass or failure |
| `scripts/build_anonymous_release.py` | Build the allowlisted anonymous ZIP and manifests | Repository files selected by its allowlist | `dist/anonymous_release_<label>/` and ZIP |
| `scripts/metrics.py` | Canonical positive-relevance NDCG and deterministic top-k handling | Score and relevance arrays | Imported functions |

The direct-version manifests under `environment/` separate the CPU, local
model, cloud-launcher, and in-container Granite dependencies. They are not
fully resolved transitive lockfiles.

## Input preparation

| Script | Purpose | Required inputs | Main outputs |
|---|---|---|---|
| `scripts/build_windows.py` | Build modern W1-W7 or historical H1-H13 complete-case FF5 residual panels | `data/universe.parquet`, Yahoo histories, daily FF5 | `results/windows_v3/W*/` or `results/ea_historical/windows/H*/`, plus `windows_summary.json` |
| `scripts/summarize_window_metadata.py` | Validate and hash saved return-window shapes and metadata | Prepared H1-H13 directories | `results/historical_windows_summary.json` or a caller-selected output |
| `scripts/sec_filing_text.py` | Access SEC submissions and filings, choose the latest annual filing before a cutoff, and extract a capped Item 1 section | Caller-supplied SEC identity, ticker/CIK, cutoff date | Imported acquisition and extraction helpers |
| `scripts/fetch_vintage_10k.py` | Build the restartable H1-H13 and W1-W7 Item 1 corpus with an 8,000-character cap | Frozen ticker panel, official SEC endpoints, `SEC_USER_AGENT` | `results/vintage_descriptions_rebuilt/<window>_descriptions.parquet`, restart cache, and summary |
| `scripts/build_dense_embedding_cache.py` | Build the universe-indexed short-context caches for four encoders | W1-W6 arrays, vintage text, four local model snapshots | `results/vintage_all_encoders_v3/embedding_cache/` and `cache_manifest.json` |
| `scripts/audit_granite_vintage_tokens.py` | Count Granite tokens in stored W1-W6 excerpts | Vintage text and Granite tokenizer | Immutable token-audit run |
| `scripts/launch_vintage_granite_longctx.py` | Package inputs, submit, monitor, and download the SageMaker Granite job | W1-W7 arrays, vintage text, Granite model, AWS permissions | `results/granite_vintage_longctx_census/<job>/` |
| `scripts/train_vintage_granite_longctx.py` | Run the long-context Granite census inside the training container | SageMaker input channels | JSON, CSV, token audit, provenance |

The public package does not include the SEC-to-GDELT workflow that produced the
2,200-name panel. The corpus driver starts from that frozen ticker list and an
official current SEC ticker/CIK mapping. The package also omits the exact source
caches used by the return-window builder.

## Primary selection and evaluation

| Script | Manuscript role | Principal frozen output |
|---|---|---|
| `scripts/run_full_return_census.py` | All-query full-return return-path, spectral, SIC-2, and random baselines | `results/full_return_census/run_20260802T_full_return_authority_v2/full_return_census.json` |
| `scripts/run_acceptance_robustness.py` | Historical TF-IDF selection, historical fusion selection, frozen modern census, k sensitivity, EWMA, two-window path, ticker screen, and eligibility reconciliation | `results/acceptance_robustness/run_20260730T032543Z/` |
| `scripts/run_adversarial_followup.py` | Stale-return control, one-to-four-window path memory, 18-transition two-window extension, and feature-time dense rescoring | `results/acceptance_followup/run_20260730T065913Z/adversarial_followup_census.json` |
| `scripts/run_full19_feature_time.py` | Nineteen-transition feature-time census and volatility grouping | `results/acceptance_full19_feature_time/run_20260730T032944Z/full19_feature_time_census.json` |

The active SIC-2 baseline uses the complete frozen 2,200-name candidate table
and reports `0.213 +/- 0.009`. The SIC-2 row in
`results/full_universe_ndcg_corrected.csv` used a legacy 1,824-name SIC map and
reports `0.201 +/- 0.008`; that row is retained for audit history but is not an
active authority. The same CSV also contains historical symmetric-universe rows
that are not produced by `run_full_return_census.py`.

## Robustness and application diagnostics

| Script | Question | Principal frozen output |
|---|---|---|
| `scripts/run_target_date_block_bootstrap.py` | Do method gaps survive common-date block perturbations with rankings fixed? | `results/acceptance_target_date_uncertainty/run_20260731T014000Z/target_date_block_bootstrap.json` |
| `scripts/run_memory_fusion_audit.py` | Does the frozen text blend still help as return memory grows? | `results/memory_fusion_audit/run_20260731T131959Z/memory_fusion_audit.json` |
| `scripts/run_downstream_peer_hedging.py` | What do frozen peer lists imply for downstream replication diagnostics? | Restricted canonical downstream run |
| `scripts/run_strict_feature_time_application_audit.py` | What happens when lists freeze before target availability and missing lists are not reranked? | `results/strict_feature_time_application/run_20260801T233929Z/strict_application_census.json` |
| `scripts/run_outlier_sensitivity.py` | Do raw-residual magnitude screens change the main ordering after full reranking? | `results/outlier_sensitivity/run_20260801T234540Z/outlier_sensitivity_census.json` |
| `scripts/run_learned_fusion_followup.py` | Does a historical-only nonlinear fusion baseline beat the fixed rank blend? | `results/learned_fusion_followup/run_20260731T012556Z/learned_fusion_followup.json` |

The gated two-tower negative result is recorded in
`experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v1/historical_selection.json`.
The release includes its frozen selection record for claim audit; it does not
claim that the original v1 training run can be repeated without the restricted
prepared inputs.

The target-date output contains conditional sensitivity envelopes. The outlier
conditions that inspect the target window are retrospective data-quality
checks. Neither should be read as deployable inference.

## Text diagnostics added for the active manuscript

| Script | Protocol | Frozen authority |
|---|---|---|
| `scripts/run_same_window_text_diagnostic.py` | Same feature-time universes and frozen text representations as the primary census, but relevance changes from W(t+1) to W(t) | `results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/same_window_text_diagnostic.json` |
| `scripts/run_pairwise_alignment.py` | Complete W1 feature-time upper triangle against W1 and W2 residual correlation, with no pair sampling | `results/pairwise_alignment/run_20260806T_pairwise_authority_v3/pairwise_alignment.json` |

The same-window diagnostic reports:

| Method | Mean NDCG@5 and population SD across six transitions |
|---|---:|
| TF-IDF | 0.296624 +/- 0.006890 |
| Granite-r2, 512 tokens | 0.257439 +/- 0.006425 |
| MiniLM | 0.212968 +/- 0.003921 |

The pairwise census contains 1,614 securities and 1,301,691 unordered pairs.
For the four frozen dense encoders, Spearman correlation ranges from 0.028947
to 0.045956 against W1 residual correlation and from 0.026655 to 0.049053
against W2. The corresponding TF-IDF values are 0.036156 and 0.035459; the
return-path values are 1.000000 and 0.102914. The pairs share securities and
target dates, so the script does not report IID p-values or confidence
intervals.

## Figures

| Script | Figure | Inputs |
|---|---|---|
| `scripts/gen_acceptance_robustness_figure.py` | `paper/figures/fig_acceptance_robustness.pdf` | Frozen blend, follow-up, and memory authorities |
| `scripts/gen_fig_regime_timeline.py` | `paper/figures/fig_regime_timeline.pdf` | Frozen 19-transition authority |

These scripts import `plots/style.py`. The style module must be present when
figures are regenerated. Tier 2 uses the released figure PDFs and does not need
the restricted arrays.

## Excluded legacy scripts

The release builder intentionally omits the earlier 200-query and
symmetric-eligibility experiment drivers. Their summaries do not reproduce the
active all-query feature-time manuscript. The cache-only replacement is
`scripts/build_dense_embedding_cache.py`; current number authorities are named
in `NUMBER_PROVENANCE.md` and `artifact_authorities.json`.

## Output conventions

Active experiment scripts create a new run directory and refuse to overwrite
it. A normal run directory contains:

```text
<result>.json
artifact_manifest.json
SHA256SUMS.txt
```

Some runs also include per-transition CSV files or compressed query-level
files. The JSON records configuration, validation checks, software versions,
runtimes, and input hashes. `scripts/check_numbers.py` reads fixed authority
paths, not the latest timestamp by directory order.
