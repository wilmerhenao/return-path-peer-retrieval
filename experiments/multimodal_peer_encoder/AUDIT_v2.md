# v2 pre-write audit

Audit date: 2026-08-02. This file records the repository, v1 result, paper,
and data facts observed before any v2 code or paper edit.

## Repository state

`git status --short --branch` returned:

```text
## main
```

`git tag -l 'multimodal-peer-*'` with peeled commit identities returned:

```text
multimodal-peer-historical-stop-v1 -> redacted-for-anonymous-release
multimodal-peer-protocol-v1-pre-results -> redacted-for-anonymous-release
```

HEAD was `redacted-for-anonymous-release` on `main`.

## Recoverable v1 component scores

The v1 selection artifact did not log component ablations. The values below
were therefore recomputed without fitting from the exact aligned H1--H13
authorities, using the v1 feature-time universes and canonical census
positive-relevance NDCG@5:

- return-only: row-L2 cosine of the standardized residual paths;
- TF-IDF-only: config-B TF-IDF fitted within each transition; and
- frozen blend: the fixed 80/20 percentile-rank blend already recorded in
  `historical_selection.json`.

TF-IDF-only is the frozen comparator's text component. It is not an ablation
of the learned Granite tower, which is not recoverable after the v1 STOP.

| Transition | N | Return-only | TF-IDF-only | Frozen blend |
|---|---:|---:|---:|---:|
| H1->H2 | 1,264 | 0.308366980309 | 0.347766730944 | 0.373791985835 |
| H2->H3 | 1,310 | 0.393149952541 | 0.345567657391 | 0.428871938536 |
| H3->H4 | 1,321 | 0.398485232279 | 0.342750758615 | 0.435295543311 |
| H4->H5 | 1,329 | 0.406478089757 | 0.352642742704 | 0.446596397152 |
| H5->H6 | 1,387 | 0.381454019102 | 0.312924704086 | 0.410273614379 |
| H6->H7 | 1,402 | 0.389209332224 | 0.335299110272 | 0.434547200047 |
| H7->H8 | 1,416 | 0.427604567821 | 0.352700610646 | 0.462786883671 |
| H8->H9 | 1,501 | 0.413774490166 | 0.318962243188 | 0.432101042959 |
| H9->H10 | 1,512 | 0.380733060155 | 0.325541593379 | 0.412613088770 |
| H10->H11 | 1,518 | 0.360212035684 | 0.316938494778 | 0.396883422686 |
| H11->H12 | 1,534 | 0.355061750602 | 0.308085270875 | 0.389708486766 |
| H12->H13 | 1,545 | 0.350001738184 | 0.283116968327 | 0.362878228694 |

Equal-transition means:

| Panel | Return-only | TF-IDF-only | Frozen blend |
|---|---:|---:|---:|
| All 12 transitions | 0.380377604068 | 0.328524740434 | 0.415528986067 |
| Six gate transitions | 0.381231273769 | 0.317557530199 | 0.409495192258 |

The expected approximately 0.4095 blend value is the mean over the six gate
transitions, not the mean over all twelve transitions.

## Unrecoverable v1 diagnostics

- Learned raw-skip weight alpha by seed and width: **not recoverable**.
- Learned per-company gate distribution: **not recoverable**.
- Learned Granite-tower-only score: **not recoverable**.

The selection JSON records losses and validation scores but not those internal
values. The failed v1 gate created neither selection checkpoints nor final
checkpoints. They will not be reconstructed by retraining.

## Verbatim v1 gate definition

The following is reproduced verbatim from `PROTOCOL.md`:

> The sole model-selection metric is canonical positive-relevance NDCG@5 from
> `scripts.metrics.ndcg_at_k`, using every eligible query, self-exclusion, and
> the repository's deterministic seeded tie jitter. A transition-level value is
> the macro mean over its queries. A panel mean gives each transition equal
> weight. Comparisons use unrounded values throughout.
>
> The folds are fixed and are not rolled, reordered, or augmented:
>
> | Fold | Training transitions | Validation transitions |
> |---|---|---|
> | 1 | 1-6 (H1->H2 through H6->H7) | 7-8 (H7->H8, H8->H9) |
> | 2 | 1-8 (H1->H2 through H8->H9) | 9-10 (H9->H10, H10->H11) |
> | 3 | 1-10 (H1->H2 through H10->H11) | 11-12 (H11->H12, H12->H13) |
>
> Each width is fitted in every fold with the three selection seeds 11, 29, and
> 47. For each of the six validation transitions, the three seed-specific
> NDCG@5 values are averaged before comparison with the frozen 80/20
> return-path/TF-IDF baseline on the same transition and universe. Thus a
> "validation slot" means one transition, not one seed-transition observation.
> A strict positive difference is a win; zero is a tie and is not a win.
>
> For each width, the selection statistic is the equal-weight mean of its six
> slot-level advantages over the frozen blend. The width with the larger mean
> advantage is selected. An exact tie is resolved in favor of width 32. There is
> no fallback to the other width if the selected width subsequently misses a
> gate.
>
> Historical work continues only if the selected width satisfies both of these
> conditions:
>
> 1. its mean validation advantage over the frozen blend is at least 0.005
>    NDCG@5; and
> 2. it wins at least four of the six validation slots.
>
> If either condition fails, the experiment stops before any modern target is
> opened. The complete historical result is retained as a negative result, and
> the frozen manuscript is unchanged.

The gate universe is the sorted intersection of the feature-window universe,
target-window universe, and eligible feature-time vintage-filing universe.
Every eligible query is evaluated with self-exclusion. Gate-transition Ns are
1,416, 1,501, 1,512, 1,518, 1,534, and 1,545. They are not pooled when forming
the equal-transition gate mean.

## v1 diagnosis

The selected width-32 expanding-fold result has mean NDCG@5
0.376883334951 on the six gate transitions. Return-only is 0.381231273769, so
model minus return is -0.004347938818. The absolute difference is below 0.005,
and the v1 model loses to return-only on all six transitions. The frozen blend
beats return-only on all six and by +0.028263918489 on average.

Verdict: v1 behaved approximately like a slightly degraded return-only model
and failed to capture the complementarity delivered by the fixed rank blend.
This is a score-level diagnosis, not evidence that alpha exceeded 0.9; alpha
saturation cannot be claimed from the retained artifacts. There was no final
v1 model because the historical gate prevented the all-history refit.

## Paper identity before the v1-result edit

- Authoritative TeX: `paper/main.tex`
- TeX SHA-256: `d90c09eb37eeb79c5fbdebea0a672bef0ff8eac31170cbaf1c09c37aafc0dd77`
- Authoritative PDF: `paper/main.pdf`
- PDF SHA-256: `bd1e4fc2b97131c717152c62c4a7f793015da3b2fba4d3b8a6c2f35b3ac5745e`
- PDF pages: 8, US Letter
- Repository submission check: no more than 8 pages.

The literal proposed anchors are stale after prior prose editing:

```text
grep -F "HistGradientBoosting" paper/main.tex: 0 hits
grep -F "comprehensive benchmark of learned peer models" paper/main.tex: 0 hits
```

The unique current semantic anchor is the paragraph headed
`Historically trained nonlinear fusion.` It ends with the sentence
`the experiment does not cover the full range of learned peer models.` The
paper consistently calls the training panel `twelve 2020--2024 transitions` or
`the same twelve historical transitions`.

Because the external mandate was explicitly advisory, the paper phase may use
that unique current sentence as the insertion anchor. It must still require a
one-hit positive sentinel before editing and must not guess if that current
anchor is not unique.

## Historical feature availability for v2

All H windows use complete-case panels. `ts_features.npy` contains row-wise
standardized FF5 residual paths, `corr_target.npy` contains the corresponding
Ledoit--Wolf correlation, and metadata records the common window length and
shrinkage. The stored historical authorities do not contain `raw_resid.npy` or
the pre-standardization residual scale, so idiosyncratic volatility is not
recoverable without rebuilding the authorities and will not be used in v2.

Raw next-window Pearson correlation remains exactly recoverable from the
target-window standardized paths. Across H1->H2 through H12->H13, multiplying
that raw correlation's off-diagonal entries by one minus the recorded
Ledoit--Wolf shrinkage reproduces `corr_target.npy` with maximum absolute error
between 1.19e-7 and 1.64e-7.

The complete-case policy makes pairwise overlap equal to the window length for
every pair, so `n_overlap_days` is not a separate feature. Window length and
feature-time filing character length are available. These facts must be bound
in `PROTOCOL_v2.md` before its pre-results tag.
