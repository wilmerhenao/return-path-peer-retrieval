# Historical protocol for baseline-anchored multimodal heads (v2)

**Declared 2 August 2026, before any v2 implementation or fit.** This document
governs v2 under `experiments/multimodal_peer_encoder/`. It ends at the
historical gate. It does not authorize access to any modern target, a modern
evaluation, a paper edit based on v2, or a further rescue model.

## Status, scope, and interpretation

The v1 residual-gated encoder family is closed. Its selected historical model
did not clear the fixed continuation gate and will not be retrained, retuned,
or reinterpreted. The retained v1 result is described in `AUDIT_v2.md`.

V2 asks a narrower question: can a very small, baseline-anchored head use a
frozen filing similarity and context variables to correct the exact residual
return ranking? It tests exactly two parameterizations, H-A and H-B, below.
There is no learned return encoder and no text fine-tuning in v2.

This is a prospective specification relative to v2 code and v2 results, but it
is not an independent replication. The v2 design was proposed after the v1
historical scores, including the six gate-transition outcomes, were known.
The development/gate separation below prevents programmatic v2 selection on
those six gate targets; it does not make them a researcher-blind holdout.
Results will be described as a historical diagnostic, not as confirmatory
evidence or a new paper result.

The application is candidate generation for comparable-company analysis and
residual-comovement monitoring. NDCG@5 measures whether the five retrieved
companies align with next-window residual co-movement. It does not measure
valuation accuracy, analyst forecasts, portfolio profit, transaction costs,
or a causal effect. Filing length and window length are availability/context
variables, not economic mechanisms. Any coefficient or gate association with
them is descriptive.

## Locked historical data and metric

The twelve historical transitions are H1->H2 through H12->H13. At Ht->H(t+1),
the model receives only:

- the Ht factor-residual path in `ts_features.npy`;
- the frozen 768-dimensional Granite-r2 embedding of the eligible vintage 10-K
  Item 1 excerpt, from the already sealed v1 historical cache;
- the stored Item 1 excerpt itself, only to compute its character length; and
- the Ht common window length from `meta.json`.

For each aligned transition, `cos_raw` is row-L2 cosine similarity of the
standardized feature-window residual paths and `cos_text` is row-L2 cosine
similarity of the frozen Granite embeddings. The text encoder remains frozen.
The eligible universe is the sorted intersection of the feature-window
universe, target-window universe, and eligible feature-time vintage-filing
universe.

The heads may be directional because the query and candidate filing-length
coefficients are separate. This is permitted because the task is
query-conditioned retrieval; learned score matrices are not forced to be
symmetric. Raw, text, and target correlation matrices remain symmetric.

The filing-length feature is `log1p(len(item1_text))`, where `len` is the
number of Unicode code points in the stored excerpt. The excerpts are subject
to the collection cap, so this is a censored input-availability proxy rather
than a measure of disclosure quality or complexity. Within each fit, it is
standardized using the pooled security-transition records in that fit's
training transitions. The same training mean and population standard
deviation are applied to held-out transitions. A missing length or a zero
fitted standard deviation is an error. Window context is the fixed transform
`(window_length - 84) / 10`; it is not fitted.

Two proposed features are unavailable and are dropped before this protocol is
tagged:

- `idio_vol`: the historical authorities contain standardized paths but not
  the pre-standardization residual scale; reconstructing the scale would
  require rebuilding the authorities.
- `n_overlap_days`: every historical window uses a complete-case panel, so a
  pair's overlap is exactly the common window length and supplies no distinct
  information.

The primary evaluation relevance is always
`max(0, next-window Ledoit-Wolf residual correlation)`, including when raw
correlation is selected as a training label. Every eligible query is
evaluated, its self-candidate is excluded, and canonical positive-relevance
NDCG@5 comes from `scripts.metrics.ndcg_at_k`. A transition score is the macro
mean over valid census queries; a panel mean gives every transition equal
weight. All comparisons use unrounded values. Deterministic tie handling is
the repository's canonical query-seeded jitter.

The frozen comparators are unchanged:

- return-only: `cos_raw`;
- text-only: config-B TF-IDF cosine and its query-wise percentile ranking; and
- frozen blend: 80 percent return-path percentile rank plus 20 percent TF-IDF
  percentile rank.

TF-IDF is fitted independently within each feature-time transition. The v2
heads use Granite text cosine; the TF-IDF component is retained only because
it defines the frozen comparator that v1 and the paper use.

## Open hypotheses

Exactly two architecture families are open.

### H-A: additive boosting head

For query i and candidate j,

```text
s_ij = cos_raw_ij + lambda * f_theta(phi_ij)
phi_ij = [cos_raw_ij, cos_text_ij, zloglen_i, zloglen_j, window_context]
f_theta = Linear(5, 16) -> tanh -> Linear(16, 1) -> tanh
```

`lambda` is an unconstrained scalar initialized to zero. The MLP uses the
seeded default PyTorch linear initialization. There is no dropout, batch
normalization, or other layer. At initialization, the score is bitwise the raw
score regardless of the random MLP values.

### H-B: centered cross-modal correction

For query i and candidate j,

```text
gamma_ij = sigmoid(w dot psi_ij + b) - sigmoid(-4)
psi_ij = [window_context, zloglen_i, zloglen_j]
s_ij = cos_raw_ij + gamma_ij * (cos_text_ij - cos_raw_ij)
```

`w` is initialized to zero and the trainable intercept `b` is initialized to
-4. The subtracted `sigmoid(-4)` is a fixed scalar, not `sigmoid(b)`.
Centering is necessary: the uncentered formula proposed for review would begin
at gamma approximately 0.018 rather than at the raw baseline and would
contradict the mandatory initialization test. The centered form begins at
exactly zero and has nonzero gradients with respect to `w` and `b`. It is a
signed, baseline-centered correction, not a probability or a strictly convex
mixing weight; its mathematical range is approximately [-0.018, 0.982]. The
trainable intercept lets the head learn a context-independent shift toward or
away from text, while `w` learns context variation.

Both parameterizations therefore equal the raw baseline exactly at
initialization, while the development split and gate test whether movement
away from it generalizes historically. Initialization does not guarantee that
an optimizer moves only when a correction has out-of-sample value, and no such
guarantee is claimed. The narrower motivation is that v1 could not express a
direct baseline-plus-correction form and its retained score behaved like a
slightly degraded raw path.

## Predeclared toggles and 48-fit development grid

Only the following settings vary:

- architecture: H-A or H-B;
- T1 target softness: raw-scale next-window residual correlation or stored
  Ledoit-Wolf-shrunk next-window residual correlation;
- T2 application: global or constrained rank-band reranking;
- learning rate: 0.003 or 0.03; and
- seed: 11, 29, or 47.

This is 2 x 2 x 2 x 2 x 3 = 48 development fits. No width, hidden size,
feature, epoch count, optimizer, temperature, regularizer, threshold, or
candidate-range choice is tunable.

After the two family winners are fixed, the outer walk-forward stage adds
2 families x 3 folds x 3 seeds = 18 fits. A complete run therefore contains 66
fresh fits and 13,440 transition-level optimizer updates. The completed-fit
roster must match these counts exactly.

For T1 raw-scale targets, target-window raw correlation is reconstructed as
row-L2 cosine of the aligned target-window standardized residual paths. For T1
LW targets, the aligned stored `corr_target.npy` is used. Both are clamped
below at zero before entering the loss. Within a transition, LW shrinkage is a
positive off-diagonal scaling of the raw matrix, so T1 does not change the
target order; it changes only the concentration of the fixed-temperature soft
labels. If `c_t = 1 - shrinkage_t`, an LW label at target temperature 0.1 is
equivalent to the raw label at temperature `0.1 / c_t`. The run ledger records
`c_t` and this equivalent temperature. Gate evaluation always uses stored LW
relevance. T1 is therefore named target softness, not a distinct label or
business hypothesis.

For T2 global, the learned correction may change the order of every eligible
candidate. For T2 constrained rank-band reranking, one-based raw positions are
formed after self-exclusion with canonical tie-breaking. Raw positions 1-2
remain fixed, positions 3-50 are reordered by the corrected score, and
positions greater than 50 remain fixed in their raw order. Corrected candidates
cannot cross either boundary. A corrected-score tie inside positions 3-50 is
resolved by the earlier raw position. Rank-band training applies the
differentiable corrected scores and listwise loss only to the 48 candidates at
raw positions 3-50 for each query; the correction is defined as zero outside
that band. Thus T2 changes both the application policy and the training
candidate set. The discrete splice is used only at evaluation. It is encoded
as a finite integer rank score: among the N-1 eligible candidates, the
candidate at final position p receives score N-p and the excluded self receives
-N. The gaps are much larger than canonical tie jitter, so
`scripts.metrics.ndcg_at_k` recovers exactly the declared spliced order.

## Development selection, then the fixed six-slot gate

The 48 development fits use H1->H2 through H4->H5 for training and H5->H6 and
H6->H7 for development evaluation. Each architecture/T1/T2/learning-rate
configuration therefore has three seeded fits. Each fit produces one state
after H1->H2 through H4->H5; that same frozen state scores both H5->H6 and
H6->H7, with no update or refit between the two development transitions.
Seed-specific NDCG@5 is first averaged within each of the two development
transitions, then the two transition means receive equal weight. The winning
configuration has the largest equal-transition mean advantage over the frozen
blend. Exact ties are resolved by the following fixed preference order: H-B
before H-A, rank-band before global, LW before raw-scale, and learning rate
0.003 before 0.03.

One winner is also retained within each architecture family for descriptive
H-A/H-B gate columns. The global winner is chosen across all sixteen
non-seed configurations before any v2 gate score is produced. The other family
winner is diagnostic only: it cannot replace or rescue the global winner,
regardless of its gate score. There is no fallback to a runner-up.

After development selection, the two family winners are fitted under the same
three expanding historical folds used by v1:

| Fold | Training transitions | Validation transitions |
|---|---|---|
| 1 | 1-6 (H1->H2 through H6->H7) | 7-8 (H7->H8, H8->H9) |
| 2 | 1-8 (H1->H2 through H8->H9) | 9-10 (H9->H10, H10->H11) |
| 3 | 1-10 (H1->H2 through H10->H11) | 11-12 (H11->H12, H12->H13) |

Each family winner is fitted in every fold with seeds 11, 29, and 47. Each
validation transition is scored by the applicable fold models and averaged
over its three seeds. This produces six slot scores per family. Only the
preselected global winner is tested by the continuation gate. Looking at the
diagnostic family column cannot change the gate decision or trigger another
fit. Later folds may train on targets that appeared as validation targets in
earlier folds; this is the declared chronological walk-forward sequence, not
six simultaneously untouched slots. The complete sequence runs automatically
under sealed code, without an intermediate human decision.

The applicable v1 metric and threshold passages below are reproduced verbatim
from `PROTOCOL.md`:

> The sole model-selection metric is canonical positive-relevance NDCG@5 from
> `scripts.metrics.ndcg_at_k`, using every eligible query, self-exclusion, and
> the repository's deterministic seeded tie jitter. A transition-level value is
> the macro mean over its queries. A panel mean gives each transition equal
> weight. Comparisons use unrounded values throughout.
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

For v2, `selected width` in the quoted threshold passage means the globally
selected v2 configuration. The operative folds and seed aggregation are stated
immediately above. The unchanged gate components are the six validation
transitions, three-seed averaging within each slot, frozen comparator,
universe, metric, unrounded comparison, +0.005 margin, and 4/6 strict-win
requirement. Gate-transition universe sizes are expected to be 1,416, 1,501,
1,512, 1,518, 1,534, and 1,545 and are asserted against the v1 authority. This
gate is a fixed screening rule, not a statistical-significance test; the seeds
and temporally adjacent slots are not independent observations.

## Fixed optimization

Every fit runs exactly 40 epochs. There is no validation-gradient access,
early stopping, checkpoint selection, scheduler, dropout, or augmentation.
Within each epoch, training transitions are visited in a seed-determined order
and each transition contributes exactly one AdamW update. The listwise loss is
the existing `listwise_softmax_loss`, with score and target temperatures both
0.1 and query reduction by arithmetic mean. The optimizer uses the grid's
learning rate, weight decay 0.0001, float32 arithmetic, and gradient-norm
clipping at 1.0. PyTorch CPU execution uses eight intra-operation threads and
deterministic algorithms. Training-transition order and parameter
initialization use the declared seed. Every fit and fold begins from a fresh
initialization; weights, optimizer state, and feature scalers are never warm
started from another fit.

Dense query-candidate computation may be split into deterministic query
chunks solely for memory. Chunk losses must be weighted by their query counts,
their gradients accumulated, and one optimizer step taken only after every
query in that transition has contributed. Chunk size cannot change the
mathematical mean loss or create additional optimizer steps.

## Pre-results implementation boundary

This protocol is committed alone and receives the annotated tag
`multimodal-peer-protocol-v2-pre-results` before v2 implementation begins.
After implementation and synthetic-only unit checks, but before S1-S4 or any
fit, the configuration, model, staged H-only loader, runner, tests, report
generator, and protocol are committed together and receive the annotated tag
`multimodal-peer-v2-code-pre-results`. The run manifest records the peeled
commit, every source SHA-256, and `git status --short`. Mandatory tests and all
fits must execute from that sealed commit with a clean worktree. A protocol tag
without this implementation seal is insufficient provenance.

## Mandatory self-tests before fitting

All four tests must pass after implementation and before the first development
fit. Their complete output is retained under the new run directory.

S1, init equals baseline:

- with initialization seed 11, on the first 200 eligible H1->H2 queries, H-A
  and H-B under both application policies must produce exactly the same
  complete per-query candidate order as return-only, including tie resolution;
  and
- feature-only score/order checks on each feature/vintage eligible universe
  must establish the same ranking on H1 through H7 without opening a later
  window; and
- on the six pre-gate transitions H1->H2 through H6->H7, each initialized
  head/application combination must equal the corresponding return-only
  NDCG@5 in `AUDIT_v2.md` within 1e-6, both per transition and after
  equal-transition aggregation. No H8-H13 path of any kind may be opened for
  S1 because its standardized residual path can reconstruct an earlier gate
  target.

S2, learnability: on a deterministic synthetic fixture, one optimization step
must change each head's score matrix. H-A must have a finite, nonzero gradient
on `lambda`, and a second step must produce a finite, nonzero gradient on its
MLP parameters after `lambda` has moved. H-B must have finite, nonzero gradients
on both `w` and `b`.

S3, leakage guard: before any path is opened, the H-only loader must reject
every descendant of `results/windows_v3/`. The test must deliberately request
`results/windows_v3/W1/corr_target.npy` while instrumenting file open and must
show that the guard raises before any open occurs.

S4, provenance: the report generator must contain no hardcoded displayed
result literal. The binding sentinel is
`grep -E '0\.[0-9]{3}' <report-generator-source>`, which must return zero hits.
Every result in the report is read from a run JSON artifact. The generator
must also render a fixture artifact, render it again after every fixture result
value is changed, and demonstrate that every displayed result changes. This
substitution check complements the literal grep sentinel.

Failure of S1-S4 stops the experiment before training. A mechanical
implementation error may be corrected and the tests rerun only while no fit
has begun. Once the first fit begins, architecture, data, selection, and report
logic are frozen.

## Outputs and interpretability

The run directory is
`experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2/` and is
created without overwriting an existing path. It must retain:

- a manifest with the protocol tag, source hashes, configuration, input
  identities, environment, test log, start/end times, and completed fit roster;
- all 48 development results and the deterministic selection record, with
  architecture, label, application, learning rate, seed, epoch losses, and
  unrounded development scores;
- all 18 fold/seed gate fits for the two family winners; and
- machine-readable results plus a generated `RUN_REPORT.md` and either `PASS`
  or `STOP`.

The gate table contains, for every one of the six slots, return-only,
TF-IDF-only, frozen blend, H-A, H-B, and the locked global winner. It also gives
equal-transition means, seed dispersion, and unrounded winner-minus-blend
differences.

For every trained H-A fold/seed model, report `lambda` and the active-pair
correction distribution (mean, standard deviation, p10, p50, and p90). For
every H-B fold/seed model, report `w`, `b`, the active-pair centered-gamma
distribution using the same summaries, and its association with each
component of `psi`. For `zloglen_i` and `zloglen_j`, association means Pearson
correlation within each transition/seed. Because window context is constant
within a transition, its association is computed across the six slot-level
mean gammas after seed averaging. A zero-variance correlation is recorded as
undefined, not zero. `Active pair` means every off-diagonal pair for global and
raw ranks 3-50 for rank-band. Aggregate summaries first within a
transition/seed and then with equal transition weighting. Pairwise records are
descriptive and are not treated as independent observations. No subgroup is a
selection or gate criterion.

`lambda` is not interpreted by itself because its scale interacts with the
bounded MLP output. H-A also reports the maximum absolute effective correction,
its positive fraction, and the active-pair derivative of the score with
respect to `cos_text`. H-B reports the fractions with gamma below zero, with
absolute gamma below 0.01, and with gamma above 0.5, plus its distribution for
retrieved top-five pairs and all active pairs. For both families, a fixed
no-retraining semantic ablation replaces each query row's off-diagonal
`cos_text` values with that row's off-diagonal mean, then reports the NDCG@5
change. It is diagnostic only and cannot affect selection or the gate.

For each gate transition and family, the report also gives top-five overlap
with return-only and the frozen blend, defined as intersection size divided by
five, and the fraction of queries whose top-five list differs from return-only.
For candidates promoted into a model top five from outside the raw top five,
report original raw-rank p10, median, p90, and fractions in ranks 6-10, 11-20,
21-50, and above 50. One application proxy is fixed in advance: for each query,
average the five signed next-window LW query-to-peer correlations, then report
the transition mean and the fraction of queries whose mean is positive.
Return-only, the frozen blend, H-A, H-B, and the global winner receive the same
calculation. This is called mean signed query-to-peer correlation, not basket
correlation, and it is not a valuation, forecast, or trading outcome.

The context audit records filing-length quantiles, the fraction with stored
length at least the 8,000-character collection cap, window lengths, fitted
training scalers, and the absence of missing values. It may show
within-transition descriptive correlations between context and the
learned correction, but no context subgroup changes selection. The report also
records trainable parameter count, wall time, peak resident memory, and the
wall time of each full-transition scoring call so the computational scope is
auditable.

## Stop rule, data lock, and budget

Whether the historical gate passes or fails, v2 stops after writing its record
and commit. A pass receives the annotated tag
`multimodal-peer-historical-pass-v2`; a failure receives
`multimodal-peer-historical-stop-v2`. A pass does not authorize a final
all-history fit, modern evaluation, manuscript change, v3, or another tuning
round. Those require a new human instruction.

Before and throughout v2, no code may open, hash, stat, glob for metadata, or
otherwise inspect any descendant of `results/windows_v3/`. General loaders
that eagerly inspect that tree are prohibited. The deliberate S3 request is
allowed only because it must be rejected before filesystem access. Gate-target
loading must also be staged: tests and development selection may load only
H1->H2 through H6->H7 targets. No H8-H13 residual path or correlation matrix
may be loaded until the global development winner is written to an immutable
selection artifact.

Execution is fixed to the local deterministic CPU reference, so incremental
compute cost is zero. The external budget ceiling remains USD 10, with no Spot
instance and no paid resource authorized or needed. The wall-clock target is
one day. A hardware interruption may repeat only the incomplete fit with the
same inputs, source, configuration, and seed; completed fits are never dropped
or selectively rerun. The run is incomplete unless all 66 declared fits are
present.
