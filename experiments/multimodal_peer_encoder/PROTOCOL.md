# Historical-first protocol for the multimodal peer encoder

**Declared 2 August 2026, before the new encoder is evaluated on any modern
target.** This document governs the isolated experiment under
`experiments/multimodal_peer_encoder/`. The manuscript and its existing result
authorities remain frozen while the experiment is run.

This is a historical-first test, not a claim that the modern period has never
been seen by the research team. The published modern result for the frozen
80/20 return-path and TF-IDF blend (macro NDCG@5 approximately 0.373) was known
when this model was proposed. The narrower and auditable claim is that every
new-model parameter, hyperparameter, checkpoint, and reporting rule is locked
before the model's first access to a `W`-period target or target-derived
new-model result. Existing transition-level scores and audited universe counts
for that frozen comparator may be embedded in the evaluator solely as
reproduction assertions; they are not model-selection inputs.

## Fixed task, inputs, and score

The twelve historical transitions are numbered in chronological order:
H1->H2 through H12->H13. The later evaluation panel is W1->W2 through W6->W7.
At a transition, the model may use only information available at the start of
the target window: the feature-window factor-residual return path and the
feature-time filing representation. Target-window returns are used only to
construct relevance after a ranking has been fixed.

The text input is a 768-dimensional Granite-r2 embedding of the first 512
tokens of the eligible vintage 10-K Item 1 excerpt. The missing H1--H12 cache
is built once from the local model snapshot, content-deduplicated, hashed, and
then frozen. To match the active modern Granite caches, the historical builder
uses CPU inference and explicitly upcasts the frozen bfloat16 checkpoint to
float32 before encoding. It requires the authoritative selected model-tree
hash, the pinned software versions, and float32 inference parameters. Granite
is not fine-tuned, and neither precision, device, cache, nor token limit is
changed in response to a result. The trainable model is
the compact residual-gated retriever. Its width is bound to both
`embedding_dim` and `return_channels`; the only width candidates are 32 and
64. All other architecture and optimization choices must be identical across
the two candidates and recorded before the first fit.

Training uses the ListNet-style listwise softmax cross-entropy implemented by
`listwise_softmax_loss`, with both score and target temperatures fixed at 0.1.
For a query security, candidate relevance is
`max(0, next-window Ledoit-Wolf residual correlation)`. Self-candidates are
excluded. Each transition loss is averaged over its census queries, and every
training transition contributes exactly one optimizer update in every epoch.
Thus transitions have equal update frequency even though they have different
numbers of securities; the optimizer steps are sequential rather than one
joint gradient average.
Validation targets never receive gradients and are not used for checkpoint or
epoch selection. Every selection and final fit runs exactly 40 epochs. Within
each epoch the historical transitions are visited in a seed-determined order,
with one optimizer update per transition. The optimizer is AdamW with learning
rate 0.001, weight decay 0.0001, float32 arithmetic, and gradient-norm clipping
at 1.0. The per-company return gate and global raw-return skip both start at
0.8. No scheduler, dropout, early stopping, or validation checkpoint choice is
used. The reference selection run uses deterministic PyTorch CPU execution
with eight intra-operation threads. Device substitution is not allowed after
historical results have been inspected.

The sole model-selection metric is canonical positive-relevance NDCG@5 from
`scripts.metrics.ndcg_at_k`, using every eligible query, self-exclusion, and
the repository's deterministic seeded tie jitter. A transition-level value is
the macro mean over its queries. A panel mean gives each transition equal
weight. Comparisons use unrounded values throughout.

## Expanding historical folds

The folds are fixed and are not rolled, reordered, or augmented:

| Fold | Training transitions | Validation transitions |
|---|---|---|
| 1 | 1-6 (H1->H2 through H6->H7) | 7-8 (H7->H8, H8->H9) |
| 2 | 1-8 (H1->H2 through H8->H9) | 9-10 (H9->H10, H10->H11) |
| 3 | 1-10 (H1->H2 through H10->H11) | 11-12 (H11->H12, H12->H13) |

Each width is fitted in every fold with the three selection seeds 11, 29, and
47. For each of the six validation transitions, the three seed-specific
NDCG@5 values are averaged before comparison with the frozen 80/20
return-path/TF-IDF baseline on the same transition and universe. Thus a
"validation slot" means one transition, not one seed-transition observation.
A strict positive difference is a win; zero is a tie and is not a win.

For each width, the selection statistic is the equal-weight mean of its six
slot-level advantages over the frozen blend. The width with the larger mean
advantage is selected. An exact tie is resolved in favor of width 32. There is
no fallback to the other width if the selected width subsequently misses a
gate.

Historical work continues only if the selected width satisfies both of these
conditions:

1. its mean validation advantage over the frozen blend is at least 0.005
   NDCG@5; and
2. it wins at least four of the six validation slots.

If either condition fails, the experiment stops before any modern target is
opened. The complete historical result is retained as a negative result, and
the frozen manuscript is unchanged.

## Final historical fit and lock

After a historical pass, the selected width is refitted on all twelve
historical transitions with five disjoint final seeds: 101, 211, 307, 401, and
503. All five runs must complete. No seed may be selected, dropped, or replaced
because of its score. Each seed is evaluated separately; reported transition
scores are arithmetic means across the five seeds, not the score of a
post-hoc ensemble.

Before modern evaluation, a lock manifest must record and hash at least the
protocol and JSON configuration, source code, data contract, historical input
identities, frozen Granite cache identity, selected width, complete training
settings, five checkpoints, evaluator, historical slot results, and the
historical gate decision. Before the first fit, the production runner loads the
historical panel twice and requires identical file hashes and exact aligned
ticker, return, text-embedding, text, and target-array identities. It then
recomputes that source/input snapshot after selection, after final fitting, and
immediately before lock construction. Rebound execution callables cannot
receive a production attestation. Each serialized checkpoint is reloaded and
its tensor-state digest must equal the just-fitted in-memory model. The lock is
bound to its project-relative run location, so copying the run directory does
not create another valid one-shot marker location. Injected data, trainer,
evaluator, checkpoint-writer, or source seams are test-only and can create only
a `TEST_ONLY` manifest, never a production lock. The modern evaluator
independently recomputes the six-slot selection decision and verifies these
attestations before it imports or calls a target loader.

Before the lock, the new experiment must not open, hash, probe, or derive
metadata from `results/windows_v3/W*/corr_target.npy`. It must also avoid
general helpers that eagerly load or hash those targets, including
`core.build_transition`, `core.load_window`, `check_data_contract`, and
`collect_input_provenance`. A feature-only loader is required. The frozen
comparator's existing aggregate and transition-level scores and audited
universe counts are permitted prior information and may be used only as
reproduction assertions. Raw W targets, per-query target data, and any W
target-derived score for the new model are not permitted before the lock.

## One-shot modern evaluation and paper gate

Once locked, the five final checkpoints are evaluated once on W1->W2 through
W6->W7. The canonical one-shot marker is claimed immediately after lock
verification and before this evaluator opens any modern feature, cache, or
target byte. This does not erase the disclosed feature-only cache compatibility
check performed before sealing; that check used no target or target-derived
availability. Each transition then proceeds chronologically: its pinned
feature inputs are verified and loaded, all model and comparator peer lists
are frozen, and only then are that transition's target files hashed and loaded.
Target hashes are appended to the existing marker as access occurs. The comparator is the
historically selected 80% return-path, 20%
TF-IDF percentile-rank blend. Its active authority reports an unrounded modern
macro NDCG@5 of 0.3729682197528556 (0.373 when displayed). For each modern
transition, the proposed model's score is the five-seed mean; a win requires
that mean to be strictly greater than the blend's unrounded score for the same
transition.

The model may enter the modern paper only if every condition below holds:

1. its equal-transition modern mean exceeds the frozen blend mean by at least
   0.010 NDCG@5;
2. it beats the frozen blend in at least five of the six modern transitions;
3. the result survives the strict feature-time application audit: peer lists
   are frozen before target availability is known, missing query-list pairs are
   dropped without reranking, there is no target-dependent selection, and the
   model-minus-blend equal-transition mean difference in query-basket residual
   correlation remains positive;
4. its gates remain directly interpretable; and
5. hashes and run records establish that no modern tuning occurred.

Gate interpretability is a completeness requirement, not an invitation to
choose a favorable subgroup. For every modern transition and final seed, the
report must include the return-gate mean, standard deviation, p10, p50, p90,
and fractions below 0.05 and above 0.95, together with the global raw-return
skip weight. The same report must show fixed, no-retraining score ablations for
the full model, learned fusion with the raw skip disabled, and the raw-return
skip alone. All gate values must be finite and retain their stated [0, 1]
meaning. Collapsed or unstable gates are reported as such; they are not grounds
for changing the model after modern evaluation.

Passing the gate permits the result to be considered for the paper. It does
not convert the modern panel into an untouched external validation set, and it
does not justify IID inference over queries or adjacent transitions.

## Failure and stop rules

- A historical gate failure ends the experiment before W-target access. No
  modern run is made and no rescue configuration is tried.
- A failure of any modern inclusion condition leaves the current paper frozen.
  The full modern result is archived, but the model is not added to the main
  claims and there is no tuning or second attempt on the same W panel.
- Any pre-lock W-target access, including an eager provenance hash, invalidates
  the confirmatory status of the experiment and stops paper inclusion under
  this protocol.
- All candidate widths, folds, and required seeds must complete. A hardware
  interruption may be rerun only with the identical seed, inputs, code, and
  configuration. Seeds are never removed from a denominator.
- A purely mechanical failure may be corrected before any target-dependent
  output is inspected, with the correction and rerun logged. Once a modern
  target-dependent output has been inspected, a material code, data, loss,
  width, seed, checkpoint, or evaluation change invalidates the paper gate.
- Further modeling after a stop is exploratory work. It requires a new
  protocol and a genuinely later evaluation period; it cannot reuse W1->W7 as
  a fresh holdout.
