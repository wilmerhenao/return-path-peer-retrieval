# V2 protocol clarification 1

**Declared 2 August 2026 after the protocol tag and before any historical
self-test or fit.** This clarification changes no data, model, feature,
hyperparameter, split, seed, metric, threshold, or reporting output. It resolves
two process ambiguities found during the final implementation audit. Together,
the original tag `multimodal-peer-protocol-v2-pre-results` and this file govern
the run; this file takes precedence only for the two points below.

## Source-tree cleanliness during artifact creation

The implementation-seal commit and tag require a completely clean worktree.
After the exact run directory is created, tests and fits necessarily create
declared run artifacts before those artifacts can be committed. During that
interval, the binding cleanliness rule is:

1. `git diff --quiet` and `git diff --cached --quiet` must both succeed; and
2. every untracked path reported by
   `git ls-files --others --exclude-standard` must be a descendant of the exact
   declared run directory
   `experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2/`.

The runner asserts this rule before the mandatory historical self-tests, before
the first fit, after development selection, and before final report creation.
An untracked or modified path anywhere else aborts the run. This is the meaning
of a clean source tree once declared run artifacts exist.

## S4 artifact-substitution test

S4 does not require invariant labels, prose, provenance identifiers, or section
structure to change. For each artifact-sourced displayed numeric or result
field in the report fixture, the test mutates that field in isolation and
requires its corresponding rendered value to change. It also requires every
such fixture field to appear in the rendered report. This one-field-at-a-time
substitution check, together with the literal grep sentinel, is the binding S4
test.
