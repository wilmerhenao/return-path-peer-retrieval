# Baseline-anchored multimodal peer retrieval

This directory contains historical-only multimodal experiments that combine a
return-path similarity with a frozen filing-text similarity. The experiments
sit beside the ICAIF 2026 submission. They do not change the paper's active
result authorities.

V1 trained a residual-gated encoder and failed its historical continuation
gate. Its retained record is under
`artifacts/runs/run_20260802_v1/`, and `AUDIT_v2.md` explains why that model
family was closed. V2 asks a smaller question: can either of two tiny heads
make a useful correction to the raw return ranking? V2 stops after the same
historical gate, whether it passes or fails. Read `PROTOCOL_v2.md` and
`PROTOCOL_v2_CLARIFICATION_1.md` before running it.

## 1. Check out the sealed code

Run from the repository root. The production runner accepts only the commit
identified by the annotated code-seal tag:

```bash
git status --short -- . \
  ':(exclude)results/windows_v3' ':(exclude)results/windows_v3/**'
git switch --detach multimodal-peer-v2-code-pre-results
git rev-parse HEAD
git rev-parse 'multimodal-peer-v2-code-pre-results^{commit}'
```

The first command must print nothing, and the two commit identifiers must be
identical. The runner also verifies that the base protocol tags still resolve
to their sealed commits. Repository history is useful for this check, but it
is not included in the anonymous ZIP.

## 2. Install the pinned environment

The reference run uses Python 3.13.2 and a deterministic CPU implementation.
Create a clean environment and install the experiment lock file:

```bash
python3.13 -m venv .venv-v2
source .venv-v2/bin/activate
python -m pip install -r experiments/multimodal_peer_encoder/requirements.lock
python -c "import numpy, polars, sklearn, torch; print(numpy.__version__, polars.__version__, sklearn.__version__, torch.__version__)"
```

The required versions are NumPy 2.4.2, Polars 1.39.3, scikit-learn 1.8.0,
PyTorch 2.11.0, and pytest 9.0.2. Cache construction additionally pins
sentence-transformers 5.3.0 and transformers 5.5.0. The production runner
checks the versions again and refuses a mismatch.

The source-only tests do not open the historical or modern experiment data:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  experiments/multimodal_peer_encoder/tests/test_v2_*.py
```

## 3. Supply the historical inputs

The canonical paths are fixed. Place the following prepared inputs at these
locations before starting the run:

```text
results/ea_historical/windows/H1/ ... H13/
results/vintage_descriptions/H1_descriptions.parquet ... H12_descriptions.parquet
models/granite-embedding-english-r2/
```

Each H-window directory contains `universe.parquet`, `meta.json`, and
`ts_features.npy`. H2 through H13 also supply `corr_target.npy`. The vintage
Parquet files contain the point-in-time 10-K Item 1 excerpts used at feature
time. The local Granite directory must be the frozen
`ibm-granite/granite-embedding-english-r2` snapshot expected by the code.

These files are restricted inputs and are not part of the anonymous release.
The runner records their SHA-256 identities in the run artifacts.

## 4. Prepare the verified filing-embedding cache

If the verified H1-H12 cache is not already present, build it once:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m \
  experiments.multimodal_peer_encoder.build_historical_embeddings
```

The output is
`experiments/multimodal_peer_encoder/artifacts/historical_text_cache/`.
Before writing it, the builder checks the pinned package versions, model-weight
hash, selected model-tree hash, filing vintage dates, row order, and expected
source identities. It encodes each of the 6,169 distinct filing texts once on
CPU, expands them to the H1-H12 eligible rows, verifies finite normalized
float32 vectors, and writes a manifest and checksums. The v2 loader rechecks
the manifest, source and input identities, artifact hashes, shapes, dtypes,
and ticker alignment whenever it uses the cache.

Do not pass `--overwrite` merely to restart a run. A nonempty cache should be
reused after it validates. Under the protocol, it may not be regenerated after
any v2 result has been observed.

## 5. Run or resume the historical experiment

The canonical command has no data-dependent options:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m \
  experiments.multimodal_peer_encoder.run_v2_historical
```

On a fresh run, the command creates only
`experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2/`. On an
interrupted run, invoke the same command from the same sealed checkout. The
runner supports validated resume:
it binds the partial run to the code seal, configuration, environment, input
hashes, selection record, and completed-fit roster before it reuses anything.
It keeps valid completed fits and starts at the first incomplete fit. It
refuses a changed, malformed, selectively pruned, or unexpected partial run.
Do not delete a poor intermediate result, rerun a completed seed, or copy the
directory to obtain another attempt.

The fixed job consists of four mandatory self-tests, 48 development fits, and
18 expanding-window gate fits. Every fit uses 40 epochs. The final directory
contains:

```text
RUN_START.json
SELF_TESTS.log
self_tests.json
development_results.json
development_selection.json
training_history/development/   # 48 JSON records
training_history/gate/          # 18 JSON records
FINALIZATION_START.json
RUN_MANIFEST.json
historical_results.json
RUN_REPORT.md
RUN_RECEIPT.json
PASS or STOP
```

`PASS` means that the development-selected configuration beat the frozen
blend by at least 0.005 mean NDCG@5 and won strictly in at least four of six
historical gate slots. `STOP` means that one or both conditions failed. Both
are completed, reportable historical outcomes. Neither result authorizes a
modern evaluation, a paper edit, an all-history refit, or another tuning round.

## 6. Verify and tag the result

First verify the self-hashed receipt and the artifacts it names:

```bash
python -c "from pathlib import Path; from experiments.multimodal_peer_encoder.run_v2_historical import sha256_file, validate_manifest; p=Path('experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2'); r=validate_manifest(p/'RUN_RECEIPT.json'); assert all(sha256_file(p/n)==h for n,h in r['artifact_sha256'].items()); print(r['decision'], 'verified')"
git diff --quiet -- . \
  ':(exclude)results/windows_v3' ':(exclude)results/windows_v3/**'
git diff --cached --quiet -- . \
  ':(exclude)results/windows_v3' ':(exclude)results/windows_v3/**'
git status --short -- . \
  ':(exclude)results/windows_v3' ':(exclude)results/windows_v3/**'
```

Only files below the declared run directory should appear in the final status.
Commit the complete directory, then create exactly one annotated outcome tag:

```bash
git add -f experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2
git commit -m "v2: retain sealed historical gate result"
```

For a `PASS` marker:

```bash
git tag -a multimodal-peer-historical-pass-v2 \
  -m "Archive sealed v2 experiment after passing the historical gate"
```

For a `STOP` marker:

```bash
git tag -a multimodal-peer-historical-stop-v2 \
  -m "Archive sealed v2 experiment after failing the historical gate"
```

Do not create both tags. After tagging, repeat the path-limited
`git status --short` command above; it must again print nothing.

## Compute and data boundaries

V2 uses the local deterministic CPU reference with eight PyTorch threads. It
does not call AWS, use a GPU, or incur a cloud charge. Its incremental billed
compute cost is therefore USD 0 on the already available machine; electricity
and host opportunity cost are not estimated. The protocol allows at most USD
10 externally, but no paid resource is needed or authorized for this run.

The code may use only H1-H13 historical return windows, H1-H12 vintage filing
excerpts, and the verified H1-H12 Granite cache. It must never open, hash,
stat, or enumerate any descendant of `results/windows_v3/`. That modern tree,
raw price data, filing text, model weights, dense embedding arrays, and any
checkpoints are excluded from the anonymous artifact. The release may contain
the source, protocols, configuration, tests, and completed JSON/Markdown run
records because those records contain hashes and aggregate results rather than
the restricted inputs themselves.

The application remains candidate generation for comparable-company analysis
and residual-comovement monitoring. NDCG@5 does not establish valuation
accuracy, analyst forecast quality, trading profit, or a causal effect. The
experiment also remains conditional on the paper's ex-post company panel and
complete-window filters.
