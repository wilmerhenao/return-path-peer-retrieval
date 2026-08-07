# Reproduction Guide

Run all commands from the extracted artifact root unless a section says
otherwise.

## Support tiers

### Tier 1: frozen-result verification

Tier 1 checks the released numerical record against the current manuscript. It
does not require market data, filing text, model weights, a GPU, or network
access.

```bash
python3 scripts/check_numbers.py
```

Expected final line:

```text
PASS: 109 checked claims; checksum manifests and standalone abstract verified.
```

### Tier 2: manuscript rebuild

Tier 2 rebuilds the PDF from the released LaTeX, bibliography, and figures. It
requires TeX Live 2026, `acmart` v2.19 dated 2026-06-27, and Poppler.

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
python3 ../scripts/submission_preflight.py
cd ..
```

The preflight expects pdfTeX, letter-size pages, embedded fonts, anonymous ACM
metadata, no unresolved references, and at most eight total pages.

### Tier 3: experiment recomputation from prepared inputs

Tier 3 requires the restricted input tree in `DATA.md`. It recomputes the
reported metrics from the prepared arrays and filing corpora. It is not
available from the public archive alone.

## Python environments

### Core CPU environment

The active numerical runs used Python 3.13.2. Create a small environment for
window construction, TF-IDF, metrics, tables, and plots:

```bash
python3.13 -m venv .venv-core
source .venv-core/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/core.lock
```

Run the unit tests after installation:

```bash
python -m unittest discover -s tests -v
```

### Local model environment

The short-context embedding scripts also need Torch, Transformers, and
Sentence Transformers:

```bash
python3.13 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/models.lock
```

Download current model snapshots into the expected directories:

```bash
hf download sentence-transformers/all-MiniLM-L12-v2 \
  --local-dir models/all-MiniLM-L12-v2
hf download BAAI/bge-base-en-v1.5 \
  --local-dir models/bge-base-en-v1.5
hf download nlpaueb/sec-bert-base \
  --local-dir models/sec-bert-base
hf download ibm-granite/granite-embedding-english-r2 \
  --local-dir models/granite-embedding-english-r2
```

Compare the downloaded files with the hashes in `DATA.md`. The artifact does
not record a retrievable commit revision for every model, so a current snapshot
can differ from the submitted one.

### Optional SageMaker launcher environment

The full-stored-excerpt Granite census was run in a managed GPU job. The local
launcher needs only boto3:

```bash
python3.13 -m venv .venv-cloud
source .venv-cloud/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/cloud-launch.lock
```

The launcher requires working AWS credentials, permission to create and stop a
SageMaker training job, permission to read and write the selected S3 bucket,
and these environment variables:

```bash
export SAGEMAKER_ROLE_ARN='<authorized-sagemaker-role-arn>'
export SAGEMAKER_S3_BUCKET='<authorized-bucket-name>'
export SAGEMAKER_TRAINING_IMAGE='<regional-pytorch-2.6-gpu-training-image-uri>'
python3 scripts/launch_vintage_granite_longctx.py --wait
```

The fixed job configuration uses region `us-east-1`, instance
`ml.g4dn.xlarge`, a 30 GB volume, a three-hour hard limit, and the AWS PyTorch
2.6.0 GPU Python 3.12 CUDA 12.6 SageMaker training image supplied by the
reproducer. AWS publishes different registry URIs by region, so an
account-shaped registry identifier is intentionally not embedded in the
anonymous launcher. The same value may be passed with `--training-image`.
This step incurs cloud charges and is not needed for Tier 1 or Tier 2.

The recorded in-container add-on package set is listed in
`environment/granite-sagemaker.requirements`. To prepare that base image
manually, install it with:

```bash
python -m pip install -r environment/granite-sagemaker.requirements
```

Torch 2.6.0+cu126 comes from the base image. The files under `environment/`
pin direct dependencies but do not record hashes or every resolved transitive
package. They are version manifests, not a guarantee of a bitwise-identical
environment. The root `requirements.lock` remains as a compatibility manifest
for the local model stack; the tier-specific files are clearer for new runs.

## Input checks

Before Tier 3, confirm that the following paths exist:

```text
data/universe.parquet
results/windows_v3/W1/meta.json
results/windows_v3/W7/meta.json
results/ea_historical/windows/H1/meta.json
results/ea_historical/windows/H13/meta.json
results/historical_windows_summary.json
results/vintage_descriptions/H1_descriptions.parquet
results/vintage_descriptions/W7_descriptions.parquet
results/vintage_all_encoders_v3/embedding_cache/
```

The frozen result JSON files contain SHA-256 maps for the exact submitted
inputs. Check those maps before interpreting numerical differences.

## Return-window construction

If the exact modern source cache and universe are supplied, rebuild W1-W7 with:

```bash
python3 scripts/build_windows.py \
  --universe data/universe.parquet \
  --output-dir results/windows_v3 \
  --source-cache results/source_cache \
  --missing-policy complete
```

Use `--force` only when an intentional rebuild should replace existing derived
window files. A rebuild from newly downloaded Yahoo Finance histories is a
fresh-data replication and may not reproduce the submitted hashes or numbers.

For a fresh source-level historical replication, build H1-H13 with:

```bash
python3 scripts/build_windows.py \
  --window-set historical \
  --universe data/universe.parquet \
  --output-dir results/ea_historical/windows \
  --source-cache results/source_cache \
  --missing-policy complete
```

The same caveat applies to both grids: rebuilding from current Yahoo Finance
histories is a fresh-data replication, not a reconstruction of the submitted
arrays. No hash inventory for the original historical source cache is released.
For an input-identical experiment rerun, place the supplied four-file H1-H13
directories under `results/ea_historical/windows/` and skip this step.

Check those directories against the released historical summary:

```bash
python3 scripts/summarize_window_metadata.py \
  --input-dir results/ea_historical/windows \
  --output results/historical_windows_summary.reproduction.json
```

Compare each `artifact_sha256` field with
`results/historical_windows_summary.json`. The separate output path preserves
the frozen authority.

## SEC Item 1 corpus construction

In a clean artifact tree, set an honest project identity and monitored contact
address before making SEC requests:

```bash
export SEC_USER_AGENT='<project name> <monitored contact email>'
python3 scripts/fetch_vintage_10k.py \
  --universe data/universe.parquet \
  --output-dir results/vintage_descriptions_rebuilt \
  --max-rps 2 \
  --batch-size 20
```

Replace both placeholders before running. The command processes H1-H13 and
W1-W7 by default. It is restartable and writes
`results/vintage_descriptions_rebuilt/vintage_summary.json` after completing
the requested windows. Company submission metadata is loaded once per CIK,
extracted sections are cached under `results/source_cache/sec_item1_v1/`, and
rows with transient request errors are retried on the next invocation. Use
`--windows W1,W2` to select a subset and `--limit` only for a smoke test.

The separate output directory prevents an accidental mix of the submitted
seven-column files and the fresh eleven-column files. Validate the completed
fresh corpus before using it in a clean numerical rerun:

```bash
python3 scripts/check_data_contract.py \
  --tier primary \
  --vintage-dir results/vintage_descriptions_rebuilt
```

The experiment scripts expect the selected corpus at
`results/vintage_descriptions/`. In a separate reproduction working tree,
place either the supplied submitted corpus or the validated fresh corpus at
that path. Do not merge rows from the two schemas or overwrite a frozen input
bundle in place.

The output schema and official SEC endpoints are listed in `DATA.md`. The
driver enforces the 8,000-character Item 1 cap. SEC acquisition time was not
recorded and depends on the number of historical submission pages, retries,
and the request ceiling. This step rebuilds filing text for a supplied frozen
universe; it does not reconstruct the 2,200-name candidate ticker list or its
original ticker-to-CIK mapping.

## Short-context dense cache

If the restricted filing corpora and the four model snapshots are present, the
cache can be rebuilt with:

```bash
source .venv-model/bin/activate
python3 scripts/build_dense_embedding_cache.py --device cpu
```

The cache builder validates filing dates, fixes each model's token limit and
pooling rule, preserves return-universe row order, and writes a content-addressed
cache manifest. The current dense result is produced by feature-time rescoring
in `run_adversarial_followup.py`.

## Active CPU command sequence

Activate the core environment and run:

```bash
source .venv-core/bin/activate

python3 scripts/run_full_return_census.py
python3 scripts/run_acceptance_robustness.py
python3 scripts/run_adversarial_followup.py
python3 scripts/run_full19_feature_time.py
python3 scripts/run_target_date_block_bootstrap.py --replicates 1000
python3 scripts/run_memory_fusion_audit.py
python3 scripts/run_downstream_peer_hedging.py
python3 scripts/run_strict_feature_time_application_audit.py
python3 scripts/run_outlier_sensitivity.py
python3 scripts/run_learned_fusion_followup.py
python3 scripts/run_same_window_text_diagnostic.py
python3 scripts/run_pairwise_alignment.py
```

The token-length audit additionally requires the Granite tokenizer files:

```bash
source .venv-model/bin/activate
python3 scripts/audit_granite_vintage_tokens.py
```

Every active script creates a new immutable timestamped directory and refuses
to overwrite an existing run. Some downstream scripts deliberately load the
frozen historical-selection authority under
`results/acceptance_robustness/run_20260730T032543Z/`. This keeps modern targets
out of tuning. It also means that a new primary run is not automatically wired
into every later diagnostic.

The two narrow diagnostics accept explicit authority paths and run labels:

```bash
python3 scripts/run_same_window_text_diagnostic.py \
  --run-id run_reproduction_same_window \
  --wf-robustness results/acceptance_robustness/run_20260730T032543Z/02_blend_historical_tuning_modern_frozen.json \
  --wf-dense results/acceptance_followup/run_20260730T065913Z/adversarial_followup_census.json

python3 scripts/run_pairwise_alignment.py \
  --run-id run_reproduction_pairwise \
  --tfidf-selection results/acceptance_robustness/run_20260730T032543Z/01_tfidf_historical_tuning.json
```

The frozen versions used by the paper have different run IDs. Do not reuse a
frozen run ID because the scripts reject an existing output directory.

## Figure regeneration

The figure scripts require `plots/style.py` and the input authorities named in
their source. With those files and the restricted data present, run:

```bash
python3 scripts/gen_acceptance_robustness_figure.py
python3 scripts/gen_fig_regime_timeline.py
```

The released figure PDFs are sufficient for Tier 2.

## Recorded runtimes

| Script or stage | Recorded wall time | Main hardware requirement |
|---|---:|---|
| `fetch_vintage_10k.py` | Not recorded | Network; SEC request pacing |
| `run_full_return_census.py` | 8.5 s | CPU |
| `run_acceptance_robustness.py` | 119.0 s | CPU |
| `run_adversarial_followup.py` | 25.0 s | CPU with cached embeddings |
| `run_full19_feature_time.py` | 31.5 s | CPU |
| Target-date sensitivity, 1,000 replicates | 225.2 s | CPU |
| `run_memory_fusion_audit.py` | 65.8 s | CPU |
| `run_downstream_peer_hedging.py` | 26.4 s | CPU |
| Strict application audit | 20.0 s | CPU |
| Outlier sensitivity | 80.2 s | CPU |
| Learned fusion | 48.5 s | CPU |
| Same-window text diagnostic | 12.6 s | CPU with cached embeddings |
| Pairwise alignment | 4.2 s | CPU |
| Full-excerpt Granite-r2 census | 640.5 s | Tesla T4, 15.8 GB |

The CPU entries were recorded on macOS ARM64 with Python 3.13.2 after inputs
were prepared. CPU model, RAM, BLAS thread count, market-data download time, SEC
download time, and short-context embedding time were not recorded. The timings
are reference values, not guarantees.

## Comparing a fresh run with the paper

Each fresh directory contains its own result JSON, manifest, and
`SHA256SUMS.txt`. Check it in place with:

```bash
cd results/<experiment>/<fresh-run-id>
shasum -a 256 -c SHA256SUMS.txt
cd ../../..
```

`scripts/check_numbers.py` validates the frozen authorities named in
`NUMBER_PROVENANCE.md`. It does not automatically switch to the newest run.
Compare fresh summary fields with the frozen fields listed in that document.

Differences can arise from a changed universe, revised Yahoo data, different
filing extraction, different model files, or device-dependent transformer
kernels. A run is an exact input-level reproduction only when all recorded
input hashes match.
