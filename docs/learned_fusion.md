# Historical gradient-boosting diagnostic

[`run_learned_fusion_followup.py`](../scripts/run_learned_fusion_followup.py)
implements the paper's single fixed histogram gradient-boosting classifier.
[`paper_results/learned_fusion.json`](../paper_results/learned_fusion.json)
contains the recorded configuration and aggregate results; it is a summary of
the original run, not a new training run for this release.

## Fixed experiment

The classifier trains on the **12 historical transitions H1→H2 through
H12→H13**. These are also the periods used to select TF-IDF configuration `b`
and the fixed blend weight. It evaluates once on the **six modern transitions
W1→W2 through W6→W7**, without training or configuration selection on them.
The H13→W1 bridge is not part of this experiment. Consequently these six tests
are a subset of the seven transitions outside configuration selection in the
full 19-transition comparison.

For each historical query, the binary positive label identifies its five
highest positive next-window residual-correlation peers. Training candidates
are the union of these five positives, the top 30 candidates from each input
ranking, and 30 randomly sampled candidates excluding the query and positives.
The original run produced 1,522,059 training pairs, including 85,195 positives.

The six features are each input's percentile rank and raw cosine similarity,
the smaller percentile rank, and the absolute rank difference. The inputs are
the one-window return path and vintage TF-IDF configuration `b` (at most 50,000
unigrams/bigrams, English stop words, `min_df=2`, sublinear term frequency and
Unicode accent stripping). Fixed classifier settings are:

| Setting | Value |
| --- | ---: |
| Loss | `log_loss` |
| Learning rate | 0.05 |
| Maximum iterations | 150 |
| Maximum leaf nodes | 15 |
| Minimum samples per leaf | 100 |
| L2 regularization | 1.0 |
| Class weight | `balanced` |
| Random seed | 20260730 |

The runner passes these arguments unchanged to scikit-learn; other estimator
options retain the defaults of the recorded scikit-learn 1.8.0 environment.
No model settings or blend-weight options are exposed by the release CLI.

At evaluation, the classifier scores every eligible candidate for every query,
excluding the query itself. The return path, TF-IDF, fixed 80% return / 20% text
percentile-rank blend, and classifier use identical sorted ticker universes.
Eligibility requires feature- and target-window returns and a filing strictly
before the feature window starts. Evaluation uses graded positive-correlation
NDCG@5, not the binary training labels.

The original six-transition mean NDCG@5 was 0.308 for the classifier, 0.336 for
the return path, 0.300 for TF-IDF and 0.373 for the blend. The classifier's
binary objective differs from the graded evaluation metric, but the experiment
does not isolate the reason for its weaker performance. It supports conclusions
about this tested configuration, not learned fusion in general.

## Inputs and execution

Use the repository's existing `requirements.lock`; this runner adds no packages.
The recorded experiment used Python 3.13.2, NumPy 2.4.2 and scikit-learn 1.8.0.
No market data, filing text, trained model or array caches are distributed.

| Default location | Required contents |
| --- | --- |
| `results/ea_historical/windows/H1/` … `H13/` | Each: `ts_features.npy`, `corr_target.npy`, `universe.parquet`, `meta.json` |
| `results/windows_v3/W1/` … `W7/` | The same four files per window |
| `results/vintage_descriptions/` | `H1_descriptions.parquet` … `H12_descriptions.parquet` and `W1_descriptions.parquet` … `W6_descriptions.parquet` |

The shared window loader requires finite arrays aligned with unique `ticker`
rows in `universe.parquet`, and ISO `start`/`end` dates in `meta.json`. Vintage
tables require `ticker`, `source`, `item1_text`, and `filing_date`; eligible text
has `source="10K"`, nonempty text and a pre-window filing date. The same input
alignment, minimum-universe and vintage checks as the primary experiment apply.
Target-only windows H13 and W7 need no filing table for this runner.

From the repository root:

```bash
python scripts/run_learned_fusion_followup.py --help
python scripts/run_learned_fusion_followup.py \
  --historical-windows-dir /path/to/historical/windows \
  --modern-windows-dir /path/to/modern/windows \
  --vintage-dir /path/to/vintage_descriptions \
  --output-root results/learned_fusion_followup
```

Omit input flags when using the default layout. Running without `--help`
trains the model and requires the empirical inputs; it is not a lightweight
demonstration. Data collection today can differ from the archived paper inputs.

Each successful run writes a fresh UTC-named directory containing
`learned_fusion_followup.json`, `artifact_manifest.json`, and `SHA256SUMS.txt`.
It refuses to overwrite a run directory. All required inputs are checked and
hashed before computation, and their hashes must remain unchanged afterward.
Hashes use logical repository paths even when the data live outside the checkout.
The JSON records per-transition results, training counts, fixed parameters,
software versions, runtime and source/input hashes. The model itself is not saved.

For this public packaging, the original numerical, sampling and evaluation code
is unchanged. The added CLI input paths and before/after provenance checks make
the existing experiment usable with externally supplied inputs. Packaging tests
use synthetic file fixtures; they do not retrain the empirical model.
