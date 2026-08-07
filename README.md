# Retrieving Behavioral Equity Peers from Filing Text and Return History

This is the anonymous artifact for an ICAIF 2026 submission. It contains the
eight-page manuscript, its figures, immutable result summaries, claim checks,
and the active experiment and diagnostic scripts.

Start with the compiled [`paper/main.pdf`](paper/main.pdf). Its LaTeX source is
[`paper/main.tex`](paper/main.tex).

## What this release supports

The package has three support tiers.

| Tier | Supported in the public archive | Network access | Restricted inputs |
|---|---|---:|---:|
| 1. Frozen-result verification | Yes | No | No |
| 2. Manuscript rebuild | Yes, with the stated TeX tools | No | No |
| 3. Experiment recomputation from prepared inputs | Conditional on supplied inputs; see known limits | Optional for fresh downloads | Yes |

Tier 1 checks the released result files against every active numerical claim in
the manuscript. Tier 2 rebuilds and checks the PDF. Tier 3 reruns the numerical
experiments from prepared return panels, filing corpora, and embedding caches.

Raw prices, SEC filing text, derived return matrices, model weights, and dense
embedding caches are not redistributed. The run artifacts record SHA-256 hashes
for the prepared files and model weights consumed by the active experiments;
they do not give a complete hash inventory of every upstream download. A hash
identifies a submitted input but does not make it available. A fresh Yahoo
Finance download may differ from the data used for the submission.

See the companion documents before attempting Tier 3:

- `DATA.md` lists data sources, schemas, hashes, and withheld inputs.
- `REPRODUCTION.md` gives the environments and command sequence.
- `CODE_MAP.md` maps active claims to scripts and outputs.
- `LICENSE_NOTICE.md` maps files to their licenses and lists exclusions.
- `NUMBER_PROVENANCE.md` maps each manuscript number to an artifact field or
  derivation.
- `RESULTS_VINTAGE.md` gives a readable account of the active results and their
  limits.

## Quick verification

Run from the extracted artifact root:

```bash
python3 scripts/check_numbers.py
```

This check uses only the Python standard library. The expected final line is:

```text
PASS: 109 checked claims; checksum manifests and standalone abstract verified.
```

After installing the core environment described in `REPRODUCTION.md`, run:

```bash
python3 scripts/verify_artifact.py --strict-file-set
```

The comprehensive verifier checks the release manifest and authority index,
repeats the manuscript-claim audit, and runs the unit suite. The suite covers
the retrieval metric, return-path geometry, window boundaries, simple-return
construction, complete-case window policy, and SEC Item 1 extraction.

These commands verify the frozen authorities included in this release. They do
not rerun the experiments from raw market data or filing text.

## Manuscript rebuild

The submitted PDF was produced with pdfTeX from TeX Live 2026 and `acmart`
v2.19 dated 2026-06-27. The preflight also requires the Poppler commands
`pdfinfo`, `pdftotext`, and `pdffonts`.

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
python3 ../scripts/submission_preflight.py
```

The third post-BibTeX pass settles the final ACM page labels. The preflight
checks the PDF producer, ACM class version, anonymity, page count, page size,
embedded fonts, unresolved references, and the manuscript numbers. The output
must remain at most eight total pages, including figures and references.

## Experiment recomputation from prepared inputs

Tier 3 requires the prepared files listed in `DATA.md`. The public archive does
not include the SEC-to-GDELT workflow that produced the 2,200-name panel. It
does include the return-window builder and a restartable SEC Item 1 corpus
driver. With the frozen panel and exact inputs supplied, the active CPU
experiments can be rerun in about eleven minutes on the recorded ARM64
environment, excluding input construction, SEC acquisition, and dense-model
inference. Detailed commands, runtimes, and dependencies are in
`REPRODUCTION.md`.

The multimodal V2 directory is included for protocol, source, self-test, and
frozen-output audit. Its production runner is intentionally gated by private
Git seal tags, which are absent from this history-free release, so that runner
cannot be repeated from the public copy alone.

Two narrow diagnostics have standalone entry points:

```bash
python3 scripts/run_same_window_text_diagnostic.py
python3 scripts/run_pairwise_alignment.py
```

Their frozen authorities are:

```text
results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/
results/pairwise_alignment/run_20260806T_pairwise_authority_v3/
```

The same-window authority records NDCG@5 of
`0.296624 +/- 0.006890` for TF-IDF, `0.257439 +/- 0.006425` for 512-token
Granite-r2, and `0.212968 +/- 0.003921` for MiniLM. The pairwise authority uses
1,614 securities and all 1,301,691 unordered pairs. For BGE, 512-token
Granite-r2, MiniLM, and SEC-BERT, Spearman correlation ranges from 0.028947 to
0.045956 against W1 residual correlation and from 0.026655 to 0.049053 against
W2. The script, its checksummed output, and the exact cache identifiers are all
included in this package. These are descriptive diagnostics, not IID inference.

## Recorded runtime and storage

The CPU timings below came from Python 3.13.2 ARM64 runs after the required
input arrays and text files were present.

| Stage | Recorded wall time |
|---|---:|
| Full-return baseline census | 8.5 s |
| Primary selection and robustness suite | 119.0 s |
| Adversarial follow-up | 25.0 s |
| Full 19-transition census | 31.5 s |
| Target-date sensitivity, 1,000 replicates | 225.2 s |
| Memory and fusion audit | 65.8 s |
| Downstream peer calculation | 26.4 s |
| Strict feature-time application audit | 20.0 s |
| Outlier sensitivity | 80.2 s |
| Historical-only nonlinear fusion | 48.5 s |
| Same-window text diagnostic | 12.6 s |
| Pairwise alignment census | 4.2 s |

The full-stored-excerpt Granite-r2 census took 640.5 seconds on one Tesla T4
with 15.8 GB of GPU memory, using Python 3.12.8, Torch 2.6.0+cu126, FP16, and
batch size 1. CPU model and RAM were not recorded for the ARM64 run, so the
table is a reference rather than a hardware-independent estimate.

The recorded input sizes were:

| Prepared component | Disk use |
|---|---:|
| W1-W7 return windows | 109 MB |
| H1-H13 return windows | 127 MB |
| Vintage filing excerpts | 70 MB |
| Short-context embedding cache | 143 MB |
| Four model directories | About 1.8 GB |

The four prepared data components total about 449 MB, excluding model files,
fresh downloads, outputs, and temporary files.

## Result authorities

The immutable paths used by the manuscript are listed in
`NUMBER_PROVENANCE.md`. The main groups are:

```text
results/historical_windows_summary.json
results/full_return_census/run_20260802T_full_return_authority_v2/
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
results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/
results/pairwise_alignment/run_20260806T_pairwise_authority_v3/
```

Fresh runs write new immutable directories. `scripts/check_numbers.py` checks
the frozen paths above, not arbitrary fresh run directories.

## Integrity and anonymity

`ARTIFACT_MANIFEST.json` records the SHA-256 digest and size of each released
file. Each result directory also contains `SHA256SUMS.txt`. The claim checker
validates the nested manifests it uses. The SHA-256 of the final ZIP, printed
by the release builder, is the external identifier for the downloaded archive.

`ANONYMITY_REPORT.json` records the release scan. The archive is built from an
allowlist and excludes repository history, symlinks, source caches, review
notes, cloud credentials, local paths, and private keys.

## Known limits

- The 2,200-name panel was frozen on April 12, 2026 after ex-post matching of
  current SEC metadata to GDELT organization coverage and a 2025-2026 news
  filter. It is not a point-in-time security master.
- Historical results are conditional on that survivor and news-visible cohort.
  The direction of selection bias is not identified.
- The primary modern retrieval universe requires target-window return
  availability. The strict application audit separately freezes lists before
  target availability and reports attrition without reranking.
- The released filing excerpts have an 8,000-character collection cap. The
  Granite long-context result covers each stored excerpt, not necessarily the
  complete Item 1 section.
- Target-date ranges are conditional perturbation summaries, not confidence
  intervals. No IID query, pair, or transition inference is claimed.
- The outputs do not establish valuation peers, hedge effectiveness, trading
  profitability, or analyst outcomes.
- A public, bitwise rerun from raw Yahoo Finance data is not supported. Vendor
  histories can be revised after the submission run.
- The active full-return SIC-2 result is `0.213 +/- 0.009`, computed from the
  complete frozen 2,200-name candidate table. The older
  `results/full_universe_ndcg_corrected.csv` used a legacy 1,824-name SIC map;
  its `0.201 +/- 0.008` SIC-2 row is superseded and is not an active claim.

## Citation

No DOI has been assigned. During anonymous review, cite the artifact as:

```text
Anonymous. 2026. Retrieving Behavioral Equity Peers from Filing Text and
Return History. Anonymous artifact for ICAIF 2026, release <release-label>,
ZIP SHA-256 <archive-sha256>.
```

Replace the placeholders with the release label and hash printed by the final
archive builder. Author names and an archival identifier can be added after
double-blind review.

## License

Project-authored software is licensed under Apache-2.0. Project-authored
manuscript content, documentation, figures, and frozen research outputs are
licensed under CC BY 4.0. Upstream data, model weights, dependencies, ACM/TeX
materials, fonts, trademarks, and cited works remain under their original
terms. See `LICENSE_NOTICE.md` for the exact path-based scope and attribution.
