# Research code

This repository contains the source code for the return-path, filing-text, and
retrieval experiments in the accompanying submission.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
```

## Quick check

```bash
python -m unittest discover -s tests -v
```

## Entry points

```bash
python scripts/build_windows.py --help
python scripts/run_acceptance_robustness.py --help
python scripts/run_full_return_census.py --help
python scripts/run_same_window_text_diagnostic.py --help
python scripts/run_pairwise_alignment.py --help
```

The scripts accept paths to user-supplied source data and local caches. This
repository intentionally includes source code only: no manuscript, data,
model files, result files, or generated artifacts are distributed here.
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
