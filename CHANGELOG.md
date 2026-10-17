# Changes

## v1.1.0 — 2026-10-17

- Add the unchanged 80/20 blend to the full 19-transition/regime comparison, four retrieval depths, and ticker-syntax exclusion. Explicitly distinguish 12 configuration-selection transitions from seven held out from selection.
- Publish the fixed gradient-boosting diagnostic and the paper's CNN V1 through its historical continuation decision. The CNN's six continuation tests fall inside the blend's selection period; no modern CNN test or later variant is included.
- Add four aggregate result extracts, five frozen-blend authority extracts, checksums, source identities, input requirements, and an arithmetic/partition verifier. Raw data, filing text, model weights, caches, and private execution records remain excluded.
- Preserve original dependency pins and scientific configurations. Public input-path and provenance adaptations are documented; no hyperparameter selection or model retraining was performed for this update.

Validation: 86 tests and 15 subtests passed. Both public blend runners reproduced their original transition scores and summary dictionaries exactly, including all bridge, depth, primary-score and ticker-exclusion comparator checks. The HGB numerical functions and CNN training/model/evaluation functions match the original implementations; these models were not retrained. Public extracts were checked against the exact retained fields of the original results.

The validation environment used Python 3.13.2 and NumPy 2.4.4; the original lockfile pins NumPy 2.4.2. This difference is disclosed and does not replace the recorded environment. Run `python -m pytest tests` after installing `requirements-dev.txt`, and `python scripts/verify_paper_results.py` for the data-free result checks. CNN tests use synthetic inputs and embedding stubs. Two dependency deprecation warnings from SWIG do not affect the test outcome.

The v1.0.0 tag and its existing Zenodo archive remain unchanged. Cite a version-specific DOI when referring to archived code and results.
