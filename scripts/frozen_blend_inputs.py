"""Verified public authority extracts and frozen inputs for blend audits.

An extract's checksum identifies its public bytes. Its source checksum identifies
the original result; these are deliberately different provenance claims.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import run_acceptance_robustness as core

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = PROJECT_ROOT
AUTHORITY_ROOT = PROJECT_ROOT / "evidence/frozen_blend/authorities"
ALPHA = 0.8
CONFIG = {"config_id": "b", "kwargs": {"max_features": 50000, "min_df": 2,
    "ngram_range": (1, 2), "stop_words": "english", "strip_accents": "unicode", "sublinear_tf": True}}

AUTHORITY_SOURCES = {
    "01_tfidf_historical_tuning.json": ("results/acceptance_robustness/run_20260730T032543Z/01_tfidf_historical_tuning.json", "2f730a48d3e995d675a4a4d4591222c0ba09ae85db1f5bb548cda54f52ff48ba"),
    "02_blend_historical_tuning_modern_frozen.json": ("results/acceptance_robustness/run_20260730T032543Z/02_blend_historical_tuning_modern_frozen.json", "32f43b42592758071bfd7b57411ec1f0f0f481290be88b9432afbff8ce3eaf64"),
    "03_modern_k_sensitivity.json": ("results/acceptance_robustness/run_20260730T032543Z/03_modern_k_sensitivity.json", "4f33aa6b47e767720f5c098a1567bfc6860b6be6fab2b0169bb8944a38801542"),
    "06_common_share_heuristic_exclusion.json": ("results/acceptance_robustness/run_20260730T032543Z/06_common_share_heuristic_exclusion.json", "eb5400bbc1b78601c0c62bf7ac7241bbfcc8641a0aa33b4e3cc8bc1c8aa775d5"),
    "full19_feature_time_census.json": ("results/acceptance_full19_feature_time/run_20260730T032944Z/full19_feature_time_census.json", "fa34b8d9425864101c3ea46ad846b318c8d5f30944079352582bb759ba5f3f02"),
}
# Reviewed differences: metrics comments/docstring; core comments/docstrings,
# external-path labels and provenance collection. Numerical functions/settings
# are unchanged. Only these exact published files are accepted for execution.
SOURCE_COMPATIBILITY = {
    "scripts/run_acceptance_robustness.py": {
        "archived_sha256": "55ed41a9379df7a8a74f5dd25233c209de45eacac26c8d487a8d7020650d10c3",
        "execution_sha256": "1cac259ad2eeac963cd40f58127c888baae3b5d69d6768f405230c66a33e2c34",
        "reviewed_changes": "documentation, external path labels, provenance collection; numerical functions unchanged",
    },
    "scripts/metrics.py": {
        "archived_sha256": "827e411d77466675d5837285f3c73a83d7a28fdd4d9739076dc8fd8b384b62c6",
        "execution_sha256": "75f1d072fc04ea029a3400a0cd32a2ba88319d0f4669d236bfe7e3ba436f0acb",
        "reviewed_changes": "comments and module docstring only",
    },
}
PRODUCER_ONLY = {"scripts/run_full19_feature_time.py": "ead37b0e80a4f7b5e2df1c4c9ad65da541d83fa2cc5155999120ba5039245e07"}


def configure(input_root=None, authority_root=None):
    global INPUT_ROOT, AUTHORITY_ROOT
    INPUT_ROOT = Path(input_root or PROJECT_ROOT).resolve()
    AUTHORITY_ROOT = Path(authority_root or PROJECT_ROOT / "evidence/frozen_blend/authorities").resolve()
    core.WINDOWS_MODERN_DIR = INPUT_ROOT / "results/windows_v3"
    core.WINDOWS_HISTORICAL_DIR = INPUT_ROOT / "results/ea_historical/windows"
    core.VINTAGE_DIR = INPUT_ROOT / "results/vintage_descriptions"
    core.WINDOW_CACHE.clear()
    return {name: AUTHORITY_ROOT / name for name in AUTHORITY_SOURCES}


def rel(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def verify_artifact(path):
    path = Path(path)
    if path.name not in AUTHORITY_SOURCES:
        raise AssertionError("unknown frozen authority")
    manifest = path.parent / "SHA256SUMS.txt"
    expected = {name.lstrip("*"): digest for digest, name in
        (line.split(maxsplit=1) for line in manifest.read_text().splitlines())}
    if core.sha256_file(path) != expected[path.name]:
        raise AssertionError(f"public authority checksum mismatch: {path.name}")
    wrapper = json.loads(path.read_text())
    original_path, original_hash = AUTHORITY_SOURCES[path.name]
    if (wrapper.get("schema_version") != "frozen-blend-authority-extract-v1"
        or wrapper.get("source_artifact_path") != original_path
        or wrapper.get("source_artifact_sha256") != original_hash):
        raise AssertionError(f"incorrect authority source identity: {path.name}")
    return wrapper["payload"]


def verify_inputs(blend, comparison):
    a, b = blend["provenance"]["sha256"], comparison["provenance"]["sha256"]
    common = set(a) & set(b)
    if any(a[p] != b[p] for p in common):
        raise AssertionError("source authorities disagree on shared input hashes")
    recorded = {**a, **b}
    executed = {}
    for name, policy in SOURCE_COMPATIBILITY.items():
        if recorded.get(name) != policy["archived_sha256"]:
            raise AssertionError(f"unexpected historical source identity: {name}")
        actual = core.sha256_file(PROJECT_ROOT / name)
        if actual != policy["execution_sha256"]:
            raise AssertionError(f"unreviewed execution source: {name}")
        executed[name] = dict(policy)
    numeric, dependencies, metadata, producer = {}, {}, {}, {}
    known_authorities = {source_path: (name, digest) for name, (source_path, digest) in AUTHORITY_SOURCES.items()}
    for name, expected in sorted(recorded.items()):
        if name in SOURCE_COMPATIBILITY:
            continue
        if name in PRODUCER_ONLY:
            if expected != PRODUCER_ONLY[name]:
                raise AssertionError(f"unknown historical producer: {name}")
            producer[name] = {"archived_sha256": expected, "executed_in_this_run": False}
        elif name == "requirements.lock":
            metadata[name] = {"archived_sha256": expected, "execution_sha256": core.sha256_file(PROJECT_ROOT / name)}
        elif name in known_authorities:
            filename, original_hash = known_authorities[name]
            if expected != original_hash:
                raise AssertionError(f"unexpected dependent authority identity: {name}")
            path = AUTHORITY_ROOT / filename
            verify_artifact(path)
            dependencies[name] = {"original_sha256": original_hash, "public_extract_sha256": core.sha256_file(path)}
        else:
            # Every frozen numerical data input remains required, byte-for-byte.
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not name.startswith("results/"):
                raise AssertionError(f"unsupported input path: {name}")
            actual = core.sha256_file(INPUT_ROOT / relative)
            if actual != expected:
                raise AssertionError(f"frozen numerical input changed: {name}")
            numeric[name] = actual
    return {"shared_authority_hashes_identical": len(common),
        "numerical_input_paths_checked": len(numeric), "numerical_input_sha256": numeric,
        "reviewed_execution_source_compatibility": executed,
        "historical_producer_sources_not_executed": producer,
        "dependent_authorities": dependencies, "environment_lockfile_identity": metadata}


def authority_provenance(paths):
    return {rel(path): {"public_extract_sha256": core.sha256_file(path),
        "source_artifact_sha256": AUTHORITY_SOURCES[Path(path).name][1]} for path in paths}
