"""Build a history-free, allowlisted anonymous artifact archive.

The builder never copies repository history, symlinks, source-data caches, raw
filings, model weights, review notes, or local logs. It fails closed when a
requested file is absent or a common identity/credential pattern is detected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"

REQUIRED_FILES = [
    "artifact_authorities.json",
    "artifact_release/.gitignore",
    "artifact_release/README.md",
    "artifact_release/CODE_MAP.md",
    "artifact_release/DATA.md",
    "artifact_release/LICENSE",
    "artifact_release/LICENSE-CC-BY-4.0.txt",
    "artifact_release/LICENSE_NOTICE.md",
    "artifact_release/REPRODUCTION.md",
    "NUMBER_PROVENANCE.md",
    "RESULTS_VINTAGE.md",
    "requirements.lock",
    "environment/core.lock",
    "environment/models.lock",
    "environment/cloud-launch.lock",
    "environment/granite-sagemaker.requirements",
    "experiments/multimodal_peer_encoder/.gitignore",
    "experiments/multimodal_peer_encoder/AUDIT_v2.md",
    "experiments/multimodal_peer_encoder/PROTOCOL.md",
    "experiments/multimodal_peer_encoder/PROTOCOL_v2.md",
    "experiments/multimodal_peer_encoder/PROTOCOL_v2_CLARIFICATION_1.md",
    "experiments/multimodal_peer_encoder/README.md",
    "experiments/multimodal_peer_encoder/__init__.py",
    "experiments/multimodal_peer_encoder/build_historical_embeddings.py",
    "experiments/multimodal_peer_encoder/config/v2_historical_protocol.json",
    "experiments/multimodal_peer_encoder/evaluation.py",
    "experiments/multimodal_peer_encoder/historical_data.py",
    "experiments/multimodal_peer_encoder/model.py",
    "experiments/multimodal_peer_encoder/report_v2.py",
    "experiments/multimodal_peer_encoder/requirements.lock",
    "experiments/multimodal_peer_encoder/run_v2_historical.py",
    "experiments/multimodal_peer_encoder/v2_data.py",
    "experiments/multimodal_peer_encoder/v2_diagnostics.py",
    "experiments/multimodal_peer_encoder/v2_model.py",
    "experiments/multimodal_peer_encoder/v2_self_tests.py",
    "experiments/multimodal_peer_encoder/v2_training.py",
    "experiments/multimodal_peer_encoder/tests/test_v2_data.py",
    "experiments/multimodal_peer_encoder/tests/test_v2_model.py",
    "experiments/multimodal_peer_encoder/tests/test_v2_report.py",
    "experiments/multimodal_peer_encoder/tests/test_v2_runner.py",
    "experiments/multimodal_peer_encoder/tests/test_v2_training.py",
    "experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v1/RUN_REPORT.md",
    "experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v1/STOP",
    "experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v1/historical_selection.json",
    "paper/abstract_submitted.tex",
    "paper/main.pdf",
    "paper/main.tex",
    "paper/references.bib",
    "paper/figures/fig_acceptance_robustness.pdf",
    "paper/figures/fig_regime_timeline.pdf",
    "plots/__init__.py",
    "plots/style.py",
    "scripts/__init__.py",
    "scripts/build_anonymous_release.py",
    "scripts/build_dense_embedding_cache.py",
    "scripts/build_windows.py",
    "scripts/check_data_contract.py",
    "scripts/check_numbers.py",
    "scripts/audit_granite_vintage_tokens.py",
    "scripts/fetch_vintage_10k.py",
    "scripts/gen_acceptance_robustness_figure.py",
    "scripts/gen_fig_regime_timeline.py",
    "scripts/launch_vintage_granite_longctx.py",
    "scripts/metrics.py",
    "scripts/run_acceptance_robustness.py",
    "scripts/run_adversarial_followup.py",
    "scripts/run_full19_feature_time.py",
    "scripts/run_full_return_census.py",
    "scripts/run_learned_fusion_followup.py",
    "scripts/run_memory_fusion_audit.py",
    "scripts/run_pairwise_alignment.py",
    "scripts/run_same_window_text_diagnostic.py",
    "scripts/run_target_date_block_bootstrap.py",
    "scripts/run_strict_feature_time_application_audit.py",
    "scripts/run_downstream_peer_hedging.py",
    "scripts/run_outlier_sensitivity.py",
    "scripts/sec_filing_text.py",
    "scripts/summarize_window_metadata.py",
    "scripts/submission_preflight.py",
    "scripts/train_vintage_granite_longctx.py",
    "scripts/verify_artifact.py",
    "results/__init__.py",
    "results/historical_windows_summary.json",
    "results/acceptance_full19_feature_time/run_20260730T032944Z/SHA256SUMS.txt",
    "results/acceptance_full19_feature_time/run_20260730T032944Z/artifact_manifest.json",
    "results/acceptance_full19_feature_time/run_20260730T032944Z/full19_feature_time_census.json",
    "results/acceptance_robustness/run_20260730T032543Z/01_tfidf_historical_tuning.json",
    "results/acceptance_robustness/run_20260730T032543Z/02_blend_historical_tuning_modern_frozen.json",
    "results/acceptance_robustness/run_20260730T032543Z/03_modern_k_sensitivity.json",
    "results/acceptance_robustness/run_20260730T032543Z/04_fixed_ewma_return_path.json",
    "results/acceptance_robustness/run_20260730T032543Z/05_two_window_concatenated_return_path.json",
    "results/acceptance_robustness/run_20260730T032543Z/06_common_share_heuristic_exclusion.json",
    "results/acceptance_robustness/run_20260730T032543Z/07_symmetric_vs_feature_time_reconciliation.json",
    "results/acceptance_robustness/run_20260730T032543Z/SHA256SUMS.txt",
    "results/acceptance_robustness/run_20260730T032543Z/artifact_manifest.json",
    "results/acceptance_followup/run_20260730T065913Z/SHA256SUMS.txt",
    "results/acceptance_followup/run_20260730T065913Z/adversarial_followup_census.json",
    "results/acceptance_followup/run_20260730T065913Z/artifact_manifest.json",
    "results/acceptance_followup/run_20260730T065913Z/fig_acceptance_robustness.provenance.json",
    "results/acceptance_target_date_uncertainty/run_20260731T014000Z/SHA256SUMS.txt",
    "results/acceptance_target_date_uncertainty/run_20260731T014000Z/artifact_manifest.json",
    "results/acceptance_target_date_uncertainty/run_20260731T014000Z/target_date_block_bootstrap.json",
    "results/memory_fusion_audit/run_20260731T131959Z/SHA256SUMS.txt",
    "results/memory_fusion_audit/run_20260731T131959Z/artifact_manifest.json",
    "results/memory_fusion_audit/run_20260731T131959Z/memory_fusion_audit.json",
    "results/memory_fusion_audit/run_20260731T131959Z/fig_acceptance_robustness.provenance.json",
    "results/strict_feature_time_application/run_20260801T233929Z/SHA256SUMS.txt",
    "results/strict_feature_time_application/run_20260801T233929Z/artifact_manifest.json",
    "results/strict_feature_time_application/run_20260801T233929Z/strict_application_census.json",
    "results/strict_feature_time_application/run_20260801T233929Z/strict_application_per_query.csv.gz",
    "results/outlier_sensitivity/run_20260801T234540Z/SHA256SUMS.txt",
    "results/outlier_sensitivity/run_20260801T234540Z/artifact_manifest.json",
    "results/outlier_sensitivity/run_20260801T234540Z/outlier_sensitivity_census.json",
    "results/outlier_sensitivity/run_20260801T234540Z/per_transition.csv",
    "results/outlier_sensitivity/run_20260801T234540Z/publication_summary.json",
    "results/learned_fusion_followup/run_20260731T012556Z/SHA256SUMS.txt",
    "results/learned_fusion_followup/run_20260731T012556Z/artifact_manifest.json",
    "results/learned_fusion_followup/run_20260731T012556Z/learned_fusion_followup.json",
    "results/granite_vintage_token_audit/run_20260731T132708Z/SHA256SUMS.txt",
    "results/granite_vintage_token_audit/run_20260731T132708Z/summary.json",
    "results/granite_vintage_token_audit/run_20260731T132708Z/token_length_stats.csv",
    "results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/SHA256SUMS.txt",
    "results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/granite_longctx_census.json",
    "results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/per_query_ndcg.csv",
    "results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/per_transition.csv",
    "results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/provenance_manifest.json",
    "results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825/token_length_stats.csv",
    "results/full_return_census/run_20260802T_full_return_authority_v2/SHA256SUMS.txt",
    "results/full_return_census/run_20260802T_full_return_authority_v2/artifact_manifest.json",
    "results/full_return_census/run_20260802T_full_return_authority_v2/full_return_census.json",
    "results/full_return_census/run_20260802T_full_return_authority_v2/per_transition.csv",
    "results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/SHA256SUMS.txt",
    "results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/artifact_manifest.json",
    "results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2/same_window_text_diagnostic.json",
    "results/pairwise_alignment/run_20260806T_pairwise_authority_v3/SHA256SUMS.txt",
    "results/pairwise_alignment/run_20260806T_pairwise_authority_v3/artifact_manifest.json",
    "results/pairwise_alignment/run_20260806T_pairwise_authority_v3/pairwise_alignment.json",
    # Do not publish the superseded sampled-query/symmetric-eligibility *_v3
    # summaries.  The timestamped census runs above are the result authorities.
    "results/windows_v3/windows_summary.json",
    "tests/test_embedding_geometry.py",
    "tests/test_metrics.py",
    "tests/test_release_builder.py",
    "tests/test_sec_filing_text.py",
    "tests/test_windows.py",
]

OPTIONAL_FILES: list[str] = []

V2_RUN_DIRECTORY = (
    "experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v2"
)
V2_RUN_CORE_FILES = {
    "RUN_START.json",
    "SELF_TESTS.log",
    "self_tests.json",
    "development_results.json",
    "development_selection.json",
    "FINALIZATION_START.json",
    "RUN_MANIFEST.json",
    "historical_results.json",
    "RUN_REPORT.md",
    "RUN_RECEIPT.json",
}
V2_DEVELOPMENT_FIT_PATTERN = re.compile(
    r"dev-h(?:a|b)-(?:raw_scale|lw)-(?:global|rank_band)-"
    r"lr(?:0p003|0p03)-seed(?:11|29|47)\.json\Z"
)
V2_GATE_FIT_PATTERN = re.compile(
    r"gate-h(?:a|b)-(?:raw_scale|lw)-(?:global|rank_band)-"
    r"lr(?:0p003|0p03)-fold(?:1|2|3)-seed(?:11|29|47)\.json\Z"
)
V2_RELEASE_SOURCE_REBIND_ALLOWLIST = {
    # The runner is text-redacted in the anonymous copy. The release builder
    # and its test may receive post-result anonymity-only fixes, but scientific
    # sources must remain bitwise identical to the sealed run.
    # The root ignore file is likewise replaced with release-workflow metadata;
    # it does not participate in model training or evaluation.
    ".gitignore",
    "experiments/multimodal_peer_encoder/run_v2_historical.py",
    "scripts/build_anonymous_release.py",
    "tests/test_release_builder.py",
}

DESTINATION_OVERRIDES = {
    "artifact_release/.gitignore": ".gitignore",
    "artifact_release/README.md": "README.md",
    "artifact_release/CODE_MAP.md": "CODE_MAP.md",
    "artifact_release/DATA.md": "DATA.md",
    "artifact_release/LICENSE": "LICENSE",
    "artifact_release/LICENSE-CC-BY-4.0.txt": "LICENSE-CC-BY-4.0.txt",
    "artifact_release/LICENSE_NOTICE.md": "LICENSE_NOTICE.md",
    "artifact_release/REPRODUCTION.md": "REPRODUCTION.md",
}

TEXT_SUFFIXES = {
    ".bib", ".csv", ".json", ".lock", ".log", ".md", ".py", ".tex", ".txt",
    ".yml", ".yaml",
}

FORBIDDEN_PATH_PARTS = {
    ".git", ".claude", "__pycache__", "data", "data_local", "logs", "models",
    "reviews", "source_cache",
}

TEXT_PATTERNS = {
    "absolute macOS user path": re.compile(r"/" + r"Users/[A-Za-z0-9._-]+/"),
    "absolute Linux home path": re.compile(r"/" + r"home/[A-Za-z0-9._-]+/"),
    "AWS ARN": re.compile(r"\b" + r"arn:" + r"aws(?:-[a-z]+)?:[^\s\"']+"),
    "AWS access-key shape": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Generic documentation templates such as s3://{bucket}/{key} are safe;
    # concrete bucket names are not.
    "S3 URL": re.compile(
        r"\b" + r"s3:" + r"//(?![\[{(])[^\s\"']+"
    ),
    # Do not flag a 12-digit substring inside a decimal result or SHA digest.
    "12-digit account number": re.compile(
        # AWS account IDs are standalone numeric tokens.  Exclude DOI suffixes
        # and other identifiers in which the digits touch any letter or digit.
        r"(?<![A-Za-z0-9.])\d{12}(?![A-Za-z0-9.])"
    ),
    "non-anonymous email": re.compile(
        r"\b(?!anonymous@example\.com\b)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
    # Even an abbreviated commit can link the anonymous artifact back to a
    # private or later-published source repository.
    "source-control commit reference": re.compile(
        r"\b(?:commit|revision|git(?:\s+commit)?)\s*[:=#]?\s*[0-9a-f]{7,40}\b",
        re.IGNORECASE,
    ),
}

BINARY_MARKERS = [
    b"/" + b"Users/",
    b"/" + b"home/",
    b"arn:" + b"aws",
    b"BEGIN " + b"PRIVATE KEY",
    b"BEGIN RSA " + b"PRIVATE KEY",
    b"s3:" + b"//",
]

REDACTED_JSON_KEYS = {
    # A source-control hash can link a supposedly anonymous artifact back to a
    # non-anonymous repository if the same commit is published elsewhere.
    "project_git_commit",
    # Internal run manifests may record the workstation's absolute path.
    "run_directory",
}

# V2 uses more specific names for Git identities. Keep these fields present so
# the released result schema and report fixture remain usable, but replace the
# linkable value in the history-free anonymous copy.
ANONYMIZED_JSON_KEYS = {
    "code_commit",
    "peeled_commit",
    "base_protocol_commit",
    "clarification_commit",
}
ANONYMOUS_GIT_IDENTITY = "redacted-for-anonymous-release"
ANONYMOUS_USER_HOME = "<redacted-user-home>/"

ANONYMIZED_TEXT_FILES = {
    "experiments/multimodal_peer_encoder/AUDIT_v2.md",
    "experiments/multimodal_peer_encoder/artifacts/runs/run_20260802_v1/RUN_REPORT.md",
    "experiments/multimodal_peer_encoder/run_v2_historical.py",
    f"{V2_RUN_DIRECTORY}/SELF_TESTS.log",
    f"{V2_RUN_DIRECTORY}/RUN_REPORT.md",
}
GIT_IDENTITY_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")

RELEASE_RESULT_MANIFESTS = {
    (
        "results/acceptance_robustness/run_20260730T032543Z/"
        "artifact_manifest.json"
    ): [
        "01_tfidf_historical_tuning.json",
        "02_blend_historical_tuning_modern_frozen.json",
        "03_modern_k_sensitivity.json",
        "04_fixed_ewma_return_path.json",
        "05_two_window_concatenated_return_path.json",
        "06_common_share_heuristic_exclusion.json",
        "07_symmetric_vs_feature_time_reconciliation.json",
    ],
    (
        "results/acceptance_full19_feature_time/run_20260730T032944Z/"
        "artifact_manifest.json"
    ): ["full19_feature_time_census.json"],
    (
        "results/acceptance_followup/run_20260730T065913Z/"
        "artifact_manifest.json"
    ): ["adversarial_followup_census.json"],
    (
        "results/acceptance_target_date_uncertainty/run_20260731T014000Z/"
        "artifact_manifest.json"
    ): ["target_date_block_bootstrap.json"],
    (
        "results/memory_fusion_audit/run_20260731T131959Z/"
        "artifact_manifest.json"
    ): ["memory_fusion_audit.json"],
    (
        "results/strict_feature_time_application/run_20260801T233929Z/"
        "artifact_manifest.json"
    ): [
        "strict_application_census.json",
        "strict_application_per_query.csv.gz",
    ],
    (
        "results/outlier_sensitivity/run_20260801T234540Z/"
        "artifact_manifest.json"
    ): [
        "outlier_sensitivity_census.json",
        "per_transition.csv",
        "publication_summary.json",
    ],
    (
        "results/learned_fusion_followup/run_20260731T012556Z/"
        "artifact_manifest.json"
    ): ["learned_fusion_followup.json"],
    (
        "results/full_return_census/"
        "run_20260802T_full_return_authority_v2/artifact_manifest.json"
    ): ["full_return_census.json", "per_transition.csv"],
    (
        "results/same_window_text_diagnostic/"
        "run_20260802T_same_window_authority_v2/artifact_manifest.json"
    ): ["same_window_text_diagnostic.json"],
    (
        "results/pairwise_alignment/"
        "run_20260806T_pairwise_authority_v3/artifact_manifest.json"
    ): ["pairwise_alignment.json"],
}

RELEASE_CHECKSUM_GROUPS = {
    "results/acceptance_robustness/run_20260730T032543Z": [
        "01_tfidf_historical_tuning.json",
        "02_blend_historical_tuning_modern_frozen.json",
        "03_modern_k_sensitivity.json",
        "04_fixed_ewma_return_path.json",
        "05_two_window_concatenated_return_path.json",
        "06_common_share_heuristic_exclusion.json",
        "07_symmetric_vs_feature_time_reconciliation.json",
        "artifact_manifest.json",
    ],
    "results/acceptance_full19_feature_time/run_20260730T032944Z": [
        "artifact_manifest.json",
        "full19_feature_time_census.json",
    ],
    "results/acceptance_followup/run_20260730T065913Z": [
        "adversarial_followup_census.json",
        "artifact_manifest.json",
        "fig_acceptance_robustness.provenance.json",
    ],
    "results/acceptance_target_date_uncertainty/run_20260731T014000Z": [
        "artifact_manifest.json",
        "target_date_block_bootstrap.json",
    ],
    "results/memory_fusion_audit/run_20260731T131959Z": [
        "artifact_manifest.json",
        "memory_fusion_audit.json",
    ],
    "results/strict_feature_time_application/run_20260801T233929Z": [
        "artifact_manifest.json",
        "strict_application_census.json",
        "strict_application_per_query.csv.gz",
    ],
    "results/outlier_sensitivity/run_20260801T234540Z": [
        "artifact_manifest.json",
        "outlier_sensitivity_census.json",
        "per_transition.csv",
        "publication_summary.json",
    ],
    "results/learned_fusion_followup/run_20260731T012556Z": [
        "artifact_manifest.json",
        "learned_fusion_followup.json",
    ],
    "results/granite_vintage_token_audit/run_20260731T132708Z": [
        "summary.json",
        "token_length_stats.csv",
    ],
    "results/granite_vintage_longctx_census/granite-longctx-20260730-034718-3825": [
        "granite_longctx_census.json",
        "per_query_ndcg.csv",
        "per_transition.csv",
        "provenance_manifest.json",
        "token_length_stats.csv",
    ],
    "results/full_return_census/run_20260802T_full_return_authority_v2": [
        "artifact_manifest.json",
        "full_return_census.json",
        "per_transition.csv",
    ],
    "results/same_window_text_diagnostic/run_20260802T_same_window_authority_v2": [
        "artifact_manifest.json",
        "same_window_text_diagnostic.json",
    ],
    "results/pairwise_alignment/run_20260806T_pairwise_authority_v3": [
        "artifact_manifest.json",
        "pairwise_alignment.json",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe allowlist path: {relative}")
    if FORBIDDEN_PATH_PARTS.intersection(path.parts):
        raise ValueError(f"forbidden allowlist path: {relative}")
    return path


def _require_regular_files(paths: list[Path], *, label: str) -> None:
    missing = [path for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise RuntimeError(f"incomplete {label}: {names}")


def _canonical_json_sha256(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("canonical_payload_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_self_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} is not a JSON object")
    recorded = payload.get("canonical_payload_sha256")
    if recorded != _canonical_json_sha256(payload):
        raise RuntimeError(f"{path.name} canonical SHA-256 mismatch")
    return payload


def discover_complete_v2_run(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Return a complete, bounded v2 result set or no v2 results at all.

    A partial run is intentionally omitted from the anonymous archive. Once a
    decision marker exists, every declared core artifact and all 66 fit records
    become mandatory. No cache, checkpoint, array, or caller-chosen filename is
    discovered by this function.
    """

    run_relative = validate_relative_path(V2_RUN_DIRECTORY)
    run_dir = Path(project_root) / run_relative
    if not run_dir.exists():
        return []
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise RuntimeError("canonical v2 run path is not a regular directory")

    decisions = [name for name in ("PASS", "STOP") if (run_dir / name).is_file()]
    if not decisions:
        return []
    if len(decisions) != 1:
        raise RuntimeError("v2 run must contain exactly one PASS or STOP marker")
    decision = decisions[0]

    core_names = sorted(V2_RUN_CORE_FILES | {decision})
    core_paths = [run_dir / name for name in core_names]
    _require_regular_files(core_paths, label="completed v2 core artifact set")

    development_dir = run_dir / "training_history" / "development"
    gate_dir = run_dir / "training_history" / "gate"
    for directory in (development_dir, gate_dir):
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"v2 fit-history directory is unsafe: {directory.name}")
    # Strict resume preserves an interrupted publication under a hidden
    # ``.incomplete-*`` or ``.partial-*`` name. Those forensic remnants are
    # not completed fits and must neither change the canonical roster nor be
    # copied into an anonymous release. Unexpected visible entries still fail
    # the fixed count/name checks below.
    development_paths = sorted(
        path for path in development_dir.iterdir() if not path.name.startswith(".")
    )
    gate_paths = sorted(
        path for path in gate_dir.iterdir() if not path.name.startswith(".")
    )
    _require_regular_files(development_paths, label="v2 development-fit roster")
    _require_regular_files(gate_paths, label="v2 gate-fit roster")
    if len(development_paths) != 48 or any(
        V2_DEVELOPMENT_FIT_PATTERN.fullmatch(path.name) is None
        for path in development_paths
    ):
        raise RuntimeError("v2 development-fit roster is not the fixed 48-file grid")
    expected_development_names = {
        f"dev-h{architecture}-{softness}-{application}-lr{learning_rate}-seed{seed}.json"
        for architecture in ("a", "b")
        for softness in ("raw_scale", "lw")
        for application in ("global", "rank_band")
        for learning_rate in ("0p003", "0p03")
        for seed in (11, 29, 47)
    }
    if {path.name for path in development_paths} != expected_development_names:
        raise RuntimeError("v2 development-fit names do not match the declared grid")
    if len(gate_paths) != 18 or any(
        V2_GATE_FIT_PATTERN.fullmatch(path.name) is None for path in gate_paths
    ):
        raise RuntimeError("v2 gate-fit roster is not the fixed 18-file grid")

    selection = json.loads(
        (run_dir / "development_selection.json").read_text(encoding="utf-8")
    )
    if not isinstance(selection, dict):
        raise RuntimeError("v2 development selection is not a JSON object")
    selection_canonical = dict(selection)
    recorded_selection_sha = selection_canonical.pop("selection_sha256", None)
    selection_encoded = json.dumps(
        selection_canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if recorded_selection_sha != hashlib.sha256(selection_encoded).hexdigest():
        raise RuntimeError("v2 development-selection SHA-256 mismatch")
    if selection.get("development_results_sha256") != sha256_file(
        run_dir / "development_results.json"
    ):
        raise RuntimeError("v2 selection is not bound to development results")
    family_winners = selection.get("family_winners")
    if not isinstance(family_winners, list) or len(family_winners) != 2:
        raise RuntimeError("v2 selection lacks the two family winners")
    expected_gate_names: set[str] = set()
    for winner in family_winners:
        if not isinstance(winner, dict):
            raise RuntimeError("v2 family winner is not an object")
        architecture = str(winner.get("architecture", ""))
        if architecture not in {"H-A", "H-B"}:
            raise RuntimeError("v2 selection contains an unknown family")
        softness = str(winner.get("target_softness", ""))
        application = str(winner.get("application", ""))
        learning_rate = {0.003: "0p003", 0.03: "0p03"}.get(
            winner.get("learning_rate")
        )
        if softness not in {"raw_scale", "lw"} or application not in {
            "global",
            "rank_band",
        } or learning_rate is None:
            raise RuntimeError("v2 selection contains an undeclared gate setting")
        architecture_id = architecture.replace("-", "").lower()
        expected_gate_names.update(
            f"gate-{architecture_id}-{softness}-{application}-lr{learning_rate}-"
            f"fold{fold}-seed{seed}.json"
            for fold in (1, 2, 3)
            for seed in (11, 29, 47)
        )
    if {path.name for path in gate_paths} != expected_gate_names:
        raise RuntimeError("v2 gate-fit names do not match the locked family winners")

    for path in (
        run_dir / "RUN_START.json",
        run_dir / "self_tests.json",
        run_dir / "FINALIZATION_START.json",
        *development_paths,
        *gate_paths,
    ):
        _validate_self_hashed_json(path)

    prefinal_paths = [
        run_dir / "RUN_START.json",
        run_dir / "SELF_TESTS.log",
        run_dir / "self_tests.json",
        run_dir / "development_results.json",
        run_dir / "development_selection.json",
        *development_paths,
        *gate_paths,
    ]
    expected_prefinal_hashes = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in prefinal_paths
    }
    finalization = _validate_self_hashed_json(
        run_dir / "FINALIZATION_START.json"
    )
    if finalization.get("decision") != decision:
        raise RuntimeError("v2 finalization state and decision marker disagree")
    if finalization.get("prefinal_artifact_sha256") != expected_prefinal_hashes:
        raise RuntimeError("v2 finalization state does not bind the prefinal artifacts")

    manifest = _validate_self_hashed_json(run_dir / "RUN_MANIFEST.json")
    roster = manifest.get("fit_roster")
    if not isinstance(roster, dict) or {
        "development_completed": roster.get("development_completed"),
        "gate_completed": roster.get("gate_completed"),
        "total_completed": roster.get("total_completed"),
    } != {
        "development_completed": 48,
        "gate_completed": 18,
        "total_completed": 66,
    }:
        raise RuntimeError("v2 manifest does not attest the complete fit roster")
    if manifest.get("decision") != decision:
        raise RuntimeError("v2 manifest and decision marker disagree")
    if manifest.get("prefinal_artifact_sha256") != expected_prefinal_hashes:
        raise RuntimeError("v2 manifest does not bind the prefinal artifacts")
    if manifest.get("finalization_state_sha256") != sha256_file(
        run_dir / "FINALIZATION_START.json"
    ):
        raise RuntimeError("v2 manifest does not bind the finalization state")

    results = json.loads(
        (run_dir / "historical_results.json").read_text(encoding="utf-8")
    )
    if (
        results.get("run", {}).get("status") != decision
        or results.get("gate", {}).get("decision") != decision
    ):
        raise RuntimeError("v2 results and decision marker disagree")

    receipt = _validate_self_hashed_json(run_dir / "RUN_RECEIPT.json")
    if receipt.get("decision") != decision:
        raise RuntimeError("v2 receipt and decision marker disagree")
    receipt_hashes = receipt.get("artifact_sha256")
    if not isinstance(receipt_hashes, dict):
        raise RuntimeError("v2 receipt lacks artifact identities")
    expected_receipt_names = {
        "RUN_MANIFEST.json",
        "historical_results.json",
        "RUN_REPORT.md",
        decision,
    }
    if set(receipt_hashes) != expected_receipt_names:
        raise RuntimeError("v2 receipt names an unexpected final artifact set")
    for name, expected in receipt_hashes.items():
        if not isinstance(expected, str) or sha256_file(run_dir / name) != expected:
            raise RuntimeError(f"v2 receipt hash mismatch: {name}")

    selected = [*core_paths, *development_paths, *gate_paths]
    return sorted(path.relative_to(project_root).as_posix() for path in selected)


def scan_file(path: Path, relative: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    payload = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        for marker in BINARY_MARKERS:
            if marker in payload:
                findings.append(
                    {
                        "file": relative,
                        "kind": "binary marker",
                        "match": marker.decode(),
                    }
                )
        return findings
    text = payload.decode("utf-8", errors="replace")
    for label, pattern in TEXT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            findings.append(
                {"file": relative, "kind": label, "match": match.group(0)}
            )
    return findings


def redact_json_provenance(value: object) -> tuple[object, int]:
    """Remove repository/workstation identifiers without changing result data."""
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        count = 0
        for key, child in value.items():
            if key in REDACTED_JSON_KEYS:
                count += 1
                continue
            if key in ANONYMIZED_JSON_KEYS:
                redacted[key] = ANONYMOUS_GIT_IDENTITY
                count += 1
                continue
            clean_child, child_count = redact_json_provenance(child)
            redacted[key] = clean_child
            count += child_count
        return redacted, count
    if isinstance(value, list):
        redacted_list = []
        count = 0
        for child in value:
            clean_child, child_count = redact_json_provenance(child)
            redacted_list.append(clean_child)
            count += child_count
        return redacted_list, count
    if isinstance(value, str):
        return GIT_IDENTITY_PATTERN.subn(ANONYMOUS_GIT_IDENTITY, value)
    return value, 0


def anonymize_text_provenance(text: str, relative: str) -> tuple[str, int]:
    """Remove Git and workstation identities from named generated/audit prose."""

    if relative not in ANONYMIZED_TEXT_FILES:
        return text, 0
    clean, count = GIT_IDENTITY_PATTERN.subn(ANONYMOUS_GIT_IDENTITY, text)
    for label in ("absolute macOS user path", "absolute Linux home path"):
        clean, replacements = TEXT_PATTERNS[label].subn(ANONYMOUS_USER_HOME, clean)
        count += replacements
    return clean, count


def copy_allowlisted(
    relative_files: list[str], destination: Path
) -> dict[str, int]:
    transformations: dict[str, int] = {}
    for relative in relative_files:
        rel_path = validate_relative_path(relative)
        destination_relative = validate_relative_path(
            DESTINATION_OVERRIDES.get(relative, relative)
        )
        source = PROJECT_ROOT / rel_path
        if source.is_symlink():
            raise RuntimeError(f"refusing symlink: {relative}")
        if not source.is_file():
            raise FileNotFoundError(relative)
        target = destination / destination_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            clean_payload, redaction_count = redact_json_provenance(payload)
            if redaction_count:
                target.write_text(
                    json.dumps(clean_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                transformations[destination_relative.as_posix()] = redaction_count
                continue
        if source.suffix.lower() in {".md", ".txt", ".log", ".py"}:
            text_payload = source.read_text(encoding="utf-8")
            clean_text, redaction_count = anonymize_text_provenance(
                text_payload, relative
            )
            if redaction_count:
                target.write_text(clean_text, encoding="utf-8")
                transformations[destination_relative.as_posix()] = redaction_count
                continue
        shutil.copy2(source, target)
    return transformations


def _write_self_hashed_json(path: Path, payload: dict) -> None:
    payload["canonical_payload_sha256"] = _canonical_json_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def refresh_v2_run_integrity(root: Path) -> list[str]:
    """Refresh v2 identities after anonymous Git provenance is replaced."""

    run_dir = root / validate_relative_path(V2_RUN_DIRECTORY)
    receipt_path = run_dir / "RUN_RECEIPT.json"
    if not receipt_path.is_file():
        return []
    refreshed: list[str] = []

    self_tests_path = run_dir / "self_tests.json"
    self_test_log_path = run_dir / "SELF_TESTS.log"
    if not self_test_log_path.is_file():
        raise RuntimeError("v2 self-test log is missing")
    self_hashed_paths = [
        run_dir / "RUN_START.json",
        self_tests_path,
        *sorted((run_dir / "training_history" / "development").glob("*.json")),
        *sorted((run_dir / "training_history" / "gate").glob("*.json")),
    ]
    for path in self_hashed_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "canonical_payload_sha256" not in payload:
            raise RuntimeError(f"v2 self-hashed artifact is malformed: {path.name}")
        if path == self_tests_path:
            if "log_sha256" not in payload:
                raise RuntimeError("v2 self-test record lacks its log identity")
            payload["log_sha256"] = sha256_file(self_test_log_path)
        _write_self_hashed_json(path, payload)
        refreshed.append(path.relative_to(root).as_posix())

    finalization_path = run_dir / "FINALIZATION_START.json"
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization_hashes = finalization.get("prefinal_artifact_sha256")
    if not isinstance(finalization_hashes, dict):
        raise RuntimeError("v2 finalization artifact identities are malformed")
    finalization["prefinal_artifact_sha256"] = {
        name: sha256_file(run_dir / validate_relative_path(name))
        for name in sorted(finalization_hashes)
    }
    _write_self_hashed_json(finalization_path, finalization)
    refreshed.append(f"{V2_RUN_DIRECTORY}/FINALIZATION_START.json")

    manifest_path = run_dir / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prefinal_hashes = manifest.get("prefinal_artifact_sha256")
    if prefinal_hashes is not None:
        if not isinstance(prefinal_hashes, dict):
            raise RuntimeError("v2 prefinal artifact identities are malformed")
        manifest["prefinal_artifact_sha256"] = {
            name: sha256_file(run_dir / validate_relative_path(name))
            for name in sorted(prefinal_hashes)
        }
    if "finalization_state_sha256" in manifest:
        manifest["finalization_state_sha256"] = sha256_file(finalization_path)
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("v2 manifest lacks source identities")
    release_source_hashes: dict[str, str] = {}
    for name in sorted(source_hashes):
        source_path = root / validate_relative_path(name)
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(f"v2 release source is missing or unsafe: {name}")
        observed_sha256 = sha256_file(source_path)
        recorded_sha256 = source_hashes[name]
        if (
            name not in V2_RELEASE_SOURCE_REBIND_ALLOWLIST
            and observed_sha256 != recorded_sha256
        ):
            raise RuntimeError(
                f"v2 scientific source differs from the sealed run: {name}"
            )
        release_source_hashes[name] = observed_sha256
    manifest["source_sha256"] = release_source_hashes
    _write_self_hashed_json(manifest_path, manifest)
    refreshed.append(f"{V2_RUN_DIRECTORY}/RUN_MANIFEST.json")

    results_path = run_dir / "historical_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    provenance = results.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("v2 results lack provenance")
    provenance["manifest_sha256"] = sha256_file(manifest_path)
    result_source_hashes = provenance.get("source_sha256")
    if (
        not isinstance(result_source_hashes, dict)
        or set(result_source_hashes) != set(release_source_hashes)
    ):
        raise RuntimeError("v2 result and manifest source rosters disagree")
    provenance["source_sha256"] = dict(release_source_hashes)
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    refreshed.append(f"{V2_RUN_DIRECTORY}/historical_results.json")

    report_path = run_dir / "RUN_REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")
    manifest_row = re.compile(r"(?m)^(\| manifest_sha256 \| )[^|]+( \|)$")
    report_text, replacements = manifest_row.subn(
        rf"\g<1>{sha256_file(manifest_path)}\g<2>", report_text
    )
    if replacements != 1:
        raise RuntimeError("v2 report lacks its unique manifest identity row")
    for name, digest in release_source_hashes.items():
        source_row = re.compile(rf"(?m)^(\| {re.escape(name)} \| )[^|]+( \|)$")
        report_text, replacements = source_row.subn(
            rf"\g<1>{digest}\g<2>", report_text
        )
        if replacements != 1:
            raise RuntimeError(f"v2 report lacks its unique source row: {name}")
    report_path.write_text(report_text, encoding="utf-8")
    refreshed.append(f"{V2_RUN_DIRECTORY}/RUN_REPORT.md")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_hashes = receipt.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError("v2 receipt lacks artifact identities")
    receipt["artifact_sha256"] = {
        name: sha256_file(run_dir / validate_relative_path(name))
        for name in sorted(artifact_hashes)
    }
    _write_self_hashed_json(receipt_path, receipt)
    refreshed.append(f"{V2_RUN_DIRECTORY}/RUN_RECEIPT.json")
    return refreshed


def refresh_release_result_integrity(root: Path) -> list[str]:
    """Make nested result manifests authoritative for the redacted release."""
    refreshed: list[str] = []
    for manifest_relative, artifact_names in RELEASE_RESULT_MANIFESTS.items():
        manifest_path = root / validate_relative_path(manifest_relative)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["artifact_sha256"] = {
            name: sha256_file(manifest_path.parent / name)
            for name in artifact_names
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        refreshed.append(manifest_relative)

    for directory_relative, filenames in RELEASE_CHECKSUM_GROUPS.items():
        directory = root / validate_relative_path(directory_relative)
        sums_path = directory / "SHA256SUMS.txt"
        sums_path.write_text(
            "".join(
                f"{sha256_file(directory / filename)}  {filename}\n"
                for filename in sorted(filenames)
            ),
            encoding="utf-8",
        )
        refreshed.append(
            f"{directory_relative}/SHA256SUMS.txt"
        )
    refreshed.extend(refresh_v2_run_integrity(root))
    return sorted(refreshed)


def build_manifest(root: Path) -> dict:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "format": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def write_deterministic_zip(source_root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="safe suffix for the new release directory",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.label):
        parser.error("--label may contain only letters, digits, period, underscore, hyphen")

    required = [validate_relative_path(value).as_posix() for value in REQUIRED_FILES]
    missing = [value for value in required if not (PROJECT_ROOT / value).is_file()]
    if missing:
        print("Missing required release files:", file=sys.stderr)
        for value in missing:
            print(f"  {value}", file=sys.stderr)
        return 2

    selected = required + [
        validate_relative_path(value).as_posix()
        for value in OPTIONAL_FILES
        if (PROJECT_ROOT / value).is_file()
    ] + discover_complete_v2_run(PROJECT_ROOT)
    selected = sorted(dict.fromkeys(selected))
    destinations = [
        validate_relative_path(DESTINATION_OVERRIDES.get(value, value)).as_posix()
        for value in selected
    ]
    if len(destinations) != len(set(destinations)):
        print("Release allowlist maps multiple sources to one destination.", file=sys.stderr)
        return 2

    DIST_DIR.mkdir(exist_ok=True)
    release_root = DIST_DIR / f"anonymous_release_{args.label}"
    archive_path = DIST_DIR / f"anonymous_release_{args.label}.zip"
    if release_root.exists() or archive_path.exists():
        print(f"Refusing to overwrite existing release: {args.label}", file=sys.stderr)
        return 2
    release_root.mkdir()

    transformations = copy_allowlisted(selected, release_root)
    refreshed_integrity_files = refresh_release_result_integrity(release_root)

    findings = []
    for path in sorted(p for p in release_root.rglob("*") if p.is_file()):
        findings.extend(scan_file(path, path.relative_to(release_root).as_posix()))
    if findings:
        print(json.dumps({"status": "FAILED", "findings": findings}, indent=2))
        print(f"Inspection directory retained at {release_root.relative_to(PROJECT_ROOT)}")
        return 1

    report = {
        "status": "PASS",
        "files_scanned": len(selected),
        "history_included": False,
        "symlinks_included": False,
        "raw_data_included": False,
        "model_weights_included": False,
        "source_tex_transformed": False,
        "json_provenance_keys_removed": transformations,
        "release_integrity_files_regenerated": refreshed_integrity_files,
        "findings": [],
    }
    (release_root / "ANONYMITY_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(release_root)
    (release_root / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_deterministic_zip(release_root, archive_path)

    print(f"Release directory: {release_root.relative_to(PROJECT_ROOT)}")
    print(f"Archive: {archive_path.relative_to(PROJECT_ROOT)}")
    print(f"Archive SHA-256: {sha256_file(archive_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
