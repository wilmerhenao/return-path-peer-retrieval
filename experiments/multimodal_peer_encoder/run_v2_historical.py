#!/usr/bin/env python3
"""Run the sealed v2 experiment through its historical gate and stop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.multimodal_peer_encoder.report_v2 import (
    SCHEMA_VERSION as REPORT_SCHEMA_VERSION,
    generate_report,
    render_report,
    validate_result_artifact,
)
from experiments.multimodal_peer_encoder.v2_data import (
    DEFAULT_CACHE_DIR,
    DEFAULT_HISTORICAL_ROOT,
    DEFAULT_VINTAGE_ROOT,
    SELECTION_ARTIFACT_SCHEMA,
    SELECTION_ARTIFACT_STATUS,
    V2Transition,
    selection_payload_sha256,
    sha256_path_map,
    validate_selection_artifact,
    load_v2_transition,
)
from experiments.multimodal_peer_encoder.v2_diagnostics import (
    aggregate_model_diagnostics,
    average_retrieval_diagnostics,
    diagnose_head_transition,
    retrieval_diagnostic,
    safe_pearson,
)
from experiments.multimodal_peer_encoder.v2_self_tests import (
    report_source_sentinel,
    run_s1,
    run_s3,
)
from experiments.multimodal_peer_encoder.v2_model import (
    AdditiveBoostingHead,
    CenteredCrossModalHead,
    H_B_INITIAL_INTERCEPT,
    RANK_BAND_END,
    RANK_BAND_START,
)
from experiments.multimodal_peer_encoder.v2_training import (
    FitSpec,
    evaluate_fit,
    fit_head,
    frozen_baselines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "v2_historical_protocol.json"
EXPECTED_CONFIG_SHA256 = "d094b78050ecfd25760dc1304b7a5214649a13928dfb0cc8aaa14a6e4d76c0f8"
DEFAULT_RUN_DIR = (
    EXPERIMENT_DIR / "artifacts" / "runs" / "run_20260802_v2"
)
CODE_TAG = "multimodal-peer-v2-code-pre-results"
BASE_PROTOCOL_TAG = "multimodal-peer-protocol-v2-pre-results"
CLARIFICATION_TAG = "multimodal-peer-protocol-v2-clarification-1"
RESULT_SCHEMA = REPORT_SCHEMA_VERSION
RUN_START_SCHEMA = "multimodal-peer-v2-run-start-v1"
DEVELOPMENT_FIT_SCHEMA = "multimodal-peer-v2-development-fit-v1"
GATE_FIT_SCHEMA = "multimodal-peer-v2-gate-fit-v1"
SELF_TEST_SCHEMA = "multimodal-peer-v2-self-tests-v1"
FINALIZATION_SCHEMA = "multimodal-peer-v2-finalization-state-v1"
FIT_COMPLETE = "COMPLETE"
EXPECTED_ENVIRONMENT = {
    "python": "3.13.2",
    "numpy": "2.4.2",
    "torch": "2.11.0",
    "scikit-learn": "1.8.0",
    "polars": "1.39.3",
    "pytest": "9.0.2",
}
GIT_CLEAN_PATHS = (
    ".",
    ":(exclude)results/windows_v3",
    ":(exclude)results/windows_v3/**",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("canonical_payload_sha256", None)
    encoded = json.dumps(
        _json_safe(canonical),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    recorded = payload.get("canonical_payload_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("run manifest lacks its canonical SHA-256")
    if recorded != canonical_payload_sha256(payload):
        raise ValueError("run manifest canonical SHA-256 mismatch")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes_new(path: Path, encoded: bytes, *, mode: int = 0o644) -> None:
    """Publish a complete artifact atomically without replacing an existing one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.partial-{os.getpid()}-{time.time_ns()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic, no-clobber publication on this filesystem.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            os.unlink(temporary)


def write_json_atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _publish_bytes_new(path, encoded)


def write_text_atomic_new(path: Path, content: str, *, mode: int = 0o644) -> None:
    _publish_bytes_new(path, content.encode("utf-8"), mode=mode)


def _retire_incomplete(path: Path) -> Path:
    """Preserve a non-complete artifact, freeing its canonical name for retry."""

    retired = path.with_name(
        f".{path.name}.incomplete-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    os.replace(path, retired)
    _fsync_directory(path.parent)
    return retired


def _completed_payload_or_none(path: Path, schema: str) -> Mapping[str, Any] | None:
    """Load a complete self-hashed artifact; retire only demonstrably incomplete files."""

    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"artifact path is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _retire_incomplete(path)
        return None
    if not isinstance(payload, dict) or payload.get("status") != FIT_COMPLETE:
        _retire_incomplete(path)
        return None
    if payload.get("schema_version") != schema:
        raise RuntimeError(f"completed artifact has the wrong schema: {path}")
    recorded = payload.get("canonical_payload_sha256")
    if not isinstance(recorded, str) or recorded != canonical_payload_sha256(payload):
        raise RuntimeError(f"completed artifact failed its content hash: {path}")
    return payload


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("v2 configuration differs from the sealed fixed configuration")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("v2 configuration must be a JSON object")
    if value.get("schema_version") != "multimodal-peer-v2-historical-protocol-v1":
        raise ValueError("v2 configuration schema mismatch")
    expected = value["grid"]
    checks = (
        (expected["architectures"], ["H-A", "H-B"], "architectures"),
        (expected["target_softness"], ["raw_scale", "lw"], "target softness"),
        (expected["applications"], ["global", "rank_band"], "applications"),
        (expected["learning_rates"], [0.003, 0.03], "learning rates"),
        (expected["seeds"], [11, 29, 47], "seeds"),
        (expected["expected_development_fits"], 48, "development fit count"),
        (expected["expected_gate_fits"], 18, "gate fit count"),
        (expected["expected_total_fits"], 66, "total fit count"),
        (value["optimization"]["epochs"], 40, "epochs"),
        (value["optimization"]["optimizer"], "AdamW", "optimizer"),
        (value["optimization"]["weight_decay"], 0.0001, "weight decay"),
        (value["optimization"]["gradient_norm_clip"], 1.0, "gradient clipping"),
        (value["optimization"]["score_temperature"], 0.1, "score temperature"),
        (value["optimization"]["target_temperature"], 0.1, "target temperature"),
        (value["optimization"]["dtype"], "float32", "dtype"),
        (value["optimization"]["device"], "cpu", "device"),
        (value["optimization"]["torch_num_threads"], 8, "CPU thread count"),
        (value["optimization"]["deterministic_algorithms"], True, "determinism"),
        (value["optimization"]["early_stopping"], False, "early stopping"),
        (value["optimization"]["warm_start"], False, "warm start"),
        (value["gate"]["minimum_mean_advantage"], 0.005, "gate margin"),
        (value["gate"]["minimum_slot_wins"], 4, "gate wins"),
        (value["gate"]["total_slots"], 6, "gate slots"),
    )
    for observed, required, label in checks:
        if observed != required:
            raise ValueError(f"fixed v2 {label} mismatch: {observed!r}")
    h_a = AdditiveBoostingHead()
    h_b = CenteredCrossModalHead()
    model_checks = (
        (value["model"]["h_a_hidden_width"], h_a.mlp[0].out_features, "H-A width"),
        (value["model"]["h_a_initial_lambda"], float(h_a.lambda_.detach()), "H-A initialization"),
        (value["model"]["h_b_initial_intercept"], float(h_b.b.detach()), "H-B initialization"),
        (
            value["model"]["h_b_center_reference"],
            H_B_INITIAL_INTERCEPT,
            "H-B center",
        ),
        (value["model"]["rank_band_first"], RANK_BAND_START, "rank-band start"),
        (value["model"]["rank_band_last"], RANK_BAND_END, "rank-band end"),
    )
    for observed, required, label in model_checks:
        if observed != required:
            raise ValueError(f"fixed v2 {label} differs from implementation")
    return value


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def peeled_tag(tag: str) -> str:
    return _git("rev-parse", f"{tag}^{{}}").stdout.strip()


def peeled_annotated_tag(tag: str) -> str:
    if _git("cat-file", "-t", tag).stdout.strip() != "tag":
        raise RuntimeError(f"required annotated tag is not an annotated tag object: {tag}")
    return peeled_tag(tag)


def verify_code_seal() -> str:
    head = _git("rev-parse", "HEAD").stdout.strip()
    if peeled_annotated_tag(CODE_TAG) != head:
        raise RuntimeError("HEAD is not the sealed v2 implementation commit")
    if peeled_annotated_tag(BASE_PROTOCOL_TAG) != "redacted-for-anonymous-release":
        raise RuntimeError("base v2 protocol tag moved")
    if (
        peeled_annotated_tag(CLARIFICATION_TAG)
        != "redacted-for-anonymous-release"
    ):
        raise RuntimeError("v2 protocol clarification tag moved")
    return head


def _inside_run(path: str, run_dir: Path) -> bool:
    candidate = (PROJECT_ROOT / path).resolve()
    run = run_dir.resolve()
    try:
        candidate.relative_to(run)
    except ValueError:
        return False
    return True


def assert_source_clean(run_dir: Path | None = None) -> str:
    if _git("diff", "--quiet", "--", *GIT_CLEAN_PATHS, check=False).returncode != 0:
        raise RuntimeError("tracked source differs from the code seal")
    if (
        _git(
            "diff", "--cached", "--quiet", "--", *GIT_CLEAN_PATHS, check=False
        ).returncode
        != 0
    ):
        raise RuntimeError("index differs from the code seal")
    untracked = [
        line
        for line in _git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *GIT_CLEAN_PATHS,
        ).stdout.splitlines()
        if line
    ]
    if run_dir is None:
        if untracked:
            raise RuntimeError(f"untracked paths precede run creation: {untracked}")
        return "clean"
    outside = [path for path in untracked if not _inside_run(path, run_dir)]
    if outside:
        raise RuntimeError(f"untracked path outside declared run directory: {outside}")
    return "clean source; declared run artifacts only"


def git_status_short() -> str:
    """Return the exact porcelain status retained in the run record."""

    return _git("status", "--short", "--", *GIT_CLEAN_PATHS).stdout.rstrip()


def _logical_path(path: Path) -> str:
    absolute = Path(path).resolve()
    try:
        return absolute.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return absolute.name


def source_paths() -> tuple[Path, ...]:
    paths = (
        PROJECT_ROOT / ".gitignore",
        EXPERIMENT_DIR / "README.md",
        EXPERIMENT_DIR / "PROTOCOL_v2.md",
        EXPERIMENT_DIR / "PROTOCOL_v2_CLARIFICATION_1.md",
        EXPERIMENT_DIR / ".gitignore",
        EXPERIMENT_DIR / "__init__.py",
        DEFAULT_CONFIG,
        EXPERIMENT_DIR / "build_historical_embeddings.py",
        EXPERIMENT_DIR / "v2_model.py",
        EXPERIMENT_DIR / "v2_data.py",
        EXPERIMENT_DIR / "v2_training.py",
        EXPERIMENT_DIR / "v2_diagnostics.py",
        EXPERIMENT_DIR / "v2_self_tests.py",
        EXPERIMENT_DIR / "run_v2_historical.py",
        EXPERIMENT_DIR / "report_v2.py",
        EXPERIMENT_DIR / "historical_data.py",
        EXPERIMENT_DIR / "evaluation.py",
        EXPERIMENT_DIR / "model.py",
        PROJECT_ROOT / "scripts" / "__init__.py",
        PROJECT_ROOT / "scripts" / "build_anonymous_release.py",
        PROJECT_ROOT / "scripts" / "metrics.py",
        EXPERIMENT_DIR / "requirements.lock",
        PROJECT_ROOT / "environment" / "core.lock",
        PROJECT_ROOT / "environment" / "models.lock",
        *sorted((EXPERIMENT_DIR / "tests").glob("test_v2_*.py")),
        PROJECT_ROOT / "tests" / "test_release_builder.py",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"sealed source missing: {missing}")
    return paths


def environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "scikit-learn", "polars", "pytest"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **packages,
        "torch_git_version": torch.version.git_version,
        "torch_debug_build": bool(torch.version.debug),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_parallel_info": torch.__config__.parallel_info(),
        "torch_build_config": torch.__config__.show(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }


def assert_pinned_environment() -> Mapping[str, Any]:
    observed = environment_record()
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED_ENVIRONMENT.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"v2 pinned environment mismatch: {mismatches}")
    return observed


def open_or_resume_run(
    run_dir: Path,
    *,
    code_commit: str,
    config_sha256: str,
    environment: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    """Create RUN_START or validate the exact sealed identity of a resumed run."""

    start_path = run_dir / "RUN_START.json"
    expected = {
        "schema_version": RUN_START_SCHEMA,
        "run_id": run_dir.name,
        "code_commit": code_commit,
        "config_sha256": config_sha256,
        "environment": _json_safe(environment),
    }
    if run_dir.exists():
        if not run_dir.is_dir() or not start_path.is_file():
            raise RuntimeError(
                "an existing canonical run directory is resumable only with RUN_START.json"
            )
        try:
            payload = validate_manifest(start_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("existing RUN_START.json is invalid") from exc
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"RUN_START identity mismatch for {key}")
        started = payload.get("started_at_utc")
        if not isinstance(started, str) or not started:
            raise RuntimeError("RUN_START lacks its original start time")
        if set(payload) != {*expected, "started_at_utc", "canonical_payload_sha256"}:
            raise RuntimeError("RUN_START contains an undeclared field")
        return payload, True

    run_dir.mkdir(parents=True)
    payload: dict[str, Any] = {**expected, "started_at_utc": utc_now()}
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    write_json_atomic_new(start_path, payload)
    return validate_manifest(start_path), False


def load_receipted_result(run_dir: Path) -> Mapping[str, Any] | None:
    """Return a finalized result only when the complete run remains reconstructible."""

    receipt_path = run_dir / "RUN_RECEIPT.json"
    if not receipt_path.exists():
        return None
    try:
        receipt = validate_manifest(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("completed run receipt is invalid") from exc
    if receipt.get("schema_version") != "multimodal-peer-v2-final-receipt-v1":
        raise RuntimeError("completed run receipt schema mismatch")
    if receipt.get("run_id") != run_dir.name:
        raise RuntimeError("completed run receipt run identity mismatch")
    decision = receipt.get("decision")
    if decision not in {"PASS", "STOP"}:
        raise RuntimeError("completed run receipt decision is invalid")
    expected_names = {"RUN_MANIFEST.json", "historical_results.json", "RUN_REPORT.md", decision}
    hashes = receipt.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise RuntimeError("completed run receipt artifact roster mismatch")
    for name, expected_sha256 in hashes.items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"completed run artifact moved: {name}")
    manifest = validate_manifest(run_dir / "RUN_MANIFEST.json")
    if manifest.get("run_id") != run_dir.name or manifest.get("decision") != decision:
        raise RuntimeError("receipted manifest identity/decision mismatch")
    finalization_path = run_dir / "FINALIZATION_START.json"
    finalization_sha256 = manifest.get("finalization_state_sha256")
    if (
        not isinstance(finalization_sha256, str)
        or not finalization_path.is_file()
        or sha256_file(finalization_path) != finalization_sha256
    ):
        raise RuntimeError("receipted finalization state moved")
    try:
        finalization = validate_manifest(finalization_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("receipted finalization state is invalid") from exc
    if (
        finalization.get("schema_version") != FINALIZATION_SCHEMA
        or finalization.get("status") != FIT_COMPLETE
        or finalization.get("run_id") != run_dir.name
        or finalization.get("decision") != decision
    ):
        raise RuntimeError("receipted finalization identity/decision mismatch")
    manifest_prefinal = manifest.get("prefinal_artifact_sha256")
    finalization_prefinal = finalization.get("prefinal_artifact_sha256")
    if (
        not isinstance(manifest_prefinal, dict)
        or manifest_prefinal != finalization_prefinal
    ):
        raise RuntimeError("receipted prefinal artifact maps disagree")
    observed_prefinal, observed_fit_roster = prefinal_artifact_hashes(run_dir)
    if observed_prefinal != manifest_prefinal:
        raise RuntimeError("receipted prefinal artifact moved")
    if (
        manifest.get("fit_artifact_roster") != observed_fit_roster
        or finalization.get("fit_artifact_roster") != observed_fit_roster
    ):
        raise RuntimeError("receipted fit-artifact roster mismatch")
    result = json.loads((run_dir / "historical_results.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema_version") != RESULT_SCHEMA:
        raise RuntimeError("receipted historical result schema mismatch")
    if result.get("run", {}).get("status") != decision:
        raise RuntimeError("receipted historical result decision mismatch")
    if result.get("run", {}).get("run_id") != run_dir.name:
        raise RuntimeError("receipted historical result run identity mismatch")
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("receipted historical result lacks provenance")
    result_bindings = {
        "peeled_commit": manifest.get("code_commit"),
        "configuration_sha256": manifest.get("configuration_sha256"),
        "manifest_sha256": sha256_file(run_dir / "RUN_MANIFEST.json"),
        "source_sha256": manifest.get("source_sha256"),
        "input_sha256": manifest.get("input_sha256"),
        "environment": manifest.get("environment"),
        "git_status_short": manifest.get("git_status_short"),
    }
    for key, expected in result_bindings.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f"receipted result provenance mismatch for {key}")
    if result.get("self_tests") != manifest.get("self_tests"):
        raise RuntimeError("receipted self-test record differs from the manifest")
    if result.get("fit_roster") != manifest.get("fit_roster"):
        raise RuntimeError("receipted fit roster differs from the manifest")
    validate_result_artifact(result)
    if (run_dir / "RUN_REPORT.md").read_text(encoding="utf-8") != render_report(result):
        raise RuntimeError("receipted report is not reproducible from the result")
    if (run_dir / decision).read_text(encoding="utf-8") != f"{decision}\n":
        raise RuntimeError("receipted decision marker content mismatch")
    opposite = run_dir / ("STOP" if decision == "PASS" else "PASS")
    if opposite.exists():
        raise RuntimeError("receipted run contains contradictory decision markers")
    return result


def run_synthetic_suite() -> tuple[str, dict[str, Any], dict[str, Any]]:
    tests = sorted((EXPERIMENT_DIR / "tests").glob("test_v2_*.py"))
    command = [sys.executable, "-m", "pytest", "-q", *map(str, tests)]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"mandatory synthetic self-tests failed:\n{output}")
    return (
        output,
        {
            "name": "S2",
            "status": "PASS",
            "detail": "synthetic head learnability suite, including second-step H-A gradient",
        },
        {
            "name": "S4",
            "status": "PASS",
            "detail": "artifact-field substitution suite and report source sentinel",
        },
    )


def run_or_resume_self_tests(
    run_dir: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reuse only a complete, content-bound S1--S4 record."""

    log_path = run_dir / "SELF_TESTS.log"
    record_path = run_dir / "self_tests.json"
    record = _completed_payload_or_none(record_path, SELF_TEST_SCHEMA)
    if record is not None:
        if not log_path.is_file() or record.get("log_sha256") != sha256_file(log_path):
            raise RuntimeError("completed self-test log moved")
        rows = record.get("tests")
        if (
            not isinstance(rows, list)
            or [row.get("name") for row in rows if isinstance(row, dict)]
            != ["S1", "S2", "S3", "S4"]
            or any(row.get("status") != "PASS" for row in rows)
        ):
            raise RuntimeError("completed self-test roster is invalid")
        return [dict(row) for row in rows]
    if log_path.exists():
        _retire_incomplete(log_path)

    synthetic_log, s2, _ = run_synthetic_suite()
    s4 = report_source_sentinel()
    s3 = run_s3()
    s1 = run_s1(config)
    self_tests = [s1, s2, s3, s4]
    log = (
        synthetic_log
        + "\n"
        + "\n".join(
            f"{row['name']} {row['status']}: {row['detail']}" for row in self_tests
        )
        + "\n"
    )
    write_text_atomic_new(log_path, log)
    payload: dict[str, Any] = {
        "schema_version": SELF_TEST_SCHEMA,
        "status": FIT_COMPLETE,
        "tests": self_tests,
        "log_sha256": sha256_file(log_path),
    }
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    write_json_atomic_new(record_path, payload)
    return self_tests


def spec_from_mapping(value: Mapping[str, Any], seed: int) -> FitSpec:
    return FitSpec(
        architecture=value["architecture"],
        target_softness=value["target_softness"],
        application=value["application"],
        learning_rate=float(value["learning_rate"]),
        seed=int(seed),
    )


def _spec_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["architecture"],
        row["target_softness"],
        row["application"],
        float(row["learning_rate"]),
    )


def _tie_key(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, ...]:
    preference = config["selection"]["tie_preference"]
    return (
        -float(row["advantage_over_frozen_blend"]),
        preference["architecture"].index(row["architecture"]),
        preference["application"].index(row["application"]),
        preference["target_softness"].index(row["target_softness"]),
        preference["learning_rate"].index(float(row["learning_rate"])),
    )


def select_development_winners(
    fit_rows: Sequence[Mapping[str, Any]],
    baseline_by_transition: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fit_rows:
        grouped[_spec_key(row)].append(row)
    summaries: list[dict[str, Any]] = []
    seeds = config["grid"]["seeds"]
    dev_labels = ["H5->H6", "H6->H7"]
    baseline_mean = float(np.mean([baseline_by_transition[label] for label in dev_labels]))
    for key, rows in grouped.items():
        if sorted(int(row["seed"]) for row in rows) != seeds:
            raise RuntimeError(f"incomplete development seed group: {key}")
        scores = {
            label: float(np.mean([row["development_scores"][label] for row in rows]))
            for label in dev_labels
        }
        mean_score = float(np.mean(list(scores.values())))
        summaries.append(
            {
                "architecture": key[0],
                "target_softness": key[1],
                "application": key[2],
                "learning_rate": key[3],
                "development_scores": scores,
                "equal_transition_mean_ndcg_at_5": mean_score,
                "advantage_over_frozen_blend": mean_score - baseline_mean,
            }
        )
    if len(summaries) != 16:
        raise RuntimeError("development grid did not produce sixteen configurations")
    global_winner = dict(sorted(summaries, key=lambda row: _tie_key(row, config))[0])
    family_winners = []
    for architecture in config["grid"]["architectures"]:
        candidates = [row for row in summaries if row["architecture"] == architecture]
        family_winners.append(
            dict(sorted(candidates, key=lambda row: _tie_key(row, config))[0])
        )
    return global_winner, family_winners, summaries


def write_locked_selection(
    path: Path,
    *,
    global_winner: Mapping[str, Any],
    family_winners: Sequence[Mapping[str, Any]],
    development_sha256: str,
    development_input_sha256: Mapping[str, str],
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SELECTION_ARTIFACT_SCHEMA,
        "status": SELECTION_ARTIFACT_STATUS,
        "created_at_utc": utc_now(),
        "global_winner": dict(global_winner),
        "family_winners": [dict(value) for value in family_winners],
        "development_results_sha256": development_sha256,
        "development_input_sha256": dict(sorted(development_input_sha256.items())),
    }
    payload["selection_sha256"] = selection_payload_sha256(payload)
    write_json_atomic_new(path, payload)
    path.chmod(0o444)
    return validate_selection_artifact(path)


def _fit_id(prefix: str, spec: FitSpec, fold: int | None = None) -> str:
    architecture = spec.architecture.replace("-", "").lower()
    lr = format(spec.learning_rate, "g").replace(".", "p")
    fold_part = "" if fold is None else f"-fold{fold}"
    return (
        f"{prefix}-{architecture}-{spec.target_softness}-{spec.application}-"
        f"lr{lr}{fold_part}-seed{spec.seed}"
    )


def _spec_payload(spec: FitSpec) -> dict[str, Any]:
    return {
        "architecture": spec.architecture,
        "target_softness": spec.target_softness,
        "application": spec.application,
        "learning_rate": float(spec.learning_rate),
        "seed": int(spec.seed),
    }


def _require_finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not np.isfinite(number) or (nonnegative and number < 0.0):
        raise RuntimeError(f"{label} is not a valid finite value")
    return number


def _validate_scoring_calls(
    rows: Any,
    *,
    transitions: Sequence[str],
    calls_per_transition: int,
) -> None:
    if not isinstance(rows, list) or len(rows) != len(transitions) * calls_per_transition:
        raise RuntimeError("fit artifact scoring-call roster is incomplete")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("method"), str):
            raise RuntimeError("fit artifact contains an invalid scoring call")
        transition = row.get("transition")
        if transition not in transitions:
            raise RuntimeError("fit artifact scoring call names an unexpected transition")
        counts[str(transition)] += 1
        _require_finite(
            row.get("wall_time_seconds"),
            "scoring wall time",
            nonnegative=True,
        )
    if counts != {label: calls_per_transition for label in transitions}:
        raise RuntimeError("fit artifact scoring calls are not transition-complete")


def write_completed_fit_artifact(
    path: Path,
    *,
    schema: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    complete: dict[str, Any] = {
        "schema_version": schema,
        "status": FIT_COMPLETE,
        **dict(payload),
    }
    complete["canonical_payload_sha256"] = canonical_payload_sha256(complete)
    write_json_atomic_new(path, complete)
    loaded = _completed_payload_or_none(path, schema)
    if loaded is None:
        raise RuntimeError("newly published completed fit artifact disappeared")
    return loaded


def _validate_fit_binding(
    payload: Mapping[str, Any],
    *,
    fit_id: str,
    spec: FitSpec,
    code_commit: str,
    config_sha256: str,
    input_sha256: Mapping[str, str],
) -> None:
    expected = {
        "fit_id": fit_id,
        "specification": _spec_payload(spec),
        "code_commit": code_commit,
        "config_sha256": config_sha256,
        "input_sha256": dict(sorted(input_sha256.items())),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"completed fit artifact binding mismatch for {key}")


def load_completed_development_fit(
    path: Path,
    *,
    fit_id: str,
    spec: FitSpec,
    code_commit: str,
    config_sha256: str,
    input_sha256: Mapping[str, str],
    transitions: Sequence[str],
    baseline_by_transition: Mapping[str, float],
    expected_epochs: int,
) -> Mapping[str, Any] | None:
    payload = _completed_payload_or_none(path, DEVELOPMENT_FIT_SCHEMA)
    if payload is None:
        return None
    _validate_fit_binding(
        payload,
        fit_id=fit_id,
        spec=spec,
        code_commit=code_commit,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
    )
    row = payload.get("result")
    if not isinstance(row, dict):
        raise RuntimeError("completed development fit lacks its result")
    for key, value in _spec_payload(spec).items():
        if row.get(key) != value:
            raise RuntimeError(f"development fit result identity mismatch for {key}")
    scores = row.get("development_scores")
    if not isinstance(scores, dict) or set(scores) != set(transitions):
        raise RuntimeError("development fit validation-score roster mismatch")
    for label, value in scores.items():
        _require_finite(value, f"development score {label}")
    if set(baseline_by_transition) != set(transitions):
        raise RuntimeError("development baseline roster mismatch")
    score_mean = float(np.mean([float(scores[label]) for label in transitions]))
    baseline_mean = float(
        np.mean([float(baseline_by_transition[label]) for label in transitions])
    )
    if abs(
        _require_finite(
            row.get("equal_transition_mean_ndcg_at_5"),
            "development equal-transition mean",
        )
        - score_mean
    ) > 1e-12:
        raise RuntimeError("development equal-transition mean is inconsistent")
    if abs(
        _require_finite(
            row.get("advantage_over_frozen_blend"),
            "development baseline advantage",
        )
        - (score_mean - baseline_mean)
    ) > 1e-12:
        raise RuntimeError("development baseline advantage is inconsistent")
    losses = row.get("epoch_mean_losses")
    if not isinstance(losses, list) or len(losses) != expected_epochs:
        raise RuntimeError("development fit epoch history is incomplete")
    for value in losses:
        _require_finite(value, "development epoch loss")
    if _require_finite(row.get("final_epoch_loss"), "final development loss") != float(losses[-1]):
        raise RuntimeError("development final loss differs from its epoch history")
    _require_finite(row.get("wall_seconds"), "development wall time", nonnegative=True)
    _require_finite(row.get("peak_rss_mb"), "development peak RSS", nonnegative=True)
    parameters = row.get("trainable_parameters")
    if not isinstance(parameters, int) or parameters <= 0:
        raise RuntimeError("development parameter count is invalid")
    scaler = row.get("scaler")
    if not isinstance(scaler, dict) or not isinstance(scaler.get("n_records"), int):
        raise RuntimeError("development scaler record is invalid")
    _require_finite(scaler.get("mean"), "development scaler mean")
    _require_finite(
        scaler.get("population_std"),
        "development scaler standard deviation",
    )
    if int(scaler["n_records"]) <= 0 or float(scaler["population_std"]) <= 0.0:
        raise RuntimeError("development scaler is degenerate")
    _validate_scoring_calls(
        payload.get("scoring_calls"),
        transitions=transitions,
        calls_per_transition=1,
    )
    return payload


def load_completed_gate_fit(
    path: Path,
    *,
    fit_id: str,
    spec: FitSpec,
    fold: int,
    code_commit: str,
    config_sha256: str,
    input_sha256: Mapping[str, str],
    selection_sha256: str,
    transitions: Sequence[str],
    expected_epochs: int,
) -> Mapping[str, Any] | None:
    payload = _completed_payload_or_none(path, GATE_FIT_SCHEMA)
    if payload is None:
        return None
    _validate_fit_binding(
        payload,
        fit_id=fit_id,
        spec=spec,
        code_commit=code_commit,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
    )
    if payload.get("fold") != int(fold) or payload.get("selection_sha256") != selection_sha256:
        raise RuntimeError("completed gate fit selection/fold binding mismatch")
    history = payload.get("history")
    contributions = payload.get("contributions")
    if not isinstance(history, dict) or not isinstance(contributions, dict):
        raise RuntimeError("completed gate fit lacks reconstruction records")
    if history.get("fit_id") != fit_id or history.get("architecture") != spec.architecture:
        raise RuntimeError("gate fit history identity mismatch")
    if history.get("fold") != int(fold) or history.get("seed") != int(spec.seed):
        raise RuntimeError("gate fit history fold/seed mismatch")
    if history.get("specification") != {
        "target_softness": spec.target_softness,
        "application": spec.application,
        "learning_rate": float(spec.learning_rate),
    }:
        raise RuntimeError("gate fit history specification mismatch")
    validation_scores = history.get("validation_scores")
    if not isinstance(validation_scores, dict) or set(validation_scores) != set(transitions):
        raise RuntimeError("gate fit validation-score roster mismatch")
    for label, value in validation_scores.items():
        _require_finite(value, f"gate score {label}")
    losses = history.get("epoch_mean_losses")
    if not isinstance(losses, list) or len(losses) != expected_epochs:
        raise RuntimeError("gate fit epoch history is incomplete")
    for value in losses:
        _require_finite(value, "gate epoch loss")
    if _require_finite(history.get("final_epoch_loss"), "final gate loss") != float(losses[-1]):
        raise RuntimeError("gate final loss differs from its epoch history")
    _require_finite(history.get("wall_seconds"), "gate wall time", nonnegative=True)
    _require_finite(history.get("peak_rss_mb"), "gate peak RSS", nonnegative=True)
    parameters = history.get("trainable_parameters")
    if not isinstance(parameters, int) or parameters <= 0:
        raise RuntimeError("gate parameter count is invalid")

    seed_rows = contributions.get("gate_seed_rows")
    if not isinstance(seed_rows, list) or len(seed_rows) != len(transitions):
        raise RuntimeError("gate fit seed-row reconstruction is incomplete")
    if {row.get("transition") for row in seed_rows if isinstance(row, dict)} != set(transitions):
        raise RuntimeError("gate fit seed-row transition roster mismatch")
    for row in seed_rows:
        if (
            not isinstance(row, dict)
            or row.get("architecture") != spec.architecture
            or row.get("fold") != int(fold)
            or row.get("seed") != int(spec.seed)
        ):
            raise RuntimeError("gate fit seed-row identity mismatch")
        for key in (
            "ndcg_at_5",
            "semantic_ablation_ndcg_at_5",
            "semantic_ablation_ndcg_change",
        ):
            _require_finite(row.get(key), f"gate seed-row {key}")
        label = str(row["transition"])
        if abs(float(row["ndcg_at_5"]) - float(validation_scores[label])) > 1e-12:
            raise RuntimeError("gate seed row differs from validation history")
        if abs(
            float(row["semantic_ablation_ndcg_change"])
            - (
                float(row["ndcg_at_5"])
                - float(row["semantic_ablation_ndcg_at_5"])
            )
        ) > 1e-12:
            raise RuntimeError("gate semantic-ablation arithmetic is inconsistent")

    diagnostic = contributions.get("interpretability_row")
    if not isinstance(diagnostic, dict):
        raise RuntimeError("gate fit interpretability reconstruction is missing")
    if diagnostic.get("fold") != int(fold) or diagnostic.get("seed") != int(spec.seed):
        raise RuntimeError("gate fit interpretability identity mismatch")
    for key, value in {
        "target_softness": spec.target_softness,
        "application": spec.application,
        "learning_rate": float(spec.learning_rate),
    }.items():
        if diagnostic.get(key) != value:
            raise RuntimeError(f"gate fit interpretability mismatch for {key}")
    retrieval = contributions.get("retrieval_seed_rows")
    if not isinstance(retrieval, list) or len(retrieval) != len(transitions):
        raise RuntimeError("gate fit retrieval reconstruction is incomplete")
    if {row.get("transition") for row in retrieval if isinstance(row, dict)} != set(transitions):
        raise RuntimeError("gate fit retrieval transition roster mismatch")
    for row in retrieval:
        if not isinstance(row, dict) or row.get("method") != spec.architecture:
            raise RuntimeError("gate fit retrieval method identity mismatch")
        for key in (
            "top5_overlap_return_only",
            "top5_overlap_frozen_blend",
            "fraction_queries_different_from_return",
        ):
            _require_finite(row.get(key), f"gate retrieval {key}")
    gamma_rows = contributions.get("h_b_transition_gamma")
    expected_gamma = len(transitions) if spec.architecture == "H-B" else 0
    if not isinstance(gamma_rows, list) or len(gamma_rows) != expected_gamma:
        raise RuntimeError("gate fit gamma reconstruction roster mismatch")
    for row in gamma_rows:
        if (
            not isinstance(row, dict)
            or row.get("transition") not in transitions
            or row.get("seed") != int(spec.seed)
            or not isinstance(row.get("window_length"), int)
        ):
            raise RuntimeError("gate fit gamma identity mismatch")
        _require_finite(row.get("mean_gamma"), "gate mean gamma")
    contexts = contributions.get("context_by_transition")
    if not isinstance(contexts, dict) or set(contexts) != set(transitions):
        raise RuntimeError("gate fit context reconstruction is incomplete")
    for label, row in contexts.items():
        if not isinstance(row, dict) or row.get("transition") != label:
            raise RuntimeError("gate fit context identity mismatch")
    _validate_scoring_calls(
        contributions.get("scoring_calls"),
        transitions=transitions,
        calls_per_transition=2,
    )
    return payload


def _distribution_context_row(
    transition: V2Transition,
    scaler: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    lengths = np.asarray([len(text) for text in transition.texts], dtype=np.float64)
    cap = int(config["reporting"]["filing_collection_cap_chars"])
    return {
        "transition": transition.transition,
        "filing_length": {
            "p10": float(np.percentile(lengths, 10)),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "fraction_at_collection_cap": float(np.mean(lengths >= cap)),
            "missing_count": 0,
        },
        "window_length": int(transition.window_length),
        "training_scaler_mean": float(scaler.mean),
        "training_scaler_population_sd": float(scaler.population_std),
    }


def _target_softness_row(
    transition: V2Transition,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    scale = 1.0 - float(transition.shrinkage)
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"{transition.transition}: invalid target shrinkage scale")
    target_temperature = float(config["optimization"]["target_temperature"])
    return {
        "transition": transition.transition,
        "lw_shrinkage": float(transition.shrinkage),
        "c_t": scale,
        "target_temperature": target_temperature,
        "raw_equivalent_target_temperature": target_temperature / scale,
    }


def _collect_input_hashes(transitions: Sequence[V2Transition]) -> dict[str, str]:
    result: dict[str, str] = {}
    for transition in transitions:
        for path, digest in transition.input_sha256.items():
            logical = _logical_path(Path(path))
            value = str(digest)
            previous = result.get(logical)
            if previous is not None and previous != value:
                raise RuntimeError(f"input changed between transition loads: {logical}")
            result[logical] = value
    return dict(sorted(result.items()))


def _rehash_inputs(
    transitions: Sequence[V2Transition],
    *,
    selection_artifact: Path | None = None,
) -> dict[str, str]:
    paths = tuple(path for transition in transitions for path in transition.input_paths)
    absolute = sha256_path_map(paths, selection_artifact=selection_artifact)
    return dict(
        sorted((_logical_path(Path(path)), digest) for path, digest in absolute.items())
    )


def prefinal_artifact_hashes(run_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Bind every completed fit and pre-finalization run artifact."""

    development_histories = sorted(
        (run_dir / "training_history" / "development").glob("*.json")
    )
    gate_histories = sorted((run_dir / "training_history" / "gate").glob("*.json"))
    if len(development_histories) != 48 or len(gate_histories) != 18:
        raise RuntimeError("artifact history roster does not contain all 66 fits")
    core = (
        run_dir / "RUN_START.json",
        run_dir / "SELF_TESTS.log",
        run_dir / "self_tests.json",
        run_dir / "development_results.json",
        run_dir / "development_selection.json",
    )
    missing = [path for path in core if not path.is_file()]
    if missing:
        raise RuntimeError(f"prefinal run artifacts are missing: {missing}")
    paths = (*core, *development_histories, *gate_histories)
    identities = {
        path.relative_to(run_dir).as_posix(): sha256_file(path) for path in paths
    }
    roster = {
        "development_fit_ids": [path.stem for path in development_histories],
        "gate_fit_ids": [path.stem for path in gate_histories],
    }
    return dict(sorted(identities.items())), roster


def open_finalization_state(
    run_dir: Path,
    *,
    decision: str,
    artifact_sha256: Mapping[str, str],
    fit_roster: Mapping[str, Sequence[str]],
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    """Freeze volatile final fields so an interrupted publication is repeatable."""

    path = run_dir / "FINALIZATION_START.json"
    existing = _completed_payload_or_none(path, FINALIZATION_SCHEMA)
    expected = {
        "run_id": run_dir.name,
        "decision": decision,
        "prefinal_artifact_sha256": dict(sorted(artifact_sha256.items())),
        "fit_artifact_roster": _json_safe(fit_roster),
    }
    if existing is not None:
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(f"finalization state binding mismatch for {key}")
        _require_finite(
            existing.get("run_wall_time_seconds"),
            "finalization run wall time",
            nonnegative=True,
        )
        if not isinstance(existing.get("completed_at_utc"), str):
            raise RuntimeError("finalization state lacks completion time")
        return existing
    payload = {
        **expected,
        "completed_at_utc": utc_now(),
        "run_wall_time_seconds": float(elapsed_seconds),
    }
    return write_completed_fit_artifact(
        path,
        schema=FINALIZATION_SCHEMA,
        payload=payload,
    )


def _ensure_json_exact(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"partial finalization artifact differs: {path.name}")
        return
    _publish_bytes_new(path, encoded)


def _ensure_text_exact(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"partial finalization artifact differs: {path.name}")
        return
    _publish_bytes_new(path, encoded)


def write_final_receipt(run_dir: Path, decision: str) -> Mapping[str, Any]:
    """Write a self-hashed receipt over every final user-facing artifact."""

    opposite = run_dir / ("STOP" if decision == "PASS" else "PASS")
    if opposite.exists():
        raise RuntimeError("cannot receipt contradictory decision markers")
    artifact_names = (
        "RUN_MANIFEST.json",
        "historical_results.json",
        "RUN_REPORT.md",
        decision,
    )
    paths = [run_dir / name for name in artifact_names]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("cannot receipt an incomplete final artifact set")
    payload: dict[str, Any] = {
        "schema_version": "multimodal-peer-v2-final-receipt-v1",
        "run_id": run_dir.name,
        "decision": decision,
        "artifact_sha256": {
            path.name: sha256_file(path) for path in paths
        },
    }
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    receipt_path = run_dir / "RUN_RECEIPT.json"
    write_json_atomic_new(receipt_path, payload)
    return validate_manifest(receipt_path)


def validate_selection_binding(
    selection_path: Path,
    development_path: Path,
    development_input_sha256: Mapping[str, str],
) -> Mapping[str, Any]:
    locked = validate_selection_artifact(selection_path)
    if locked["development_results_sha256"] != sha256_file(development_path):
        raise RuntimeError("locked development-results identity moved")
    if dict(locked["development_input_sha256"]) != dict(development_input_sha256):
        raise RuntimeError("locked development input identities moved")
    return locked


def assert_frozen_blend_authority(
    baselines: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """Fail before fitting if the frozen v1 comparator has moved."""

    expected = config["data"]["expected_frozen_blend_ndcg"]
    if set(baselines) - set(expected):
        raise RuntimeError("frozen-blend check received an undeclared transition")
    for transition, row in baselines.items():
        observed = float(row["frozen_blend_ndcg_at_5"])
        authority = float(expected[transition])
        if abs(observed - authority) > 1e-12:
            raise RuntimeError(
                f"{transition}: frozen blend moved: {observed!r} != {authority!r}"
            )


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Execute tests, 48 development fits, 18 gate fits, report, and stop."""

    if Path(config_path).resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("production v2 requires the canonical configuration")
    if Path(run_dir).resolve() != DEFAULT_RUN_DIR.resolve():
        raise ValueError("production v2 requires the canonical run directory")
    config = load_config(config_path)
    torch.set_num_threads(int(config["optimization"]["torch_num_threads"]))
    torch.use_deterministic_algorithms(
        bool(config["optimization"]["deterministic_algorithms"])
    )
    code_commit = verify_code_seal()
    environment = assert_pinned_environment()
    config_sha256 = sha256_file(config_path)
    if run_dir.exists():
        assert_source_clean(run_dir)
    else:
        assert_source_clean()
    run_start, resumed = open_or_resume_run(
        run_dir,
        code_commit=code_commit,
        config_sha256=config_sha256,
        environment=environment,
    )
    started_at = str(run_start["started_at_utc"])
    assert_source_clean(run_dir)
    completed_result = load_receipted_result(run_dir)
    if completed_result is not None:
        print(f"[v2] verified completed run {run_dir.name}; no fits rerun", flush=True)
        return dict(completed_result)
    if resumed:
        print(f"[v2] resuming sealed run {run_dir.name}", flush=True)

    self_tests = run_or_resume_self_tests(run_dir, config)
    assert_source_clean(run_dir)

    development = tuple(
        load_v2_transition(index, stage="development") for index in range(1, 7)
    )
    development_input_sha256 = _collect_input_hashes(development)
    dev_train = development[:4]
    dev_validation = development[4:]
    dev_baselines = {row.transition: frozen_baselines(row) for row in dev_validation}
    assert_frozen_blend_authority(dev_baselines, config)
    baseline_by_transition = {
        label: float(value["frozen_blend_ndcg_at_5"])
        for label, value in dev_baselines.items()
    }

    development_rows: list[dict[str, Any]] = []
    scoring_calls: list[dict[str, Any]] = []
    peak_rss_mb = 0.0
    parameter_counts: dict[str, int] = {}
    for architecture, target_softness, application, learning_rate, seed in itertools.product(
        config["grid"]["architectures"],
        config["grid"]["target_softness"],
        config["grid"]["applications"],
        config["grid"]["learning_rates"],
        config["grid"]["seeds"],
    ):
        spec = FitSpec(
            architecture=architecture,
            target_softness=target_softness,
            application=application,
            learning_rate=float(learning_rate),
            seed=int(seed),
        )
        fit_id = _fit_id("dev", spec)
        artifact_path = run_dir / "training_history" / "development" / f"{fit_id}.json"
        validation_labels = [transition.transition for transition in dev_validation]
        completed = load_completed_development_fit(
            artifact_path,
            fit_id=fit_id,
            spec=spec,
            code_commit=code_commit,
            config_sha256=config_sha256,
            input_sha256=development_input_sha256,
            transitions=validation_labels,
            baseline_by_transition=baseline_by_transition,
            expected_epochs=int(config["optimization"]["epochs"]),
        )
        if completed is not None:
            row = dict(completed["result"])
            development_rows.append(row)
            scoring_calls.extend(dict(value) for value in completed["scoring_calls"])
            peak_rss_mb = max(peak_rss_mb, float(row["peak_rss_mb"]))
            parameter_counts[architecture] = int(row["trainable_parameters"])
            print(f"[v2] development_fit {fit_id} resume-skip", flush=True)
            continue

        print(f"[v2] development_fit {fit_id}", flush=True)
        fit = fit_head(dev_train, spec, config, progress_prefix="v2-dev")
        scores: dict[str, float] = {}
        fit_scoring_calls: list[dict[str, Any]] = []
        for transition in dev_validation:
            evaluated = evaluate_fit(fit, spec, transition, config)
            scores[transition.transition] = evaluated.ndcg_at_5
            fit_scoring_calls.append(
                {
                    "transition": transition.transition,
                    "method": f"development/{fit_id}",
                    "wall_time_seconds": evaluated.wall_seconds,
                }
            )
        equal_mean = float(np.mean(list(scores.values())))
        row = {
            "architecture": architecture,
            "target_softness": target_softness,
            "application": application,
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "development_scores": scores,
            "equal_transition_mean_ndcg_at_5": equal_mean,
            "advantage_over_frozen_blend": equal_mean
            - float(np.mean(list(baseline_by_transition.values()))),
            "final_epoch_loss": float(fit.epoch_mean_losses[-1]),
            "epoch_mean_losses": list(fit.epoch_mean_losses),
            "wall_seconds": fit.wall_seconds,
            "peak_rss_mb": fit.peak_rss_mb,
            "trainable_parameters": fit.trainable_parameters,
            "scaler": {
                "mean": fit.scaler.mean,
                "population_std": fit.scaler.population_std,
                "n_records": fit.scaler.n_records,
            },
        }
        write_completed_fit_artifact(
            artifact_path,
            schema=DEVELOPMENT_FIT_SCHEMA,
            payload={
                "fit_id": fit_id,
                "specification": _spec_payload(spec),
                "code_commit": code_commit,
                "config_sha256": config_sha256,
                "input_sha256": development_input_sha256,
                "result": row,
                "scoring_calls": fit_scoring_calls,
            },
        )
        if load_completed_development_fit(
            artifact_path,
            fit_id=fit_id,
            spec=spec,
            code_commit=code_commit,
            config_sha256=config_sha256,
            input_sha256=development_input_sha256,
            transitions=validation_labels,
            baseline_by_transition=baseline_by_transition,
            expected_epochs=int(config["optimization"]["epochs"]),
        ) is None:
            raise RuntimeError("published development fit could not be validated")
        development_rows.append(row)
        scoring_calls.extend(fit_scoring_calls)
        peak_rss_mb = max(peak_rss_mb, fit.peak_rss_mb)
        parameter_counts[architecture] = fit.trainable_parameters

    if len(development_rows) != int(config["grid"]["expected_development_fits"]):
        raise RuntimeError("development fit roster is incomplete")
    global_winner, family_winners, configuration_summaries = select_development_winners(
        development_rows, baseline_by_transition, config
    )
    development_payload = {
        "fits": development_rows,
        "configuration_summaries": configuration_summaries,
        "global_winner": global_winner,
        "family_winners": family_winners,
    }
    development_path = run_dir / "development_results.json"
    if _rehash_inputs(development) != development_input_sha256:
        raise RuntimeError("development inputs changed during development fitting")
    selection_path = run_dir / "development_selection.json"
    if development_path.exists():
        try:
            existing_development = json.loads(
                development_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if selection_path.exists():
                raise RuntimeError(
                    "locked development results became unreadable"
                ) from exc
            _retire_incomplete(development_path)
            existing_development = None
        if existing_development is not None and existing_development != development_payload:
            raise RuntimeError("existing development aggregate differs from fit artifacts")
    else:
        existing_development = None
    if existing_development is None:
        write_json_atomic_new(development_path, development_payload)

    if selection_path.exists():
        locked_selection = validate_selection_binding(
            selection_path, development_path, development_input_sha256
        )
        if (
            locked_selection["global_winner"] != global_winner
            or locked_selection["family_winners"] != family_winners
        ):
            raise RuntimeError("locked selection differs from reconstructed development winners")
    else:
        locked_selection = write_locked_selection(
            selection_path,
            global_winner=global_winner,
            family_winners=family_winners,
            development_sha256=sha256_file(development_path),
            development_input_sha256=development_input_sha256,
        )
    locked_selection = validate_selection_binding(
        selection_path, development_path, development_input_sha256
    )
    global_winner = dict(locked_selection["global_winner"])
    family_winners = [dict(value) for value in locked_selection["family_winners"]]
    assert_source_clean(run_dir)

    gate_transitions = tuple(
        load_v2_transition(
            index,
            stage="gate",
            selection_artifact=selection_path,
        )
        for index in range(1, 13)
    )
    gate_input_sha256 = _collect_input_hashes(gate_transitions)
    for input_path, digest in development_input_sha256.items():
        gate_digest = gate_input_sha256.get(input_path)
        if gate_digest is not None and gate_digest != digest:
            raise RuntimeError(f"development/gate input identity mismatch: {input_path}")
    gate_by_index = {index + 1: value for index, value in enumerate(gate_transitions)}
    gate_validation = gate_transitions[6:]
    expected_sizes = config["data"]["expected_gate_universe_sizes"]
    if [len(item.tickers) for item in gate_validation] != expected_sizes:
        raise RuntimeError("gate universe sizes differ from the v1 authority")
    gate_baselines = {
        transition.transition: frozen_baselines(transition)
        for transition in gate_validation
    }
    assert_frozen_blend_authority(gate_baselines, config)

    gate_seed_rows: list[dict[str, Any]] = []
    interpretability: dict[str, list[dict[str, Any]]] = {"H-A": [], "H-B": []}
    h_b_transition_gamma: list[dict[str, Any]] = []
    retrieval_seed_rows: list[dict[str, Any]] = []
    context_by_transition: dict[str, dict[str, Any]] = {}
    gate_fit_count = 0
    selection_sha256 = str(locked_selection["selection_sha256"])

    family_by_architecture = {row["architecture"]: row for row in family_winners}
    for architecture in config["grid"]["architectures"]:
        winner = family_by_architecture[architecture]
        for fold in config["data"]["gate_folds"]:
            train_set = tuple(gate_by_index[index] for index in fold["train_indices"])
            validation_set = tuple(
                gate_by_index[index] for index in fold["validation_indices"]
            )
            for seed in config["grid"]["seeds"]:
                spec = spec_from_mapping(winner, int(seed))
                fit_id = _fit_id("gate", spec, int(fold["fold"]))
                artifact_path = run_dir / "training_history" / "gate" / f"{fit_id}.json"
                validation_labels = [item.transition for item in validation_set]
                completed = load_completed_gate_fit(
                    artifact_path,
                    fit_id=fit_id,
                    spec=spec,
                    fold=int(fold["fold"]),
                    code_commit=code_commit,
                    config_sha256=config_sha256,
                    input_sha256=gate_input_sha256,
                    selection_sha256=selection_sha256,
                    transitions=validation_labels,
                    expected_epochs=int(config["optimization"]["epochs"]),
                )
                if completed is None:
                    print(f"[v2] gate_fit {fit_id}", flush=True)
                    fit = fit_head(train_set, spec, config, progress_prefix="v2-gate")
                    model_diag_rows = []
                    fit_scores: dict[str, float] = {}
                    fit_gate_seed_rows: list[dict[str, Any]] = []
                    fit_scoring_calls: list[dict[str, Any]] = []
                    fit_gamma_rows: list[dict[str, Any]] = []
                    fit_retrieval_rows: list[dict[str, Any]] = []
                    fit_context: dict[str, dict[str, Any]] = {}
                    for transition in validation_set:
                        evaluated = evaluate_fit(fit, spec, transition, config)
                        ablated = evaluate_fit(
                            fit,
                            spec,
                            transition,
                            config,
                            semantic_ablation=True,
                        )
                        semantic_change = evaluated.ndcg_at_5 - ablated.ndcg_at_5
                        fit_scores[transition.transition] = evaluated.ndcg_at_5
                        fit_gate_seed_rows.append(
                            {
                                "architecture": architecture,
                                "fold": int(fold["fold"]),
                                "seed": int(seed),
                                "transition": transition.transition,
                                "ndcg_at_5": evaluated.ndcg_at_5,
                                "semantic_ablation_ndcg_at_5": ablated.ndcg_at_5,
                                "semantic_ablation_ndcg_change": semantic_change,
                            }
                        )
                        fit_scoring_calls.extend(
                            (
                                {
                                    "transition": transition.transition,
                                    "method": f"{architecture}/seed{seed}",
                                    "wall_time_seconds": evaluated.wall_seconds,
                                },
                                {
                                    "transition": transition.transition,
                                    "method": f"{architecture}/seed{seed}/semantic-ablation",
                                    "wall_time_seconds": ablated.wall_seconds,
                                },
                            )
                        )
                        diagnostic = diagnose_head_transition(
                            fit,
                            spec,
                            transition,
                            config,
                            frozen_scores=evaluated.scores,
                            semantic_ablation_ndcg_change=semantic_change,
                        )
                        model_diag_rows.append(diagnostic)
                        if architecture == "H-B":
                            fit_gamma_rows.append(
                                {
                                    "transition": transition.transition,
                                    "seed": int(seed),
                                    "window_length": transition.window_length,
                                    "mean_gamma": diagnostic["active_pair_gamma"]["mean"],
                                }
                            )
                        baseline = gate_baselines[transition.transition]
                        fit_retrieval_rows.append(
                            retrieval_diagnostic(
                                transition=transition.transition,
                                method=architecture,
                                scores=evaluated.scores,
                                return_scores=baseline["return_scores"],
                                blend_scores=baseline["blend_scores"],
                                lw_target=transition.lw_target,
                            )
                        )
                        fit_context[transition.transition] = _distribution_context_row(
                            transition, fit.scaler, config
                        )
                    interpretability_row = aggregate_model_diagnostics(
                        fit, spec, int(fold["fold"]), model_diag_rows
                    )
                    history_row = {
                        "fit_id": fit_id,
                        "architecture": architecture,
                        "fold": int(fold["fold"]),
                        "seed": int(seed),
                        "specification": {
                            "target_softness": spec.target_softness,
                            "application": spec.application,
                            "learning_rate": spec.learning_rate,
                        },
                        "validation_scores": fit_scores,
                        "epoch_mean_losses": list(fit.epoch_mean_losses),
                        "final_epoch_loss": fit.epoch_mean_losses[-1],
                        "wall_seconds": fit.wall_seconds,
                        "peak_rss_mb": fit.peak_rss_mb,
                        "trainable_parameters": fit.trainable_parameters,
                    }
                    write_completed_fit_artifact(
                        artifact_path,
                        schema=GATE_FIT_SCHEMA,
                        payload={
                            "fit_id": fit_id,
                            "fold": int(fold["fold"]),
                            "specification": _spec_payload(spec),
                            "code_commit": code_commit,
                            "config_sha256": config_sha256,
                            "input_sha256": gate_input_sha256,
                            "selection_sha256": selection_sha256,
                            "history": history_row,
                            "contributions": {
                                "gate_seed_rows": fit_gate_seed_rows,
                                "interpretability_row": interpretability_row,
                                "h_b_transition_gamma": fit_gamma_rows,
                                "retrieval_seed_rows": fit_retrieval_rows,
                                "scoring_calls": fit_scoring_calls,
                                "context_by_transition": fit_context,
                            },
                        },
                    )
                    completed = load_completed_gate_fit(
                        artifact_path,
                        fit_id=fit_id,
                        spec=spec,
                        fold=int(fold["fold"]),
                        code_commit=code_commit,
                        config_sha256=config_sha256,
                        input_sha256=gate_input_sha256,
                        selection_sha256=selection_sha256,
                        transitions=validation_labels,
                        expected_epochs=int(config["optimization"]["epochs"]),
                    )
                    if completed is None:
                        raise RuntimeError("published gate fit could not be validated")
                else:
                    print(f"[v2] gate_fit {fit_id} resume-skip", flush=True)

                history_row = completed["history"]
                contributions = completed["contributions"]
                gate_fit_count += 1
                peak_rss_mb = max(peak_rss_mb, float(history_row["peak_rss_mb"]))
                parameter_counts[architecture] = int(history_row["trainable_parameters"])
                gate_seed_rows.extend(
                    dict(row) for row in contributions["gate_seed_rows"]
                )
                interpretability[architecture].append(
                    dict(contributions["interpretability_row"])
                )
                h_b_transition_gamma.extend(
                    dict(row) for row in contributions["h_b_transition_gamma"]
                )
                retrieval_seed_rows.extend(
                    dict(row) for row in contributions["retrieval_seed_rows"]
                )
                scoring_calls.extend(
                    dict(row) for row in contributions["scoring_calls"]
                )
                for label, row in contributions["context_by_transition"].items():
                    previous = context_by_transition.get(label)
                    if previous is not None and previous != row:
                        raise RuntimeError(
                            f"gate context reconstruction differs across seeds: {label}"
                        )
                    context_by_transition[label] = dict(row)

    if gate_fit_count != int(config["grid"]["expected_gate_fits"]):
        raise RuntimeError("gate fit roster is incomplete")

    slot_rows = []
    for transition in gate_validation:
        label = transition.transition
        baseline = gate_baselines[label]
        family_values: dict[str, list[float]] = {}
        for architecture in config["grid"]["architectures"]:
            values = [
                float(row["ndcg_at_5"])
                for row in gate_seed_rows
                if row["architecture"] == architecture and row["transition"] == label
            ]
            if len(values) != 3:
                raise RuntimeError(f"incomplete seed slot: {architecture} {label}")
            family_values[architecture] = values
        global_values = family_values[global_winner["architecture"]]
        global_mean = float(np.mean(global_values))
        blend = float(baseline["frozen_blend_ndcg_at_5"])
        slot_rows.append(
            {
                "transition": label,
                "n": len(transition.tickers),
                "return_only_ndcg_at_5": baseline["return_only_ndcg_at_5"],
                "tfidf_only_ndcg_at_5": baseline["tfidf_only_ndcg_at_5"],
                "frozen_blend_ndcg_at_5": blend,
                "h_a_seed_mean_ndcg_at_5": float(np.mean(family_values["H-A"])),
                "h_a_seed_sd_ndcg_at_5": float(np.std(family_values["H-A"], ddof=0)),
                "h_b_seed_mean_ndcg_at_5": float(np.mean(family_values["H-B"])),
                "h_b_seed_sd_ndcg_at_5": float(np.std(family_values["H-B"], ddof=0)),
                "global_winner_seed_mean_ndcg_at_5": global_mean,
                "global_winner_seed_sd_ndcg_at_5": float(np.std(global_values, ddof=0)),
                "winner_minus_frozen_blend": global_mean - blend,
            }
        )

    mean_advantage = float(np.mean([row["winner_minus_frozen_blend"] for row in slot_rows]))
    strict_wins = int(sum(row["winner_minus_frozen_blend"] > 0.0 for row in slot_rows))
    gate_passed = (
        mean_advantage >= float(config["gate"]["minimum_mean_advantage"])
        and strict_wins >= int(config["gate"]["minimum_slot_wins"])
    )
    decision = "PASS" if gate_passed else "STOP"

    mean_keys = {
        "return_only_ndcg_at_5": "return_only_ndcg_at_5",
        "tfidf_only_ndcg_at_5": "tfidf_only_ndcg_at_5",
        "frozen_blend_ndcg_at_5": "frozen_blend_ndcg_at_5",
        "h_a_ndcg_at_5": "h_a_seed_mean_ndcg_at_5",
        "h_b_ndcg_at_5": "h_b_seed_mean_ndcg_at_5",
        "global_winner_ndcg_at_5": "global_winner_seed_mean_ndcg_at_5",
        "global_winner_seed_dispersion": "global_winner_seed_sd_ndcg_at_5",
        "winner_minus_frozen_blend": "winner_minus_frozen_blend",
    }
    equal_means = {
        output: float(np.mean([row[source] for row in slot_rows]))
        for output, source in mean_keys.items()
    }

    retrieval_rows: list[dict[str, Any]] = []
    for transition in gate_validation:
        baseline = gate_baselines[transition.transition]
        for method, scores in (
            ("return-only", baseline["return_scores"]),
            ("frozen blend", baseline["blend_scores"]),
        ):
            retrieval_rows.append(
                retrieval_diagnostic(
                    transition=transition.transition,
                    method=method,
                    scores=scores,
                    return_scores=baseline["return_scores"],
                    blend_scores=baseline["blend_scores"],
                    lw_target=transition.lw_target,
                )
            )
        for architecture in config["grid"]["architectures"]:
            rows = [
                row
                for row in retrieval_seed_rows
                if row["transition"] == transition.transition
                and row["method"] == architecture
            ]
            aggregated = average_retrieval_diagnostics(rows)
            retrieval_rows.append(aggregated)
            if architecture == global_winner["architecture"]:
                selected = json.loads(json.dumps(aggregated))
                selected["method"] = f"global winner ({architecture})"
                retrieval_rows.append(selected)

    gamma_slots = []
    for transition in gate_validation:
        rows = [
            row for row in h_b_transition_gamma if row["transition"] == transition.transition
        ]
        if len(rows) != 3:
            raise RuntimeError("H-B window-context diagnostic lacks three seeds")
        gamma_slots.append(
            (
                float(transition.window_length),
                float(np.mean([row["mean_gamma"] for row in rows])),
            )
        )
    window_gamma_association = safe_pearson(
        np.asarray([row[0] for row in gamma_slots]),
        np.asarray([row[1] for row in gamma_slots]),
    )

    fit_roster = {
        "development_expected": 48,
        "development_completed": len(development_rows),
        "gate_expected": 18,
        "gate_completed": gate_fit_count,
        "total_expected": 66,
        "total_completed": len(development_rows) + gate_fit_count,
        "transition_level_optimizer_updates": 13440,
    }
    source_sha256 = {
        _logical_path(path): sha256_file(path) for path in source_paths()
    }
    validate_selection_binding(selection_path, development_path, development_input_sha256)
    if _rehash_inputs(
        gate_transitions, selection_artifact=selection_path
    ) != gate_input_sha256:
        raise RuntimeError("historical inputs changed during gate fitting")
    input_sha256 = dict(sorted({**development_input_sha256, **gate_input_sha256}.items()))
    prefinal_sha256, fit_artifact_roster = prefinal_artifact_hashes(run_dir)
    started_datetime = datetime.fromisoformat(started_at)
    if started_datetime.tzinfo is None:
        raise RuntimeError("RUN_START time is not timezone-aware")
    lifecycle_wall_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - started_datetime).total_seconds(),
    )
    finalization_state = open_finalization_state(
        run_dir,
        decision=decision,
        artifact_sha256=prefinal_sha256,
        fit_roster=fit_artifact_roster,
        elapsed_seconds=lifecycle_wall_seconds,
    )
    completed_at = str(finalization_state["completed_at_utc"])
    run_wall_time_seconds = float(finalization_state["run_wall_time_seconds"])
    manifest = {
        "schema_version": "multimodal-peer-v2-run-manifest-v1",
        "run_id": run_dir.name,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "decision": decision,
        "code_tag": CODE_TAG,
        "code_commit": code_commit,
        "base_protocol_tag": BASE_PROTOCOL_TAG,
        "base_protocol_commit": peeled_tag(BASE_PROTOCOL_TAG),
        "clarification_tag": CLARIFICATION_TAG,
        "clarification_commit": peeled_tag(CLARIFICATION_TAG),
        "configuration_sha256": config_sha256,
        "configuration": config,
        "git_status_short": git_status_short(),
        "selection_sha256": validate_selection_artifact(selection_path)[
            "selection_sha256"
        ],
        "source_sha256": source_sha256,
        "input_sha256": input_sha256,
        "environment": environment,
        "self_tests": self_tests,
        "fit_roster": fit_roster,
        "fit_artifact_roster": fit_artifact_roster,
        "prefinal_artifact_sha256": prefinal_sha256,
        "finalization_state_sha256": sha256_file(
            run_dir / "FINALIZATION_START.json"
        ),
        "resume_policy": {
            "mode": "strict-completed-artifact-skip-v1",
            "completed_fits_are_never_rerun": True,
            "incomplete_fits_may_be_retried": True,
        },
    }
    manifest["canonical_payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path = run_dir / "RUN_MANIFEST.json"
    _ensure_json_exact(manifest_path, manifest)
    validate_manifest(manifest_path)
    manifest_file_sha256 = sha256_file(manifest_path)

    assert_source_clean(run_dir)
    result = {
        "schema_version": RESULT_SCHEMA,
        "run": {
            "run_id": run_dir.name,
            "status": decision,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
        },
        "provenance": {
            "protocol_tag": f"{BASE_PROTOCOL_TAG} + {CLARIFICATION_TAG}",
            "peeled_commit": code_commit,
            "git_status_short": git_status_short(),
            "configuration_sha256": config_sha256,
            "manifest_sha256": manifest_file_sha256,
            "source_sha256": source_sha256,
            "input_sha256": input_sha256,
            "environment": environment,
        },
        "self_tests": self_tests,
        "fit_roster": fit_roster,
        "development": {
            "global_winner": global_winner,
            "family_winners": family_winners,
            "fits": development_rows,
        },
        "gate": {
            "decision": decision,
            "global_winner_label": (
                f"{global_winner['architecture']}/{global_winner['target_softness']}/"
                f"{global_winner['application']}"
            ),
            "thresholds": {
                "minimum_mean_advantage": config["gate"]["minimum_mean_advantage"],
                "minimum_strict_wins": config["gate"]["minimum_slot_wins"],
            },
            "observed": {
                "mean_advantage": mean_advantage,
                "strict_wins": strict_wins,
                "slot_count": len(slot_rows),
            },
            "slots": slot_rows,
            "equal_transition_means": equal_means,
        },
        "interpretability": {
            "h_a_models": interpretability["H-A"],
            "h_b_models": interpretability["H-B"],
            "window_context_gamma_association": window_gamma_association,
        },
        "retrieval_diagnostics": retrieval_rows,
        "target_softness_audit": [
            _target_softness_row(transition, config) for transition in gate_transitions
        ],
        "context_audit": [
            context_by_transition[transition.transition]
            for transition in gate_validation
        ],
        "compute_audit": {
            "trainable_parameters": parameter_counts,
            "run_wall_time_seconds": run_wall_time_seconds,
            "peak_resident_memory_mb": peak_rss_mb,
            "full_transition_scoring_calls": scoring_calls,
        },
    }
    result_path = run_dir / "historical_results.json"
    _ensure_json_exact(result_path, result)
    report_path = run_dir / "RUN_REPORT.md"
    if report_path.exists():
        _ensure_text_exact(report_path, render_report(result))
    else:
        generate_report(result_path, report_path)
    _ensure_text_exact(run_dir / decision, f"{decision}\n")
    write_final_receipt(run_dir, decision)
    print(
        f"[v2] decision={decision} mean_advantage={mean_advantage:.12f} "
        f"wins={strict_wins}/6",
        flush=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    run_experiment(config_path=args.config, run_dir=args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
