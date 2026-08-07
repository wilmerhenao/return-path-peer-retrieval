"""Launch the actual-vintage Granite long-context census with pure boto3.

No AWS CLI or SageMaker SDK commands are used.  The default lifecycle is:

1. validate and hash the local inputs;
2. upload them under a unique, non-overwriting run prefix;
3. submit one ``ml.g4dn.xlarge`` SageMaker training job;
4. monitor to a terminal state; and
5. download JSON/CSV outputs into a job-specific local directory.

Cost controls
-------------
* SageMaker training instances terminate when the job reaches Completed, Failed,
  or Stopped; no managed warm pool is configured.
* The hard runtime ceiling is three hours.
* Waiting is the default.  Ctrl-C sends ``stop_training_job`` and continues
  polling until SageMaker confirms a terminal state.
* ``--no-wait`` is an explicit opt-out and prints recovery/status commands.

Examples
--------
Run safely through terminal-state confirmation and download::

    python3 scripts/launch_vintage_granite_longctx.py --wait

Inspect or stop a known job (both calls are boto3)::

    python3 scripts/launch_vintage_granite_longctx.py --status JOB_NAME
    python3 scripts/launch_vintage_granite_longctx.py --stop JOB_NAME

Download an already completed job without submitting anything::

    python3 scripts/launch_vintage_granite_longctx.py --download JOB_NAME
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


# ---------------------------------------------------------------------------
# Fixed job configuration
# ---------------------------------------------------------------------------

REGION = "us-east-1"
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
S3_BUCKET = os.environ.get("SAGEMAKER_S3_BUCKET")
S3_PREFIX = "icaif2026-granite-longctx-census-v1"
CHECKPOINT_PREFIX = f"{S3_PREFIX}/checkpoints"
INSTANCE_TYPE = "ml.g4dn.xlarge"
MAX_RUN_SECONDS = 10_800
VOLUME_SIZE_GB = 30
TERMINAL_STATES = {"Completed", "Failed", "Stopped"}

# AWS publishes Deep Learning Container URIs by region.  Supplying the URI at
# launch time keeps account-like registry identifiers out of the anonymous
# artifact and makes the required region explicit to the person reproducing it.
TRAINING_IMAGE = os.environ.get("SAGEMAKER_TRAINING_IMAGE")
PIP_PACKAGES = (
    "polars==1.39.3",
    "pyarrow==23.0.1",
    "transformers==5.5.0",
    "sentence-transformers==5.3.0",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WINDOWS_DIR = PROJECT_ROOT / "results" / "windows_v3"
VINTAGE_DIR = PROJECT_ROOT / "results" / "vintage_descriptions"
# This is a read-only project asset even when `models` is a symlink.
MODEL_DIR = PROJECT_ROOT / "models" / "granite-embedding-english-r2"
MODEL_NAME = "granite-embedding-english-r2"
OUT_DIR = PROJECT_ROOT / "results" / "granite_vintage_longctx_census"

WINDOWS = [f"W{i}" for i in range(1, 8)]
FEATURE_WINDOWS = WINDOWS[:-1]
EXPECTED_OUTPUT_FILES = {
    "granite_longctx_census.json",
    "per_transition.csv",
    "per_query_ndcg.csv",
    "token_length_stats.csv",
    "provenance_manifest.json",
    "failure.json",
}
MODEL_EXCLUDED_FILES = {
    ".gitattributes",
    "README.md",
    # The local checkpoint also contains the same weights in safetensors form.
    "pytorch_model.bin",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def new_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"granite-longctx-{timestamp}-{secrets.token_hex(2)}"


def required_data_files() -> dict[str, Path]:
    """Return unique channel-relative names for the exact census inputs."""
    result: dict[str, Path] = {}
    for window in WINDOWS:
        result[f"windows/{window}/universe.parquet"] = (
            WINDOWS_DIR / window / "universe.parquet"
        )
        result[f"windows/{window}/meta.json"] = (
            WINDOWS_DIR / window / "meta.json"
        )
    for window in FEATURE_WINDOWS:
        result[f"windows/{window}/ts_features.npy"] = (
            WINDOWS_DIR / window / "ts_features.npy"
        )
    for window in WINDOWS:
        result[f"vintage/{window}_descriptions.parquet"] = (
            VINTAGE_DIR / f"{window}_descriptions.parquet"
        )
    for window in WINDOWS[1:]:
        result[f"windows/{window}/corr_target.npy"] = (
            WINDOWS_DIR / window / "corr_target.npy"
        )
    result["code/train_vintage_granite_longctx.py"] = (
        SCRIPTS_DIR / "train_vintage_granite_longctx.py"
    )
    result["code/metrics.py"] = SCRIPTS_DIR / "metrics.py"
    return result


def selected_model_files() -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not MODEL_DIR.is_dir():
        return result
    for path in sorted(MODEL_DIR.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(MODEL_DIR).as_posix()
        if relative in MODEL_EXCLUDED_FILES:
            continue
        result[relative] = path
    return result


def validate_local_inputs() -> tuple[dict[str, Path], dict[str, Path]]:
    data_files = required_data_files()
    model_files = selected_model_files()
    missing = [name for name, path in data_files.items() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(f"missing census inputs:\n{formatted}")
    if "model.safetensors" not in model_files:
        raise FileNotFoundError(
            f"{MODEL_DIR}/model.safetensors is required; "
            "the redundant pytorch_model.bin is intentionally not archived"
        )
    required_model_metadata = {
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "1_Pooling/config.json",
    }
    missing_model = sorted(required_model_metadata - set(model_files))
    if missing_model:
        raise FileNotFoundError(f"Granite model lacks required files: {missing_model}")
    return data_files, model_files


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def upload_file(
    s3: Any,
    local_path: Path,
    key: str,
    *,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    digest = sha256_file(local_path)
    size_bytes = local_path.stat().st_size
    upload_metadata = {"sha256": digest}
    if metadata:
        upload_metadata.update(metadata)
    print(
        f"  upload {local_path.name} ({size_bytes / 1e6:.1f} MB) "
        f"-> {key}",
        flush=True,
    )
    s3.upload_file(
        str(local_path),
        S3_BUCKET,
        key,
        ExtraArgs={
            "ServerSideEncryption": "AES256",
            "Metadata": upload_metadata,
        },
    )
    return {
        "s3_key": key,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def model_tree_manifest(
    model_files: dict[str, Path],
) -> tuple[dict[str, str], str]:
    hashes = {
        relative: sha256_file(path)
        for relative, path in sorted(model_files.items())
    }
    tree_digest = sha256_strings(
        f"{relative}:{digest}" for relative, digest in sorted(hashes.items())
    )
    return hashes, tree_digest


def build_model_archive(
    model_files: dict[str, Path],
) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        prefix="granite-longctx-",
        suffix=".tar.gz",
        delete=False,
    )
    temporary.close()
    archive_path = Path(temporary.name)
    try:
        print("  building Granite safetensors archive ...", flush=True)
        with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
            for relative, path in sorted(model_files.items()):
                archive.add(
                    path,
                    arcname=f"{MODEL_NAME}/{relative}",
                    recursive=False,
                )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def ensure_model_asset(
    s3: Any,
    model_files: dict[str, Path],
    model_hashes: dict[str, str],
    model_tree_sha256: str,
    *,
    force: bool,
) -> tuple[str, dict[str, Any]]:
    key = (
        f"{S3_PREFIX}/assets/models/"
        f"{MODEL_NAME}-{model_tree_sha256}.tar.gz"
    )
    if not force:
        try:
            head = s3.head_object(Bucket=S3_BUCKET, Key=key)
            remote_tree = head.get("Metadata", {}).get("modeltreesha256")
            if remote_tree == model_tree_sha256:
                print(
                    f"  reuse content-addressed Granite model asset {key}",
                    flush=True,
                )
                return key, {
                    "s3_key": key,
                    "model_tree_sha256": model_tree_sha256,
                    "model_file_sha256": model_hashes,
                    "size_bytes": int(head.get("ContentLength", 0)),
                    "reused": True,
                }
            raise RuntimeError(
                f"existing content-addressed model asset has mismatched metadata: {key}"
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise

    archive_path = build_model_archive(model_files)
    try:
        uploaded = upload_file(
            s3,
            archive_path,
            key,
            metadata={"modeltreesha256": model_tree_sha256},
        )
    finally:
        archive_path.unlink(missing_ok=True)
    uploaded.update(
        {
            "model_tree_sha256": model_tree_sha256,
            "model_file_sha256": model_hashes,
            "reused": False,
        }
    )
    return key, uploaded


def upload_run_inputs(
    s3: Any,
    run_id: str,
    data_files: dict[str, Path],
    model_files: dict[str, Path],
    *,
    force_model: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Upload data/code under a unique run prefix and one content-addressed model."""
    run_input_prefix = f"{S3_PREFIX}/runs/{run_id}/input"
    uploaded: dict[str, Any] = {}

    print("=== Uploading unique run inputs ===", flush=True)
    for relative, path in sorted(data_files.items()):
        key = f"{run_input_prefix}/{relative}"
        uploaded[relative] = upload_file(s3, path, key)

    print("=== Preparing content-addressed Granite model ===", flush=True)
    model_hashes, model_tree_sha256 = model_tree_manifest(model_files)
    model_key, model_manifest = ensure_model_asset(
        s3,
        model_files,
        model_hashes,
        model_tree_sha256,
        force=force_model,
    )
    # Give SageMaker a directory prefix containing exactly one archive. A
    # server-side copy avoids the ambiguous semantics of using one object key
    # itself as an S3Prefix and does not re-upload the 300 MB asset.
    run_model_key = f"{run_input_prefix}/models/{MODEL_NAME}.tar.gz"
    print(f"  copy content-addressed model asset -> {run_model_key}", flush=True)
    s3.copy_object(
        Bucket=S3_BUCKET,
        Key=run_model_key,
        CopySource={"Bucket": S3_BUCKET, "Key": model_key},
        ServerSideEncryption="AES256",
        MetadataDirective="COPY",
    )
    model_manifest = dict(model_manifest)
    model_manifest["asset_s3_key"] = model_key
    model_manifest["s3_key"] = run_model_key
    uploaded[f"models/{MODEL_NAME}.tar.gz"] = model_manifest

    channels = {
        "windows": s3_uri(
            S3_BUCKET,
            f"{run_input_prefix}/windows/",
        ),
        "vintage": s3_uri(
            S3_BUCKET,
            f"{run_input_prefix}/vintage/",
        ),
        "code": s3_uri(
            S3_BUCKET,
            f"{run_input_prefix}/code/",
        ),
        "models": s3_uri(
            S3_BUCKET,
            f"{run_input_prefix}/models/",
        ),
    }
    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "s3_prefix": S3_PREFIX,
        "input_channels_without_bucket": {
            name: uri.split("/", 3)[-1] for name, uri in channels.items()
        },
        "uploaded_inputs": uploaded,
        "model_tree_sha256": model_tree_sha256,
        "absolute_local_paths_recorded": False,
        "s3_bucket_recorded": False,
        "role_arn_recorded": False,
    }
    return channels, manifest


def input_channel(name: str, uri: str) -> dict[str, Any]:
    return {
        "ChannelName": name,
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        "ContentType": "application/octet-stream",
    }


def bootstrap_command() -> str:
    quoted_packages = " ".join(f"'{package}'" for package in PIP_PACKAGES)
    return (
        "set -euo pipefail; "
        "python3 -m pip install --quiet --disable-pip-version-check "
        f"--no-cache-dir {quoted_packages}; "
        "python3 /opt/ml/input/data/code/train_vintage_granite_longctx.py"
    )


def submit_job(
    sm: Any,
    run_id: str,
    channels: dict[str, str],
    training_image: str,
) -> dict[str, Any]:
    output_prefix = f"{S3_PREFIX}/runs/{run_id}/output"
    request = {
        "TrainingJobName": run_id,
        "AlgorithmSpecification": {
            "TrainingImage": training_image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "-lc"],
            "ContainerArguments": [bootstrap_command()],
        },
        "RoleArn": ROLE_ARN,
        "InputDataConfig": [
            input_channel("windows", channels["windows"]),
            input_channel("vintage", channels["vintage"]),
            input_channel("models", channels["models"]),
            input_channel("code", channels["code"]),
        ],
        "OutputDataConfig": {
            "S3OutputPath": s3_uri(S3_BUCKET, output_prefix),
        },
        "ResourceConfig": {
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": VOLUME_SIZE_GB,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": MAX_RUN_SECONDS,
        },
        "Environment": {
            "SAGEMAKER_S3_BUCKET": S3_BUCKET,
            "GRANITE_CHECKPOINT_PREFIX": CHECKPOINT_PREFIX,
            "GRANITE_RUN_ID": run_id,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "Tags": [
            {"Key": "project", "Value": "icaif2026-return-path-embeddings"},
            {"Key": "experiment", "Value": "granite-longctx-census"},
        ],
    }
    # No managed warm pool is configured. Training capacity is released at a
    # terminal job state.
    response = sm.create_training_job(**request)
    print(f"\nSubmitted SageMaker job: {run_id}", flush=True)
    print(
        f"Output prefix: {output_prefix} (bucket intentionally not printed)",
        flush=True,
    )
    return {
        "job_name": run_id,
        "submitted_at_utc": utc_now(),
        "instance_type": INSTANCE_TYPE,
        "instance_count": 1,
        "max_runtime_seconds": MAX_RUN_SECONDS,
        "training_image": training_image,
        "pip_packages": list(PIP_PACKAGES),
        "output_prefix_without_bucket": output_prefix,
        "response_http_status": response.get(
            "ResponseMetadata", {}
        ).get("HTTPStatusCode"),
    }


def status_line(response: dict[str, Any]) -> str:
    status = response["TrainingJobStatus"]
    secondary = response.get("SecondaryStatus", "")
    training_seconds = int(response.get("TrainingTimeInSeconds") or 0)
    billable_seconds = int(response.get("BillableTimeInSeconds") or 0)
    return (
        f"{status:10s} secondary={secondary:22s} "
        f"training={training_seconds:5d}s billable={billable_seconds:5d}s"
    )


def describe_and_print(sm: Any, job_name: str) -> dict[str, Any]:
    response = sm.describe_training_job(TrainingJobName=job_name)
    print(f"{job_name}: {status_line(response)}", flush=True)
    if response["TrainingJobStatus"] == "Failed":
        print(f"FailureReason: {response.get('FailureReason', '')}", flush=True)
    return response


def stop_and_confirm(
    sm: Any,
    job_name: str,
    *,
    poll_seconds: int = 15,
    timeout_seconds: int = 900,
) -> tuple[str, dict[str, Any]]:
    """Request a stop if needed, then prove the job is terminal."""
    response = sm.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]
    if status in TERMINAL_STATES:
        print(f"{job_name} is already terminal: {status}", flush=True)
        return status, response

    print(f"Sending stop_training_job for {job_name} ...", flush=True)
    sm.stop_training_job(TrainingJobName=job_name)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            response = sm.describe_training_job(TrainingJobName=job_name)
            status = response["TrainingJobStatus"]
            print(f"  stop confirmation: {status_line(response)}", flush=True)
            if status in TERMINAL_STATES:
                print(
                    f"Terminal state confirmed: {status}. "
                    "No training instance remains attached to this job.",
                    flush=True,
                )
                return status, response
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{job_name} did not reach a terminal state within "
                    f"{timeout_seconds}s after stop request"
                )
            time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print(
                "\nStop is already requested; continuing until SageMaker "
                "confirms termination.",
                flush=True,
            )


def wait_for_terminal(
    sm: Any,
    job_name: str,
    *,
    poll_seconds: int,
) -> tuple[str, dict[str, Any], bool]:
    """Monitor normally; Ctrl-C becomes stop-and-confirm, not an orphaned job."""
    print(
        f"\nMonitoring {job_name} every {poll_seconds}s. "
        "Ctrl-C stops and confirms termination.",
        flush=True,
    )
    consecutive_errors = 0
    try:
        while True:
            try:
                response = sm.describe_training_job(TrainingJobName=job_name)
                consecutive_errors = 0
            except (BotoCoreError, ClientError) as exc:
                consecutive_errors += 1
                print(
                    f"  transient describe error "
                    f"({consecutive_errors}/3): {type(exc).__name__}",
                    flush=True,
                )
                if consecutive_errors >= 3:
                    raise RuntimeError(
                        "lost SageMaker monitoring after three consecutive errors"
                    ) from exc
                time.sleep(min(poll_seconds, 30))
                continue

            status = response["TrainingJobStatus"]
            print(
                f"  [{time.strftime('%H:%M:%S')}] {status_line(response)}",
                flush=True,
            )
            if status in TERMINAL_STATES:
                # One final describe is deliberate terminal-state verification.
                verified = sm.describe_training_job(TrainingJobName=job_name)
                verified_status = verified["TrainingJobStatus"]
                if verified_status not in TERMINAL_STATES:
                    raise RuntimeError(
                        f"terminal-state verification changed to {verified_status}"
                    )
                return verified_status, verified, False
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nCtrl-C received; stopping the GPU job.", flush=True)
        status, response = stop_and_confirm(sm, job_name)
        return status, response, True


def parse_s3_uri(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"s3://([^/]+)/(.+)", uri)
    if not match:
        raise ValueError(f"invalid S3 URI: {uri}")
    return match.group(1), match.group(2)


def output_tar_uri(response: dict[str, Any], job_name: str) -> str:
    output_root = response["OutputDataConfig"]["S3OutputPath"].rstrip("/")
    return f"{output_root}/{job_name}/output/output.tar.gz"


def download_results(
    s3: Any,
    sm: Any,
    job_name: str,
) -> Path:
    response = sm.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]
    if status != "Completed":
        raise RuntimeError(
            f"cannot download normal census outputs: {job_name} is {status}, "
            "not Completed"
        )

    uri = output_tar_uri(response, job_name)
    bucket, key = parse_s3_uri(uri)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = OUT_DIR / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f"{job_name}-",
        suffix=".output.tar.gz",
        delete=False,
    ) as temporary:
        tar_path = Path(temporary.name)
    try:
        print(
            f"Downloading terminal job output for {job_name} "
            "(bucket intentionally not printed) ...",
            flush=True,
        )
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                s3.download_file(bucket, key, str(tar_path))
                last_error = None
                break
            except (BotoCoreError, ClientError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                print(
                    f"  output archive not ready ({attempt}/5); retrying ...",
                    flush=True,
                )
                time.sleep(10)
        if last_error is not None:
            raise last_error

        extracted: list[str] = []
        with tarfile.open(tar_path, "r:gz") as archive:
            for member in archive.getmembers():
                basename = Path(member.name).name
                if basename not in EXPECTED_OUTPUT_FILES or not member.isfile():
                    continue
                file_object = archive.extractfile(member)
                if file_object is None:
                    continue
                content = file_object.read()
                destination = job_dir / basename
                if destination.exists():
                    existing = destination.read_bytes()
                    if existing != content:
                        raise FileExistsError(
                            f"refusing to overwrite different existing result: "
                            f"{destination}"
                        )
                    print(f"  verified existing {basename}", flush=True)
                else:
                    destination.write_bytes(content)
                    print(f"  wrote {destination.relative_to(PROJECT_ROOT)}", flush=True)
                extracted.append(basename)
        if "granite_longctx_census.json" not in extracted:
            raise FileNotFoundError(
                f"census JSON absent from output archive; extracted {extracted}"
            )
    finally:
        tar_path.unlink(missing_ok=True)
    return job_dir


def print_summary(job_dir: Path) -> None:
    result_path = job_dir / "granite_longctx_census.json"
    if not result_path.is_file():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print("\n=== Granite long-context census ===", flush=True)
    for protocol, summary in result.get("summary", {}).items():
        print(
            f"{protocol:18s} "
            f"ReturnPath={summary['return_path_mean_across_transitions']:.6f} "
            f"Granite={summary['granite_longctx_mean_across_transitions']:.6f} "
            f"ReturnPath wins={summary['return_path_transition_wins']}/"
            f"{summary['n_transitions']}",
            flush=True,
        )


def write_launch_manifest(
    run_id: str,
    upload_manifest: dict[str, Any],
    job_manifest: dict[str, Any] | None = None,
) -> Path:
    manifest_dir = OUT_DIR / "jobs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    phase = "launch" if job_manifest is not None else "upload"
    path = manifest_dir / f"{run_id}.{phase}.json"
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "upload": upload_manifest,
        "job": job_manifest,
        "notes": (
            "Local manifest omits bucket name, role ARN, account identifiers, "
            "and absolute workstation paths."
        ),
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite launch manifest: {path}")
    path.write_text(content, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure-boto3 Granite long-context SageMaker census launcher"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--download",
        metavar="JOB_NAME",
        help="download a completed existing job; do not upload or submit",
    )
    action.add_argument(
        "--status",
        metavar="JOB_NAME",
        help="describe one existing job; do not upload or submit",
    )
    action.add_argument(
        "--stop",
        metavar="JOB_NAME",
        help="stop one existing job and wait for terminal confirmation",
    )
    waiting = parser.add_mutually_exclusive_group()
    waiting.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        help="monitor through terminal state and download (default)",
    )
    waiting.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="return immediately after submit; caller owns later monitoring",
    )
    parser.set_defaults(wait=True)
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="monitoring interval, 10--60 seconds (default: 30)",
    )
    parser.add_argument(
        "--force-model-upload",
        action="store_true",
        help="re-upload the content-addressed model archive",
    )
    parser.add_argument(
        "--training-image",
        default=TRAINING_IMAGE,
        help=(
            "regional SageMaker PyTorch 2.6 GPU training-image URI; "
            "defaults to SAGEMAKER_TRAINING_IMAGE"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 10 and 60")

    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    s3 = session.client("s3")

    if args.status:
        describe_and_print(sm, args.status)
        return 0
    if args.stop:
        stop_and_confirm(sm, args.stop)
        return 0
    if args.download:
        job_dir = download_results(s3, sm, args.download)
        print_summary(job_dir)
        return 0

    if not ROLE_ARN or not S3_BUCKET:
        parser.error(
            "set SAGEMAKER_ROLE_ARN and SAGEMAKER_S3_BUCKET; "
            "account-specific identifiers are intentionally not stored in the repository"
        )
    if not args.training_image:
        parser.error(
            "set SAGEMAKER_TRAINING_IMAGE or pass --training-image with the "
            "regional AWS Deep Learning Container URI"
        )

    data_files, model_files = validate_local_inputs()
    run_id = new_run_id()
    channels, upload_manifest = upload_run_inputs(
        s3,
        run_id,
        data_files,
        model_files,
        force_model=args.force_model_upload,
    )
    write_launch_manifest(run_id, upload_manifest)

    submitted = False
    try:
        job_manifest = submit_job(
            sm,
            run_id,
            channels,
            args.training_image,
        )
        submitted = True
        write_launch_manifest(run_id, upload_manifest, job_manifest)

        if not args.wait:
            print(
                "\nWARNING: --no-wait selected. The job has not been confirmed terminal.",
                flush=True,
            )
            print(
                f"Monitor: python3 scripts/launch_vintage_granite_longctx.py "
                f"--status {run_id}",
                flush=True,
            )
            print(
                f"Stop:    python3 scripts/launch_vintage_granite_longctx.py "
                f"--stop {run_id}",
                flush=True,
            )
            return 0

        status, response, interrupted = wait_for_terminal(
            sm,
            run_id,
            poll_seconds=args.poll_seconds,
        )
        print(f"\nTerminal state verified: {status}", flush=True)
        if interrupted:
            return 130
        if status == "Completed":
            job_dir = download_results(s3, sm, run_id)
            print_summary(job_dir)
            return 0
        print(f"Failure reason: {response.get('FailureReason', '')}", flush=True)
        return 1
    except BaseException:
        # Once submission may have succeeded, any unexpected launcher failure
        # must not strand the GPU. Query by the already-known deterministic name.
        if submitted:
            print(
                "\nLauncher error after submission; initiating emergency stop.",
                flush=True,
            )
            try:
                stop_and_confirm(sm, run_id)
            except Exception as stop_error:
                print(
                    f"EMERGENCY STOP CONFIRMATION FAILED: {stop_error}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            # create_training_job can be accepted server-side before a client
            # interruption. A describe call closes that narrow race safely.
            try:
                sm.describe_training_job(TrainingJobName=run_id)
            except (BotoCoreError, ClientError):
                pass
            else:
                print(
                    "\nJob exists despite submit interruption; stopping it.",
                    flush=True,
                )
                stop_and_confirm(sm, run_id)
        raise


if __name__ == "__main__":
    sys.exit(main())
