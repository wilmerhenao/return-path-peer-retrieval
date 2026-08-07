"""Staged, historical-only data boundary for the v2 multimodal heads.

The v2 experiment has two deliberately separate historical stages.  Before
development selection is sealed, callers may load targets only through
``H6->H7``.  The later expanding-fold stage requires a content-addressed,
read-only selection artifact before *any* transition is loaded.  Modern data
are outside both stages: :func:`assert_no_modern_path` is a lexical guard and
is called before this module performs a filesystem operation on a caller-
controlled path.

This module does not train a model, evaluate a retrieval metric, or create a
selection artifact.  It reuses the v1 H-only loader so that the Granite cache
continues to receive its full manifest, model, source, input, dtype, and
software verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import polars as pl

from experiments.multimodal_peer_encoder import historical_data
from experiments.multimodal_peer_encoder.evaluation import row_l2_cosine


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_ROOT = PROJECT_ROOT / "results" / "ea_historical" / "windows"
DEFAULT_VINTAGE_ROOT = PROJECT_ROOT / "results" / "vintage_descriptions"
DEFAULT_CACHE_DIR = EXPERIMENT_DIR / "artifacts" / "historical_text_cache"

DEVELOPMENT_TRANSITION_MAX = 6
S1_FEATURE_WINDOW_MAX = 7
SELECTION_ARTIFACT_SCHEMA = "multimodal-peer-v2-development-selection-v1"
SELECTION_ARTIFACT_STATUS = "LOCKED"
PROVENANCE_CHUNK_SIZE = 1024 * 1024
LW_SCALE_ATOL = 2e-6
LW_SCALE_RTOL = 2e-5

V2Stage = Literal["development", "gate"]
_H_WINDOW = re.compile(r"H([1-9]|1[0-3])\Z")


@dataclass(frozen=True)
class V2Transition:
    """Aligned feature scores and two historical training-target scales."""

    transition: str
    feature_window: str
    target_window: str
    tickers: tuple[str, ...]
    raw_scores: np.ndarray
    text_scores: np.ndarray
    lw_target: np.ndarray
    raw_target: np.ndarray
    texts: tuple[str, ...]
    log_lengths: np.ndarray
    window_length: int
    shrinkage: float
    input_paths: tuple[Path, ...]
    input_sha256: Mapping[str, str]


@dataclass(frozen=True)
class V2FeatureWindow:
    """Feature-only H-window view used by the initialization self-test."""

    window: str
    tickers: tuple[str, ...]
    raw_scores: np.ndarray
    text_scores: np.ndarray
    texts: tuple[str, ...]
    log_lengths: np.ndarray
    window_length: int
    input_paths: tuple[Path, ...]
    input_sha256: Mapping[str, str]


def _lexically_normalized_parts(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Normalize path segments without querying the filesystem.

    Backslashes are treated as separators as well.  This makes the guard
    conservative on POSIX and preserves its intended behavior if the package
    is inspected or tested on a platform with Windows-style paths.
    """

    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError("path must be text or a text-valued PathLike")
    stack: list[str] = []
    for component in raw.replace("\\", "/").split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if stack and stack[-1] != "..":
                stack.pop()
            else:
                stack.append(component)
            continue
        stack.append(component.casefold())
    return tuple(stack)


def assert_no_modern_path(path: str | os.PathLike[str]) -> None:
    """Reject a lexical descendant of ``results/windows_v3``.

    This function is intentionally pure string processing: it does not call
    ``stat``, ``resolve``, ``glob``, ``exists``, ``open``, or any equivalent
    filesystem operation.  Callers can therefore apply it before touching an
    adversarially supplied modern path, as required by S3.
    """

    parts = _lexically_normalized_parts(path)
    forbidden = ("results", "windows_v3")
    if any(parts[index : index + 2] == forbidden for index in range(len(parts) - 1)):
        raise PermissionError(
            "v2 is historical-only: every results/windows_v3 descendant is forbidden"
        )


def reject_symlink_components(path: str | os.PathLike[str]) -> None:
    """Reject existing symlink components without following them.

    Callers must run :func:`assert_no_modern_path` first.  This closes the case
    where a lexically historical-looking root redirects into the forbidden
    modern subtree.
    """

    absolute = Path(os.path.abspath(os.path.normpath(os.fspath(path))))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"symlinked data path is forbidden: {current}")


def guard_historical_path(path: str | os.PathLike[str]) -> None:
    """Apply the lexical modern lock, then the no-symlink lock."""

    assert_no_modern_path(path)
    reject_symlink_components(path)


def selection_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical selection JSON, excluding its self-hash field."""

    if not isinstance(payload, Mapping):
        raise TypeError("selection payload must be a mapping")
    canonical_payload = dict(payload)
    canonical_payload.pop("selection_sha256", None)
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_selection_artifact(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Validate and return a content-addressed, filesystem-read-only lock.

    The runner creates this file exclusively, computes ``selection_sha256``
    with :func:`selection_payload_sha256`, and removes all write bits.  Opening
    with ``O_NOFOLLOW`` where available prevents a symlink from standing in
    for the immutable artifact.
    """

    guard_historical_path(path)
    artifact = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(artifact, flags)
    except OSError as exc:
        raise ValueError(f"selection artifact cannot be opened safely: {artifact}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("selection artifact must be a regular file")
        if before.st_mode & 0o222:
            raise ValueError("selection artifact must have no filesystem write bits")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, PROVENANCE_CHUNK_SIZE)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_after != identity_before:
            raise ValueError("selection artifact changed while it was read")
    finally:
        os.close(descriptor)

    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selection artifact must contain UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("selection artifact must contain a JSON object")
    if payload.get("schema_version") != SELECTION_ARTIFACT_SCHEMA:
        raise ValueError("selection artifact schema mismatch")
    if payload.get("status") != SELECTION_ARTIFACT_STATUS:
        raise ValueError("selection artifact is not locked")
    required_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "global_winner",
        "family_winners",
        "development_results_sha256",
        "development_input_sha256",
        "selection_sha256",
    }
    if set(payload) != required_keys:
        raise ValueError("selection artifact fields differ from the sealed schema")
    if not isinstance(payload.get("created_at_utc"), str) or not payload["created_at_utc"]:
        raise ValueError("selection artifact lacks a creation timestamp")
    winner = payload.get("global_winner")
    if not isinstance(winner, dict) or not winner:
        raise ValueError("selection artifact lacks a nonempty global winner")
    families = payload.get("family_winners")
    if not isinstance(families, list) or len(families) != 2:
        raise ValueError("selection artifact must contain exactly two family winners")
    if any(not isinstance(value, dict) for value in families):
        raise ValueError("selection family winners must be JSON objects")
    winner_keys = {
        "architecture",
        "target_softness",
        "application",
        "learning_rate",
        "development_scores",
        "equal_transition_mean_ndcg_at_5",
        "advantage_over_frozen_blend",
    }
    for value in (winner, *families):
        if set(value) != winner_keys:
            raise ValueError("selection winner fields differ from the sealed schema")
        if value["architecture"] not in {"H-A", "H-B"}:
            raise ValueError("selection winner has an invalid architecture")
        if value["target_softness"] not in {"raw_scale", "lw"}:
            raise ValueError("selection winner has an invalid target softness")
        if value["application"] not in {"global", "rank_band"}:
            raise ValueError("selection winner has an invalid application")
        if float(value["learning_rate"]) not in {0.003, 0.03}:
            raise ValueError("selection winner has an invalid learning rate")
        scores = value["development_scores"]
        if not isinstance(scores, dict) or set(scores) != {"H5->H6", "H6->H7"}:
            raise ValueError("selection winner has invalid development scores")
        numeric = (
            *scores.values(),
            value["equal_transition_mean_ndcg_at_5"],
            value["advantage_over_frozen_blend"],
        )
        try:
            finite = np.asarray(numeric, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("selection winner contains a nonnumeric result") from exc
        if not np.isfinite(finite).all():
            raise ValueError("selection winner contains a nonfinite result")
    if {value["architecture"] for value in families} != {"H-A", "H-B"}:
        raise ValueError("selection family winners must cover H-A and H-B")
    matching_family = [
        value for value in families if value["architecture"] == winner["architecture"]
    ]
    if len(matching_family) != 1 or matching_family[0] != winner:
        raise ValueError("global winner must equal its locked family winner")
    development_digest = payload.get("development_results_sha256")
    if (
        not isinstance(development_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", development_digest) is None
    ):
        raise ValueError("selection artifact lacks the development-results SHA-256")
    development_inputs = payload.get("development_input_sha256")
    if not isinstance(development_inputs, dict) or not development_inputs:
        raise ValueError("selection artifact lacks development input identities")
    for input_path, input_digest in development_inputs.items():
        if not isinstance(input_path, str) or not input_path:
            raise ValueError("selection input identity has an invalid path")
        if (
            not isinstance(input_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", input_digest) is None
        ):
            raise ValueError("selection input identity has an invalid SHA-256")
    recorded_digest = payload.get("selection_sha256")
    if (
        not isinstance(recorded_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded_digest) is None
    ):
        raise ValueError("selection artifact lacks its canonical SHA-256")
    if recorded_digest != selection_payload_sha256(payload):
        raise ValueError("selection artifact self-hash mismatch")
    return MappingProxyType(payload)


def sha256_path_map(
    paths: Sequence[str | os.PathLike[str]],
    *,
    selection_artifact: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Return deterministic absolute-path SHA-256 provenance for input files.

    Hashing an H8--H13 residual path or correlation target is target access in
    its own right, so the same immutable development lock is required here as
    in :func:`load_v2_transition`.
    """

    normalized: dict[str, Path] = {}
    needs_gate_unlock = False
    for value in paths:
        guard_historical_path(value)
        needs_gate_unlock |= _is_late_historical_target_path(value)
        # ``abspath`` and ``normpath`` are lexical; unlike ``resolve`` they do
        # not inspect symlinks or require the path to exist.
        key = Path(os.path.abspath(os.path.normpath(os.fspath(value)))).as_posix()
        assert_no_modern_path(key)
        normalized[key] = Path(value)

    if selection_artifact is not None:
        guard_historical_path(selection_artifact)
    if needs_gate_unlock:
        if selection_artifact is None:
            raise PermissionError(
                "hashing H8--H13 residual/target files requires the selection lock"
            )
        validate_selection_artifact(selection_artifact)

    result: dict[str, str] = {}
    for key in sorted(normalized):
        path = normalized[key]
        guard_historical_path(path)
        result[key] = historical_data.sha256_file(path)
    return result


def _transition_number(index: int | str) -> int:
    if isinstance(index, int) and not isinstance(index, bool):
        number = index
    elif isinstance(index, str) and _H_WINDOW.fullmatch(index):
        number = int(index[1:])
    else:
        raise TypeError("transition index must be an integer or H-window label")
    if number < 1 or number > 12:
        raise ValueError("historical transition index must lie in [1, 12]")
    return number


def _is_late_historical_target_path(path: str | os.PathLike[str]) -> bool:
    parts = _lexically_normalized_parts(path)
    if not parts or parts[-1] not in {"ts_features.npy", "corr_target.npy"}:
        return False
    return any(
        component.startswith("h")
        and component[1:].isdigit()
        and 8 <= int(component[1:]) <= 13
        for component in parts[:-1]
    )


def _feature_number(index: int | str) -> int:
    number = _transition_number(index)
    if number > S1_FEATURE_WINDOW_MAX:
        raise PermissionError("S1 may open feature windows only through H7")
    return number


def _guard_roots(*paths: str | os.PathLike[str]) -> None:
    for path in paths:
        guard_historical_path(path)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    guard_historical_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _window_length(metadata: Mapping[str, Any], *, window: str, width: int) -> int:
    if metadata.get("window", window) != window:
        raise ValueError(f"{window}: metadata window label mismatch")
    value = metadata.get("T")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{window}: metadata T must be a positive integer")
    if value != width:
        raise ValueError(f"{window}: metadata T does not match residual-path width")
    return value


def _window_population(metadata: Mapping[str, Any], *, window: str, rows: int) -> int:
    value = metadata.get("N")
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise ValueError(f"{window}: metadata N must be an integer of at least two")
    if value != rows:
        raise ValueError(f"{window}: metadata N does not match stored row count")
    return value


def _target_shrinkage(metadata: Mapping[str, Any], *, window: str) -> float:
    if metadata.get("window", window) != window:
        raise ValueError(f"{window}: metadata window label mismatch")
    try:
        value = float(metadata["lw_shrinkage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{window}: missing or invalid LW shrinkage") from exc
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{window}: LW shrinkage must lie in [0, 1)")
    return value


def _readonly_float32(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    array.setflags(write=False)
    return array


def _log_lengths(texts: tuple[str, ...]) -> np.ndarray:
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("every aligned filing excerpt must be a nonempty string")
    lengths = np.fromiter((len(text) for text in texts), dtype=np.float64)
    return _readonly_float32(np.log1p(lengths), label="filing log lengths")


def _validate_tickers(tickers: tuple[str, ...]) -> None:
    if len(tickers) < 2:
        raise ValueError("a v2 universe must contain at least two securities")
    if tickers != tuple(sorted(set(tickers))):
        raise ValueError("v2 tickers must be sorted and unique")


def _readonly_scores(features: np.ndarray, *, label: str) -> np.ndarray:
    scores = row_l2_cosine(features)
    return _readonly_float32(scores, label=label)


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    unique: list[Path] = []
    for value in paths:
        guard_historical_path(value)
        key = os.path.abspath(os.path.normpath(os.fspath(value)))
        if key not in seen:
            seen.add(key)
            unique.append(Path(value))
    return tuple(unique)


def _provenance(
    paths: tuple[Path, ...],
    *,
    selection_artifact: Path | None = None,
) -> Mapping[str, str]:
    return MappingProxyType(
        sha256_path_map(paths, selection_artifact=selection_artifact)
    )


def load_v2_transition(
    index: int | str,
    *,
    stage: V2Stage,
    selection_artifact: Path | None = None,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    vintage_root: Path = DEFAULT_VINTAGE_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> V2Transition:
    """Load one aligned v2 transition under the declared stage boundary.

    ``development`` permits only indices 1--6 and rejects a selection
    artifact.  ``gate`` requires a valid immutable artifact even when loading
    an earlier transition for an expanding-fold training set.
    """

    number = _transition_number(index)
    historical_root = Path(historical_root)
    vintage_root = Path(vintage_root)
    cache_dir = Path(cache_dir)
    selection_path = (
        None if selection_artifact is None else Path(selection_artifact)
    )
    # Screen every caller-controlled path before opening anything, including
    # arguments that a later stage-validation error would otherwise ignore.
    _guard_roots(
        historical_root,
        vintage_root,
        cache_dir,
        *((selection_path,) if selection_path is not None else ()),
    )
    if stage not in {"development", "gate"}:
        raise ValueError("stage must be 'development' or 'gate'")
    if stage == "development":
        if selection_path is not None:
            raise ValueError("development must not receive a selection artifact")
        if number > DEVELOPMENT_TRANSITION_MAX:
            raise PermissionError(
                "development may load targets only through H6->H7"
            )
    else:
        if selection_path is None:
            raise PermissionError("gate loading requires an immutable selection artifact")
    if selection_path is not None:
        validate_selection_artifact(selection_path)

    feature_window = f"H{number}"
    target_window = f"H{number + 1}"
    _guard_roots(
        historical_root / feature_window / "universe.parquet",
        historical_root / feature_window / "meta.json",
        historical_root / feature_window / "ts_features.npy",
        historical_root / target_window / "universe.parquet",
        historical_root / target_window / "meta.json",
        historical_root / target_window / "corr_target.npy",
        historical_root / target_window / "ts_features.npy",
        vintage_root / f"{feature_window}_descriptions.parquet",
        cache_dir / f"{feature_window}_eligible_tickers.json",
        cache_dir / f"{feature_window}_granite_512.npy",
        cache_dir / "cache_manifest.json",
    )

    loaded = historical_data.load_historical_transition(
        number,
        historical_root=historical_root,
        vintage_root=vintage_root,
        cache_dir=cache_dir,
        verify_cache=True,
    )
    expected_label = f"H{number}->H{number + 1}"
    if loaded.transition != expected_label:
        raise ValueError("H-only loader returned the wrong transition")
    tickers = tuple(loaded.tickers)
    _validate_tickers(tickers)
    if loaded.text_embeddings is None:
        raise ValueError("v2 requires a verified Granite embedding cache")
    if loaded.ts_features.shape[0] != len(tickers):
        raise ValueError("feature residual paths do not match the aligned universe")
    if loaded.text_embeddings.shape != (
        len(tickers),
        historical_data.GRANITE_EMBEDDING_DIMENSION,
    ):
        raise ValueError("Granite embeddings do not match the aligned universe")
    if len(loaded.texts) != len(tickers):
        raise ValueError("filing excerpts do not match the aligned universe")

    feature_meta_path = historical_root / feature_window / "meta.json"
    target_meta_path = historical_root / target_window / "meta.json"
    raw_target_path = historical_root / target_window / "ts_features.npy"
    _guard_roots(feature_meta_path, target_meta_path, raw_target_path)
    feature_metadata = _read_json_object(feature_meta_path, label="feature metadata")
    target_metadata = _read_json_object(target_meta_path, label="target metadata")

    feature_values = np.asarray(loaded.ts_features, dtype=np.float32)
    window_length = _window_length(
        feature_metadata,
        window=feature_window,
        width=feature_values.shape[1],
    )
    shrinkage = _target_shrinkage(target_metadata, window=target_window)

    raw_target_values = np.load(raw_target_path, mmap_mode="r")
    if raw_target_values.ndim != 2:
        raise ValueError(f"{target_window}: target residual paths must be two-dimensional")
    _window_population(
        target_metadata,
        window=target_window,
        rows=raw_target_values.shape[0],
    )
    _window_length(
        target_metadata,
        window=target_window,
        width=raw_target_values.shape[1],
    )
    target_indices = np.asarray(loaded.target_indices, dtype=np.int64)
    if target_indices.shape != (len(tickers),):
        raise ValueError("target row map does not match the aligned universe")
    if np.any(target_indices < 0) or np.any(target_indices >= raw_target_values.shape[0]):
        raise ValueError("target row map lies outside the target residual paths")
    aligned_target_paths = np.asarray(raw_target_values[target_indices], dtype=np.float32)

    raw_scores = _readonly_scores(feature_values, label="return scores")
    text_scores = _readonly_scores(loaded.text_embeddings, label="text scores")
    raw_target = _readonly_scores(aligned_target_paths, label="raw target")
    lw_target = _readonly_float32(loaded.corr_target, label="LW target")
    expected_shape = (len(tickers), len(tickers))
    if lw_target.shape != expected_shape:
        raise ValueError("LW target does not match the aligned universe")
    if not np.allclose(lw_target, lw_target.T, atol=2e-5, rtol=2e-5):
        raise ValueError("LW target must be symmetric")
    off_diagonal = ~np.eye(len(tickers), dtype=bool)
    if not np.allclose(
        lw_target[off_diagonal],
        (1.0 - shrinkage) * raw_target[off_diagonal],
        atol=LW_SCALE_ATOL,
        rtol=LW_SCALE_RTOL,
    ):
        raise ValueError("LW target is not the declared scaling of the raw target")
    if not np.allclose(np.diag(lw_target), 1.0, atol=2e-6, rtol=0.0):
        raise ValueError("LW target diagonal must equal one")

    texts = tuple(loaded.texts)
    input_paths = _unique_paths(
        (
            *loaded.input_paths,
            raw_target_path,
            *((selection_path,) if selection_path is not None else ()),
        )
    )
    return V2Transition(
        transition=expected_label,
        feature_window=feature_window,
        target_window=target_window,
        tickers=tickers,
        raw_scores=raw_scores,
        text_scores=text_scores,
        lw_target=lw_target,
        raw_target=raw_target,
        texts=texts,
        log_lengths=_log_lengths(texts),
        window_length=window_length,
        shrinkage=shrinkage,
        input_paths=input_paths,
        input_sha256=_provenance(
            input_paths,
            selection_artifact=selection_path,
        ),
    )


def load_feature_window_for_s1(
    index: int | str,
    *,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    vintage_root: Path = DEFAULT_VINTAGE_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> V2FeatureWindow:
    """Load H1--H7 features and vintage text without any target-window path."""

    number = _feature_number(index)
    window = f"H{number}"
    historical_root = Path(historical_root)
    vintage_root = Path(vintage_root)
    cache_dir = Path(cache_dir)
    _guard_roots(historical_root, vintage_root, cache_dir)

    feature_dir = historical_root / window
    universe_path = feature_dir / "universe.parquet"
    meta_path = feature_dir / "meta.json"
    residual_path = feature_dir / "ts_features.npy"
    vintage_path = vintage_root / f"{window}_descriptions.parquet"
    _guard_roots(universe_path, meta_path, residual_path, vintage_path)

    vintage = historical_data.load_eligible_vintage_window(
        window,
        historical_root=historical_root,
        vintage_root=vintage_root,
    )
    tickers = tuple(vintage.tickers)
    _validate_tickers(tickers)
    if len(vintage.texts) != len(tickers):
        raise ValueError("feature filing excerpts do not match the eligible universe")

    universe = pl.read_parquet(universe_path)
    if not {"ticker", "company_idx"} <= set(universe.columns):
        raise ValueError(f"{window}: universe lacks ticker/company_idx")
    raw_tickers = universe.get_column("ticker").to_list()
    if any(ticker is None for ticker in raw_tickers):
        raise ValueError(f"{window}: universe contains a null ticker")
    universe_tickers = tuple(str(ticker) for ticker in raw_tickers)
    if len(universe_tickers) != len(set(universe_tickers)):
        raise ValueError(f"{window}: universe contains duplicate tickers")
    company_indices = universe.get_column("company_idx").to_numpy()
    if not np.array_equal(company_indices, np.arange(len(universe_tickers))):
        raise ValueError(f"{window}: company_idx does not match row order")
    position = {ticker: row for row, ticker in enumerate(universe_tickers)}
    if not set(tickers) <= set(position):
        raise ValueError(f"{window}: eligible filing ticker is absent from universe")
    indices = np.fromiter((position[ticker] for ticker in tickers), dtype=np.int64)

    residual_values = np.load(residual_path, mmap_mode="r")
    if residual_values.ndim != 2 or residual_values.shape[0] != len(universe_tickers):
        raise ValueError(f"{window}: residual paths do not match the universe")
    aligned_residuals = np.asarray(residual_values[indices], dtype=np.float32)
    metadata = _read_json_object(meta_path, label="feature metadata")
    _window_population(metadata, window=window, rows=residual_values.shape[0])
    window_length = _window_length(
        metadata,
        window=window,
        width=aligned_residuals.shape[1],
    )

    expected_cache_inputs = {
        f"historical_windows/{window}/universe.parquet": historical_data.sha256_file(
            universe_path
        ),
        f"historical_windows/{window}/meta.json": historical_data.sha256_file(meta_path),
        f"vintage_descriptions/{window}_descriptions.parquet": historical_data.sha256_file(
            vintage_path
        ),
    }
    _guard_roots(
        cache_dir / f"{window}_eligible_tickers.json",
        cache_dir / f"{window}_granite_512.npy",
        cache_dir / "cache_manifest.json",
    )
    text_embeddings, cache_paths = historical_data.load_cached_embeddings(
        window,
        tickers,
        cache_dir=cache_dir,
        verify_manifest=True,
        expected_input_sha256=expected_cache_inputs,
    )
    if text_embeddings.shape != (
        len(tickers),
        historical_data.GRANITE_EMBEDDING_DIMENSION,
    ):
        raise ValueError("Granite embeddings do not match the feature-only universe")

    texts = tuple(vintage.texts)
    input_paths = _unique_paths(
        (
            residual_path,
            universe_path,
            meta_path,
            vintage_path,
            *vintage.input_paths,
            *cache_paths,
        )
    )
    return V2FeatureWindow(
        window=window,
        tickers=tickers,
        raw_scores=_readonly_scores(aligned_residuals, label="return scores"),
        text_scores=_readonly_scores(text_embeddings, label="text scores"),
        texts=texts,
        log_lengths=_log_lengths(texts),
        window_length=window_length,
        input_paths=input_paths,
        input_sha256=_provenance(input_paths),
    )
