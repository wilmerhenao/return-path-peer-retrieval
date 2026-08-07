"""Synthetic tests for the staged v2 historical-data boundary."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from experiments.multimodal_peer_encoder import historical_data
from experiments.multimodal_peer_encoder.evaluation import row_l2_cosine
from experiments.multimodal_peer_encoder.v2_data import (
    SELECTION_ARTIFACT_SCHEMA,
    SELECTION_ARTIFACT_STATUS,
    assert_no_modern_path,
    guard_historical_path,
    load_feature_window_for_s1,
    load_v2_transition,
    selection_payload_sha256,
    sha256_path_map,
    validate_selection_artifact,
)


def _write_lock(path: Path, *, writable: bool = False) -> dict:
    def winner(architecture: str) -> dict:
        return {
            "architecture": architecture,
            "target_softness": "lw",
            "application": "global",
            "learning_rate": 0.003,
            "development_scores": {"H5->H6": 0.4, "H6->H7": 0.4},
            "equal_transition_mean_ndcg_at_5": 0.4,
            "advantage_over_frozen_blend": -0.02,
        }

    global_winner = winner("H-A")
    payload = {
        "schema_version": SELECTION_ARTIFACT_SCHEMA,
        "status": SELECTION_ARTIFACT_STATUS,
        "created_at_utc": "2026-08-02T00:00:00+00:00",
        "global_winner": global_winner,
        "family_winners": [global_winner, winner("H-B")],
        "development_results_sha256": "d" * 64,
        "development_input_sha256": {"fixture/input.npy": "e" * 64},
    }
    payload["selection_sha256"] = selection_payload_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o644 if writable else 0o444)
    return payload


def _transition_fixture(tmp_path: Path):
    historical_root = tmp_path / "historical"
    vintage_root = tmp_path / "vintage"
    cache_dir = tmp_path / "cache"
    (historical_root / "H1").mkdir(parents=True)
    (historical_root / "H2").mkdir(parents=True)
    vintage_root.mkdir()
    cache_dir.mkdir()

    feature_paths = np.array(
        [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    target_full = np.array(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    target_indices = np.array([2, 0, 3], dtype=np.int64)
    aligned_raw_target = row_l2_cosine(target_full[target_indices])
    shrinkage = 0.25
    lw_target = aligned_raw_target * (1.0 - shrinkage)
    np.fill_diagonal(lw_target, 1.0)

    feature_meta = historical_root / "H1" / "meta.json"
    target_meta = historical_root / "H2" / "meta.json"
    feature_meta.write_text(
        json.dumps({"window": "H1", "N": 3, "T": 4}), encoding="utf-8"
    )
    target_meta.write_text(
        json.dumps(
            {"window": "H2", "N": 4, "T": 4, "lw_shrinkage": shrinkage}
        ),
        encoding="utf-8",
    )
    np.save(historical_root / "H2" / "ts_features.npy", target_full)
    authority = historical_root / "authority.bin"
    authority.write_bytes(b"synthetic-authority")

    text_embeddings = np.zeros((3, 768), dtype=np.float32)
    text_embeddings[0, 0] = 1.0
    text_embeddings[1, 1] = 1.0
    text_embeddings[2, 2] = 1.0
    loaded = historical_data.HistoricalTransition(
        transition="H1->H2",
        feature_window="H1",
        target_window="H2",
        tickers=("A", "B", "C"),
        ts_features=feature_paths,
        text_embeddings=text_embeddings,
        corr_target=lw_target,
        texts=("alpha", "beta beta", "gamma"),
        filing_dates=("2019-01-01", "2019-01-02", "2019-01-03"),
        feature_indices=np.arange(3, dtype=np.int64),
        target_indices=target_indices,
        input_paths=(authority, feature_meta, target_meta),
    )
    return historical_root, vintage_root, cache_dir, loaded, aligned_raw_target


@pytest.mark.parametrize(
    "path",
    [
        "results/windows_v3",
        "results/windows_v3/W1/corr_target.npy",
        "/repo/results/windows_v3/W2/raw_resid.npy",
        "safe/../results/windows_v3/W3/ts_features.npy",
        r"RESULTS\WINDOWS_V3\W4\corr_target.npy",
    ],
)
def test_lexical_modern_guard_rejects_path_variants(path: str):
    with pytest.raises(PermissionError, match="historical-only"):
        assert_no_modern_path(path)


def test_historical_guard_rejects_a_lexically_safe_symlink(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "historical-looking"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(PermissionError, match="symlinked data path"):
        guard_historical_path(alias / "corr_target.npy")


def test_modern_guard_runs_before_any_filesystem_or_loader_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    opened = False
    loaded = False

    def forbidden_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("filesystem access preceded the lexical guard")

    def forbidden_loader(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("H loader should not be called")

    monkeypatch.setattr(os, "open", forbidden_open)
    with pytest.raises(PermissionError):
        validate_selection_artifact("results/windows_v3/W1/corr_target.npy")
    assert not opened

    monkeypatch.setattr(
        historical_data, "load_historical_transition", forbidden_loader
    )
    with pytest.raises(PermissionError):
        load_v2_transition(
            1,
            stage="development",
            historical_root=Path("results/windows_v3"),
            vintage_root=Path("safe-vintage"),
            cache_dir=Path("safe-cache"),
        )
    assert not loaded

    lock = tmp_path / "DEVELOPMENT_SELECTION_LOCK.json"
    _write_lock(lock)
    with pytest.raises(PermissionError):
        load_v2_transition(
            1,
            stage="gate",
            selection_artifact=lock,
            historical_root=Path("results/windows_v3"),
            vintage_root=Path("safe-vintage"),
            cache_dir=Path("safe-cache"),
        )
    assert not opened


def test_stage_rejections_happen_before_the_historical_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def forbidden_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("stage rejection was late")

    monkeypatch.setattr(
        historical_data, "load_historical_transition", forbidden_loader
    )
    with pytest.raises(PermissionError, match="only through H6"):
        load_v2_transition(7, stage="development")
    with pytest.raises(PermissionError, match="requires an immutable"):
        load_v2_transition(1, stage="gate")
    with pytest.raises(PermissionError, match="only through H7"):
        load_feature_window_for_s1(8)
    assert calls == 0


def test_selection_artifact_is_self_hashed_read_only_and_not_a_symlink(
    tmp_path: Path,
):
    lock = tmp_path / "DEVELOPMENT_SELECTION_LOCK.json"
    expected = _write_lock(lock)
    validated = validate_selection_artifact(lock)
    assert dict(validated) == expected
    with pytest.raises(TypeError):
        validated["status"] = "changed"

    lock.chmod(0o644)
    with pytest.raises(ValueError, match="write bits"):
        validate_selection_artifact(lock)

    lock.chmod(0o644)
    tampered = json.loads(lock.read_text(encoding="utf-8"))
    tampered["global_winner"]["architecture"] = "H-B"
    lock.write_text(json.dumps(tampered), encoding="utf-8")
    lock.chmod(0o444)
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_selection_artifact(lock)

    target = tmp_path / "real-lock.json"
    _write_lock(target)
    symlink = tmp_path / "symlink-lock.json"
    symlink.symlink_to(target)
    with pytest.raises(PermissionError, match="symlinked data path"):
        validate_selection_artifact(symlink)


def test_transition_loader_aligns_raw_target_and_verifies_cache_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    historical_root, vintage_root, cache_dir, loaded, expected_raw_target = (
        _transition_fixture(tmp_path)
    )
    calls: list[dict] = []

    def fake_loader(index, **kwargs):
        calls.append({"index": index, **kwargs})
        return loaded

    monkeypatch.setattr(historical_data, "load_historical_transition", fake_loader)
    transition = load_v2_transition(
        1,
        stage="development",
        historical_root=historical_root,
        vintage_root=vintage_root,
        cache_dir=cache_dir,
    )

    assert calls == [
        {
            "index": 1,
            "historical_root": historical_root,
            "vintage_root": vintage_root,
            "cache_dir": cache_dir,
            "verify_cache": True,
        }
    ]
    assert transition.transition == "H1->H2"
    assert transition.tickers == ("A", "B", "C")
    assert transition.window_length == 4
    assert transition.shrinkage == 0.25
    np.testing.assert_allclose(transition.raw_scores, row_l2_cosine(loaded.ts_features))
    np.testing.assert_allclose(
        transition.text_scores, row_l2_cosine(loaded.text_embeddings)
    )
    np.testing.assert_allclose(transition.raw_target, expected_raw_target)
    np.testing.assert_allclose(
        transition.log_lengths,
        np.log1p(np.array([5.0, 9.0, 5.0], dtype=np.float32)),
    )
    assert not transition.raw_scores.flags.writeable
    assert not transition.raw_target.flags.writeable
    assert not transition.log_lengths.flags.writeable
    with pytest.raises(TypeError):
        transition.input_sha256["new"] = "0" * 64

    raw_target_path = historical_root / "H2" / "ts_features.npy"
    assert raw_target_path in transition.input_paths
    assert transition.input_sha256 == sha256_path_map(transition.input_paths)


def test_gate_loader_requires_and_records_the_immutable_selection_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    historical_root, vintage_root, cache_dir, loaded, _ = _transition_fixture(tmp_path)
    lock = tmp_path / "DEVELOPMENT_SELECTION_LOCK.json"
    _write_lock(lock)
    monkeypatch.setattr(
        historical_data,
        "load_historical_transition",
        lambda *args, **kwargs: loaded,
    )

    transition = load_v2_transition(
        1,
        stage="gate",
        selection_artifact=lock,
        historical_root=historical_root,
        vintage_root=vintage_root,
        cache_dir=cache_dir,
    )
    assert lock in transition.input_paths
    assert any(Path(key) == lock for key in transition.input_sha256)


def test_transition_rejects_a_target_that_is_not_the_declared_lw_scaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    historical_root, vintage_root, cache_dir, loaded, _ = _transition_fixture(tmp_path)
    corrupted = np.array(loaded.corr_target, copy=True)
    corrupted[0, 1] += 0.1
    corrupted[1, 0] += 0.1
    bad = historical_data.HistoricalTransition(
        **{**loaded.__dict__, "corr_target": corrupted}
    )
    monkeypatch.setattr(
        historical_data,
        "load_historical_transition",
        lambda *args, **kwargs: bad,
    )
    with pytest.raises(ValueError, match="declared scaling"):
        load_v2_transition(
            1,
            stage="development",
            historical_root=historical_root,
            vintage_root=vintage_root,
            cache_dir=cache_dir,
        )


def test_feature_only_h7_loader_opens_no_target_and_requires_verified_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    historical_root = tmp_path / "historical"
    vintage_root = tmp_path / "vintage"
    cache_dir = tmp_path / "cache"
    feature_dir = historical_root / "H7"
    feature_dir.mkdir(parents=True)
    vintage_root.mkdir()
    cache_dir.mkdir()

    tickers_in_storage = ["C", "A", "B"]
    pl.DataFrame(
        {"ticker": tickers_in_storage, "company_idx": [0, 1, 2]}
    ).write_parquet(feature_dir / "universe.parquet")
    (feature_dir / "meta.json").write_text(
        json.dumps({"window": "H7", "N": 3, "T": 4}), encoding="utf-8"
    )
    stored_residuals = np.array(
        [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(feature_dir / "ts_features.npy", stored_residuals)
    vintage_path = vintage_root / "H7_descriptions.parquet"
    vintage_path.write_bytes(b"synthetic-vintage-authority")

    ticker_cache = cache_dir / "H7_eligible_tickers.json"
    embedding_cache = cache_dir / "H7_granite_512.npy"
    manifest = cache_dir / "cache_manifest.json"
    ticker_cache.write_text('["A", "B", "C"]', encoding="utf-8")
    embedding_cache.write_bytes(b"synthetic-embedding-authority")
    manifest.write_text("{}", encoding="utf-8")
    embeddings = np.zeros((3, 768), dtype=np.float32)
    embeddings[np.arange(3), np.arange(3)] = 1.0

    vintage = historical_data.EligibleVintageWindow(
        window="H7",
        start=date(2022, 1, 3),
        tickers=("A", "B", "C"),
        texts=("filing A", "filing BB", "filing CCC"),
        filing_dates=("2021-01-01", "2021-01-02", "2021-01-03"),
        input_paths=(feature_dir / "universe.parquet", feature_dir / "meta.json", vintage_path),
    )
    cache_calls: list[dict] = []

    monkeypatch.setattr(
        historical_data,
        "load_eligible_vintage_window",
        lambda *args, **kwargs: vintage,
    )

    def fake_cache_loader(window, selected_tickers, **kwargs):
        cache_calls.append(
            {"window": window, "tickers": selected_tickers, **kwargs}
        )
        return embeddings, (ticker_cache, embedding_cache, manifest)

    monkeypatch.setattr(historical_data, "load_cached_embeddings", fake_cache_loader)
    opened_npy: list[Path] = []
    original_np_load = np.load

    def audited_np_load(path, *args, **kwargs):
        opened_npy.append(Path(path))
        return original_np_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_np_load)
    feature = load_feature_window_for_s1(
        7,
        historical_root=historical_root,
        vintage_root=vintage_root,
        cache_dir=cache_dir,
    )

    assert feature.window == "H7"
    assert feature.tickers == ("A", "B", "C")
    assert feature.window_length == 4
    np.testing.assert_allclose(
        feature.raw_scores,
        row_l2_cosine(stored_residuals[[1, 2, 0]]),
    )
    assert opened_npy == [feature_dir / "ts_features.npy"]
    assert all("H8" not in path.as_posix() for path in feature.input_paths)
    assert len(cache_calls) == 1
    assert cache_calls[0]["window"] == "H7"
    assert cache_calls[0]["tickers"] == ("A", "B", "C")
    assert cache_calls[0]["verify_manifest"] is True
    expected_inputs = cache_calls[0]["expected_input_sha256"]
    assert expected_inputs == {
        "historical_windows/H7/universe.parquet": historical_data.sha256_file(
            feature_dir / "universe.parquet"
        ),
        "historical_windows/H7/meta.json": historical_data.sha256_file(
            feature_dir / "meta.json"
        ),
        "vintage_descriptions/H7_descriptions.parquet": historical_data.sha256_file(
            vintage_path
        ),
    }
    assert feature.input_sha256 == sha256_path_map(feature.input_paths)


def test_provenance_hash_guard_rejects_modern_path_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
):
    called = False

    def forbidden_hash(path, *args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("modern path reached the hashing function")

    monkeypatch.setattr(historical_data, "sha256_file", forbidden_hash)
    with pytest.raises(PermissionError):
        sha256_path_map(["results/windows_v3/W1/corr_target.npy"])
    assert not called


def test_late_historical_target_cannot_be_hashed_without_selection_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "historical" / "H8" / "ts_features.npy"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"late historical target")
    called = False
    original_hash = historical_data.sha256_file

    def audited_hash(path, *args, **kwargs):
        nonlocal called
        called = True
        return original_hash(path, *args, **kwargs)

    monkeypatch.setattr(historical_data, "sha256_file", audited_hash)
    with pytest.raises(PermissionError, match="requires the selection lock"):
        sha256_path_map([target])
    assert not called

    lock = tmp_path / "DEVELOPMENT_SELECTION_LOCK.json"
    _write_lock(lock)
    hashes = sha256_path_map([target], selection_artifact=lock)
    assert called
    assert list(hashes.values()) == [original_hash(target)]
