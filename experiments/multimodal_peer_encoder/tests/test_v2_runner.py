"""Synthetic protocol and selection tests for the v2 orchestrator."""

from __future__ import annotations

import json

import pytest

from experiments.multimodal_peer_encoder import run_v2_historical as runner
from experiments.multimodal_peer_encoder.run_v2_historical import (
    DEFAULT_CONFIG,
    DEVELOPMENT_FIT_SCHEMA,
    GATE_FIT_SCHEMA,
    GIT_CLEAN_PATHS,
    PROJECT_ROOT,
    canonical_payload_sha256,
    load_completed_development_fit,
    load_completed_gate_fit,
    load_config,
    open_finalization_state,
    open_or_resume_run,
    select_development_winners,
    source_paths,
    validate_manifest,
    write_completed_fit_artifact,
    write_locked_selection,
)
from experiments.multimodal_peer_encoder.v2_data import validate_selection_artifact
from experiments.multimodal_peer_encoder.v2_training import FitSpec


def _fit_rows():
    rows = []
    for architecture in ("H-A", "H-B"):
        for softness in ("raw_scale", "lw"):
            for application in ("global", "rank_band"):
                for learning_rate in (0.003, 0.03):
                    bonus = 0.01 if architecture == "H-B" else 0.0
                    for seed in (11, 29, 47):
                        rows.append(
                            {
                                "architecture": architecture,
                                "target_softness": softness,
                                "application": application,
                                "learning_rate": learning_rate,
                                "seed": seed,
                                "development_scores": {
                                    "H5->H6": 0.4 + bonus,
                                    "H6->H7": 0.42 + bonus,
                                },
                            }
                        )
    return rows


def test_grid_selects_global_and_family_winners_before_gate():
    config = load_config()
    global_winner, family_winners, summaries = select_development_winners(
        _fit_rows(),
        {"H5->H6": 0.4, "H6->H7": 0.4},
        config,
    )
    assert len(summaries) == 16
    assert global_winner["architecture"] == "H-B"
    # All within-family scores tie, so every other preference is binding.
    assert global_winner["application"] == "rank_band"
    assert global_winner["target_softness"] == "lw"
    assert global_winner["learning_rate"] == 0.003
    assert [row["architecture"] for row in family_winners] == ["H-A", "H-B"]


def test_selection_artifact_is_exclusive_hashed_and_read_only(tmp_path):
    config = load_config()
    global_winner, family_winners, _ = select_development_winners(
        _fit_rows(),
        {"H5->H6": 0.4, "H6->H7": 0.4},
        config,
    )
    path = tmp_path / "development_selection.json"
    write_locked_selection(
        path,
        global_winner=global_winner,
        family_winners=family_winners,
        development_sha256="d" * 64,
        development_input_sha256={"fixture/input.npy": "e" * 64},
    )
    payload = validate_selection_artifact(path)
    assert payload["global_winner"]["architecture"] == "H-B"
    assert json.loads(path.read_text())["selection_sha256"] == payload["selection_sha256"]
    assert path.stat().st_mode & 0o222 == 0


def test_manifest_self_hash_excludes_itself_and_detects_tampering(tmp_path):
    path = tmp_path / "RUN_MANIFEST.json"
    payload = {"schema_version": "fixture", "decision": "STOP", "count": 66}
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_manifest(path)["decision"] == "STOP"

    payload["decision"] = "PASS"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_manifest(path)
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("tampered manifest unexpectedly validated")


def test_config_hash_binds_every_fixed_protocol_field(tmp_path):
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload["model"]["rank_band_last"] = 49
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_config(altered)
    except ValueError as exc:
        assert "sealed fixed configuration" in str(exc)
    else:
        raise AssertionError("altered configuration unexpectedly validated")


def test_git_cleanliness_pathspec_excludes_the_forbidden_modern_tree():
    assert ":(exclude)results/windows_v3" in GIT_CLEAN_PATHS
    assert ":(exclude)results/windows_v3/**" in GIT_CLEAN_PATHS


def test_source_manifest_covers_environment_and_cache_reproduction_inputs():
    logical_paths = {path.relative_to(PROJECT_ROOT).as_posix() for path in source_paths()}
    assert "experiments/multimodal_peer_encoder/README.md" in logical_paths
    assert (
        "experiments/multimodal_peer_encoder/build_historical_embeddings.py"
        in logical_paths
    )
    assert "environment/core.lock" in logical_paths
    assert "environment/models.lock" in logical_paths
    assert ".gitignore" in logical_paths
    assert "scripts/__init__.py" in logical_paths
    assert "scripts/build_anonymous_release.py" in logical_paths
    assert "tests/test_release_builder.py" in logical_paths


def test_run_start_resumes_only_the_exact_seal_config_and_environment(tmp_path):
    run_dir = tmp_path / "canonical-run"
    arguments = {
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "environment": {"python": "fixture", "torch_num_threads": 8},
    }
    created, resumed = open_or_resume_run(run_dir, **arguments)
    assert resumed is False
    loaded, resumed = open_or_resume_run(run_dir, **arguments)
    assert resumed is True
    assert loaded == created

    with pytest.raises(RuntimeError, match="config_sha256"):
        open_or_resume_run(run_dir, **{**arguments, "config_sha256": "c" * 64})
    missing_start = tmp_path / "missing-start"
    missing_start.mkdir()
    with pytest.raises(RuntimeError, match="RUN_START"):
        open_or_resume_run(missing_start, **arguments)


def _development_artifact_payload(spec, *, input_sha256):
    fit_id = "dev-ha-raw_scale-global-lr0p003-seed11"
    losses = [0.5 - index / 1000 for index in range(40)]
    row = {
        "architecture": spec.architecture,
        "target_softness": spec.target_softness,
        "application": spec.application,
        "learning_rate": spec.learning_rate,
        "seed": spec.seed,
        "development_scores": {"H5->H6": 0.4, "H6->H7": 0.42},
        "equal_transition_mean_ndcg_at_5": 0.41,
        "advantage_over_frozen_blend": 0.01,
        "final_epoch_loss": losses[-1],
        "epoch_mean_losses": losses,
        "wall_seconds": 2.0,
        "peak_rss_mb": 100.0,
        "trainable_parameters": 114,
        "scaler": {"mean": 8.0, "population_std": 1.0, "n_records": 20},
    }
    return fit_id, {
        "fit_id": fit_id,
        "specification": {
            "architecture": spec.architecture,
            "target_softness": spec.target_softness,
            "application": spec.application,
            "learning_rate": spec.learning_rate,
            "seed": spec.seed,
        },
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "input_sha256": input_sha256,
        "result": row,
        "scoring_calls": [
            {
                "transition": label,
                "method": f"development/{fit_id}",
                "wall_time_seconds": 0.1,
            }
            for label in ("H5->H6", "H6->H7")
        ],
    }


def test_complete_development_fit_is_validated_for_skip_and_never_rewritten(tmp_path):
    spec = FitSpec("H-A", "raw_scale", "global", 0.003, 11)
    inputs = {"fixture.npy": "d" * 64}
    fit_id, payload = _development_artifact_payload(spec, input_sha256=inputs)
    path = tmp_path / f"{fit_id}.json"
    write_completed_fit_artifact(
        path, schema=DEVELOPMENT_FIT_SCHEMA, payload=payload
    )
    original = path.read_bytes()
    loaded = load_completed_development_fit(
        path,
        fit_id=fit_id,
        spec=spec,
        code_commit="a" * 40,
        config_sha256="b" * 64,
        input_sha256=inputs,
        transitions=("H5->H6", "H6->H7"),
        baseline_by_transition={"H5->H6": 0.4, "H6->H7": 0.4},
        expected_epochs=40,
    )
    assert loaded["result"]["development_scores"]["H6->H7"] == 0.42
    assert path.read_bytes() == original

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text('{"status":"INCOMPLETE"}\n', encoding="utf-8")
    assert load_completed_development_fit(
        incomplete,
        fit_id=fit_id,
        spec=spec,
        code_commit="a" * 40,
        config_sha256="b" * 64,
        input_sha256=inputs,
        transitions=("H5->H6", "H6->H7"),
        baseline_by_transition={"H5->H6": 0.4, "H6->H7": 0.4},
        expected_epochs=40,
    ) is None
    assert not incomplete.exists()
    assert list(tmp_path.glob(".incomplete.json.incomplete-*"))


def _retrieval_row(transition):
    return {
        "transition": transition,
        "method": "H-B",
        "top5_overlap_return_only": 0.5,
        "top5_overlap_frozen_blend": 0.6,
        "fraction_queries_different_from_return": 0.4,
        "promoted_raw_rank": {
            "p10": 6.0,
            "median": 10.0,
            "p90": 20.0,
            "fraction_6_10": 0.5,
            "fraction_11_20": 0.5,
            "fraction_21_50": 0.0,
            "fraction_above_50": 0.0,
        },
        "application_proxy": {
            "mean_signed_query_to_peer_correlation": 0.1,
            "fraction_queries_positive": 0.6,
        },
    }


def test_gate_artifact_reconstructs_every_final_contribution_and_rejects_tampering(
    tmp_path,
):
    spec = FitSpec("H-B", "lw", "rank_band", 0.003, 11)
    fit_id = "gate-hb-lw-rank_band-lr0p003-fold1-seed11"
    transitions = ("H7->H8", "H8->H9")
    inputs = {"fixture.npy": "d" * 64}
    losses = [0.4 - index / 1000 for index in range(40)]
    seed_rows = [
        {
            "architecture": "H-B",
            "fold": 1,
            "seed": 11,
            "transition": label,
            "ndcg_at_5": 0.4,
            "semantic_ablation_ndcg_at_5": 0.39,
            "semantic_ablation_ndcg_change": 0.01,
        }
        for label in transitions
    ]
    payload = {
        "fit_id": fit_id,
        "fold": 1,
        "specification": {
            "architecture": "H-B",
            "target_softness": "lw",
            "application": "rank_band",
            "learning_rate": 0.003,
            "seed": 11,
        },
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "input_sha256": inputs,
        "selection_sha256": "e" * 64,
        "history": {
            "fit_id": fit_id,
            "architecture": "H-B",
            "fold": 1,
            "seed": 11,
            "specification": {
                "target_softness": "lw",
                "application": "rank_band",
                "learning_rate": 0.003,
            },
            "validation_scores": {label: 0.4 for label in transitions},
            "epoch_mean_losses": losses,
            "final_epoch_loss": losses[-1],
            "wall_seconds": 3.0,
            "peak_rss_mb": 110.0,
            "trainable_parameters": 4,
        },
        "contributions": {
            "gate_seed_rows": seed_rows,
            "interpretability_row": {
                "fold": 1,
                "seed": 11,
                "target_softness": "lw",
                "application": "rank_band",
                "learning_rate": 0.003,
            },
            "h_b_transition_gamma": [
                {
                    "transition": label,
                    "seed": 11,
                    "window_length": 40,
                    "mean_gamma": 0.1,
                }
                for label in transitions
            ],
            "retrieval_seed_rows": [_retrieval_row(label) for label in transitions],
            "scoring_calls": [
                {
                    "transition": label,
                    "method": method,
                    "wall_time_seconds": 0.1,
                }
                for label in transitions
                for method in ("H-B/seed11", "H-B/seed11/semantic-ablation")
            ],
            "context_by_transition": {
                label: {
                    "transition": label,
                    "filing_length": {"p10": 1, "p50": 2, "p90": 3},
                    "window_length": 40,
                    "training_scaler_mean": 8.0,
                    "training_scaler_population_sd": 1.0,
                }
                for label in transitions
            },
        },
    }
    path = tmp_path / f"{fit_id}.json"
    write_completed_fit_artifact(path, schema=GATE_FIT_SCHEMA, payload=payload)
    arguments = {
        "fit_id": fit_id,
        "spec": spec,
        "fold": 1,
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "input_sha256": inputs,
        "selection_sha256": "e" * 64,
        "transitions": transitions,
        "expected_epochs": 40,
    }
    loaded = load_completed_gate_fit(path, **arguments)
    assert len(loaded["contributions"]["scoring_calls"]) == 4
    assert len(loaded["contributions"]["retrieval_seed_rows"]) == 2

    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["contributions"]["scoring_calls"].pop()
    altered["canonical_payload_sha256"] = canonical_payload_sha256(altered)
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scoring-call roster"):
        load_completed_gate_fit(path, **arguments)


def test_finalization_state_freezes_time_and_runtime_across_resume(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    arguments = {
        "decision": "STOP",
        "artifact_sha256": {"fit.json": "a" * 64},
        "fit_roster": {"development_fit_ids": ["dev-1"], "gate_fit_ids": ["gate-1"]},
    }
    first = open_finalization_state(run_dir, elapsed_seconds=12.5, **arguments)
    second = open_finalization_state(run_dir, elapsed_seconds=999.0, **arguments)
    assert second["completed_at_utc"] == first["completed_at_utc"]
    assert second["run_wall_time_seconds"] == 12.5


def test_receipted_resume_rehashes_every_prefinal_artifact(tmp_path, monkeypatch):
    run_dir = tmp_path / "fixture-run"
    development = run_dir / "training_history" / "development"
    gate = run_dir / "training_history" / "gate"
    development.mkdir(parents=True)
    gate.mkdir(parents=True)
    for index in range(48):
        (development / f"dev-{index:02}.json").write_text(
            f"development {index}\n", encoding="utf-8"
        )
    for index in range(18):
        (gate / f"gate-{index:02}.json").write_text(
            f"gate {index}\n", encoding="utf-8"
        )
    for name in (
        "RUN_START.json",
        "SELF_TESTS.log",
        "self_tests.json",
        "development_results.json",
        "development_selection.json",
    ):
        (run_dir / name).write_text(f"fixture {name}\n", encoding="utf-8")

    prefinal, fit_artifact_roster = runner.prefinal_artifact_hashes(run_dir)
    runner.write_completed_fit_artifact(
        run_dir / "FINALIZATION_START.json",
        schema=runner.FINALIZATION_SCHEMA,
        payload={
            "run_id": run_dir.name,
            "decision": "STOP",
            "prefinal_artifact_sha256": prefinal,
            "fit_artifact_roster": fit_artifact_roster,
            "completed_at_utc": "fixture-completed",
            "run_wall_time_seconds": 1.0,
        },
    )
    self_tests = [
        {"name": f"S{index}", "status": "PASS", "detail": "fixture"}
        for index in range(1, 5)
    ]
    fit_roster = {
        "development_expected": 48,
        "development_completed": 48,
        "gate_expected": 18,
        "gate_completed": 18,
        "total_expected": 66,
        "total_completed": 66,
        "transition_level_optimizer_updates": 13_440,
    }
    manifest = {
        "schema_version": "multimodal-peer-v2-run-manifest-v1",
        "run_id": run_dir.name,
        "decision": "STOP",
        "code_commit": "a" * 40,
        "configuration_sha256": "b" * 64,
        "git_status_short": "clean source; declared run artifacts only",
        "source_sha256": {"runner.py": "c" * 64},
        "input_sha256": {"historical.npy": "d" * 64},
        "environment": {"python": "fixture"},
        "self_tests": self_tests,
        "fit_roster": fit_roster,
        "fit_artifact_roster": fit_artifact_roster,
        "prefinal_artifact_sha256": prefinal,
        "finalization_state_sha256": runner.sha256_file(
            run_dir / "FINALIZATION_START.json"
        ),
    }
    manifest["canonical_payload_sha256"] = runner.canonical_payload_sha256(manifest)
    (run_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    result = {
        "schema_version": runner.RESULT_SCHEMA,
        "run": {"run_id": run_dir.name, "status": "STOP"},
        "provenance": {
            "peeled_commit": manifest["code_commit"],
            "configuration_sha256": manifest["configuration_sha256"],
            "manifest_sha256": runner.sha256_file(run_dir / "RUN_MANIFEST.json"),
            "source_sha256": manifest["source_sha256"],
            "input_sha256": manifest["input_sha256"],
            "environment": manifest["environment"],
            "git_status_short": manifest["git_status_short"],
        },
        "self_tests": self_tests,
        "fit_roster": fit_roster,
    }
    (run_dir / "historical_results.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    expected_report = "fixture report\n"
    (run_dir / "RUN_REPORT.md").write_text(expected_report, encoding="utf-8")
    (run_dir / "STOP").write_text("STOP\n", encoding="utf-8")
    validated = []
    monkeypatch.setattr(runner, "validate_result_artifact", validated.append)
    monkeypatch.setattr(runner, "render_report", lambda artifact: expected_report)
    runner.write_final_receipt(run_dir, "STOP")

    loaded = runner.load_receipted_result(run_dir)
    assert loaded == result
    assert validated == [result]

    changed_fit = development / "dev-00.json"
    changed_fit.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prefinal artifact moved"):
        runner.load_receipted_result(run_dir)
