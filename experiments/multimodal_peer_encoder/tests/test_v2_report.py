"""S4 provenance tests for the v2 JSON-bound report generator."""

from __future__ import annotations

import copy
import inspect
import itertools
import json
import re
from pathlib import Path

import pytest

from experiments.multimodal_peer_encoder import report_v2


def _distribution(next_value):
    return {
        "mean": next_value(),
        "standard_deviation": next_value(),
        "p10": next_value(),
        "p50": next_value(),
        "p90": next_value(),
    }


def _fixture():
    counter = 0

    def next_value():
        nonlocal counter
        counter += 1
        return 0.2 + counter / 10_000.0

    def mean(rows, key):
        return sum(row[key] for row in rows) / len(rows)

    development_fits = []
    for architecture, target, application, learning_rate, seed in itertools.product(
        ("H-A", "H-B"),
        ("raw_scale", "lw"),
        ("global", "rank_band"),
        (0.003, 0.03),
        (11, 29, 47),
    ):
        h5_score = next_value()
        h6_score = next_value()
        development_fits.append(
            {
                "architecture": architecture,
                "target_softness": target,
                "application": application,
                "learning_rate": learning_rate,
                "seed": seed,
                "development_scores": {
                    "H5->H6": h5_score,
                    "H6->H7": h6_score,
                },
                "equal_transition_mean_ndcg_at_5": (h5_score + h6_score) / 2,
                "advantage_over_frozen_blend": next_value(),
                "final_epoch_loss": next_value(),
            }
        )

    h_a_spec = ("H-A", "raw_scale", "rank_band", 0.03)
    h_b_spec = ("H-B", "lw", "rank_band", 0.003)

    def winner(spec):
        return {
            "architecture": spec[0],
            "target_softness": spec[1],
            "application": spec[2],
            "learning_rate": spec[3],
            "equal_transition_mean_ndcg_at_5": next_value(),
            "advantage_over_frozen_blend": next_value(),
        }

    transitions = [f"H{index}->H{index + 1}" for index in range(7, 13)]
    advantages = [-0.001, 0.002, 0.004, 0.006, 0.007, 0.008]
    slots = []
    for position, (transition, advantage) in enumerate(
        zip(transitions, advantages, strict=True)
    ):
        blend = 0.41 - position / 1000.0
        global_score = blend + advantage
        slots.append(
            {
                "transition": transition,
                "n": 1416 + position * 17,
                "return_only_ndcg_at_5": blend - 0.02,
                "tfidf_only_ndcg_at_5": blend - 0.04,
                "frozen_blend_ndcg_at_5": blend,
                "h_a_seed_mean_ndcg_at_5": blend - 0.01,
                "h_a_seed_sd_ndcg_at_5": 0.001 + position / 10000.0,
                "h_b_seed_mean_ndcg_at_5": global_score,
                "h_b_seed_sd_ndcg_at_5": 0.002 + position / 10000.0,
                "global_winner_seed_mean_ndcg_at_5": global_score,
                "global_winner_seed_sd_ndcg_at_5": 0.002 + position / 10000.0,
                "winner_minus_frozen_blend": global_score - blend,
            }
        )

    h_a_models = []
    h_b_models = []
    for fold, seed in itertools.product((1, 2, 3), (11, 29, 47)):
        h_a_models.append(
            {
                "fold": fold,
                "seed": seed,
                "target_softness": h_a_spec[1],
                "application": h_a_spec[2],
                "learning_rate": h_a_spec[3],
                "lambda": next_value(),
                "active_pair_correction": _distribution(next_value),
                "max_abs_effective_correction": next_value(),
                "positive_correction_fraction": next_value(),
                "score_derivative_cos_text": next_value(),
                "semantic_ablation_ndcg_change": next_value(),
            }
        )
        h_b_models.append(
            {
                "fold": fold,
                "seed": seed,
                "target_softness": h_b_spec[1],
                "application": h_b_spec[2],
                "learning_rate": h_b_spec[3],
                "w": [next_value(), next_value(), next_value()],
                "b": next_value(),
                "active_pair_gamma": _distribution(next_value),
                "top_five_gamma": _distribution(next_value),
                "associations": {
                    "zloglen_query": next_value(),
                    "zloglen_candidate": next_value(),
                },
                "fraction_gamma_below_zero": next_value(),
                "fraction_abs_gamma_near_zero": next_value(),
                "fraction_gamma_above_half": next_value(),
                "semantic_ablation_ndcg_change": next_value(),
            }
        )

    retrieval_rows = []
    retrieval_methods = (
        "return-only",
        "frozen blend",
        "H-A",
        "H-B",
        "global winner (H-B)",
    )
    for transition, method in itertools.product(transitions, retrieval_methods):
        retrieval_rows.append(
            {
                "transition": transition,
                "method": method,
                "top5_overlap_return_only": next_value(),
                "top5_overlap_frozen_blend": next_value(),
                "fraction_queries_different_from_return": next_value(),
                "promoted_raw_rank": {
                    "p10": next_value(),
                    "median": next_value(),
                    "p90": next_value(),
                    "fraction_6_10": next_value(),
                    "fraction_11_20": next_value(),
                    "fraction_21_50": next_value(),
                    "fraction_above_50": next_value(),
                },
                "application_proxy": {
                    "mean_signed_query_to_peer_correlation": next_value(),
                    "fraction_queries_positive": next_value(),
                },
            }
        )

    context_rows = []
    for position, transition in enumerate(transitions):
        context_rows.append(
            {
                "transition": transition,
                "filing_length": {
                    "p10": next_value(),
                    "p50": next_value(),
                    "p90": next_value(),
                    "fraction_at_collection_cap": next_value(),
                    "missing_count": 0,
                },
                "window_length": 84 + position,
                "training_scaler_mean": next_value(),
                "training_scaler_population_sd": next_value(),
            }
        )

    target_softness_rows = []
    for index in range(1, 13):
        shrinkage = 0.1 + index / 1000.0
        scale = 1.0 - shrinkage
        target_temperature = 0.1
        target_softness_rows.append(
            {
                "transition": f"H{index}->H{index + 1}",
                "lw_shrinkage": shrinkage,
                "c_t": scale,
                "target_temperature": target_temperature,
                "raw_equivalent_target_temperature": target_temperature / scale,
            }
        )

    mean_advantage = mean(slots, "winner_minus_frozen_blend")
    strict_wins = sum(row["winner_minus_frozen_blend"] > 0.0 for row in slots)
    equal_transition_means = {
        "return_only_ndcg_at_5": mean(slots, "return_only_ndcg_at_5"),
        "tfidf_only_ndcg_at_5": mean(slots, "tfidf_only_ndcg_at_5"),
        "frozen_blend_ndcg_at_5": mean(slots, "frozen_blend_ndcg_at_5"),
        "h_a_ndcg_at_5": mean(slots, "h_a_seed_mean_ndcg_at_5"),
        "h_b_ndcg_at_5": mean(slots, "h_b_seed_mean_ndcg_at_5"),
        "global_winner_ndcg_at_5": mean(
            slots, "global_winner_seed_mean_ndcg_at_5"
        ),
        "global_winner_seed_dispersion": mean(
            slots, "global_winner_seed_sd_ndcg_at_5"
        ),
        "winner_minus_frozen_blend": mean_advantage,
    }

    scoring_calls = [
        {
            "transition": transitions[index % len(transitions)],
            "method": f"fixed-scoring-call-{index + 1}",
            "wall_time_seconds": next_value(),
        }
        for index in range(168)
    ]
    return {
        "schema_version": report_v2.SCHEMA_VERSION,
        "run": {
            "run_id": "fixture-run",
            "status": "STOP",
            "started_at_utc": "fixture-start",
            "completed_at_utc": "fixture-end",
        },
        "provenance": {
            "protocol_tag": "fixture-protocol-tag",
            "peeled_commit": "a" * 40,
            "git_status_short": "clean",
            "configuration_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "source_sha256": {"report_v2.py": "d" * 64},
            "input_sha256": {"historical-input": "e" * 64},
            "environment": {"python": "fixture-python", "torch": "fixture-torch"},
        },
        "self_tests": [
            {"name": "S1", "status": "PASS", "detail": "fixture detail"},
            {"name": "S2", "status": "PASS", "detail": "fixture learnability"},
            {"name": "S3", "status": "PASS", "detail": "fixture gradient"},
            {"name": "S4", "status": "PASS", "detail": "fixture binding"},
        ],
        "fit_roster": {
            "development_expected": 48,
            "development_completed": 48,
            "gate_expected": 18,
            "gate_completed": 18,
            "total_expected": 66,
            "total_completed": 66,
            "transition_level_optimizer_updates": 13_440,
        },
        "development": {
            "global_winner": winner(h_b_spec),
            "family_winners": [winner(h_a_spec), winner(h_b_spec)],
            "fits": development_fits,
        },
        "gate": {
            "decision": "STOP",
            "global_winner_label": "H-B/lw/rank_band",
            "thresholds": {
                "minimum_mean_advantage": 0.005,
                "minimum_strict_wins": 4,
            },
            "observed": {
                "mean_advantage": mean_advantage,
                "strict_wins": strict_wins,
                "slot_count": 6,
            },
            "slots": slots,
            "equal_transition_means": equal_transition_means,
        },
        "interpretability": {
            "h_a_models": h_a_models,
            "h_b_models": h_b_models,
            "window_context_gamma_association": next_value(),
        },
        "retrieval_diagnostics": retrieval_rows,
        "target_softness_audit": target_softness_rows,
        "context_audit": context_rows,
        "compute_audit": {
            "trainable_parameters": {"H-A": 114, "H-B": 4},
            "run_wall_time_seconds": next_value(),
            "peak_resident_memory_mb": next_value(),
            "full_transition_scoring_calls": scoring_calls,
        },
    }


def _result_paths(value, prefix=()):
    result = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result.append((prefix, value))
    elif isinstance(value, str) and prefix != ("schema_version",):
        result.append((prefix, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            path = prefix + (key,)
            result.extend(_result_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_result_paths(item, prefix + (index,)))
    return result


def _set_path(value, path, replacement):
    cursor = value
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = replacement


def test_s4_source_has_no_displayed_decimal_literal_and_no_data_tree_access():
    source = inspect.getsource(report_v2)

    assert re.search(r"0\.[0-9]{3}", source) is None
    assert "windows_v3" not in source
    assert "ea_historical" not in source
    assert "vintage_descriptions" not in source


def test_every_fixture_numeric_and_categorical_result_is_bound_to_the_artifact():
    artifact = _fixture()
    baseline = report_v2.render_report(artifact)
    paths = _result_paths(artifact)
    assert paths

    for position, (path, original_value) in enumerate(paths, start=1):
        changed = copy.deepcopy(artifact)
        if isinstance(original_value, int):
            replacement = original_value + position * 100_000 + 17
        elif isinstance(original_value, float):
            replacement = original_value + position * 1000 + 0.25
        else:
            replacement = f"substituted-result-{position}"
        _set_path(changed, path, replacement)
        rendered = report_v2.render_report(changed, validate=False)
        old_token = report_v2._display(original_value)
        new_token = report_v2._display(replacement)
        assert old_token in baseline, f"fixture result is not displayed: {path}"
        assert new_token in rendered, f"substituted result is not displayed: {path}"
        assert rendered != baseline


@pytest.mark.parametrize(
    "path",
    (
        ("gate", "observed", "mean_advantage"),
        ("gate", "observed", "strict_wins"),
        ("gate", "slots", 0, "winner_minus_frozen_blend"),
        ("gate", "equal_transition_means", "global_winner_ndcg_at_5"),
        ("fit_roster", "total_completed"),
    ),
)
def test_strict_validation_rejects_one_field_arithmetic_or_roster_tampering(path):
    artifact = _fixture()
    changed = copy.deepcopy(artifact)
    cursor = changed
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] += 1

    with pytest.raises(report_v2.ReportSchemaError):
        report_v2.render_report(changed)


@pytest.mark.parametrize(
    "section",
    (
        "interpretability",
        "retrieval_diagnostics",
        "target_softness_audit",
        "context_audit",
        "compute_audit",
    ),
)
def test_diagnostic_sections_are_required(section):
    artifact = _fixture()
    del artifact[section]

    with pytest.raises(report_v2.ReportSchemaError, match="required field"):
        report_v2.render_report(artifact)


def test_strict_validation_rejects_duplicate_self_test_rows():
    artifact = _fixture()
    artifact["self_tests"].append(copy.deepcopy(artifact["self_tests"][0]))

    with pytest.raises(report_v2.ReportSchemaError, match="four mandatory self-tests"):
        report_v2.render_report(artifact)


def test_file_generation_cannot_use_unchecked_mode(tmp_path: Path):
    artifact = _fixture()
    artifact["gate"]["observed"]["mean_advantage"] += 1
    input_path = tmp_path / "tampered.json"
    output_path = tmp_path / "RUN_REPORT.md"
    input_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(report_v2.ReportSchemaError, match="mean_advantage"):
        report_v2.generate_report(input_path, output_path)
    assert not output_path.exists()


def test_required_sections_render_and_file_creation_refuses_overwrite(tmp_path: Path):
    artifact = _fixture()
    input_path = tmp_path / "results.json"
    output_path = tmp_path / "RUN_REPORT.md"
    input_path.write_text(json.dumps(artifact), encoding="utf-8")

    rendered = report_v2.generate_report(input_path, output_path)

    assert output_path.read_text(encoding="utf-8") == rendered
    for heading in (
        "## Run status",
        "## Provenance",
        "## Mandatory self-tests",
        "## Completed fit roster",
        "## Development selection",
        "## Historical continuation gate",
        "## H-A interpretability",
        "## H-B interpretability",
        "## Retrieval and application diagnostics",
        "## Target-softness audit",
        "## Context audit",
        "## Computational audit",
        "## Scope and interpretation",
    ):
        assert heading in rendered
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        report_v2.generate_report(input_path, output_path)
