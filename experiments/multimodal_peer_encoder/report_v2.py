#!/usr/bin/env python3
"""Render the v2 historical run report from one machine-readable JSON artifact.

The renderer performs no experiment-data discovery.  It reads only the artifact
named by the caller and refuses to replace an existing report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "multimodal-peer-v2-results-v1"

EXPECTED_DEVELOPMENT_FITS = 48
EXPECTED_GATE_FITS = 18
EXPECTED_TOTAL_FITS = 66
EXPECTED_GATE_SLOTS = 6
EXPECTED_FAMILY_MODELS = 9
EXPECTED_RETRIEVAL_ROWS = 30
EXPECTED_CONTEXT_ROWS = 6
EXPECTED_TARGET_SOFTNESS_ROWS = 12
EXPECTED_TRANSITION_UPDATES = 13_440

_RELATIVE_TOLERANCE = 1e-11
_ABSOLUTE_TOLERANCE = 1e-12


class ReportSchemaError(ValueError):
    """Raised when a run artifact cannot support the required report."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportSchemaError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReportSchemaError(f"{label} must be a JSON array")
    return value


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ReportSchemaError(f"{label} lacks required field {key!r}")
    return mapping[key]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportSchemaError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportSchemaError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportSchemaError(f"{label} must be an integer")
    return value


def _expect_count(value: Sequence[Any], expected: int, label: str) -> None:
    if len(value) != expected:
        raise ReportSchemaError(
            f"{label} must contain {expected} rows; observed {len(value)}"
        )


def _expect_close(observed: Any, expected: Any, label: str) -> None:
    observed_number = _number(observed, label)
    expected_number = _number(expected, f"recomputed {label}")
    if not math.isclose(
        observed_number,
        expected_number,
        rel_tol=_RELATIVE_TOLERANCE,
        abs_tol=_ABSOLUTE_TOLERANCE,
    ):
        raise ReportSchemaError(
            f"{label} disagrees with the slot-level arithmetic: "
            f"{observed_number!r} != {expected_number!r}"
        )


def _format_number(value: int | float) -> str:
    if isinstance(value, bool):
        raise ReportSchemaError("Boolean value passed as a numeric result")
    if isinstance(value, int):
        return f"{value:,}"
    number = float(value)
    if not math.isfinite(number):
        raise ReportSchemaError("Displayed numeric results must be finite")
    return format(number, ".12g")


def _display(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(_display(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    if not headers:
        raise ReportSchemaError("A report table must have at least one column")
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ReportSchemaError("Report-table row has the wrong number of columns")
        output.append("| " + " | ".join(_display(value) for value in row) + " |")
    return output


def _key_value_table(mapping: Mapping[str, Any]) -> list[str]:
    return _table(
        ("Field", "Artifact value"),
        [(str(key), value) for key, value in mapping.items()],
    )


def _append_section(lines: list[str], title: str, body: Sequence[str]) -> None:
    lines.extend(("", f"## {title}", ""))
    lines.extend(body)


def _winner_row(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _required(value, "architecture", "winner"),
        _required(value, "target_softness", "winner"),
        _required(value, "application", "winner"),
        _required(value, "learning_rate", "winner"),
        _required(value, "equal_transition_mean_ndcg_at_5", "winner"),
        _required(value, "advantage_over_frozen_blend", "winner"),
    )


def _winner_spec(value: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    return (
        _required(value, "architecture", label),
        _required(value, "target_softness", label),
        _required(value, "application", label),
        _number(_required(value, "learning_rate", label), f"{label}.learning_rate"),
    )


def _validate_roster(roster: Mapping[str, Any]) -> None:
    expected = {
        "development_expected": EXPECTED_DEVELOPMENT_FITS,
        "development_completed": EXPECTED_DEVELOPMENT_FITS,
        "gate_expected": EXPECTED_GATE_FITS,
        "gate_completed": EXPECTED_GATE_FITS,
        "total_expected": EXPECTED_TOTAL_FITS,
        "total_completed": EXPECTED_TOTAL_FITS,
        "transition_level_optimizer_updates": EXPECTED_TRANSITION_UPDATES,
    }
    for key, required_value in expected.items():
        observed = _integer(_required(roster, key, "fit roster"), f"fit roster.{key}")
        if observed != required_value:
            raise ReportSchemaError(
                f"fit roster.{key} must be {required_value}; observed {observed}"
            )


def _validate_development(development: Mapping[str, Any]) -> Mapping[str, Any]:
    fits = [
        _object(value, "development fit")
        for value in _array(
            _required(development, "fits", "development"), "development.fits"
        )
    ]
    _expect_count(fits, EXPECTED_DEVELOPMENT_FITS, "development.fits")

    expected_grid = {
        (architecture, target, application, learning_rate, seed)
        for architecture in ("H-A", "H-B")
        for target in ("raw_scale", "lw")
        for application in ("global", "rank_band")
        for learning_rate in (3e-3, 3e-2)
        for seed in (11, 29, 47)
    }
    observed_grid: set[tuple[Any, ...]] = set()
    expected_score_labels = {"H5->H6", "H6->H7"}
    for fit in fits:
        scores = _object(
            _required(fit, "development_scores", "development fit"),
            "development fit scores",
        )
        if set(scores) != expected_score_labels:
            raise ReportSchemaError(
                "each development fit must contain exactly H5->H6 and H6->H7"
            )
        key = (
            _required(fit, "architecture", "development fit"),
            _required(fit, "target_softness", "development fit"),
            _required(fit, "application", "development fit"),
            _number(
                _required(fit, "learning_rate", "development fit"),
                "development fit.learning_rate",
            ),
            _integer(_required(fit, "seed", "development fit"), "development fit.seed"),
        )
        if key in observed_grid:
            raise ReportSchemaError(f"duplicate development fit: {key!r}")
        observed_grid.add(key)
    if observed_grid != expected_grid:
        raise ReportSchemaError("development fit ledger does not match the fixed grid")

    global_winner = _object(
        _required(development, "global_winner", "development"),
        "development.global_winner",
    )
    family_winners = [
        _object(value, "development.family_winner")
        for value in _array(
            _required(development, "family_winners", "development"),
            "development.family_winners",
        )
    ]
    _expect_count(family_winners, 2, "development.family_winners")
    family_by_architecture = {
        _required(value, "architecture", "family winner"): value
        for value in family_winners
    }
    if set(family_by_architecture) != {"H-A", "H-B"}:
        raise ReportSchemaError("family winners must contain one H-A and one H-B row")
    global_architecture = _required(global_winner, "architecture", "global winner")
    if global_architecture not in family_by_architecture:
        raise ReportSchemaError("global winner architecture is not a family winner")
    if _winner_spec(global_winner, "global winner") != _winner_spec(
        family_by_architecture[global_architecture], "selected family winner"
    ):
        raise ReportSchemaError("global winner specification differs from its family winner")
    return global_winner


def _validate_gate(
    gate: Mapping[str, Any],
    run: Mapping[str, Any],
    global_winner: Mapping[str, Any],
) -> None:
    slots = [
        _object(value, "gate slot")
        for value in _array(_required(gate, "slots", "gate"), "gate.slots")
    ]
    _expect_count(slots, EXPECTED_GATE_SLOTS, "gate.slots")
    expected_transitions = {f"H{index}->H{index + 1}" for index in range(7, 13)}
    observed_transitions = {
        _required(slot, "transition", "gate slot") for slot in slots
    }
    if observed_transitions != expected_transitions:
        raise ReportSchemaError("gate slots do not match the six fixed transitions")

    global_architecture = _required(global_winner, "architecture", "global winner")
    global_family_field = {
        "H-A": "h_a_seed_mean_ndcg_at_5",
        "H-B": "h_b_seed_mean_ndcg_at_5",
    }.get(global_architecture)
    if global_family_field is None:
        raise ReportSchemaError("global winner architecture must be H-A or H-B")

    for slot in slots:
        label = f"gate slot {_required(slot, 'transition', 'gate slot')}"
        sample_size = _integer(_required(slot, "n", label), f"{label}.n")
        if sample_size <= 0:
            raise ReportSchemaError(f"{label}.n must be positive")
        _expect_close(
            _required(slot, "global_winner_seed_mean_ndcg_at_5", label),
            _required(slot, global_family_field, label),
            f"{label}.global_winner_seed_mean_ndcg_at_5",
        )
        recomputed_advantage = _number(
            _required(slot, "global_winner_seed_mean_ndcg_at_5", label),
            f"{label}.global_winner_seed_mean_ndcg_at_5",
        ) - _number(
            _required(slot, "frozen_blend_ndcg_at_5", label),
            f"{label}.frozen_blend_ndcg_at_5",
        )
        _expect_close(
            _required(slot, "winner_minus_frozen_blend", label),
            recomputed_advantage,
            f"{label}.winner_minus_frozen_blend",
        )

    advantages = [
        _number(
            _required(slot, "winner_minus_frozen_blend", "gate slot"),
            "gate slot.winner_minus_frozen_blend",
        )
        for slot in slots
    ]
    recomputed_mean_advantage = sum(advantages) / len(advantages)
    recomputed_strict_wins = sum(value > 0.0 for value in advantages)

    observed = _object(_required(gate, "observed", "gate"), "gate.observed")
    if _integer(
        _required(observed, "slot_count", "gate.observed"),
        "gate.observed.slot_count",
    ) != EXPECTED_GATE_SLOTS:
        raise ReportSchemaError("gate.observed.slot_count must be six")
    _expect_close(
        _required(observed, "mean_advantage", "gate.observed"),
        recomputed_mean_advantage,
        "gate.observed.mean_advantage",
    )
    if _integer(
        _required(observed, "strict_wins", "gate.observed"),
        "gate.observed.strict_wins",
    ) != recomputed_strict_wins:
        raise ReportSchemaError("gate.observed.strict_wins disagrees with the slots")

    summary = _object(
        _required(gate, "equal_transition_means", "gate"),
        "gate.equal_transition_means",
    )
    summary_fields = {
        "return_only_ndcg_at_5": "return_only_ndcg_at_5",
        "tfidf_only_ndcg_at_5": "tfidf_only_ndcg_at_5",
        "frozen_blend_ndcg_at_5": "frozen_blend_ndcg_at_5",
        "h_a_ndcg_at_5": "h_a_seed_mean_ndcg_at_5",
        "h_b_ndcg_at_5": "h_b_seed_mean_ndcg_at_5",
        "global_winner_ndcg_at_5": "global_winner_seed_mean_ndcg_at_5",
        "global_winner_seed_dispersion": "global_winner_seed_sd_ndcg_at_5",
        "winner_minus_frozen_blend": "winner_minus_frozen_blend",
    }
    if set(summary) != set(summary_fields):
        raise ReportSchemaError("gate.equal_transition_means has an unexpected schema")
    for output_field, slot_field in summary_fields.items():
        recomputed = sum(
            _number(
                _required(slot, slot_field, "gate slot"),
                f"gate slot.{slot_field}",
            )
            for slot in slots
        ) / len(slots)
        _expect_close(
            _required(summary, output_field, "gate.equal_transition_means"),
            recomputed,
            f"gate.equal_transition_means.{output_field}",
        )

    thresholds = _object(
        _required(gate, "thresholds", "gate"), "gate.thresholds"
    )
    minimum_advantage = _number(
        _required(thresholds, "minimum_mean_advantage", "gate.thresholds"),
        "gate.thresholds.minimum_mean_advantage",
    )
    minimum_wins = _integer(
        _required(thresholds, "minimum_strict_wins", "gate.thresholds"),
        "gate.thresholds.minimum_strict_wins",
    )
    _expect_close(
        minimum_advantage,
        5e-3,
        "gate.thresholds.minimum_mean_advantage",
    )
    if minimum_wins != 4:
        raise ReportSchemaError("gate.thresholds.minimum_strict_wins must be four")
    expected_decision = (
        "PASS"
        if recomputed_mean_advantage >= minimum_advantage
        and recomputed_strict_wins >= minimum_wins
        else "STOP"
    )
    decision = _required(gate, "decision", "gate")
    if decision != expected_decision:
        raise ReportSchemaError("gate decision disagrees with its fixed thresholds")
    if _required(run, "status", "run") != expected_decision:
        raise ReportSchemaError("run status disagrees with the gate decision")
    expected_label = "/".join(
        str(value) for value in _winner_spec(global_winner, "global winner")[:3]
    )
    if _required(gate, "global_winner_label", "gate") != expected_label:
        raise ReportSchemaError("gate global-winner label disagrees with development")


def _validate_fixed_diagnostics(
    artifact: Mapping[str, Any],
    interpretation: Mapping[str, Any],
    retrieval: Sequence[Any],
    target_softness: Sequence[Any],
    context: Sequence[Any],
    compute: Mapping[str, Any],
    global_winner: Mapping[str, Any],
) -> None:
    h_a_models = [
        _object(value, "H-A model")
        for value in _array(
            _required(interpretation, "h_a_models", "interpretability"),
            "interpretability.h_a_models",
        )
    ]
    h_b_models = [
        _object(value, "H-B model")
        for value in _array(
            _required(interpretation, "h_b_models", "interpretability"),
            "interpretability.h_b_models",
        )
    ]
    for label, rows in (("H-A models", h_a_models), ("H-B models", h_b_models)):
        _expect_count(rows, EXPECTED_FAMILY_MODELS, label)
        fold_seed = {
            (
                _integer(_required(row, "fold", label), f"{label}.fold"),
                _integer(_required(row, "seed", label), f"{label}.seed"),
            )
            for row in rows
        }
        if fold_seed != {
            (fold, seed) for fold in (1, 2, 3) for seed in (11, 29, 47)
        }:
            raise ReportSchemaError(f"{label} do not cover the fixed fold/seed grid")
    context_association = _required(
        interpretation,
        "window_context_gamma_association",
        "interpretability",
    )
    if context_association is not None:
        _number(
            context_association,
            "interpretability.window_context_gamma_association",
        )

    retrieval_rows = [
        _object(value, "retrieval diagnostic") for value in retrieval
    ]
    _expect_count(retrieval_rows, EXPECTED_RETRIEVAL_ROWS, "retrieval diagnostics")
    gate_transitions = {f"H{index}->H{index + 1}" for index in range(7, 13)}
    global_architecture = _required(global_winner, "architecture", "global winner")
    expected_methods = {
        "return-only",
        "frozen blend",
        "H-A",
        "H-B",
        f"global winner ({global_architecture})",
    }
    retrieval_by_transition: dict[Any, set[Any]] = {}
    for row in retrieval_rows:
        transition = _required(row, "transition", "retrieval diagnostic")
        method = _required(row, "method", "retrieval diagnostic")
        methods = retrieval_by_transition.setdefault(transition, set())
        if method in methods:
            raise ReportSchemaError(
                f"duplicate retrieval method for {transition}: {method}"
            )
        methods.add(method)
    if set(retrieval_by_transition) != gate_transitions or any(
        methods != expected_methods for methods in retrieval_by_transition.values()
    ):
        raise ReportSchemaError(
            "retrieval diagnostics must contain five fixed methods for every gate slot"
        )

    context_rows = [_object(value, "context audit") for value in context]
    _expect_count(context_rows, EXPECTED_CONTEXT_ROWS, "context audit")
    if {
        _required(row, "transition", "context audit") for row in context_rows
    } != gate_transitions:
        raise ReportSchemaError("context audit does not match the six gate slots")

    target_rows = [_object(value, "target-softness audit") for value in target_softness]
    _expect_count(
        target_rows, EXPECTED_TARGET_SOFTNESS_ROWS, "target-softness audit"
    )
    expected_target_transitions = {
        f"H{index}->H{index + 1}" for index in range(1, 13)
    }
    if {
        _required(row, "transition", "target-softness audit")
        for row in target_rows
    } != expected_target_transitions:
        raise ReportSchemaError("target-softness audit does not cover H1 through H13")
    for row in target_rows:
        label = f"target-softness {_required(row, 'transition', 'target-softness audit')}"
        shrinkage = _number(_required(row, "lw_shrinkage", label), f"{label}.lw_shrinkage")
        scale = _number(_required(row, "c_t", label), f"{label}.c_t")
        temperature = _number(
            _required(row, "target_temperature", label),
            f"{label}.target_temperature",
        )
        _expect_close(temperature, 1e-1, f"{label}.target_temperature")
        _expect_close(scale, 1.0 - shrinkage, f"{label}.c_t")
        if scale <= 0.0:
            raise ReportSchemaError(f"{label}.c_t must be positive")
        _expect_close(
            _required(row, "raw_equivalent_target_temperature", label),
            temperature / scale,
            f"{label}.raw_equivalent_target_temperature",
        )

    parameters = _object(
        _required(compute, "trainable_parameters", "compute audit"),
        "compute trainable parameters",
    )
    if set(parameters) != {"H-A", "H-B"}:
        raise ReportSchemaError("compute parameter counts must contain H-A and H-B")
    expected_parameters = {"H-A": 114, "H-B": 4}
    for architecture, expected in expected_parameters.items():
        if _integer(
            parameters[architecture], f"compute trainable parameters.{architecture}"
        ) != expected:
            raise ReportSchemaError("trainable parameter count differs from the sealed heads")
    calls = _array(
        _required(compute, "full_transition_scoring_calls", "compute audit"),
        "compute full-transition scoring calls",
    )
    _expect_count(calls, 168, "compute full-transition scoring calls")


def _validate_production_artifact(
    artifact: Mapping[str, Any],
    run: Mapping[str, Any],
    roster: Mapping[str, Any],
    development: Mapping[str, Any],
    gate: Mapping[str, Any],
    interpretation: Mapping[str, Any],
    retrieval: Sequence[Any],
    target_softness: Sequence[Any],
    context: Sequence[Any],
    compute: Mapping[str, Any],
) -> None:
    tests = [
        _object(value, "self-test")
        for value in _array(
            _required(artifact, "self_tests", "artifact"), "self_tests"
        )
    ]
    if len(tests) != 4 or {
        (
            _required(value, "name", "self-test"),
            _required(value, "status", "self-test"),
        )
        for value in tests
    } != {(f"S{index}", "PASS") for index in range(1, 5)}:
        raise ReportSchemaError("the four mandatory self-tests must all pass")
    _validate_roster(roster)
    global_winner = _validate_development(development)
    _validate_gate(gate, run, global_winner)
    _validate_fixed_diagnostics(
        artifact,
        interpretation,
        retrieval,
        target_softness,
        context,
        compute,
        global_winner,
    )


def _render_development(development: Mapping[str, Any]) -> list[str]:
    global_winner = _object(
        _required(development, "global_winner", "development"),
        "development.global_winner",
    )
    family_winners = [
        _object(value, "development.family_winner")
        for value in _array(
            _required(development, "family_winners", "development"),
            "development.family_winners",
        )
    ]
    lines = ["### Locked winners", ""]
    lines.extend(
        _table(
            (
                "Architecture",
                "Target softness",
                "Application",
                "Learning rate",
                "Development mean NDCG@5",
                "Advantage over frozen blend",
            ),
            [
                ("Global: " + _display(global_winner["architecture"]),)
                + _winner_row(global_winner)[1:],
                *[_winner_row(value) for value in family_winners],
            ],
        )
    )

    fits = [
        _object(value, "development.fit")
        for value in _array(
            _required(development, "fits", "development"),
            "development.fits",
        )
    ]
    score_labels = sorted(
        {
            str(label)
            for fit in fits
            for label in _object(
                _required(fit, "development_scores", "development fit"),
                "development fit scores",
            )
        }
    )
    lines.extend(("", "### Development fit ledger", ""))
    headers = (
        "Architecture",
        "Target softness",
        "Application",
        "Learning rate",
        "Seed",
        *score_labels,
        "Equal-transition mean",
        "Advantage over frozen blend",
        "Final epoch loss",
    )
    rows = []
    for fit in fits:
        scores = _object(fit["development_scores"], "development fit scores")
        rows.append(
            (
                _required(fit, "architecture", "development fit"),
                _required(fit, "target_softness", "development fit"),
                _required(fit, "application", "development fit"),
                _required(fit, "learning_rate", "development fit"),
                _required(fit, "seed", "development fit"),
                *(scores.get(label) for label in score_labels),
                _required(fit, "equal_transition_mean_ndcg_at_5", "development fit"),
                _required(fit, "advantage_over_frozen_blend", "development fit"),
                _required(fit, "final_epoch_loss", "development fit"),
            )
        )
    lines.extend(_table(headers, rows))
    return lines


def _render_gate(gate: Mapping[str, Any]) -> list[str]:
    thresholds = _object(
        _required(gate, "thresholds", "gate"), "gate.thresholds"
    )
    observed = _object(_required(gate, "observed", "gate"), "gate.observed")
    lines = [
        "### Decision",
        "",
        *_key_value_table(
            {
                "decision": _required(gate, "decision", "gate"),
                "global_winner": _required(gate, "global_winner_label", "gate"),
                **{f"threshold.{key}": value for key, value in thresholds.items()},
                **{f"observed.{key}": value for key, value in observed.items()},
            }
        ),
        "",
        "### Six-slot historical table",
        "",
    ]
    slots = [
        _object(value, "gate slot")
        for value in _array(_required(gate, "slots", "gate"), "gate.slots")
    ]
    slot_fields = (
        "transition",
        "n",
        "return_only_ndcg_at_5",
        "tfidf_only_ndcg_at_5",
        "frozen_blend_ndcg_at_5",
        "h_a_seed_mean_ndcg_at_5",
        "h_a_seed_sd_ndcg_at_5",
        "h_b_seed_mean_ndcg_at_5",
        "h_b_seed_sd_ndcg_at_5",
        "global_winner_seed_mean_ndcg_at_5",
        "global_winner_seed_sd_ndcg_at_5",
        "winner_minus_frozen_blend",
    )
    slot_headers = (
        "Transition",
        "N",
        "Return",
        "TF-IDF",
        "Frozen blend",
        "H-A",
        "H-A seed SD",
        "H-B",
        "H-B seed SD",
        "Global winner",
        "Winner seed SD",
        "Winner minus blend",
    )
    lines.extend(
        _table(
            slot_headers,
            [
                tuple(_required(slot, field, "gate slot") for field in slot_fields)
                for slot in slots
            ],
        )
    )
    means = _object(
        _required(gate, "equal_transition_means", "gate"),
        "gate.equal_transition_means",
    )
    lines.extend(("", "### Equal-transition summary", ""))
    lines.extend(_key_value_table(means))
    return lines


def _distribution_values(value: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    return tuple(
        _required(value, key, label)
        for key in ("mean", "standard_deviation", "p10", "p50", "p90")
    )


def _render_h_a(rows_raw: Any) -> list[str]:
    rows = [_object(value, "H-A model") for value in _array(rows_raw, "H-A models")]
    headers = (
        "Fold",
        "Seed",
        "Target softness",
        "Application",
        "Learning rate",
        "Lambda",
        "Correction mean",
        "Correction SD",
        "Correction p10",
        "Correction p50",
        "Correction p90",
        "Maximum absolute correction",
        "Positive fraction",
        "Score derivative wrt text cosine",
        "Semantic-ablation NDCG change",
    )
    output_rows = []
    for row in rows:
        distribution = _object(
            _required(row, "active_pair_correction", "H-A model"),
            "H-A correction distribution",
        )
        output_rows.append(
            (
                _required(row, "fold", "H-A model"),
                _required(row, "seed", "H-A model"),
                _required(row, "target_softness", "H-A model"),
                _required(row, "application", "H-A model"),
                _required(row, "learning_rate", "H-A model"),
                _required(row, "lambda", "H-A model"),
                *_distribution_values(distribution, "H-A correction distribution"),
                _required(row, "max_abs_effective_correction", "H-A model"),
                _required(row, "positive_correction_fraction", "H-A model"),
                _required(row, "score_derivative_cos_text", "H-A model"),
                _required(row, "semantic_ablation_ndcg_change", "H-A model"),
            )
        )
    return _table(headers, output_rows)


def _render_h_b(rows_raw: Any, context_association: Any) -> list[str]:
    rows = [_object(value, "H-B model") for value in _array(rows_raw, "H-B models")]
    headers = (
        "Fold",
        "Seed",
        "Target softness",
        "Application",
        "Learning rate",
        "w",
        "b",
        "Active gamma mean",
        "Active gamma SD",
        "Active gamma p10",
        "Active gamma p50",
        "Active gamma p90",
        "Top-five gamma mean",
        "Top-five gamma SD",
        "Top-five gamma p10",
        "Top-five gamma p50",
        "Top-five gamma p90",
        "Association: query length",
        "Association: candidate length",
        "Fraction gamma below zero",
        "Fraction absolute gamma near zero",
        "Fraction gamma above half",
        "Semantic-ablation NDCG change",
    )
    output_rows = []
    for row in rows:
        active = _object(
            _required(row, "active_pair_gamma", "H-B model"),
            "H-B active gamma distribution",
        )
        top_five = _object(
            _required(row, "top_five_gamma", "H-B model"),
            "H-B top-five gamma distribution",
        )
        associations = _object(
            _required(row, "associations", "H-B model"), "H-B associations"
        )
        output_rows.append(
            (
                _required(row, "fold", "H-B model"),
                _required(row, "seed", "H-B model"),
                _required(row, "target_softness", "H-B model"),
                _required(row, "application", "H-B model"),
                _required(row, "learning_rate", "H-B model"),
                _required(row, "w", "H-B model"),
                _required(row, "b", "H-B model"),
                *_distribution_values(active, "H-B active gamma distribution"),
                *_distribution_values(top_five, "H-B top-five gamma distribution"),
                _required(associations, "zloglen_query", "H-B associations"),
                _required(associations, "zloglen_candidate", "H-B associations"),
                _required(row, "fraction_gamma_below_zero", "H-B model"),
                _required(row, "fraction_abs_gamma_near_zero", "H-B model"),
                _required(row, "fraction_gamma_above_half", "H-B model"),
                _required(row, "semantic_ablation_ndcg_change", "H-B model"),
            )
        )
    lines = _table(headers, output_rows)
    lines.extend(("", "Window-context association after slot/seed aggregation: " + _display(context_association)))
    return lines


def _render_retrieval(rows_raw: Any) -> list[str]:
    rows = [
        _object(value, "retrieval diagnostic")
        for value in _array(rows_raw, "retrieval diagnostics")
    ]
    headers = (
        "Transition",
        "Method",
        "Top-five overlap with return",
        "Top-five overlap with blend",
        "Queries differing from return",
        "Promoted raw-rank p10",
        "Promoted raw-rank median",
        "Promoted raw-rank p90",
        "Promoted ranks six through ten",
        "Promoted ranks eleven through twenty",
        "Promoted ranks twenty-one through fifty",
        "Promoted ranks above fifty",
        "Mean signed query-to-peer correlation",
        "Fraction positive",
    )
    output_rows = []
    for row in rows:
        promoted = _object(
            _required(row, "promoted_raw_rank", "retrieval diagnostic"),
            "promoted raw-rank diagnostic",
        )
        application = _object(
            _required(row, "application_proxy", "retrieval diagnostic"),
            "application proxy",
        )
        output_rows.append(
            (
                _required(row, "transition", "retrieval diagnostic"),
                _required(row, "method", "retrieval diagnostic"),
                _required(row, "top5_overlap_return_only", "retrieval diagnostic"),
                _required(row, "top5_overlap_frozen_blend", "retrieval diagnostic"),
                _required(row, "fraction_queries_different_from_return", "retrieval diagnostic"),
                _required(promoted, "p10", "promoted raw-rank diagnostic"),
                _required(promoted, "median", "promoted raw-rank diagnostic"),
                _required(promoted, "p90", "promoted raw-rank diagnostic"),
                _required(promoted, "fraction_6_10", "promoted raw-rank diagnostic"),
                _required(promoted, "fraction_11_20", "promoted raw-rank diagnostic"),
                _required(promoted, "fraction_21_50", "promoted raw-rank diagnostic"),
                _required(promoted, "fraction_above_50", "promoted raw-rank diagnostic"),
                _required(application, "mean_signed_query_to_peer_correlation", "application proxy"),
                _required(application, "fraction_queries_positive", "application proxy"),
            )
        )
    return _table(headers, output_rows)


def _render_context(rows_raw: Any) -> list[str]:
    rows = [_object(value, "context audit") for value in _array(rows_raw, "context audit")]
    headers = (
        "Transition",
        "Length p10",
        "Length p50",
        "Length p90",
        "Fraction at collection cap",
        "Missing lengths",
        "Window length",
        "Training-scaler mean",
        "Training-scaler population SD",
    )
    output_rows = []
    for row in rows:
        length = _object(
            _required(row, "filing_length", "context audit"),
            "filing-length audit",
        )
        output_rows.append(
            (
                _required(row, "transition", "context audit"),
                _required(length, "p10", "filing-length audit"),
                _required(length, "p50", "filing-length audit"),
                _required(length, "p90", "filing-length audit"),
                _required(length, "fraction_at_collection_cap", "filing-length audit"),
                _required(length, "missing_count", "filing-length audit"),
                _required(row, "window_length", "context audit"),
                _required(row, "training_scaler_mean", "context audit"),
                _required(row, "training_scaler_population_sd", "context audit"),
            )
        )
    return _table(headers, output_rows)


def _render_target_softness(rows_raw: Any) -> list[str]:
    rows = [
        _object(value, "target-softness audit")
        for value in _array(rows_raw, "target-softness audit")
    ]
    lines = [
        "The LW target rescales the raw correlation target by `c_t`. The last",
        "column gives the raw-scale temperature that has the same softness.",
        "",
    ]
    lines.extend(
        _table(
            (
                "Transition",
                "LW shrinkage",
                "Target scale c_t",
                "Fixed target temperature",
                "Raw-equivalent target temperature",
            ),
            [
                (
                    _required(row, "transition", "target-softness audit"),
                    _required(row, "lw_shrinkage", "target-softness audit"),
                    _required(row, "c_t", "target-softness audit"),
                    _required(row, "target_temperature", "target-softness audit"),
                    _required(
                        row,
                        "raw_equivalent_target_temperature",
                        "target-softness audit",
                    ),
                )
                for row in rows
            ],
        )
    )
    return lines


def _render_compute(value: Mapping[str, Any]) -> list[str]:
    summary = {
        "run_wall_time_seconds": _required(value, "run_wall_time_seconds", "compute audit"),
        "peak_resident_memory_mb": _required(value, "peak_resident_memory_mb", "compute audit"),
    }
    parameters = _object(
        _required(value, "trainable_parameters", "compute audit"),
        "compute trainable parameters",
    )
    summary.update({f"trainable_parameters.{key}": item for key, item in parameters.items()})
    lines = _key_value_table(summary)
    calls = [
        _object(item, "scoring call")
        for item in _array(
            _required(value, "full_transition_scoring_calls", "compute audit"),
            "full-transition scoring calls",
        )
    ]
    lines.extend(("", "### Full-transition scoring calls", ""))
    lines.extend(
        _table(
            ("Transition", "Method", "Wall time seconds"),
            [
                (
                    _required(item, "transition", "scoring call"),
                    _required(item, "method", "scoring call"),
                    _required(item, "wall_time_seconds", "scoring call"),
                )
                for item in calls
            ],
        )
    )
    return lines


def validate_result_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fixed production roster and all redundant gate arithmetic."""

    artifact = _object(artifact, "artifact")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ReportSchemaError("Unsupported v2 result-artifact schema")
    run = _object(_required(artifact, "run", "artifact"), "run")
    roster = _object(
        _required(artifact, "fit_roster", "artifact"), "fit_roster"
    )
    development = _object(
        _required(artifact, "development", "artifact"), "development"
    )
    gate = _object(_required(artifact, "gate", "artifact"), "gate")
    interpretation = _object(
        _required(artifact, "interpretability", "artifact"), "interpretability"
    )
    retrieval = _array(
        _required(artifact, "retrieval_diagnostics", "artifact"),
        "retrieval diagnostics",
    )
    target_softness = _array(
        _required(artifact, "target_softness_audit", "artifact"),
        "target-softness audit",
    )
    context = _array(
        _required(artifact, "context_audit", "artifact"), "context audit"
    )
    compute = _object(
        _required(artifact, "compute_audit", "artifact"), "compute audit"
    )
    _validate_production_artifact(
        artifact,
        run,
        roster,
        development,
        gate,
        interpretation,
        retrieval,
        target_softness,
        context,
        compute,
    )


def render_report(artifact: Mapping[str, Any], *, validate: bool = True) -> str:
    """Render only artifact values, validating production shape by default.

    The explicit unchecked mode exists solely for the S4 one-field substitution
    test. Production report creation always uses strict validation.
    """

    artifact = _object(artifact, "artifact")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ReportSchemaError("Unsupported v2 result-artifact schema")
    if validate:
        validate_result_artifact(artifact)
    run = _object(_required(artifact, "run", "artifact"), "run")
    provenance = _object(
        _required(artifact, "provenance", "artifact"), "provenance"
    )
    tests = [
        _object(value, "self-test")
        for value in _array(
            _required(artifact, "self_tests", "artifact"), "self_tests"
        )
    ]
    roster = _object(
        _required(artifact, "fit_roster", "artifact"), "fit_roster"
    )
    development = _object(
        _required(artifact, "development", "artifact"), "development"
    )
    gate = _object(_required(artifact, "gate", "artifact"), "gate")
    interpretation = _object(
        _required(artifact, "interpretability", "artifact"), "interpretability"
    )
    retrieval = _array(
        _required(artifact, "retrieval_diagnostics", "artifact"),
        "retrieval diagnostics",
    )
    target_softness = _array(
        _required(artifact, "target_softness_audit", "artifact"),
        "target-softness audit",
    )
    context = _array(
        _required(artifact, "context_audit", "artifact"), "context audit"
    )
    compute = _object(
        _required(artifact, "compute_audit", "artifact"), "compute audit"
    )

    lines = [
        "# Baseline-anchored multimodal heads: historical run report",
        "",
        "This report is generated from the machine-readable run artifact. It is a",
        "historical diagnostic and does not authorize a modern evaluation or paper edit.",
    ]
    _append_section(
        lines,
        "Run status",
        _key_value_table(
            {
                "run_id": _required(run, "run_id", "run"),
                "status": _required(run, "status", "run"),
                "started_at_utc": _required(run, "started_at_utc", "run"),
                "completed_at_utc": _required(run, "completed_at_utc", "run"),
            }
        ),
    )

    provenance_summary = {
        "protocol_tag": _required(provenance, "protocol_tag", "provenance"),
        "peeled_commit": _required(provenance, "peeled_commit", "provenance"),
        "git_status_short": _required(provenance, "git_status_short", "provenance"),
        "configuration_sha256": _required(
            provenance, "configuration_sha256", "provenance"
        ),
        "manifest_sha256": _required(provenance, "manifest_sha256", "provenance"),
    }
    provenance_lines = _key_value_table(provenance_summary)
    for title, key in (
        ("Source identities", "source_sha256"),
        ("Input identities", "input_sha256"),
        ("Environment", "environment"),
    ):
        provenance_lines.extend(("", f"### {title}", ""))
        provenance_lines.extend(
            _key_value_table(
                _object(_required(provenance, key, "provenance"), f"provenance.{key}")
            )
        )
    _append_section(lines, "Provenance", provenance_lines)

    _append_section(
        lines,
        "Mandatory self-tests",
        _table(
            ("Test", "Status", "Artifact detail"),
            [
                (
                    _required(value, "name", "self-test"),
                    _required(value, "status", "self-test"),
                    _required(value, "detail", "self-test"),
                )
                for value in tests
            ],
        ),
    )
    _append_section(lines, "Completed fit roster", _key_value_table(roster))
    _append_section(lines, "Development selection", _render_development(development))
    _append_section(lines, "Historical continuation gate", _render_gate(gate))

    _append_section(
        lines,
        "H-A interpretability",
        _render_h_a(_required(interpretation, "h_a_models", "interpretability")),
    )
    _append_section(
        lines,
        "H-B interpretability",
        _render_h_b(
            _required(interpretation, "h_b_models", "interpretability"),
            _required(
                interpretation,
                "window_context_gamma_association",
                "interpretability",
            ),
        ),
    )
    _append_section(
        lines,
        "Retrieval and application diagnostics",
        _render_retrieval(retrieval),
    )
    _append_section(
        lines,
        "Target-softness audit",
        _render_target_softness(target_softness),
    )
    _append_section(lines, "Context audit", _render_context(context))
    _append_section(lines, "Computational audit", _render_compute(compute))

    _append_section(
        lines,
        "Scope and interpretation",
        (
            "The experiment measures historical next-window residual-peer retrieval.",
            "It does not measure valuation accuracy, analyst forecasts, portfolio profit,",
            "transaction costs, or a causal effect. Context associations and pairwise",
            "diagnostics are descriptive and do not alter selection or the gate.",
        ),
    )
    return "\n".join(lines).rstrip() + "\n"


def load_artifact(path: Path) -> Mapping[str, Any]:
    """Read exactly one caller-specified JSON result artifact."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportSchemaError(f"Cannot read result artifact: {path}") from exc
    return _object(value, "artifact")


def generate_report(input_path: Path, output_path: Path) -> str:
    """Render ``input_path`` and create ``output_path`` without replacement."""

    report = render_report(load_artifact(input_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = report.encode("utf-8")
    try:
        descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to replace report: {output_path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    generate_report(args.input, args.output)
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
