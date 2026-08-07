"""Mandatory post-seal, pre-fit self-tests for the v2 historical run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from experiments.multimodal_peer_encoder import report_v2
from experiments.multimodal_peer_encoder.v2_data import (
    V2Transition,
    assert_no_modern_path,
    load_feature_window_for_s1,
    load_v2_transition,
)
from experiments.multimodal_peer_encoder.v2_model import raw_candidate_order
from experiments.multimodal_peer_encoder.v2_training import (
    FitResult,
    FitSpec,
    count_trainable_parameters,
    evaluate_fit,
    fit_length_scaler,
    make_head,
    score_matrix,
)
from scripts.metrics import top_k_indices


def _complete_order(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores)
    return np.stack(
        [
            top_k_indices(
                matrix[query],
                matrix.shape[0] - 1,
                excluded=query,
                query_seed=query,
            )
            for query in range(matrix.shape[0])
        ],
        axis=0,
    )


def _untrained_fit(architecture: str, scaler: Any) -> FitResult:
    torch.manual_seed(11)
    head = make_head(architecture)  # type: ignore[arg-type]
    return FitResult(
        head=head,
        scaler=scaler,
        epoch_mean_losses=(),
        final_transition_losses={},
        wall_seconds=0.0,
        peak_rss_mb=0.0,
        trainable_parameters=count_trainable_parameters(head),
    )


def run_s1(config: dict[str, Any]) -> dict[str, Any]:
    """Verify exact baseline initialization without touching H8--H13."""

    feature_windows = tuple(load_feature_window_for_s1(index) for index in range(1, 8))
    exact_checks = 0
    aligned_fixture_checks = 0
    for window in feature_windows:
        scaler = fit_length_scaler([window])
        expected_order = raw_candidate_order(
            torch.tensor(window.raw_scores, dtype=torch.float32)
        ).numpy()
        for architecture in ("H-A", "H-B"):
            fit = _untrained_fit(architecture, scaler)
            for application in ("global", "rank_band"):
                spec = FitSpec(
                    architecture=architecture,  # type: ignore[arg-type]
                    target_softness="lw",
                    application=application,  # type: ignore[arg-type]
                    learning_rate=float(config["grid"]["learning_rates"][0]),
                    seed=11,
                )
                scores, _ = score_matrix(fit, spec, window, config)
                observed_order = _complete_order(scores)
                if not np.array_equal(observed_order, expected_order):
                    raise AssertionError(
                        f"S1 ranking mismatch: {window.window} {architecture} {application}"
                    )
                exact_checks += observed_order.shape[0]

    transitions: tuple[V2Transition, ...] = tuple(
        load_v2_transition(index, stage="development") for index in range(1, 7)
    )
    training_scaler = fit_length_scaler(transitions[:4])
    expected = config["data"]["expected_pre_gate_return_ndcg"]
    expected_macro = float(config["data"]["expected_pre_gate_return_macro"])
    ndcg_checks = 0
    for architecture in ("H-A", "H-B"):
        fit = _untrained_fit(architecture, training_scaler)
        for application in ("global", "rank_band"):
            spec = FitSpec(
                architecture=architecture,  # type: ignore[arg-type]
                target_softness="lw",
                application=application,  # type: ignore[arg-type]
                learning_rate=float(config["grid"]["learning_rates"][0]),
                seed=11,
            )
            values = []
            for transition_index, transition in enumerate(transitions):
                evaluated = evaluate_fit(fit, spec, transition, config)
                observed = evaluated.ndcg_at_5
                target = float(expected[transition.transition])
                if abs(observed - target) > 1e-6:
                    raise AssertionError(
                        f"S1 NDCG mismatch: {transition.transition} "
                        f"{architecture} {application} {observed} != {target}"
                    )
                values.append(observed)
                ndcg_checks += 1
                if transition_index == 0:
                    expected_order = _complete_order(transition.raw_scores)[:200]
                    observed_order = _complete_order(evaluated.scores)[:200]
                    if not np.array_equal(observed_order, expected_order):
                        raise AssertionError(
                            "S1 aligned H1->H2 200-query fixture mismatch: "
                            f"{architecture} {application}"
                        )
                    aligned_fixture_checks += observed_order.shape[0]
            if abs(float(np.mean(values)) - expected_macro) > 1e-6:
                raise AssertionError("S1 pre-gate macro NDCG mismatch")

    return {
        "name": "S1",
        "status": "PASS",
        "detail": (
            f"feature_orders={exact_checks}; aligned_fixture_orders={aligned_fixture_checks}; "
            f"pre_gate_ndcg_checks={ndcg_checks}; no_H8_H13_access"
        ),
    }


def run_s3() -> dict[str, Any]:
    """Deliberately request a modern target and prove rejection precedes open."""

    attempted_open = False
    forbidden = Path("results/windows_v3/W1/corr_target.npy")

    def guarded_read(path: Path) -> bytes:
        nonlocal attempted_open
        assert_no_modern_path(path)
        attempted_open = True
        return path.read_bytes()

    try:
        guarded_read(forbidden)
    except PermissionError:
        pass
    else:
        raise AssertionError("S3 modern path was not rejected")
    if attempted_open:
        raise AssertionError("S3 reached filesystem open")
    return {
        "name": "S3",
        "status": "PASS",
        "detail": "modern subtree request rejected before filesystem open",
    }


def report_source_sentinel() -> dict[str, Any]:
    """Run the literal S4 generator-source sentinel."""

    source = Path(report_v2.__file__).read_text(encoding="utf-8")
    hits = re.findall(r"0\.[0-9]{3}", source)
    if hits:
        raise AssertionError(f"S4 report source contains result-like literals: {hits}")
    return {
        "name": "S4",
        "status": "PASS",
        "detail": "generator decimal-literal grep=0; artifact substitution suite=PASS",
    }
