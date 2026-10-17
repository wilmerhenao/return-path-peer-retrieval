"""Fast synthetic tests for the historical-only selection runner."""

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.multimodal_peer_encoder import train_historical as historical
from experiments.multimodal_peer_encoder.train_historical import (
    DEFAULT_CONFIG,
    FitResult,
    _validate_fixed_protocol,
    _training_relevance,
    canonical_payload_sha256,
    run_historical_experiment,
)


@dataclass(frozen=True)
class SyntheticTransition:
    transition: str
    tickers: tuple[str, ...]
    ts_features: np.ndarray
    text_embeddings: np.ndarray
    corr_target: np.ndarray
    texts: tuple[str, ...]
    input_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FakeModel:
    width: int
    seed: int


def _transitions() -> tuple[SyntheticTransition, ...]:
    generator = np.random.default_rng(20260802)
    result = []
    tickers = tuple(f"T{i}" for i in range(6))
    texts = tuple(
        f"common filing business segment company token{i % 2}" for i in range(6)
    )
    for index in range(1, 13):
        returns = generator.normal(size=(6, 8)).astype(np.float32)
        text = generator.normal(size=(6, 768)).astype(np.float32)
        raw = generator.uniform(0.05, 0.9, size=(6, 6))
        target = ((raw + raw.T) / 2.0).astype(np.float32)
        np.fill_diagonal(target, 1.0)
        result.append(
            SyntheticTransition(
                transition=f"H{index}->H{index + 1}",
                tickers=tickers,
                ts_features=returns,
                text_embeddings=text,
                corr_target=target,
                texts=texts,
            )
        )
    return tuple(result)


def _trainer_log(log):
    def trainer(transitions, width, seed, _config):
        log.append((tuple(item.transition for item in transitions), width, seed))
        return FitResult(FakeModel(width, seed), tuple([0.5] * 40))

    return trainer




def test_failed_gate_writes_stop_and_never_runs_final_seeds(tmp_path):
    calls = []
    output = tmp_path / "failed"
    result = run_historical_experiment(
        config_path=DEFAULT_CONFIG,
        output_dir=output,
        transitions=_transitions(),
        trainer=_trainer_log(calls),
        evaluator=lambda _model, _transition: 0.0,
        test_mode=True,
    )

    assert result["status"] == "STOP"
    assert result["selected_width"] == 32
    assert (output / "STOP").is_file()
    assert (output / "historical_selection.json").is_file()
    assert not (output / "LOCKED_MANIFEST.json").exists()
    assert not (output / "checkpoints").exists()
    assert len(calls) == 18
    assert not ({101, 211, 307, 401, 503} & {call[2] for call in calls})


def test_training_relevance_clamps_negatives_and_excludes_self():
    target = np.array(
        [
            [1.0, -0.5, 0.4],
            [-0.5, 1.0, 0.2],
            [0.4, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    relevance, candidate_mask = _training_relevance(
        target,
        device=torch.device("cpu"),
    )

    assert relevance[0, 1].item() == 0.0
    assert relevance[0, 2].item() == np.float32(0.4)
    assert not torch.any(torch.diagonal(candidate_mask))
    assert torch.all(candidate_mask[~torch.eye(3, dtype=torch.bool)])


def test_aligned_array_mutation_prevents_even_a_test_lock(tmp_path):
    changed = False

    def mutating_trainer(transitions, width, seed, _config):
        nonlocal changed
        if not changed:
            transitions[0].returns[0, 0] += np.float32(1.0)
            changed = True
        return FitResult(FakeModel(width, seed), tuple([0.5] * 40))

    output = tmp_path / "mutated"
    with pytest.raises(RuntimeError, match="snapshot changed"):
        run_historical_experiment(
            config_path=DEFAULT_CONFIG,
            output_dir=output,
            transitions=_transitions(),
            trainer=mutating_trainer,
            evaluator=lambda _model, _transition: 1.0,
                    test_mode=True,
        )
    assert not (output / "LOCKED_MANIFEST.json").exists()
    assert not (output / "TEST_ONLY_MANIFEST.json").exists()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("historical_selection", "continuation_gate", "minimum_mean_validation_advantage"), 0.0),
        (("historical_selection", "selection_seeds"), [11]),
        (("model", "training", "epochs"), 41),
        (("final_historical_fit", "seeds"), [101, 211, 307, 401]),
    ],
)
def test_consequential_protocol_changes_are_rejected(path, replacement):
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement

    with pytest.raises(ValueError, match="protocol"):
        _validate_fixed_protocol(changed)



def test_passing_gate_reports_decision_without_final_fits_or_modern_access(tmp_path):
    calls = []
    output = tmp_path / "passed"
    result = run_historical_experiment(
        config_path=DEFAULT_CONFIG, output_dir=output,
        transitions=_transitions(), trainer=_trainer_log(calls),
        evaluator=lambda model, _: 1.0 if model.width == 64 else 0.0,
        test_mode=True,
    )
    assert result["status"] == "HISTORICAL_GATE_PASSED"
    assert result["selected_width"] == 64
    assert result["continuation_gate"]["passed"] is True
    assert len(calls) == 18
    assert {call[2] for call in calls} == {11, 29, 47}
    assert not (output / "STOP").exists()
    assert not (output / "checkpoints").exists()
    assert not (output / "MODERN_TARGET_ACCESSED.json").exists()
    assert not (output / "LOCKED_MANIFEST.json").exists()


def test_injected_execution_requires_explicit_test_mode(tmp_path):
    with pytest.raises(ValueError, match="test_mode"):
        run_historical_experiment(
            config_path=DEFAULT_CONFIG, output_dir=tmp_path / "rejected",
            transitions=_transitions(),
        )


def test_public_runner_has_no_modern_loader_or_private_lock_dependency():
    source = Path(historical.__file__).read_text()
    assert "evaluate_locked_modern" not in source
    assert "windows_v3" not in source
    assert "data_contract.json" not in source
    assert "environment/models.lock" not in source


def test_cli_help_does_not_need_private_inputs():
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    for module in ("train_historical", "build_historical_embeddings"):
        result = subprocess.run(
            [sys.executable, "-m", "experiments.multimodal_peer_encoder." + module, "--help"],
            cwd=root, text=True, capture_output=True, check=True,
        )
        assert "--historical-root" in result.stdout
