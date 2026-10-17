"""Public replay of the paper's CNN V1 historical continuation experiment.

Architecture, training, scoring, folds and continuation thresholds are preserved.
Only the historical stage is exposed: no private seals, final-fit machinery or
modern evaluator is included. See docs/cnn.md for scope and required inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from experiments.multimodal_peer_encoder.evaluation import (
    census_ndcg_at_5,
    frozen_rank_blend,
    row_l2_cosine,
)
from experiments.multimodal_peer_encoder.model import (
    ResidualGatedMultimodalRetriever,
    listwise_softmax_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "historical_protocol.json"
DEFAULT_HISTORICAL_ROOT = PROJECT_ROOT / "results" / "ea_historical" / "windows"
DEFAULT_VINTAGE_ROOT = PROJECT_ROOT / "results" / "vintage_descriptions"
DEFAULT_CACHE_DIR = EXPERIMENT_DIR / "artifacts" / "historical_text_cache"

TFIDF_CONFIG_B: dict[str, Any] = {
    "max_features": 50_000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "min_df": 2,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}


@dataclass(frozen=True)
class PreparedTransition:
    """Aligned tensors and text for one historical feature-target transition."""

    transition: str
    tickers: tuple[str, ...]
    returns: np.ndarray
    text_embeddings: np.ndarray
    realized: np.ndarray
    texts: tuple[str, ...]
    input_paths: tuple[Path, ...]



@dataclass
class FitResult:
    """Final-epoch model and the fixed forty-epoch training trace."""

    model: Any
    epoch_mean_losses: tuple[float, ...]



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    """Hash a JSON payload using the lock verifier's canonical encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def sha256_array(values: np.ndarray) -> str:
    """Hash an aligned numeric array including dtype and shape."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()



def sha256_strings(values: Sequence[str]) -> str:
    """Hash an ordered sequence without delimiter ambiguity."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()



def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return _logical_path(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value



def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()



def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )



def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_fixed_protocol(config)
    return config



def _validate_fixed_protocol(config: dict[str, Any]) -> None:
    """Fail if a consequential setting differs from the declared protocol."""

    model = config["model"]
    training = model["training"]
    objective = model["objective"]
    selection = config["historical_selection"]
    final = config["final_historical_fit"]
    modern_lock = config["modern_target_lock"]
    modern = config["modern_evaluation"]
    expected_transitions = [f"H{i}->H{i + 1}" for i in range(1, 13)]
    expected_modern_transitions = [f"W{i}->W{i + 1}" for i in range(1, 7)]
    expected_training = {
        "epochs": 40,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_norm_clip": 1.0,
        "dtype": "float32",
        "reference_device": "cpu",
        "torch_num_threads": 8,
        "updates": (
            "one optimizer update per transition; seed-determined transition "
            "order within each epoch"
        ),
        "scheduler": None,
        "dropout": 0.0,
        "early_stopping": False,
        "deterministic_algorithms": True,
    }

    checks = (
        (
            config["schema_version"],
            "multimodal-peer-encoder-historical-protocol-v1",
            "schema version",
        ),
        (config["data"]["historical_transitions"], expected_transitions, "transitions"),
        (
            config["data"]["modern_transitions"],
            expected_modern_transitions,
            "modern transitions",
        ),
        (config["data"]["text_embedding"]["dimension"], 768, "text dimension"),
        (config["data"]["text_embedding"]["token_cap"], 512, "text token cap"),
        (
            config["data"]["text_embedding"]["historical_inference_device"],
            "cpu",
            "historical text device",
        ),
        (
            config["data"]["text_embedding"]["source_parameter_dtype"],
            "torch.bfloat16",
            "source text dtype",
        ),
        (
            config["data"]["text_embedding"]["inference_parameter_dtype"],
            "torch.float32",
            "inference text dtype",
        ),
        (
            config["data"]["text_embedding"]["cpu_dtype_policy"],
            "CPU-only; explicitly call model.float() before encoding",
            "text dtype policy",
        ),
        (
            config["data"]["text_embedding"]["selected_model_tree_sha256"],
            "198d7df8b72a9fd77f9823e5e278f60df77c4577c08dd54e6df9888ade0c6d99",
            "selected Granite model tree",
        ),
        (
            config["data"]["text_embedding"]["pinned_embedding_software"],
            {
                "numpy": "2.4.2",
                "torch": "2.11.0",
                "sentence-transformers": "5.3.0",
                "transformers": "5.5.0",
            },
            "embedding software",
        ),
        (config["data"]["text_embedding"]["frozen"], True, "frozen text encoder"),
        (
            config["data"]["text_embedding"]["fine_tuning_allowed"],
            False,
            "text fine tuning",
        ),
        (model["class"], "ResidualGatedMultimodalRetriever", "model class"),
        (model["candidate_widths"], [32, 64], "candidate widths"),
        (
            model["width_binding"],
            {"embedding_dim": "width", "return_channels": "width"},
            "width binding",
        ),
        (selection["selection_seeds"], [11, 29, 47], "selection seeds"),
        (final["seeds"], [101, 211, 307, 401, 503], "final seeds"),
        (training, expected_training, "training settings"),
        (objective["name"], "ListNet-style listwise softmax cross-entropy", "objective"),
        (objective["implementation"], "listwise_softmax_loss", "loss implementation"),
        (objective["score_temperature"], 0.1, "score temperature"),
        (objective["target_temperature"], 0.1, "target temperature"),
        (objective["query_reduction"], "mean", "query reduction"),
        (
            objective["transition_reduction"],
            "one equal-frequency optimizer update per transition per epoch; "
            "arithmetic mean only for the epoch training trace",
            "transition contribution",
        ),
        (objective["validation_used_for_gradients"], False, "validation gradients"),
        (
            objective["validation_used_for_checkpoint_selection"],
            False,
            "validation checkpoint selection",
        ),
        (model["initial_return_gate"], 0.8, "initial return gate"),
        (model["initial_raw_return_skip"], 0.8, "initial raw skip"),
        (selection["n_validation_slots"], 6, "validation slots"),
        (
            selection["continuation_gate"]["minimum_mean_validation_advantage"],
            0.005,
            "historical mean-advantage gate",
        ),
        (
            selection["continuation_gate"]["minimum_validation_slot_wins"],
            4,
            "historical win gate",
        ),
        (
            selection["continuation_gate"]["total_validation_slots"],
            6,
            "historical gate denominator",
        ),
        (
            selection["baseline"]["return_path_weight"],
            0.8,
            "historical return baseline weight",
        ),
        (
            selection["baseline"]["tfidf_weight"],
            0.2,
            "historical text baseline weight",
        ),
        (
            final["train_transition_indices"],
            list(range(1, 13)),
            "final training transitions",
        ),
        (final["required_completed_seed_count"], 5, "final seed count"),
        (final["seed_selection_allowed"], False, "final seed selection"),
        (final["post_hoc_score_ensemble"], False, "post-hoc score ensemble"),
        (
            modern_lock["required_before_modern_target_access"],
            True,
            "modern lock requirement",
        ),
        (
            modern_lock["evaluator_must_verify_lock_before_target_loader_import"],
            True,
            "pre-import lock verification",
        ),
        (
            modern_lock["one_shot_marker_before_any_modern_data_access"],
            True,
            "one-shot marker ordering",
        ),
        (modern_lock["feature_only_loader_required"], True, "feature-only loader"),
        (modern["attempts_allowed_under_this_protocol"], 1, "modern attempts"),
        (modern["tuning_allowed"], False, "modern tuning"),
        (
            modern["baseline"]["macro_ndcg_at_5_unrounded"],
            0.3729682197528556,
            "modern baseline",
        ),
        (
            modern["paper_inclusion_gate"]["minimum_mean_advantage_over_baseline"],
            0.01,
            "modern mean-advantage gate",
        ),
        (
            modern["paper_inclusion_gate"]["minimum_transition_wins"],
            5,
            "modern win gate",
        ),
        (
            modern["paper_inclusion_gate"]["total_transitions"],
            6,
            "modern gate denominator",
        ),
    )
    for observed, expected, label in checks:
        if observed != expected:
            raise ValueError(f"protocol {label} changed: {observed!r} != {expected!r}")

    expected_folds = [
        (list(range(1, 7)), [7, 8]),
        (list(range(1, 9)), [9, 10]),
        (list(range(1, 11)), [11, 12]),
    ]
    folds = selection["folds"]
    if len(folds) != 3:
        raise ValueError("protocol must contain exactly three historical folds")
    for fold_number, (fold, (train_indices, validation_indices)) in enumerate(
        zip(folds, expected_folds), start=1
    ):
        expected_train_labels = [expected_transitions[index - 1] for index in train_indices]
        expected_validation_labels = [
            expected_transitions[index - 1] for index in validation_indices
        ]
        if fold["fold"] != fold_number:
            raise ValueError("historical fold number changed")
        if fold["train_transition_indices"] != train_indices:
            raise ValueError("historical training fold changed")
        if fold["validation_transition_indices"] != validation_indices:
            raise ValueError("historical validation fold changed")
        if fold["train_transitions"] != expected_train_labels:
            raise ValueError("historical training labels changed")
        if fold["validation_transitions"] != expected_validation_labels:
            raise ValueError("historical validation labels changed")



def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)



def prepare_transition(value: Any) -> PreparedTransition:
    label = str(_field(value, "transition"))
    tickers = tuple(str(item) for item in _field(value, "tickers"))
    returns = np.asarray(_field(value, "ts_features"), dtype=np.float32)
    text_value = _field(value, "text_embeddings")
    if text_value is None:
        raise ValueError(f"{label}: historical Granite cache is required")
    text_embeddings = np.asarray(text_value, dtype=np.float32)
    realized = np.asarray(_field(value, "corr_target"), dtype=np.float32)
    texts = tuple(str(item) for item in _field(value, "texts"))
    input_paths = tuple(Path(item) for item in _field(value, "input_paths"))
    n_items = len(tickers)

    if not label.startswith("H") or "->H" not in label:
        raise ValueError(f"non-historical transition rejected: {label}")
    if n_items < 2 or len(set(tickers)) != n_items or tickers != tuple(sorted(tickers)):
        raise ValueError(f"{label}: tickers must be unique and lexicographically sorted")
    if returns.ndim != 2 or returns.shape[0] != n_items:
        raise ValueError(f"{label}: return-path shape mismatch")
    if text_embeddings.ndim != 2 or text_embeddings.shape != (n_items, 768):
        raise ValueError(f"{label}: expected text embeddings with shape ({n_items}, 768)")
    if realized.shape != (n_items, n_items):
        raise ValueError(f"{label}: target shape mismatch")
    if len(texts) != n_items or not all(texts):
        raise ValueError(f"{label}: filing-text alignment mismatch")
    if not (
        np.isfinite(returns).all()
        and np.isfinite(text_embeddings).all()
        and np.isfinite(realized).all()
    ):
        raise ValueError(f"{label}: non-finite input")
    if not np.allclose(realized, realized.T, atol=2e-5, rtol=2e-5):
        raise ValueError(f"{label}: target must be symmetric")
    for path in input_paths:
        if any(part[1:].isdigit() for part in path.parts if part.startswith("W")):
            raise ValueError(f"{label}: non-historical input path rejected")

    return PreparedTransition(
        transition=label,
        tickers=tickers,
        returns=returns,
        text_embeddings=text_embeddings,
        realized=realized,
        texts=texts,
        input_paths=input_paths,
    )



def load_transitions(
    *,
    historical_root: Path,
    vintage_root: Path,
    cache_dir: Path,
) -> tuple[PreparedTransition, ...]:
    """Load all twelve transitions through the narrow H-only data module."""

    from experiments.multimodal_peer_encoder.historical_data import (
        load_all_historical_transitions,
    )

    loaded = load_all_historical_transitions(
        historical_root=historical_root,
        vintage_root=vintage_root,
        cache_dir=cache_dir,
        verify_cache=True,
    )
    return tuple(prepare_transition(item) for item in loaded)



def compute_frozen_baseline(transition: PreparedTransition) -> dict[str, Any]:
    """Recompute config-b TF-IDF and the fixed 80/20 rank blend from scratch."""

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG_B)
    matrix = normalize(vectorizer.fit_transform(transition.texts), norm="l2")
    tfidf_scores = np.asarray((matrix @ matrix.T).toarray(), dtype=np.float32)
    return_scores = row_l2_cosine(transition.returns)
    blend_scores = frozen_rank_blend(return_scores, tfidf_scores)
    evaluation = census_ndcg_at_5(
        blend_scores,
        transition.realized,
        require_symmetric_scores=False,
    )
    return {
        "transition": transition.transition,
        "n_universe": len(transition.tickers),
        "tfidf_config_id": "b",
        "tfidf_config": _json_safe(TFIDF_CONFIG_B),
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "mean_ndcg": evaluation["mean_ndcg"],
        "n_valid": evaluation["n_valid"],
    }



def _training_relevance(
    realized: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    relevance = torch.as_tensor(realized, dtype=torch.float32, device=device).clamp_min(0.0)
    candidate_mask = ~torch.eye(relevance.shape[0], dtype=torch.bool, device=device)
    return relevance, candidate_mask



def fit_historical_model(
    transitions: Sequence[PreparedTransition],
    width: int,
    seed: int,
    config: dict[str, Any],
) -> FitResult:
    """Fit exactly forty epochs without validation or checkpoint selection."""

    training = config["model"]["training"]
    objective = config["model"]["objective"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(training["torch_num_threads"]))
    torch.use_deterministic_algorithms(bool(training["deterministic_algorithms"]))
    device = torch.device("cpu")

    model = ResidualGatedMultimodalRetriever(
        text_dim=768,
        embedding_dim=width,
        return_channels=width,
        initial_return_gate=float(config["model"]["initial_return_gate"]),
        initial_raw_skip=float(config["model"]["initial_raw_return_skip"]),
        detach_text=True,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    order_rng = random.Random(seed)
    epoch_losses: list[float] = []

    n_epochs = int(training["epochs"])
    for epoch_index in range(n_epochs):
        order = list(range(len(transitions)))
        order_rng.shuffle(order)
        transition_losses: list[float] = []
        for index in order:
            transition = transitions[index]
            returns = torch.as_tensor(transition.returns, dtype=torch.float32, device=device)
            text = torch.as_tensor(
                transition.text_embeddings, dtype=torch.float32, device=device
            )
            relevance, candidate_mask = _training_relevance(
                transition.realized,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            scores = model.full_score_matrix(
                returns,
                text,
                exclude_self=False,
            )
            loss = listwise_softmax_loss(
                scores,
                relevance,
                candidate_mask=candidate_mask,
                temperature=float(objective["score_temperature"]),
                target_temperature=float(objective["target_temperature"]),
                reduction="mean",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["gradient_norm_clip"]),
            )
            optimizer.step()
            transition_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(transition_losses)))
        completed_epoch = epoch_index + 1
        if completed_epoch == 1 or completed_epoch % 10 == 0:
            print(
                "[fit] "
                f"width={width} seed={seed} epoch={completed_epoch}/{n_epochs} "
                f"mean_loss={epoch_losses[-1]:.8f}",
                flush=True,
            )

    return FitResult(model=model, epoch_mean_losses=tuple(epoch_losses))



def evaluate_model(model: Any, transition: PreparedTransition) -> float:
    model.eval()
    with torch.no_grad():
        returns = torch.as_tensor(transition.returns, dtype=torch.float32)
        text = torch.as_tensor(transition.text_embeddings, dtype=torch.float32)
        scores = model.full_score_matrix(
            returns,
            text,
            exclude_self=False,
        ).cpu().numpy()
    return float(census_ndcg_at_5(scores, transition.realized)["mean_ndcg"])



def _fit_summary(fit: FitResult) -> dict[str, Any]:
    losses = [float(value) for value in fit.epoch_mean_losses]
    if not losses or not np.isfinite(losses).all():
        raise ValueError("training history is empty or non-finite")
    if len(losses) != 40:
        raise ValueError(f"fixed protocol requires 40 epochs, received {len(losses)}")
    return {
        "n_epochs": len(losses),
        "first_epoch_mean_loss": losses[0],
        "final_epoch_mean_loss": losses[-1],
        "epoch_mean_losses": losses,
    }



def _unique_input_paths(
    transitions: Sequence[PreparedTransition],
) -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                path.resolve()
                for transition in transitions
                for path in transition.input_paths
            },
            key=lambda path: path.as_posix(),
        )
    )



def _aligned_transition_sha256(
    transition: PreparedTransition,
) -> dict[str, Any]:
    components = {
        "transition": transition.transition,
        "n_tickers": len(transition.tickers),
        "tickers_sha256": sha256_strings(transition.tickers),
        "returns_sha256": sha256_array(transition.returns),
        "text_embeddings_sha256": sha256_array(transition.text_embeddings),
        "realized_sha256": sha256_array(transition.realized),
        "texts_sha256": sha256_strings(transition.texts),
    }
    components["combined_sha256"] = canonical_payload_sha256(components)
    return components



def _execution_snapshot(
    *,
    config_path: Path,
    source_paths: Sequence[Path],
    transitions: Sequence[PreparedTransition],
) -> dict[str, Any]:
    """Attest sources, files, and exact aligned arrays used by a fit."""

    protocol_path = PROJECT_ROOT / "docs" / "cnn.md"
    source_paths = tuple(Path(path).resolve() for path in source_paths)
    missing_sources = [path for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"snapshot source missing: {missing_sources}")
    input_paths = _unique_input_paths(transitions)
    missing_inputs = [path for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"snapshot input missing: {missing_inputs}")

    payload = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(protocol_path),
        "source_sha256": {
            _logical_path(path): sha256_file(path) for path in source_paths
        },
        "input_sha256": {
            _logical_path(path): sha256_file(path) for path in input_paths
        },
        "aligned_transition_sha256": {
            transition.transition: _aligned_transition_sha256(transition)
            for transition in transitions
        },
    }
    payload["snapshot_sha256"] = canonical_payload_sha256(payload)
    return payload



def _require_unchanged_snapshot(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    phase: str,
) -> None:
    if before != after:
        raise RuntimeError(
            f"execution snapshot changed during {phase}; refusing to report"
        )



def run_historical_experiment(
    *,
    config_path: Path,
    output_dir: Path,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    vintage_root: Path = DEFAULT_VINTAGE_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    transitions: Sequence[Any] | None = None,
    trainer: Callable[..., FitResult] | None = None,
    evaluator: Callable[..., float] | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Replay the original V1 historical selection; never open modern inputs.

    The published run failed this gate. This public extraction also ends at the
    decision if a new run passes; it does not authorize or implement modern
    evaluation or the original private final-fit/lock machinery.
    """

    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if not test_mode and any(value is not None for value in (transitions, trainer, evaluator)):
        raise ValueError("injected execution seams require explicit test_mode=True")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory {output_dir}")
    config = load_config(config_path)
    source_paths = tuple(EXPERIMENT_DIR / name for name in (
        "train_historical.py", "historical_data.py", "build_historical_embeddings.py",
        "model.py", "evaluation.py",
    )) + (PROJECT_ROOT / "scripts" / "metrics.py",)
    loader = load_transitions
    snapshotter = _execution_snapshot
    if transitions is not None:
        prepared = tuple(prepare_transition(item) for item in transitions)
        discovery_snapshot = None
    else:
        discovery = tuple(loader(
            historical_root=historical_root, vintage_root=vintage_root, cache_dir=cache_dir,
        ))
        discovery_snapshot = snapshotter(
            config_path=config_path, source_paths=source_paths, transitions=discovery,
        )
        prepared = tuple(loader(
            historical_root=historical_root, vintage_root=vintage_root, cache_dir=cache_dir,
        ))
    expected_labels = tuple(config["data"]["historical_transitions"])
    observed_labels = tuple(item.transition for item in prepared)
    if observed_labels != expected_labels:
        raise ValueError(
            f"historical transition order mismatch: {observed_labels} != {expected_labels}"
        )
    if len(prepared) != 12:
        raise ValueError("exactly twelve historical transitions are required")
    pre_fit_snapshot = snapshotter(
        config_path=config_path,
        source_paths=source_paths,
        transitions=prepared,
    )
    if discovery_snapshot is not None:
        _require_unchanged_snapshot(
            discovery_snapshot,
            pre_fit_snapshot,
            phase="stable historical reload",
        )

    output_dir.mkdir(parents=True)
    print(f"[historical] run_directory={output_dir}", flush=True)
    print(
        "[historical] aligned_transitions=12 "
        f"total_rows={sum(len(item.tickers) for item in prepared)}",
        flush=True,
    )

    trainer = trainer or fit_historical_model
    evaluator = evaluator or evaluate_model
    baseline_rows = []
    for item in prepared:
        print(f"[historical] recomputing_baseline transition={item.transition}", flush=True)
        baseline_rows.append(compute_frozen_baseline(item))
    baseline_by_label = {
        row["transition"]: float(row["mean_ndcg"]) for row in baseline_rows
    }

    seed_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    selection = config["historical_selection"]
    for width in config["model"]["candidate_widths"]:
        for fold in selection["folds"]:
            train_set = tuple(
                prepared[index - 1] for index in fold["train_transition_indices"]
            )
            validation_set = tuple(
                prepared[index - 1]
                for index in fold["validation_transition_indices"]
            )
            for seed in selection["selection_seeds"]:
                print(
                    "[historical] selection_fit "
                    f"width={width} fold={fold['fold']} seed={seed} "
                    f"train_transitions={len(train_set)}",
                    flush=True,
                )
                fit = trainer(train_set, int(width), int(seed), config)
                training_rows.append(
                    {
                        "phase": "selection",
                        "width": width,
                        "fold": fold["fold"],
                        "seed": seed,
                        **_fit_summary(fit),
                    }
                )
                for transition in validation_set:
                    model_ndcg = float(evaluator(fit.model, transition))
                    if not np.isfinite(model_ndcg) or not 0.0 <= model_ndcg <= 1.0:
                        raise ValueError("evaluator returned invalid NDCG")
                    baseline_ndcg = baseline_by_label[transition.transition]
                    seed_rows.append(
                        {
                            "width": width,
                            "fold": fold["fold"],
                            "seed": seed,
                            "transition": transition.transition,
                            "model_ndcg": model_ndcg,
                            "baseline_ndcg": baseline_ndcg,
                            "advantage": model_ndcg - baseline_ndcg,
                        }
                    )
                    print(
                        "[historical] validation "
                        f"width={width} fold={fold['fold']} seed={seed} "
                        f"transition={transition.transition} ndcg={model_ndcg:.8f}",
                        flush=True,
                    )

    slot_rows: list[dict[str, Any]] = []
    width_rows: list[dict[str, Any]] = []
    for width in config["model"]["candidate_widths"]:
        width_seed_rows = [row for row in seed_rows if row["width"] == width]
        labels = []
        for fold in selection["folds"]:
            labels.extend(fold["validation_transitions"])
        for label in labels:
            group = [row for row in width_seed_rows if row["transition"] == label]
            if len(group) != len(selection["selection_seeds"]):
                raise RuntimeError(f"incomplete validation slot: width={width}, {label}")
            seed_mean = float(np.mean([row["model_ndcg"] for row in group]))
            baseline = baseline_by_label[label]
            slot_rows.append(
                {
                    "width": width,
                    "transition": label,
                    "seed_mean_model_ndcg": seed_mean,
                    "baseline_ndcg": baseline,
                    "advantage": seed_mean - baseline,
                    "win": seed_mean > baseline,
                }
            )
        width_slots = [row for row in slot_rows if row["width"] == width]
        width_rows.append(
            {
                "width": width,
                "mean_validation_advantage": float(
                    np.mean([row["advantage"] for row in width_slots])
                ),
                "validation_slot_wins": sum(bool(row["win"]) for row in width_slots),
                "n_validation_slots": len(width_slots),
            }
        )

    selected = sorted(
        width_rows,
        key=lambda row: (-row["mean_validation_advantage"], row["width"]),
    )[0]
    gate_config = selection["continuation_gate"]
    gate_passed = (
        selected["mean_validation_advantage"]
        >= float(gate_config["minimum_mean_validation_advantage"])
        and selected["validation_slot_wins"]
        >= int(gate_config["minimum_validation_slot_wins"])
    )
    post_selection_snapshot = snapshotter(
        config_path=config_path,
        source_paths=source_paths,
        transitions=prepared,
    )
    _require_unchanged_snapshot(
        pre_fit_snapshot,
        post_selection_snapshot,
        phase="historical selection",
    )
    selection_attestation = {
        "mode": "test" if test_mode else "public_historical_replay",
        "public_scope": "historical selection and continuation decision only",
        "modern_targets_accessed": False,
        "stable_input_reload_performed": discovery_snapshot is not None,
        "pre_fit_snapshot": pre_fit_snapshot,
        "post_selection_snapshot_sha256": post_selection_snapshot["snapshot_sha256"],
    }
    result = {
        "schema_version": "multimodal-peer-historical-selection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": pre_fit_snapshot["config_sha256"],
        "baseline_recomputed_from_historical_inputs": True,
        "baseline": baseline_rows,
        "seed_validation_results": seed_rows,
        "validation_slots": slot_rows,
        "width_summaries": width_rows,
        "selected_width": selected["width"],
        "continuation_gate": {
            "minimum_mean_validation_advantage": gate_config[
                "minimum_mean_validation_advantage"
            ],
            "minimum_validation_slot_wins": gate_config[
                "minimum_validation_slot_wins"
            ],
            "observed_mean_validation_advantage": selected[
                "mean_validation_advantage"
            ],
            "observed_validation_slot_wins": selected["validation_slot_wins"],
            "passed": gate_passed,
        },
        "training_runs": training_rows,
        "execution_attestation": selection_attestation,
    }
    selection_path = output_dir / "historical_selection.json"
    _write_json_new(selection_path, result)
    print(
        "[historical] continuation_gate "
        f"selected_width={selected['width']} "
        f"mean_advantage={selected['mean_validation_advantage']:.8f} "
        f"wins={selected['validation_slot_wins']}/6 passed={gate_passed}",
        flush=True,
    )

    if not gate_passed:
        stop = {
            "status": "STOP",
            "reason": "historical continuation gate failed",
            "historical_selection_sha256": sha256_file(selection_path),
            "selected_width": selected["width"],
        }
        _write_json_new(output_dir / "STOP", stop)
        return {**result, "status": "STOP", "output_dir": output_dir}

    return {
        **result,
        "status": "HISTORICAL_GATE_PASSED",
        "output_dir": output_dir,
        "public_scope": "historical decision only; no final fitting or modern evaluation",
    }



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--vintage-root", type=Path, default=DEFAULT_VINTAGE_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    result = run_historical_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        historical_root=args.historical_root,
        vintage_root=args.vintage_root,
        cache_dir=args.cache_dir,
    )
    print(f"status={result['status']}")
    print(f"selected_width={result['selected_width']}")
    print(f"output_dir={result['output_dir']}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
