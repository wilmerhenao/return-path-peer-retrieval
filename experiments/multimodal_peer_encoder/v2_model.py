"""Baseline-anchored pairwise heads for the sealed v2 historical study.

The module is deliberately limited to tensor transformations.  It performs no
file access and contains no target-dependent evaluation.  A caller supplies
feature-time cosine matrices and standardized filing-length context, possibly
for only a chunk of query rows, and receives pairwise scores from one of the
two heads declared in :mod:`PROTOCOL_v2.md`.

The constrained rank-band policy is represented separately from the learned
heads.  Its candidate set is determined from the frozen raw-return ranking;
training gathers differentiable learned scores for that fixed set, whereas
evaluation splices the reordered band back into the otherwise unchanged raw
ranking and encodes the resulting permutation with exact integer scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from scripts.metrics import top_k_indices


ADDITIVE_FEATURE_DIM = 5
CENTERED_CONTEXT_DIM = 3
RANK_BAND_START = 3
RANK_BAND_END = 50
H_B_INITIAL_INTERCEPT = -4.0

ApplicationPolicy = Literal["global", "rank-band"]


@dataclass(frozen=True)
class PairFeatures:
    """Materialized pair features for a query chunk and candidate collection.

    ``query_indices`` and ``candidate_indices`` refer to positions in the full
    transition universe.  Candidate indices are always stored row-wise, even
    when every query uses the same full candidate universe.  This makes the
    same object suitable for global scoring and for the query-specific 3--50
    training band.
    """

    cos_raw: Tensor
    cos_text: Tensor
    phi: Tensor
    psi: Tensor
    query_indices: Tensor
    candidate_indices: Tensor

    @property
    def shape(self) -> torch.Size:
        """Return the common ``[queries, candidates]`` pair shape."""

        return self.cos_raw.shape


def _validate_float_matrix(value: Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [items, items]")
    if value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use float32 under the sealed protocol")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _as_indices(
    indices: Tensor | None,
    *,
    size: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if indices is None:
        return torch.arange(size, device=device, dtype=torch.long)
    result = torch.as_tensor(indices, device=device, dtype=torch.long)
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape [count]")
    if result.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    if torch.any(result < 0) or torch.any(result >= size):
        raise IndexError(f"{name} contains an index outside [0, {size})")
    return result


def build_pair_features(
    cos_raw: Tensor,
    cos_text: Tensor,
    zloglen: Tensor,
    window_context: float | Tensor,
    *,
    query_indices: Tensor | None = None,
    candidate_indices: Tensor | None = None,
) -> PairFeatures:
    """Construct the locked H-A and H-B features for a query chunk.

    ``cos_raw`` and ``cos_text`` are full square transition matrices and
    ``zloglen`` contains one training-standardized value per security.
    ``query_indices`` may select any row chunk while leaving
    ``candidate_indices`` unset to score that chunk against the full universe.
    A two-dimensional candidate index tensor instead supplies a separate list
    for every query, as required for rank-band training.
    """

    _validate_float_matrix(cos_raw, "cos_raw")
    _validate_float_matrix(cos_text, "cos_text")
    if cos_text.shape != cos_raw.shape:
        raise ValueError("cos_raw and cos_text must have equal shape")
    if cos_text.device != cos_raw.device:
        raise ValueError("cos_raw and cos_text must use the same device")

    n_items = cos_raw.shape[0]
    if zloglen.ndim != 1 or zloglen.shape[0] != n_items:
        raise ValueError("zloglen must have shape [items]")
    if zloglen.dtype != torch.float32:
        raise TypeError("zloglen must use float32 under the sealed protocol")
    if zloglen.device != cos_raw.device:
        raise ValueError("zloglen and cosine matrices must use the same device")
    if not torch.isfinite(zloglen).all():
        raise ValueError("zloglen must contain only finite values")

    queries = _as_indices(
        query_indices,
        size=n_items,
        device=cos_raw.device,
        name="query_indices",
    )
    n_queries = queries.numel()

    if candidate_indices is None:
        shared_candidates = torch.arange(
            n_items, device=cos_raw.device, dtype=torch.long
        )
        candidates = shared_candidates.unsqueeze(0).expand(n_queries, -1)
    else:
        supplied = torch.as_tensor(
            candidate_indices, device=cos_raw.device, dtype=torch.long
        )
        if supplied.ndim == 1:
            if supplied.numel() == 0:
                raise ValueError("candidate_indices cannot be empty")
            candidates = supplied.unsqueeze(0).expand(n_queries, -1)
        elif supplied.ndim == 2:
            if supplied.shape[0] != n_queries or supplied.shape[1] == 0:
                raise ValueError(
                    "query-specific candidate_indices must have shape "
                    "[queries, nonzero candidates]"
                )
            candidates = supplied
        else:
            raise ValueError("candidate_indices must be one- or two-dimensional")
        if torch.any(candidates < 0) or torch.any(candidates >= n_items):
            raise IndexError(
                f"candidate_indices contains an index outside [0, {n_items})"
            )

    raw_rows = cos_raw.index_select(0, queries)
    text_rows = cos_text.index_select(0, queries)
    selected_raw = torch.gather(raw_rows, 1, candidates)
    selected_text = torch.gather(text_rows, 1, candidates)

    query_length = zloglen.index_select(0, queries).unsqueeze(1).expand_as(
        selected_raw
    )
    candidate_length = zloglen[candidates]
    context_scalar = torch.as_tensor(
        window_context, device=cos_raw.device, dtype=torch.float32
    )
    if context_scalar.ndim != 0 or not torch.isfinite(context_scalar):
        raise ValueError("window_context must be one finite scalar")
    context = torch.full_like(selected_raw, context_scalar.item())

    phi = torch.stack(
        (
            selected_raw,
            selected_text,
            query_length,
            candidate_length,
            context,
        ),
        dim=-1,
    )
    psi = torch.stack((context, query_length, candidate_length), dim=-1)
    return PairFeatures(
        cos_raw=selected_raw,
        cos_text=selected_text,
        phi=phi,
        psi=psi,
        query_indices=queries,
        candidate_indices=candidates,
    )


def _validate_pair_features(features: PairFeatures) -> None:
    if features.cos_raw.ndim != 2 or features.cos_text.shape != features.cos_raw.shape:
        raise ValueError("pair cosine tensors must have the same [queries, candidates] shape")
    expected_pair_shape = (*features.cos_raw.shape,)
    if features.phi.shape != (*expected_pair_shape, ADDITIVE_FEATURE_DIM):
        raise ValueError("phi must have shape [queries, candidates, 5]")
    if features.psi.shape != (*expected_pair_shape, CENTERED_CONTEXT_DIM):
        raise ValueError("psi must have shape [queries, candidates, 3]")
    for name, value in (
        ("cos_raw", features.cos_raw),
        ("cos_text", features.cos_text),
        ("phi", features.phi),
        ("psi", features.psi),
    ):
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must use float32 under the sealed protocol")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")


class AdditiveBoostingHead(nn.Module):
    """H-A: bounded nonlinear correction with a zero-initialized scale."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ADDITIVE_FEATURE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        self.lambda_ = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def unscaled_correction(self, features: PairFeatures) -> Tensor:
        """Return ``f_theta(phi_ij)`` before multiplication by ``lambda``."""

        _validate_pair_features(features)
        return self.mlp(features.phi).squeeze(-1)

    def effective_correction(self, features: PairFeatures) -> Tensor:
        """Return the reportable additive correction ``lambda * f_theta``."""

        return self.lambda_ * self.unscaled_correction(features)

    def score(self, features: PairFeatures) -> Tensor:
        """Return baseline raw cosine plus the learned additive correction."""

        return features.cos_raw + self.effective_correction(features)

    def forward(self, features: PairFeatures) -> Tensor:
        return self.score(features)


class CenteredCrossModalHead(nn.Module):
    """H-B: signed, baseline-centered correction toward or away from text."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(CENTERED_CONTEXT_DIM, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(H_B_INITIAL_INTERCEPT, dtype=torch.float32))
        self.register_buffer(
            "centering_constant",
            torch.sigmoid(torch.tensor(H_B_INITIAL_INTERCEPT, dtype=torch.float32)),
        )

    def gamma(self, features: PairFeatures) -> Tensor:
        """Return the signed centered gamma associated with every pair."""

        _validate_pair_features(features)
        logits = torch.einsum("qcd,d->qc", features.psi, self.w) + self.b
        return torch.sigmoid(logits) - self.centering_constant

    def effective_correction(self, features: PairFeatures) -> Tensor:
        """Return ``gamma_ij * (cos_text_ij - cos_raw_ij)``."""

        return self.gamma(features) * (features.cos_text - features.cos_raw)

    def score(self, features: PairFeatures) -> Tensor:
        """Return raw cosine plus the centered cross-modal correction."""

        return features.cos_raw + self.effective_correction(features)

    def forward(self, features: PairFeatures) -> Tensor:
        return self.score(features)


def raw_candidate_order(
    cos_raw: Tensor,
    query_indices: Tensor | None = None,
    *,
    query_seeds: Tensor | None = None,
) -> Tensor:
    """Return the canonical complete raw order for each requested query.

    Ordering delegates to :func:`scripts.metrics.top_k_indices`, including its
    float64 conversion, query-seeded jitter, stable sort, and self-exclusion.
    Raw ordering is intentionally nondifferentiable and is computed only to
    bind the rank-band candidate set and evaluation splice.
    """

    _validate_float_matrix(cos_raw, "cos_raw")
    n_items = cos_raw.shape[0]
    if n_items < 2:
        raise ValueError("candidate ordering requires at least two items")
    queries = _as_indices(
        query_indices,
        size=n_items,
        device=cos_raw.device,
        name="query_indices",
    )
    if query_seeds is None:
        seeds = queries
    else:
        seeds = torch.as_tensor(query_seeds, device=cos_raw.device, dtype=torch.long)
        if seeds.ndim != 1 or seeds.shape != queries.shape:
            raise ValueError("query_seeds must have one value per query")

    rows = cos_raw.index_select(0, queries).detach().cpu().numpy()
    query_values = queries.detach().cpu().tolist()
    seed_values = seeds.detach().cpu().tolist()
    orders = [
        top_k_indices(
            row,
            n_items - 1,
            excluded=int(query),
            query_seed=int(seed),
        )
        for row, query, seed in zip(rows, query_values, seed_values, strict=True)
    ]
    return torch.as_tensor(
        np.stack(orders, axis=0), device=cos_raw.device, dtype=torch.long
    )


def _validate_complete_order(
    order: Tensor,
    query_indices: Tensor,
    *,
    n_candidates: int,
) -> tuple[Tensor, Tensor]:
    order = torch.as_tensor(order, dtype=torch.long)
    queries = torch.as_tensor(query_indices, dtype=torch.long, device=order.device)
    if order.ndim != 2:
        raise ValueError("candidate order must have shape [queries, items - 1]")
    if queries.ndim != 1 or queries.shape[0] != order.shape[0]:
        raise ValueError("query_indices must have one entry per order row")
    if order.shape[1] != n_candidates - 1:
        raise ValueError("each order row must contain every non-self candidate")
    if torch.any(queries < 0) or torch.any(queries >= n_candidates):
        raise IndexError("query index lies outside the candidate universe")
    if torch.any(order < 0) or torch.any(order >= n_candidates):
        raise IndexError("candidate order contains an out-of-range index")

    seen = torch.zeros(
        (order.shape[0], n_candidates), device=order.device, dtype=torch.int64
    )
    seen.scatter_add_(1, order, torch.ones_like(order, dtype=torch.int64))
    expected = torch.ones_like(seen)
    expected.scatter_(1, queries.unsqueeze(1), 0)
    if not torch.equal(seen, expected):
        raise ValueError(
            "each order row must contain every non-self candidate exactly once"
        )
    return order, queries


def rank_band_candidate_indices(
    raw_order: Tensor,
    *,
    start: int = RANK_BAND_START,
    end: int = RANK_BAND_END,
) -> Tensor:
    """Return candidate indices at the inclusive one-based raw rank band."""

    if raw_order.ndim != 2:
        raise ValueError("raw_order must have shape [queries, items - 1]")
    if not 1 <= start <= end <= raw_order.shape[1]:
        raise ValueError("rank-band bounds must lie within the eligible order")
    return raw_order[:, start - 1 : end]


def active_rank_band_mask(
    raw_order: Tensor,
    query_indices: Tensor,
    *,
    n_candidates: int | None = None,
    start: int = RANK_BAND_START,
    end: int = RANK_BAND_END,
) -> Tensor:
    """Return the fixed Boolean training mask for raw ranks ``start..end``.

    The mask itself is intentionally determined by the nondifferentiable raw
    ranking.  Applying it as ``candidate_mask`` in the listwise loss, or using
    :func:`gather_active_rank_band`, leaves learned scores within the band fully
    differentiable and gives scores outside the band zero training gradient.
    """

    inferred = raw_order.shape[1] + 1
    n_items = inferred if n_candidates is None else int(n_candidates)
    order, queries = _validate_complete_order(
        raw_order, query_indices, n_candidates=n_items
    )
    active = rank_band_candidate_indices(order, start=start, end=end)
    mask = torch.zeros(
        (order.shape[0], n_items), device=order.device, dtype=torch.bool
    )
    mask.scatter_(1, active, True)
    if torch.any(mask.gather(1, queries.unsqueeze(1))):
        raise AssertionError("self candidate cannot enter the active rank band")
    return mask


def gather_active_rank_band(
    values: Tensor,
    raw_order: Tensor,
    *,
    start: int = RANK_BAND_START,
    end: int = RANK_BAND_END,
) -> Tensor:
    """Differentiably gather query-specific values at active raw ranks."""

    if values.ndim != 2 or values.shape[0] != raw_order.shape[0]:
        raise ValueError("values and raw_order must have the same query count")
    if values.shape[1] != raw_order.shape[1] + 1:
        raise ValueError("values must have one column for every universe item")
    active = rank_band_candidate_indices(raw_order, start=start, end=end)
    return torch.gather(values, 1, active)


def splice_rank_band_order(
    raw_order: Tensor,
    corrected_scores: Tensor,
    query_indices: Tensor,
    *,
    start: int = RANK_BAND_START,
    end: int = RANK_BAND_END,
) -> Tensor:
    """Splice a corrected stable ordering into the protected raw order.

    Only candidates originally at one-based raw positions ``start..end`` may
    move.  Stable descending sorting resolves an exact corrected-score tie in
    favor of the candidate's earlier raw position.
    """

    if corrected_scores.ndim != 2:
        raise ValueError("corrected_scores must have shape [queries, items]")
    n_items = corrected_scores.shape[1]
    order, _ = _validate_complete_order(
        raw_order, query_indices, n_candidates=n_items
    )
    if corrected_scores.shape[0] != order.shape[0]:
        raise ValueError("corrected_scores and raw_order must have equal query count")
    active = rank_band_candidate_indices(order, start=start, end=end)
    active_scores = torch.gather(corrected_scores, 1, active)
    if not torch.isfinite(active_scores).all():
        raise ValueError("corrected scores in the active rank band must be finite")
    within_band = torch.argsort(
        active_scores, dim=1, descending=True, stable=True
    )
    reordered = torch.gather(active, 1, within_band)
    result = order.clone()
    result[:, start - 1 : end] = reordered
    return result


def encode_integer_rank_scores(
    final_order: Tensor,
    query_indices: Tensor,
    *,
    n_candidates: int | None = None,
) -> Tensor:
    """Encode a complete permutation with the protocol's exact integer scores.

    For a universe of size ``N``, the candidate at one-based position ``p``
    receives ``N - p`` and the excluded query receives ``-N``.  The returned
    tensor uses ``int64``; callers may convert it to float64 for the canonical
    metric without changing any integer or allowing its tiny jitter to bridge
    a rank gap.
    """

    inferred = final_order.shape[1] + 1
    n_items = inferred if n_candidates is None else int(n_candidates)
    order, queries = _validate_complete_order(
        final_order, query_indices, n_candidates=n_items
    )
    scores = torch.full(
        (order.shape[0], n_items),
        -n_items,
        device=order.device,
        dtype=torch.int64,
    )
    values = torch.arange(
        n_items - 1, 0, -1, device=order.device, dtype=torch.int64
    ).unsqueeze(0).expand_as(order)
    scores.scatter_(1, order, values)
    if not torch.all(scores.gather(1, queries.unsqueeze(1)) == -n_items):
        raise AssertionError("integer encoding did not preserve the self sentinel")
    return scores


def apply_application_policy(
    cos_raw: Tensor,
    corrected_scores: Tensor,
    query_indices: Tensor | None = None,
    *,
    policy: ApplicationPolicy,
    raw_order: Tensor | None = None,
    start: int = RANK_BAND_START,
    end: int = RANK_BAND_END,
) -> Tensor:
    """Apply global scoring or the exact constrained rank-band splice.

    Global application returns the learned floating-point score block without
    altering it.  Rank-band application returns an ``int64`` score block whose
    ranking exactly represents the declared splice.
    """

    _validate_float_matrix(cos_raw, "cos_raw")
    n_items = cos_raw.shape[0]
    queries = _as_indices(
        query_indices,
        size=n_items,
        device=cos_raw.device,
        name="query_indices",
    )
    if corrected_scores.ndim != 2 or corrected_scores.shape != (
        queries.numel(),
        n_items,
    ):
        raise ValueError(
            "corrected_scores must have shape [requested queries, all candidates]"
        )
    if not torch.isfinite(corrected_scores).all():
        raise ValueError("corrected_scores must contain only finite values")
    if policy == "global":
        return corrected_scores
    if policy != "rank-band":
        raise ValueError("policy must be 'global' or 'rank-band'")

    bound_order = raw_candidate_order(cos_raw, queries) if raw_order is None else raw_order
    spliced = splice_rank_band_order(
        bound_order,
        corrected_scores,
        queries,
        start=start,
        end=end,
    )
    return encode_integer_rank_scores(
        spliced, queries, n_candidates=n_items
    )
