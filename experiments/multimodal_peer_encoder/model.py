"""Compact multimodal encoder for behavioral peer retrieval.

The model consumes two measurements available at a retrieval date:

* a (possibly padded) path of factor-model return residuals; and
* a frozen 768-dimensional filing-text embedding.

Only the small return encoder, projection layers, and gates are trained.  The
text encoder itself is deliberately outside this module so that experiments can
use a fixed, auditable embedding cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EncodedBatch:
    """Representations and gates produced for one batch of securities."""

    return_embedding: Tensor
    text_embedding: Tensor
    fused_embedding: Tensor
    raw_return_embedding: Tensor
    return_gate: Tensor


def _validate_and_make_mask(
    returns: Tensor,
    mask: Tensor | None,
    lengths: Tensor | None,
) -> Tensor:
    """Return a Boolean observation mask with shape ``[batch, time]``."""

    if returns.ndim != 2:
        raise ValueError(
            f"returns must have shape [batch, time], received {tuple(returns.shape)}"
        )
    if returns.shape[1] < 1:
        raise ValueError("each return path must contain at least one time position")
    if mask is not None and lengths is not None:
        raise ValueError("provide either mask or lengths, not both")

    batch_size, time_steps = returns.shape
    if lengths is not None:
        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError("lengths must have shape [batch]")
        lengths = lengths.to(device=returns.device)
        if torch.any(lengths < 1) or torch.any(lengths > time_steps):
            raise ValueError("every path length must lie in [1, time]")
        positions = torch.arange(time_steps, device=returns.device)
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    elif mask is None:
        mask = torch.ones_like(returns, dtype=torch.bool)
    else:
        if mask.shape != returns.shape:
            raise ValueError(
                f"mask must have shape {tuple(returns.shape)}, received {tuple(mask.shape)}"
            )
        mask = mask.to(device=returns.device, dtype=torch.bool)

    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each return path must include at least one observed value")
    return mask


class MaskedReturnTower(nn.Module):
    """Encode a residual path with a shallow temporal convolutional network.

    Masking is applied before the first convolution and after every convolution.
    The latter detail prevents activations at padding positions from leaking back
    into valid boundary positions in the next layer.
    """

    def __init__(self, channels: int = 32, embedding_dim: int = 64) -> None:
        super().__init__()
        if channels < 1 or embedding_dim < 1:
            raise ValueError("channels and embedding_dim must be positive")

        self.conv1 = nn.Conv1d(1, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.projection = nn.Sequential(
            nn.Linear(2 * channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, returns: Tensor, mask: Tensor) -> Tensor:
        mask_3d = mask.unsqueeze(1)
        observed = torch.where(mask, returns, torch.zeros_like(returns))
        hidden = observed.unsqueeze(1)

        hidden = F.gelu(self.conv1(hidden))
        hidden = hidden * mask_3d.to(dtype=hidden.dtype)
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden * mask_3d.to(dtype=hidden.dtype)

        counts = mask.sum(dim=1, keepdim=True).to(dtype=hidden.dtype)
        mean_pool = hidden.sum(dim=2) / counts
        max_pool = hidden.masked_fill(~mask_3d, -torch.inf).amax(dim=2)
        pooled = torch.cat((mean_pool, max_pool), dim=1)
        return F.normalize(self.projection(pooled), dim=1)


class ResidualGatedMultimodalRetriever(nn.Module):
    """Fuse residual paths and frozen filing embeddings for peer retrieval.

    ``return_gate`` is the per-security weight on the learned return embedding;
    one minus this value is the text weight.  A second, global scalar supplies a
    direct raw-return cosine skip connection at scoring time.  Both quantities
    are bounded in ``[0, 1]`` and can therefore be reported without a post-hoc
    transformation.
    """

    def __init__(
        self,
        *,
        text_dim: int = 768,
        embedding_dim: int = 64,
        return_channels: int = 32,
        initial_return_gate: float = 0.5,
        initial_raw_skip: float = 0.5,
        detach_text: bool = True,
    ) -> None:
        super().__init__()
        if text_dim < 1 or embedding_dim < 1:
            raise ValueError("text_dim and embedding_dim must be positive")
        for name, value in (
            ("initial_return_gate", initial_return_gate),
            ("initial_raw_skip", initial_raw_skip),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")

        self.text_dim = text_dim
        self.embedding_dim = embedding_dim
        self.detach_text = detach_text
        self.return_tower = MaskedReturnTower(return_channels, embedding_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.modality_gate = nn.Linear(2 * embedding_dim, 1)
        self.raw_skip_logit = nn.Parameter(
            torch.tensor(_logit(initial_raw_skip), dtype=torch.float32)
        )

        # A constant initial gate gives every security the declared starting
        # return/text mixture. Training then learns cross-sectional differences.
        nn.init.zeros_(self.modality_gate.weight)
        nn.init.constant_(self.modality_gate.bias, _logit(initial_return_gate))

    @property
    def raw_skip_weight(self) -> Tensor:
        """Global weight on raw-return cosine similarity."""

        return torch.sigmoid(self.raw_skip_logit)

    def encode(
        self,
        returns: Tensor,
        text_embeddings: Tensor,
        *,
        mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> EncodedBatch:
        """Encode a padded batch while retaining components for diagnostics."""

        if not returns.is_floating_point():
            raise TypeError("returns must be a floating-point tensor")
        if text_embeddings.ndim != 2:
            raise ValueError("text_embeddings must have shape [batch, text_dim]")
        if text_embeddings.shape != (returns.shape[0], self.text_dim):
            raise ValueError(
                "text_embeddings must have shape "
                f"[{returns.shape[0]}, {self.text_dim}], received "
                f"{tuple(text_embeddings.shape)}"
            )
        if text_embeddings.device != returns.device:
            raise ValueError("returns and text_embeddings must be on the same device")

        observation_mask = _validate_and_make_mask(returns, mask, lengths)
        return_embedding = self.return_tower(returns, observation_mask)

        text_input = text_embeddings.detach() if self.detach_text else text_embeddings
        text_input = text_input.to(dtype=self.text_projection[0].weight.dtype)
        text_embedding = F.normalize(self.text_projection(text_input), dim=1)

        gate_features = torch.cat((return_embedding, text_embedding), dim=1)
        return_gate = torch.sigmoid(self.modality_gate(gate_features))
        fused = return_gate * return_embedding + (1.0 - return_gate) * text_embedding
        fused_embedding = F.normalize(fused, dim=1)

        observed_returns = torch.where(
            observation_mask, returns, torch.zeros_like(returns)
        )
        raw_return_embedding = F.normalize(observed_returns, dim=1)
        return EncodedBatch(
            return_embedding=return_embedding,
            text_embedding=text_embedding,
            fused_embedding=fused_embedding,
            raw_return_embedding=raw_return_embedding,
            return_gate=return_gate,
        )

    def forward(
        self,
        returns: Tensor,
        text_embeddings: Tensor,
        *,
        mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> EncodedBatch:
        """Alias for :meth:`encode`, following the standard PyTorch interface."""

        return self.encode(returns, text_embeddings, mask=mask, lengths=lengths)

    def score_encoded(
        self,
        queries: EncodedBatch,
        candidates: EncodedBatch,
    ) -> Tensor:
        """Return every query-candidate score as a dense matrix."""

        if queries.fused_embedding.shape[1] != candidates.fused_embedding.shape[1]:
            raise ValueError("query and candidate embedding dimensions differ")
        if queries.raw_return_embedding.shape[1] != candidates.raw_return_embedding.shape[1]:
            raise ValueError("query and candidate return horizons differ")

        fused_cosine = queries.fused_embedding @ candidates.fused_embedding.T
        raw_cosine = queries.raw_return_embedding @ candidates.raw_return_embedding.T
        raw_weight = self.raw_skip_weight.to(dtype=fused_cosine.dtype)
        return raw_weight * raw_cosine + (1.0 - raw_weight) * fused_cosine

    def full_score_matrix(
        self,
        returns: Tensor,
        text_embeddings: Tensor,
        *,
        mask: Tensor | None = None,
        lengths: Tensor | None = None,
        exclude_self: bool = True,
    ) -> Tensor:
        """Score every pair in one universe, optionally masking the diagonal."""

        encoded = self.encode(returns, text_embeddings, mask=mask, lengths=lengths)
        scores = self.score_encoded(encoded, encoded)
        if exclude_self:
            scores = scores.clone()
            scores.fill_diagonal_(-torch.inf)
        return scores

    def listwise_scores(
        self,
        returns: Tensor,
        text_embeddings: Tensor,
        *,
        query_indices: Tensor | None = None,
        candidate_indices: Tensor | None = None,
        mask: Tensor | None = None,
        lengths: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        exclude_self: bool = True,
    ) -> Tensor:
        """Score a shared or query-specific candidate list.

        ``candidate_indices`` may have shape ``[candidates]`` (one shared list)
        or ``[queries, candidates]`` (one list per query).  Omitting it scores
        every security.  This indexing interface keeps the universe encoding
        separate from construction of the historical training lists.
        """

        encoded = self.encode(returns, text_embeddings, mask=mask, lengths=lengths)
        all_scores = self.score_encoded(encoded, encoded)
        n_items = returns.shape[0]
        device = returns.device

        if query_indices is None:
            query_indices = torch.arange(n_items, device=device)
        else:
            query_indices = query_indices.to(device=device, dtype=torch.long)
        if query_indices.ndim != 1:
            raise ValueError("query_indices must have shape [queries]")
        _validate_indices(query_indices, n_items, "query_indices")

        if candidate_indices is None:
            candidate_indices = torch.arange(n_items, device=device)
        else:
            candidate_indices = candidate_indices.to(device=device, dtype=torch.long)
        if candidate_indices.ndim == 1:
            _validate_indices(candidate_indices, n_items, "candidate_indices")
            candidate_indices = candidate_indices.unsqueeze(0).expand(
                query_indices.shape[0], -1
            )
        elif candidate_indices.ndim == 2:
            if candidate_indices.shape[0] != query_indices.shape[0]:
                raise ValueError(
                    "query-specific candidate_indices must have one row per query"
                )
            _validate_indices(candidate_indices, n_items, "candidate_indices")
        else:
            raise ValueError("candidate_indices must be one- or two-dimensional")

        scores = all_scores[query_indices.unsqueeze(1), candidate_indices]
        valid = torch.ones_like(scores, dtype=torch.bool)
        if exclude_self:
            valid &= candidate_indices != query_indices.unsqueeze(1)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
            if candidate_mask.ndim == 1:
                if candidate_mask.shape[0] != scores.shape[1]:
                    raise ValueError("one-dimensional candidate_mask has wrong length")
                candidate_mask = candidate_mask.unsqueeze(0).expand_as(scores)
            if candidate_mask.shape != scores.shape:
                raise ValueError(
                    f"candidate_mask must have shape {tuple(scores.shape)}"
                )
            valid &= candidate_mask
        return scores.masked_fill(~valid, -torch.inf)


def listwise_softmax_loss(
    scores: Tensor,
    relevance: Tensor,
    *,
    candidate_mask: Tensor | None = None,
    temperature: float = 1.0,
    target_temperature: float = 1.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """ListNet-style cross entropy between predicted and target rankings.

    Non-finite entries (including self scores set to ``-inf``) are excluded.
    Continuous future-window relevance scores can therefore be supplied without
    first converting them to a single positive label.
    """

    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must have the same [queries, candidates] shape")
    if temperature <= 0.0 or target_temperature <= 0.0:
        raise ValueError("temperatures must be positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    valid = torch.isfinite(scores) & torch.isfinite(relevance)
    if candidate_mask is not None:
        if candidate_mask.shape != scores.shape:
            raise ValueError("candidate_mask must have the same shape as scores")
        valid &= candidate_mask.to(device=scores.device, dtype=torch.bool)
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("each query must retain at least one valid candidate")

    floor = torch.finfo(scores.dtype).min
    prediction_logits = torch.where(valid, scores / temperature, floor)
    target_logits = torch.where(valid, relevance / target_temperature, floor)
    log_probabilities = F.log_softmax(prediction_logits, dim=1)
    target_probabilities = F.softmax(target_logits, dim=1)
    losses = -(target_probabilities * log_probabilities).sum(dim=1)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def count_trainable_parameters(module: nn.Module) -> int:
    """Count scalar parameters updated by an optimizer."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _logit(probability: float) -> float:
    value = torch.tensor(probability, dtype=torch.float64)
    return float(torch.log(value) - torch.log1p(-value))


def _validate_indices(indices: Tensor, size: int, name: str) -> None:
    if indices.numel() and (torch.any(indices < 0) or torch.any(indices >= size)):
        raise IndexError(f"{name} contains an index outside [0, {size})")
