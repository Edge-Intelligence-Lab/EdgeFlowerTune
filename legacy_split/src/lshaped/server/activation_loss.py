from __future__ import annotations

import torch
import torch.nn.functional as F


def dot_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == b.shape[-1]
    return a @ b.T


def activation_contrastive_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor | None,
    temperature: float,
    use_in_batch_negatives: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert query.ndim == 2 and positive.ndim == 2
    assert query.shape == positive.shape
    assert temperature > 0.0

    query = query.float()
    positive = positive.float()
    if negatives is not None:
        negatives = negatives.float()

    logits_parts = [(query * positive).sum(dim=-1, keepdim=True)]

    if use_in_batch_negatives and query.shape[0] > 1:
        in_batch = dot_similarity(query, positive)
        diag = torch.eye(query.shape[0], device=query.device, dtype=torch.bool)
        in_batch = in_batch.masked_fill(diag, torch.finfo(in_batch.dtype).min)
        logits_parts.append(in_batch)

    if negatives is not None and negatives.numel() > 0:
        logits_parts.append(dot_similarity(query, negatives))

    logits = torch.cat(logits_parts, dim=1) / temperature
    targets = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
    loss = F.cross_entropy(logits, targets)
    return loss, logits


def multiple_choice_accuracy(
    hidden: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    assert hidden.ndim == 2
    assert candidate_embeddings.shape == (4, hidden.shape[-1])
    assert labels.ndim == 1
    hidden = hidden.float()
    candidate_embeddings = candidate_embeddings.float()
    scores = hidden @ candidate_embeddings.T
    pred = scores.argmax(dim=-1)
    return float((pred == labels).float().mean().item())
