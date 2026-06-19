import torch
import torch.nn.functional as F
from torch import Tensor


def position_loss(
    pred: Tensor,
    target: Tensor,
    batch: Tensor | None = None,
) -> Tensor:
    """MSE loss for positions, averaged over atoms."""
    loss = F.mse_loss(pred, target, reduction="none").sum(dim=-1)  # [N]
    if batch is not None:
        # Average per-graph, then average over graphs
        from torch_geometric.utils import scatter
        n_per_graph = scatter(torch.ones_like(batch, dtype=torch.float), batch, reduce="sum")
        per_graph = scatter(loss, batch, reduce="sum") / n_per_graph
        return per_graph.mean()
    return loss.mean()


def categorical_loss(
    pred_logits: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Cross-entropy loss, optionally masked.
    mask: [N] bool — when provided, loss only on True positions.
    """
    if mask is not None:
        if mask.sum() == 0:
            return pred_logits.sum() * 0.0
        return F.cross_entropy(pred_logits[mask], target[mask])
    return F.cross_entropy(pred_logits, target)


def score_matching_loss(
    pred_score: Tensor,
    noise: Tensor,
    std: Tensor,
) -> Tensor:
    """Denoising score matching: ||s_θ + ε/σ||²"""
    target = -noise / std
    return F.mse_loss(pred_score, target)
