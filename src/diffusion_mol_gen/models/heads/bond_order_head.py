import torch
import torch.nn as nn
from torch import Tensor


class BondOrderHead(nn.Module):
    """Invariant head predicting bond order logits from edge features."""

    def __init__(self, hidden_channels: int, num_bond_types: int):
        super().__init__()
        # Combines source node, target node, and edge features
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, num_bond_types),
        )

    def forward(self, h: Tensor, edge_index: Tensor, e: Tensor) -> Tensor:
        """
        Args:
            h:          [N, hidden] node features
            edge_index: [2, E]
            e:          [E, hidden] edge features
        Returns:
            [E, num_bond_types] logits
        """
        row, col = edge_index
        edge_in = torch.cat([h[row], h[col], e], dim=-1)
        return self.mlp(edge_in)
