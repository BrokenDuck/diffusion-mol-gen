"""
Heads to read out the updated predictions for categorical (atom type, bond order and atom charge)
and continuous variables (position).
"""

import torch
import torch.nn as nn
from torch import Tensor


class AtomTypeHead(nn.Module):
    """Invariant head predicting atom type logits from node features."""

    def __init__(self, hidden_channels: int, num_atom_types: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, num_atom_types),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: [N, hidden] node features
        Returns:
            [N, num_atom_types] logits
        """
        return self.mlp(h)


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


class ChargeHead(nn.Module):
    """Invariant head predicting formal charge logits from node features."""

    def __init__(self, hidden_channels: int, num_charges: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, num_charges),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: [N, hidden] node features
        Returns:
            [N, num_charges] logits
        """
        return self.mlp(h)


class PositionHead(nn.Module):
    """
    Equivariant prediction head for positions.

    The backbone's updated positions encode the equivariant coordinate delta
    (pos_out - pos_in). This head applies a scalar gate derived from node
    features to weight the coordinate update, keeping the output equivariant.

    Depending on the diffusion view, the output represents:
      - Variational: predicted clean positions x̂_0
      - Score-based:  score s(x_t, t)  [equivariant vector ∝ -∇_x log p_t]
      - Flow-based:   velocity v(x_t, t) or predicted endpoint x̂_1
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, pos_delta: Tensor, h: Tensor) -> Tensor:
        """
        Args:
            pos_delta: [N, 3] coordinate delta from backbone (equivariant)
            h:         [N, hidden] node scalar features (invariant)
        Returns:
            [N, 3] equivariant position prediction
        """
        scale = self.gate(h)  # [N, 1] scalar
        return pos_delta * scale  # [N, 3] equivariant
