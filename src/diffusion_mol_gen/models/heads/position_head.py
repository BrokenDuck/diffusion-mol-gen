import torch
import torch.nn as nn
from torch import Tensor


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
        scale = self.gate(h)          # [N, 1] scalar — invariant
        return pos_delta * scale      # [N, 3] equivariant
