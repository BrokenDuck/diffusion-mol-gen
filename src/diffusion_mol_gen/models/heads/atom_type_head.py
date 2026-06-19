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
