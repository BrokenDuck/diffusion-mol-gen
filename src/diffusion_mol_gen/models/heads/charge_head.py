import torch.nn as nn
from torch import Tensor


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
