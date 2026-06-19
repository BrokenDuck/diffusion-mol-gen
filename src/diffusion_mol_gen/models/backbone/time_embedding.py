import math
import torch
import torch.nn as nn
from torch import Tensor


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by 2-layer MLP projection."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: [B] float timestep in [0, 1] or integer timestep
        Returns:
            [B, hidden_dim] time embeddings
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        # [B, half]
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]
        return self.mlp(emb)
