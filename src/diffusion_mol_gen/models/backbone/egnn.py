import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter


class CoorsNorm(nn.Module):
    """Normalise coordinate vectors by their L2 norm with a learnable scale."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, coors: Tensor) -> Tensor:
        norm = coors.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self.scale * coors / norm


class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network layer (Satorras et al. 2021).

    For each directed edge (i→j) in edge_index, row=i and col=j.
    Messages m_ij are computed from (h_i, h_j, ||x_i-x_j||², e_ij) and
    aggregated back at node i (scatter onto row), so:

        m_ij  = φ_e(h_i, h_j, ||x_i − x_j||², e_ij)   [SiLU final]
        x_i'  = x_i + Σ_j  norm(x_i − x_j) · φ_x(m_ij)
        h_i'  = LayerNorm( h_i + φ_h(h_i, Σ_j m_ij) )
    """

    def __init__(self, hidden_channels: int, edge_feat_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden_channels

        edge_in_dim = 2 * hidden_channels + 1 + edge_feat_dim

        # φ_e — 2-layer MLP, final SiLU (matches lucidrains reference)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_in_dim * 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(edge_in_dim * 2, hidden_channels),
            nn.SiLU(),
        )

        # Soft attention gate on messages
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

        # φ_x — scalar coordinate weight per message (matches lucidrains 2-layer style)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, 1, bias=False),
        )

        # Normalise relative coordinates before weighting (CoorsNorm)
        self.coors_norm = CoorsNorm()

        # φ_h — 2-layer node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels * 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )

        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            h:          [N, hidden] node features
            pos:        [N, 3] positions
            edge_index: [2, E]  row=i (central node), col=j (neighbour)
            edge_attr:  [E, edge_feat_dim]
        Returns:
            h_out:  [N, hidden]
            pos_out: [N, 3]
        """
        row, col = edge_index           # row=i, col=j
        diff = pos[row] - pos[col]      # [E, 3]  x_i − x_j
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # --- messages m_ij = φ_e(h_i, h_j, ||x_i−x_j||², e_ij) ---
        edge_in = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=-1)
        m = self.edge_mlp(edge_in)      # [E, hidden]
        m = m * self.att_mlp(m)         # soft gate

        # --- equivariant coordinate update (scatter onto i = row) ---
        norm_diff = self.coors_norm(diff)               # [E, 3]
        coord_weight = self.coord_mlp(m)                # [E, 1]
        coord_agg = scatter(
            norm_diff * coord_weight, row,
            dim=0, dim_size=h.size(0), reduce="sum",
        )
        pos_out = pos + coord_agg

        # --- invariant node update (aggregate at i = row) ---
        h_agg = scatter(m, row, dim=0, dim_size=h.size(0), reduce="sum")
        h_out = self.dropout(self.node_mlp(torch.cat([h, h_agg], dim=-1)))
        h_out = self.norm(h + h_out)

        return h_out, pos_out
