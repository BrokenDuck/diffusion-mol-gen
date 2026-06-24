"""
Neural network to denoise the molecule throughout the generations process.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter

from diffusion_mol_gen.configs import ModelConfig


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by 2-layer MLP projection."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.half_embed_dim = embed_dim // 2
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
        # Frequencies are log-spaced to allow different resolution levels
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(self.half_embed_dim, device=t.device, dtype=t.dtype)
            / self.half_embed_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]
        return self.mlp(emb)


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
    E(n) Equivariant Graph Neural Network layer.

    See https://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf

    For each directed edge (i→j) in edge_index, row=i and col=j.
    Messages m_ij are computed from (h_i, h_j, ||x_i-x_j||², e_ij) and
    aggregated back at node i (scatter onto row), so:

        m_ij  = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)
        x_i'  = x_i + Σ_j  norm(x_i - x_j) · φ_x(m_ij)
        h_i'  = LayerNorm( h_i + φ_h(h_i, Σ_j m_ij) )
    """

    def __init__(self, hidden_channels: int, edge_feat_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden_channels

        edge_in_dim = 2 * hidden_channels + 1 + edge_feat_dim

        # φ_e = 2-layer MLP
        # We use a final SiLU because it feeds into another MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_in_dim * 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(edge_in_dim * 2, hidden_channels),
            nn.SiLU(),
        )

        # Soft attention gate on messages
        # Suppresses uninformative messages
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

        # φ_x = scalar coordinate weight per message
        # How much node i move towards/away from node j
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, 1, bias=False),
        )

        # Normalise relative coordinates before weighting
        # Prevents coordinate explosion
        self.coors_norm = CoorsNorm()

        # φ_h = 2-layer node update MLP
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
        # Compute geometric quantities
        row, col = edge_index  # row=i, col=j
        diff = pos[row] - pos[col]  # [E, 3]
        dist_sq = (diff**2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Compute messages m_ij = φ_e(h_i, h_j, ||x_i-x_j||², e_ij)
        edge_in = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=-1)  # [E, 2H+1+H]
        m = self.edge_mlp(edge_in)  # [E, H]
        m = m * self.att_mlp(m)  # [E, H]

        # Equivariant coordinate update
        norm_diff = self.coors_norm(diff)  # [E, 3]
        coord_weight = self.coord_mlp(m)  # [E, 1]
        coord_agg = scatter(
            norm_diff * coord_weight,
            row,
            dim=0,
            dim_size=h.size(0),
            reduce="sum",
        )  # [N, 3]
        pos_out = pos + coord_agg  # Residual connection

        # Invariant node update
        h_agg = scatter(m, row, dim=0, dim_size=h.size(0), reduce="sum")  # [N, H]
        h_out = self.dropout(self.node_mlp(torch.cat([h, h_agg], dim=-1)))  # [N, H]
        h_out = self.norm(h + h_out)  # Residual connection

        # The residual connection is redundand because the node_mlp already has access to h.
        # However, the inialization of node_mlp is zero, meaning the residual connection ensures the gradient flow correctly.

        # We only apply dropout to node_mlp.
        # On the edge_mlp it is too destructive as early in the computation.
        # On the coord_mlp is does not make sense as it just causes instability.

        return h_out, pos_out


class MolGNN(nn.Module):
    """
    Equivariant molecular GNN backbone.

    Embeds noisy features, runs EGNN message passing, and returns updated
    node scalars, coordinate delta, and edge features for prediction heads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        H = config.hidden_channels

        # +1 on each embedding for the mask token used in flow/score views
        self.atom_embed = nn.Embedding(config.num_atom_types + 1, H)
        self.charge_embed = nn.Embedding(config.num_charges + 1, H)
        self.bond_embed = nn.Embedding(config.num_bond_types + 1, H)

        self.time_embed = TimeEmbedding(config.time_embed_dim, H)

        # Input projection after concatenating all node embeddings + time
        self.input_proj = nn.Sequential(
            nn.Linear(3 * H, H),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList(
            [EGNNLayer(H, H, dropout=config.dropout) for _ in range(config.num_layers)]
        )

        # Edge feature update MLP applied after each layer
        self.edge_update = nn.Sequential(
            nn.Linear(2 * H + H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

    def forward(
        self,
        pos_t: Tensor,
        atom_type_t: Tensor,
        charge_t: Tensor,
        bond_order_t: Tensor,
        edge_index: Tensor,
        t: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            pos_t:        [N, 3] noisy positions
            atom_type_t:  [N] noisy atom type indices
            charge_t:     [N] noisy charge indices
            bond_order_t: [E] noisy bond order indices
            edge_index:   [2, E]
            t:            [B] timestep per molecule (int or float)
            batch:        [N] batch index per node
        Returns:
            h:      [N, hidden] final node features
            pos:    [N, 3] updated positions
            e:      [E, hidden] final edge features
        """
        # Time embedding broadcast to nodes
        t_emb = self.time_embed(t.float())  # [B, H]
        t_node = t_emb[batch]  # [N, H]

        # Embed all the categorical variables and build embedding vector
        h_atom = self.atom_embed(atom_type_t)  # [N, H]
        h_charge = self.charge_embed(charge_t)  # [N, H]
        e = self.bond_embed(bond_order_t)  # [E, H]
        h = self.input_proj(torch.cat([h_atom, h_charge, t_node], dim=-1))  # [N, H]

        # Forward pass through model
        pos = pos_t
        row, col = edge_index
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, e)
            # Update edge features with updated node features
            e_in = torch.cat([h[row], h[col], e], dim=-1)
            e += self.edge_update(e_in)  # Skip connection

        return h, pos, e
