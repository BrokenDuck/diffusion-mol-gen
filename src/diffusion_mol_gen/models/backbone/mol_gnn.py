import torch
import torch.nn as nn
from torch import Tensor

from diffusion_mol_gen.configs.base import ModelConfig
from diffusion_mol_gen.models.backbone.egnn import EGNNLayer
from diffusion_mol_gen.models.backbone.time_embedding import TimeEmbedding


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

        self.layers = nn.ModuleList([
            EGNNLayer(H, H, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])

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
        t_emb = self.time_embed(t.float())   # [B, H]
        t_node = t_emb[batch]               # [N, H]

        # Node input
        h_atom = self.atom_embed(atom_type_t)
        h_charge = self.charge_embed(charge_t)
        h = self.input_proj(torch.cat([h_atom, h_charge, t_node], dim=-1))

        # Edge input
        e = self.bond_embed(bond_order_t)

        pos = pos_t
        row, col = edge_index

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, e)
            # Update edge features with updated node features
            e_in = torch.cat([h[row], h[col], e], dim=-1)
            e = e + self.edge_update(e_in)  # residual

        return h, pos, e
