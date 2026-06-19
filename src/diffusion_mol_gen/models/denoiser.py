import torch
import torch.nn as nn
from torch import Tensor

from diffusion_mol_gen.configs.base import ModelConfig
from diffusion_mol_gen.models.backbone.mol_gnn import MolGNN
from diffusion_mol_gen.models.heads.position_head import PositionHead
from diffusion_mol_gen.models.heads.atom_type_head import AtomTypeHead
from diffusion_mol_gen.models.heads.charge_head import ChargeHead
from diffusion_mol_gen.models.heads.bond_order_head import BondOrderHead


class Denoiser(nn.Module):
    """
    Combines the equivariant GNN backbone with all prediction heads.

    Single forward pass returns predictions for all feature types.
    What these predictions represent depends on the diffusion view:
      - Variational: (x̂_0, â_0_logits, ĉ_0_logits, ê_0_logits)
      - Score-based: (score, â_logits, ĉ_logits, ê_logits)
      - Flow-based:  (velocity or x̂_1, â_1_logits, ĉ_1_logits, ê_1_logits)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        H = config.hidden_channels
        self.backbone = MolGNN(config)
        self.pos_head = PositionHead(H)
        self.atom_head = AtomTypeHead(H, config.num_atom_types)
        self.charge_head = ChargeHead(H, config.num_charges)
        self.bond_head = BondOrderHead(H, config.num_bond_types)

    def forward(
        self,
        pos_t: Tensor,
        atom_type_t: Tensor,
        charge_t: Tensor,
        bond_order_t: Tensor,
        edge_index: Tensor,
        t: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            pos_t:        [N, 3] noisy positions
            atom_type_t:  [N] noisy atom type indices
            charge_t:     [N] noisy charge indices
            bond_order_t: [E] noisy bond order indices
            edge_index:   [2, E]
            t:            [B] timestep per graph
            batch:        [N] batch index per node
        Returns:
            pred_pos:    [N, 3]
            pred_atom:   [N, num_atom_types]
            pred_charge: [N, num_charges]
            pred_bond:   [E, num_bond_types]
        """
        pos_in = pos_t

        h, pos_out, e = self.backbone(
            pos_t, atom_type_t, charge_t, bond_order_t, edge_index, t, batch
        )
        pos_delta = pos_out - pos_in

        pred_pos = self.pos_head(pos_delta, h)
        pred_atom = self.atom_head(h)
        pred_charge = self.charge_head(h)
        pred_bond = self.bond_head(h, edge_index, e)

        return pred_pos, pred_atom, pred_charge, pred_bond
