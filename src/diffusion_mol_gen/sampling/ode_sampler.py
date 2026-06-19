from __future__ import annotations

import torch
from torch import Tensor

from diffusion_mol_gen.models.denoiser import Denoiser
from diffusion_mol_gen.diffusion.unified import UnifiedDiffusion
from diffusion_mol_gen.diffusion.continuous.flow_matching import FlowMatchingContinuous
from diffusion_mol_gen.diffusion.categorical.ctmc import CTMCFlow
from diffusion_mol_gen.sampling.utils import build_fully_connected


class ODESampler:
    """
    Euler ODE integration sampler for the flow-based view.

    Integrates from t=0 (all masked / noise) to t=1 (data).
    """

    def __init__(
        self,
        denoiser: Denoiser,
        diffusion: UnifiedDiffusion,
        num_steps: int = 100,
    ):
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.num_steps = num_steps

        assert isinstance(diffusion.continuous, FlowMatchingContinuous)
        assert isinstance(diffusion.cat_atom, CTMCFlow)

    @torch.no_grad()
    def sample(
        self,
        num_atoms_list: list[int],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        N = sum(num_atoms_list)
        B = len(num_atoms_list)

        batch = torch.repeat_interleave(
            torch.arange(B, device=device),
            torch.tensor(num_atoms_list, device=device),
        )
        edge_index, edge_batch = build_fully_connected(num_atoms_list, device)
        E = edge_index.shape[1]

        # Prior: Gaussian noise for positions, all masked for categoricals
        pos = torch.randn(N, 3, device=device)
        atom = torch.full((N,), self.diffusion.atom_mask_idx, device=device)
        charge = torch.full((N,), self.diffusion.charge_mask_idx, device=device)
        bond = torch.full((E,), self.diffusion.bond_mask_idx, device=device)

        dt = 1.0 / self.num_steps
        cont = self.diffusion.continuous
        cat_atom = self.diffusion.cat_atom
        cat_charge = self.diffusion.cat_charge
        cat_bond = self.diffusion.cat_bond

        assert isinstance(cont, FlowMatchingContinuous)
        assert isinstance(cat_atom, CTMCFlow)
        assert isinstance(cat_charge, CTMCFlow)
        assert isinstance(cat_bond, CTMCFlow)

        for i in range(self.num_steps):
            t = i * dt
            t_graph = torch.full((B,), t, dtype=torch.float, device=device)

            pred_pos, pred_atom, pred_charge, pred_bond = self.denoiser(
                pos, atom, charge, bond, edge_index, t_graph, batch
            )

            # Convert endpoint prediction to velocity, then Euler step
            t_node = torch.full((N,), t, device=device)
            velocity = cont.pred_x0_to_velocity(pred_pos, pos, t_node)
            pos = cont.euler_step(velocity, pos, dt)

            atom = cat_atom.sample_step(pred_atom, atom, t, dt)
            charge = cat_charge.sample_step(pred_charge, charge, t, dt)
            bond = cat_bond.sample_step(pred_bond, bond, t, dt)

        return pos, atom, charge, bond, edge_index, batch
