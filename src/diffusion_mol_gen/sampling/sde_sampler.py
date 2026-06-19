from __future__ import annotations

import torch
from torch import Tensor

from diffusion_mol_gen.models.denoiser import Denoiser
from diffusion_mol_gen.diffusion.unified import UnifiedDiffusion
from diffusion_mol_gen.diffusion.continuous.score_sde import ScoreSDE
from diffusion_mol_gen.diffusion.categorical.absorbing import AbsorbingStateDiffusion
from diffusion_mol_gen.sampling.utils import build_fully_connected


class SDESampler:
    """
    Euler-Maruyama reverse SDE sampler for the score-based view.

    Optionally applies Langevin corrector steps at each noise level.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        diffusion: UnifiedDiffusion,
        num_steps: int = 500,
        corrector_steps: int = 0,
        snr: float = 0.1,
    ):
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.num_steps = num_steps
        self.corrector_steps = corrector_steps
        self.snr = snr

        assert isinstance(diffusion.continuous, ScoreSDE)
        assert isinstance(diffusion.cat_atom, AbsorbingStateDiffusion)

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
        edge_index, _ = build_fully_connected(num_atoms_list, device)
        E = edge_index.shape[1]

        # Sample from prior: N(0, I) for positions
        pos = torch.randn(N, 3, device=device)
        mask_atom = self.diffusion.atom_mask_idx
        mask_charge = self.diffusion.charge_mask_idx
        mask_bond = self.diffusion.bond_mask_idx
        atom = torch.full((N,), mask_atom, device=device)
        charge = torch.full((N,), mask_charge, device=device)
        bond = torch.full((E,), mask_bond, device=device)

        dt = 1.0 / self.num_steps
        cont = self.diffusion.continuous
        cat_atom = self.diffusion.cat_atom
        cat_charge = self.diffusion.cat_charge
        cat_bond = self.diffusion.cat_bond

        assert isinstance(cont, ScoreSDE)
        assert isinstance(cat_atom, AbsorbingStateDiffusion)
        assert isinstance(cat_charge, AbsorbingStateDiffusion)
        assert isinstance(cat_bond, AbsorbingStateDiffusion)

        T = self.diffusion.T

        for i in range(self.num_steps):
            t_scalar = 1.0 - i * dt
            t_node_float = torch.full((N,), t_scalar, device=device)

            # Integer timestep for categorical reverse
            t_int_val = max(0, int(t_scalar * T) - 1)
            t_node_int = torch.full((N,), t_int_val, dtype=torch.long, device=device)
            t_edge_int = torch.full((E,), t_int_val, dtype=torch.long, device=device)
            t_graph = torch.full((B,), t_int_val, dtype=torch.long, device=device)

            pred_pos, pred_atom, pred_charge, pred_bond = self.denoiser(
                pos, atom, charge, bond, edge_index, t_graph, batch
            )

            pos = cont.reverse_em_step(pred_pos, pos, t_node_float, dt)

            atom = cat_atom.p_sample(pred_atom, atom, t_node_int)
            charge = cat_charge.p_sample(pred_charge, charge, t_node_int)
            bond = cat_bond.p_sample(pred_bond, bond, t_edge_int)

            # Optional Langevin corrector for positions
            for _ in range(self.corrector_steps):
                noise = torch.randn_like(pos)
                grad_norm = pred_pos.norm()
                noise_norm = noise.norm()
                step_size = 2 * (self.snr * noise_norm / (grad_norm + 1e-8)) ** 2
                pos = pos + step_size * pred_pos + (2 * step_size) ** 0.5 * noise

        return pos, atom, charge, bond, edge_index, batch
