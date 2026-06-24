import torch
from torch import Tensor

from diffusion_mol_gen.models.denoiser import Denoiser
from diffusion_mol_gen.diffusion.unified import BaseDiffusion
from diffusion_mol_gen.diffusion.continuous.variational import VariationalContinuous
from diffusion_mol_gen.diffusion.categorical.d3pm import D3PM
from diffusion_mol_gen.sampling.utils import build_fully_connected


class VariationalSampler:
    """
    DDPM reverse-process sampler.

    Iterates from t=T-1 down to t=0, predicting x̂_0 at each step and
    computing the posterior mean for the reverse step.
    """

    def __init__(self, denoiser: Denoiser, diffusion: BaseDiffusion):
        self.denoiser = denoiser
        self.diffusion = diffusion
        assert isinstance(diffusion.continuous, VariationalContinuous)
        assert isinstance(diffusion.cat_atom, D3PM)
        self.T = diffusion.T

    @torch.no_grad()
    def sample(
        self,
        num_atoms_list: list[int],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generate molecules.

        Args:
            num_atoms_list: list of atom counts per molecule
            device: target device
        Returns:
            pos, atom_type, charge, bond_order, edge_index, batch
        """
        N = sum(num_atoms_list)
        B = len(num_atoms_list)

        batch = torch.repeat_interleave(
            torch.arange(B, device=device),
            torch.tensor(num_atoms_list, device=device),
        )

        edge_index, edge_batch = build_fully_connected(num_atoms_list, device)
        E = edge_index.shape[1]

        # Initialise from prior
        pos = torch.randn(N, 3, device=device)
        atom = torch.randint(0, 5, (N,), device=device)
        charge = torch.randint(0, 6, (N,), device=device)
        bond = torch.randint(0, 5, (E,), device=device)

        cont = self.diffusion.continuous
        cat_atom = self.diffusion.cat_atom
        cat_charge = self.diffusion.cat_charge
        cat_bond = self.diffusion.cat_bond

        assert isinstance(cont, VariationalContinuous)
        assert isinstance(cat_atom, D3PM)
        assert isinstance(cat_charge, D3PM)
        assert isinstance(cat_bond, D3PM)

        for t_int in reversed(range(self.T)):
            t_graph = torch.full((B,), t_int, dtype=torch.long, device=device)
            t_node = t_graph[batch]
            t_edge = t_graph[edge_batch]

            pred_pos, pred_atom, pred_charge, pred_bond = self.denoiser(
                pos, atom, charge, bond, edge_index, t_graph, batch
            )

            pos = cont.p_sample(pred_pos, pos, t_node)
            atom = cat_atom.p_sample(pred_atom, atom, t_node)
            charge = cat_charge.p_sample(pred_charge, charge, t_node)
            bond = cat_bond.p_sample(pred_bond, bond, t_edge)

        return pos, atom, charge, bond, edge_index, batch
