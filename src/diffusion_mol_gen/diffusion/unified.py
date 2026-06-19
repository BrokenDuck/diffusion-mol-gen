from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from diffusion_mol_gen.configs.base import DiffusionConfig, ModelConfig
from diffusion_mol_gen.diffusion.noise_schedules import NoiseSchedule, VPSDEParams
from diffusion_mol_gen.diffusion.continuous.variational import VariationalContinuous
from diffusion_mol_gen.diffusion.continuous.score_sde import ScoreSDE
from diffusion_mol_gen.diffusion.continuous.flow_matching import FlowMatchingContinuous
from diffusion_mol_gen.diffusion.categorical.d3pm import D3PM
from diffusion_mol_gen.diffusion.categorical.absorbing import AbsorbingStateDiffusion
from diffusion_mol_gen.diffusion.categorical.ctmc import CTMCFlow


@dataclass
class ForwardBatch:
    """All noisy features + metadata produced by the forward process."""
    pos_t: Tensor
    atom_t: Tensor
    charge_t: Tensor
    bond_t: Tensor
    t: Tensor               # [B] graph-level timestep
    t_node: Tensor          # [N] node-level timestep (broadcast)
    t_edge: Tensor          # [E] edge-level timestep (broadcast)
    t_cont: Tensor          # continuous t in [0,1] for SDE/flow (node-level)
    # View-specific auxiliary data
    noise: Tensor | None           = None   # Gaussian noise (variational/score/flow)
    std: Tensor | None             = None   # σ_t (score view)
    atom_mask: Tensor | None       = None   # masked positions (absorbing/ctmc)
    charge_mask: Tensor | None     = None
    bond_mask: Tensor | None       = None
    target_velocity: Tensor | None = None   # flow-matching velocity target


class UnifiedDiffusion:
    """
    Combines a continuous diffusion process (for positions) and three categorical
    processes (atom types, charges, bond orders) into a single unified forward/reverse
    interface parameterised by `view`.

    view = "variational" → DDPM (continuous) + D3PM (categorical)
    view = "score"       → VP-SDE (continuous) + AbsorbingState (categorical)
    view = "flow"        → FlowMatching (continuous) + CTMC (categorical)
    """

    def __init__(self, diffusion_config: DiffusionConfig, model_config: ModelConfig):
        self.view = diffusion_config.view
        self.T = diffusion_config.num_timesteps

        schedule = NoiseSchedule(self.T, diffusion_config.schedule_type)
        self.schedule = schedule

        Ka = model_config.num_atom_types
        Kc = model_config.num_charges
        Kb = model_config.num_bond_types

        if self.view == "variational":
            self.continuous: Union[VariationalContinuous, ScoreSDE, FlowMatchingContinuous] = VariationalContinuous(schedule)
            self.cat_atom: Union[D3PM, AbsorbingStateDiffusion, CTMCFlow] = D3PM(Ka, schedule, diffusion_config.cat_transition)
            self.cat_charge: Union[D3PM, AbsorbingStateDiffusion, CTMCFlow] = D3PM(Kc, schedule, diffusion_config.cat_transition)
            self.cat_bond: Union[D3PM, AbsorbingStateDiffusion, CTMCFlow] = D3PM(Kb, schedule, diffusion_config.cat_transition)

        elif self.view == "score":
            sde = VPSDEParams(diffusion_config.beta_min, diffusion_config.beta_max)
            self.continuous = ScoreSDE(sde)
            self.cat_atom = AbsorbingStateDiffusion(Ka, schedule)
            self.cat_charge = AbsorbingStateDiffusion(Kc, schedule)
            self.cat_bond = AbsorbingStateDiffusion(Kb, schedule)

        elif self.view == "flow":
            self.continuous = FlowMatchingContinuous(diffusion_config.sigma_min)
            self.cat_atom = CTMCFlow(Ka)
            self.cat_charge = CTMCFlow(Kc)
            self.cat_bond = CTMCFlow(Kb)

        else:
            raise ValueError(f"Unknown view: {self.view}")

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def forward_process(
        self,
        pos: Tensor,        # [N, 3]
        atom: Tensor,       # [N]
        charge: Tensor,     # [N]
        bond: Tensor,       # [E]
        batch: Tensor,      # [N] graph index
        edge_batch: Tensor, # [E] graph index
    ) -> ForwardBatch:
        """Apply forward noise to all features and return a ForwardBatch."""
        device = pos.device
        B = batch.max().item() + 1  # type: ignore[operator]

        # Sample graph-level integer timesteps
        t = torch.randint(0, self.T, (int(B),), device=device)  # [B]
        t_node = t[batch]       # [N]
        t_edge = t[edge_batch]  # [E]

        # Continuous time in [0,1] for SDE / flow
        t_cont = t_node.float() / self.T

        if self.view == "variational":
            assert isinstance(self.continuous, VariationalContinuous)
            pos_t, noise = self.continuous.q_sample(pos, t_node)
            atom_t = self.cat_atom.q_sample(atom, t_node)          # type: ignore[union-attr]
            charge_t = self.cat_charge.q_sample(charge, t_node)    # type: ignore[union-attr]
            bond_t = self.cat_bond.q_sample(bond, t_edge)          # type: ignore[union-attr]
            return ForwardBatch(
                pos_t=pos_t, atom_t=atom_t, charge_t=charge_t, bond_t=bond_t,
                t=t, t_node=t_node, t_edge=t_edge, t_cont=t_cont, noise=noise,
            )

        elif self.view == "score":
            assert isinstance(self.continuous, ScoreSDE)
            pos_t, noise, std = self.continuous.q_sample(pos, t_cont)
            atom_t, atom_mask = self.cat_atom.q_sample(atom, t_node)        # type: ignore[union-attr]
            charge_t, charge_mask = self.cat_charge.q_sample(charge, t_node)  # type: ignore[union-attr]
            bond_t, bond_mask = self.cat_bond.q_sample(bond, t_edge)        # type: ignore[union-attr]
            return ForwardBatch(
                pos_t=pos_t, atom_t=atom_t, charge_t=charge_t, bond_t=bond_t,
                t=t, t_node=t_node, t_edge=t_edge, t_cont=t_cont,
                noise=noise, std=std,
                atom_mask=atom_mask, charge_mask=charge_mask, bond_mask=bond_mask,
            )

        else:  # flow
            assert isinstance(self.continuous, FlowMatchingContinuous)
            pos_t, noise, target_velocity = self.continuous.interpolate(pos, t_cont)
            atom_t, atom_mask = self.cat_atom.interpolate(atom, t_cont)        # type: ignore[union-attr]
            charge_t, charge_mask = self.cat_charge.interpolate(charge, t_cont)  # type: ignore[union-attr]
            bond_t, bond_mask = self.cat_bond.interpolate(bond, t_cont[edge_batch])  # type: ignore[union-attr]
            # edge_batch needed to broadcast t to edges for flow
            return ForwardBatch(
                pos_t=pos_t, atom_t=atom_t, charge_t=charge_t, bond_t=bond_t,
                t=t, t_node=t_node, t_edge=t_edge, t_cont=t_cont,
                noise=noise, target_velocity=target_velocity,
                atom_mask=atom_mask, charge_mask=charge_mask, bond_mask=bond_mask,
            )

    # ------------------------------------------------------------------
    # Mask-token indices for initialising sampling
    # ------------------------------------------------------------------

    @property
    def atom_mask_idx(self) -> int:
        if isinstance(self.cat_atom, (AbsorbingStateDiffusion, CTMCFlow)):
            return self.cat_atom.mask_idx
        raise AttributeError("No mask index for D3PM")

    @property
    def charge_mask_idx(self) -> int:
        if isinstance(self.cat_charge, (AbsorbingStateDiffusion, CTMCFlow)):
            return self.cat_charge.mask_idx
        raise AttributeError("No mask index for D3PM")

    @property
    def bond_mask_idx(self) -> int:
        if isinstance(self.cat_bond, (AbsorbingStateDiffusion, CTMCFlow)):
            return self.cat_bond.mask_idx
        raise AttributeError("No mask index for D3PM")
