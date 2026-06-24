from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from diffusion_mol_gen.configs import DiffusionConfig, ModelConfig
from diffusion_mol_gen.diffusion.noise_schedules import NoiseSchedule, VPSDEParams
from diffusion_mol_gen.diffusion.continuous.variational import VariationalContinuous
from diffusion_mol_gen.diffusion.continuous.score_sde import ScoreSDE
from diffusion_mol_gen.diffusion.continuous.flow_matching import FlowMatchingContinuous
from diffusion_mol_gen.diffusion.categorical.d3pm import D3PM
from diffusion_mol_gen.diffusion.categorical.absorbing import AbsorbingStateDiffusion
from diffusion_mol_gen.diffusion.categorical.ctmc import CTMCFlow
from diffusion_mol_gen.models.denoiser import Denoiser

type ContinuousDiffusion = VariationalContinuous | ScoreSDE | FlowMatchingContinuous
type DiscreteDiffusion = D3PM | AbsorbingStateDiffusion | CTMCFlow


@dataclass
class ForwardBatch:
    """All noisy features + metadata produced by the forward process."""

    pos_t: Tensor
    atom_t: Tensor
    charge_t: Tensor
    bond_t: Tensor
    t: Tensor  # [B] graph-level timestep
    t_node: Tensor  # [N] node-level timestep (broadcast)
    t_edge: Tensor  # [E] edge-level timestep (broadcast)
    t_cont: Tensor  # continuous t in [0,1] for SDE/flow (node-level)
    # View-specific auxiliary data
    noise: Tensor | None = None  # Gaussian noise (variational/score/flow)
    std: Tensor | None = None  # σ_t (score view)
    atom_mask: Tensor | None = None  # masked positions (absorbing/ctmc)
    charge_mask: Tensor | None = None
    bond_mask: Tensor | None = None
    target_velocity: Tensor | None = None  # flow-matching velocity target


def _zero_com_noise(pos: Tensor, batch: Tensor) -> Tensor:
    """Sample Gaussian noise projected onto the zero center-of-mass subspace."""
    from torch_geometric.utils import scatter

    noise = torch.randn_like(pos)
    # Subtract per-graph mean so noise preserves CoM=0
    mean = scatter(noise, batch, dim=0, reduce="mean")
    noise = noise - mean[batch]
    return noise


class BaseDiffusion(ABC):
    """
    Abstract base for unified diffusion processes.

    Combines a continuous process (positions) and three categorical processes
    (atom types, charges, bond orders). Subclasses implement view-specific
    forward process logic.
    """

    view: str
    T: int
    continuous: ContinuousDiffusion
    cat_atom: DiscreteDiffusion
    cat_charge: DiscreteDiffusion
    cat_bond: DiscreteDiffusion

    @abstractmethod
    def forward_process(
        self,
        pos: Tensor,  # [N, 3]
        atom: Tensor,  # [N]
        charge: Tensor,  # [N]
        bond: Tensor,  # [E]
        batch: Tensor,  # [N] graph index
        edge_batch: Tensor,  # [E] graph index
    ) -> ForwardBatch:
        """Apply forward noise to all features and return a ForwardBatch."""

    @abstractmethod
    def build_sampler(
        self,
        denoiser: Denoiser,
        num_steps: int | None = None,
        corrector_steps: int = 0,
        snr: float = 0.1,
    ):
        """Instantiate the correct sampler for this diffusion view."""

    def _sample_timesteps(
        self, batch: Tensor, edge_batch: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample graph-level timesteps and broadcast to nodes/edges/continuous."""
        B = int(batch.max().item()) + 1
        t = torch.randint(0, self.T, (B,), device=device)  # [B]
        t_node = t[batch]  # [N]
        t_edge = t[edge_batch]  # [E]
        t_cont = t_node.float() / self.T  # [N] in [0, 1]
        return t, t_node, t_edge, t_cont

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


class VariationalDiffusion(BaseDiffusion):
    """DDPM (continuous positions) + D3PM (categorical features)."""

    view = "variational"
    continuous: VariationalContinuous
    cat_atom: D3PM
    cat_charge: D3PM
    cat_bond: D3PM

    def __init__(self, diffusion_config: DiffusionConfig, model_config: ModelConfig):
        self.T = diffusion_config.num_timesteps
        schedule = NoiseSchedule(self.T, diffusion_config.schedule_type)
        self.continuous = VariationalContinuous(schedule)
        self.cat_atom = D3PM(
            model_config.num_atom_types, schedule, diffusion_config.cat_transition
        )
        self.cat_charge = D3PM(
            model_config.num_charges, schedule, diffusion_config.cat_transition
        )
        self.cat_bond = D3PM(
            model_config.num_bond_types, schedule, diffusion_config.cat_transition
        )

    def forward_process(
        self, pos, atom, charge, bond, batch, edge_batch
    ) -> ForwardBatch:
        t, t_node, t_edge, t_cont = self._sample_timesteps(
            batch, edge_batch, pos.device
        )
        noise = _zero_com_noise(pos, batch)
        pos_t, noise = self.continuous.q_sample(pos, t_node, noise=noise)
        atom_t = self.cat_atom.q_sample(atom, t_node)
        charge_t = self.cat_charge.q_sample(charge, t_node)
        bond_t = self.cat_bond.q_sample(bond, t_edge)
        return ForwardBatch(
            pos_t=pos_t,
            atom_t=atom_t,
            charge_t=charge_t,
            bond_t=bond_t,
            t=t,
            t_node=t_node,
            t_edge=t_edge,
            t_cont=t_cont,
            noise=noise,
        )

    def build_sampler(
        self, denoiser: Denoiser, num_steps=None, corrector_steps=0, snr=0.1
    ):  # noqa: ARG002
        from diffusion_mol_gen.sampling import VariationalSampler

        return VariationalSampler(denoiser, self)


class ScoreDiffusion(BaseDiffusion):
    """VP-SDE (continuous positions) + absorbing-state diffusion (categorical features)."""

    view = "score"
    continuous: ScoreSDE
    cat_atom: AbsorbingStateDiffusion
    cat_charge: AbsorbingStateDiffusion
    cat_bond: AbsorbingStateDiffusion

    def __init__(self, diffusion_config: DiffusionConfig, model_config: ModelConfig):
        self.T = diffusion_config.num_timesteps
        schedule = NoiseSchedule(self.T, diffusion_config.schedule_type)
        sde = VPSDEParams(diffusion_config.beta_min, diffusion_config.beta_max)
        self.continuous = ScoreSDE(sde)
        self.cat_atom = AbsorbingStateDiffusion(model_config.num_atom_types, schedule)
        self.cat_charge = AbsorbingStateDiffusion(model_config.num_charges, schedule)
        self.cat_bond = AbsorbingStateDiffusion(model_config.num_bond_types, schedule)

    def forward_process(
        self, pos, atom, charge, bond, batch, edge_batch
    ) -> ForwardBatch:
        t, t_node, t_edge, t_cont = self._sample_timesteps(
            batch, edge_batch, pos.device
        )
        noise = _zero_com_noise(pos, batch)
        pos_t, noise, std = self.continuous.q_sample(pos, t_cont, noise=noise)
        atom_t, atom_mask = self.cat_atom.q_sample(atom, t_node)
        charge_t, charge_mask = self.cat_charge.q_sample(charge, t_node)
        bond_t, bond_mask = self.cat_bond.q_sample(bond, t_edge)
        return ForwardBatch(
            pos_t=pos_t,
            atom_t=atom_t,
            charge_t=charge_t,
            bond_t=bond_t,
            t=t,
            t_node=t_node,
            t_edge=t_edge,
            t_cont=t_cont,
            noise=noise,
            std=std,
            atom_mask=atom_mask,
            charge_mask=charge_mask,
            bond_mask=bond_mask,
        )

    def build_sampler(
        self, denoiser: Denoiser, num_steps=None, corrector_steps=0, snr=0.1
    ):
        from diffusion_mol_gen.sampling import SDESampler

        return SDESampler(denoiser, self, num_steps or 500, corrector_steps, snr)


class FlowDiffusion(BaseDiffusion):
    """OT flow matching (continuous positions) + CTMC masking (categorical features)."""

    view = "flow"
    continuous: FlowMatchingContinuous
    cat_atom: CTMCFlow
    cat_charge: CTMCFlow
    cat_bond: CTMCFlow

    def __init__(self, diffusion_config: DiffusionConfig, model_config: ModelConfig):
        self.T = diffusion_config.num_timesteps
        self.continuous = FlowMatchingContinuous(diffusion_config.sigma_min)
        self.cat_atom = CTMCFlow(model_config.num_atom_types)
        self.cat_charge = CTMCFlow(model_config.num_charges)
        self.cat_bond = CTMCFlow(model_config.num_bond_types)

    def forward_process(
        self, pos, atom, charge, bond, batch, edge_batch
    ) -> ForwardBatch:
        t, t_node, t_edge, t_cont = self._sample_timesteps(
            batch, edge_batch, pos.device
        )
        noise = _zero_com_noise(pos, batch)
        pos_t, noise, target_velocity = self.continuous.interpolate(pos, t_cont, noise=noise)
        atom_t, atom_mask = self.cat_atom.interpolate(atom, t_cont)
        charge_t, charge_mask = self.cat_charge.interpolate(charge, t_cont)
        # edge_batch broadcasts node-level t_cont to edges
        bond_t, bond_mask = self.cat_bond.interpolate(bond, t_cont[edge_batch])
        return ForwardBatch(
            pos_t=pos_t,
            atom_t=atom_t,
            charge_t=charge_t,
            bond_t=bond_t,
            t=t,
            t_node=t_node,
            t_edge=t_edge,
            t_cont=t_cont,
            noise=noise,
            target_velocity=target_velocity,
            atom_mask=atom_mask,
            charge_mask=charge_mask,
            bond_mask=bond_mask,
        )

    def build_sampler(
        self, denoiser: Denoiser, num_steps=None, corrector_steps=0, snr=0.1
    ):  # noqa: ARG002
        from diffusion_mol_gen.sampling import ODESampler

        return ODESampler(denoiser, self, num_steps or 100)


def make_diffusion(
    diffusion_config: DiffusionConfig, model_config: ModelConfig
) -> BaseDiffusion:
    """Instantiate the correct diffusion class for the given config."""
    classes = {
        "variational": VariationalDiffusion,
        "score": ScoreDiffusion,
        "flow": FlowDiffusion,
    }
    cls = classes.get(diffusion_config.view)
    if cls is None:
        raise ValueError(f"Unknown view: {diffusion_config.view!r}")
    return cls(diffusion_config, model_config)
