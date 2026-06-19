from __future__ import annotations

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from diffusion_mol_gen.configs.base import ModelConfig, DiffusionConfig, TrainingConfig
from diffusion_mol_gen.models.denoiser import Denoiser
from diffusion_mol_gen.diffusion.unified import UnifiedDiffusion
from diffusion_mol_gen.diffusion.continuous.variational import VariationalContinuous
from diffusion_mol_gen.diffusion.continuous.score_sde import ScoreSDE
from diffusion_mol_gen.diffusion.continuous.flow_matching import FlowMatchingContinuous
from diffusion_mol_gen.diffusion.categorical.absorbing import AbsorbingStateDiffusion
from diffusion_mol_gen.diffusion.categorical.ctmc import CTMCFlow
from diffusion_mol_gen.training.losses import position_loss, categorical_loss, score_matching_loss


class MolGenLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for molecular generation.

    Supports three diffusion views selected via DiffusionConfig.view:
      - "variational": DDPM + D3PM
      - "score":       VP-SDE + absorbing-state
      - "flow":        flow matching + CTMC
    """

    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.training_config = training_config

        self.denoiser = Denoiser(model_config)
        self.diffusion = UnifiedDiffusion(diffusion_config, model_config)

        self.ema = ExponentialMovingAverage(
            self.denoiser.parameters(), decay=training_config.ema_decay
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        loss, log_dict = self._shared_step(batch)
        self.log_dict({f"train/{k}": v for k, v in log_dict.items()}, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        with self.ema.average_parameters():
            loss, log_dict = self._shared_step(batch)
        self.log_dict({f"val/{k}": v for k, v in log_dict.items()}, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def _shared_step(self, batch):
        # Compute edge-level batch index
        edge_batch = batch.batch[batch.edge_index[0]]

        # Forward diffusion
        fb = self.diffusion.forward_process(
            batch.pos, batch.atom_type, batch.charge, batch.edge_attr,
            batch.batch, edge_batch,
        )

        # Network forward pass (timestep broadcast to graph level)
        pred_pos, pred_atom, pred_charge, pred_bond = self.denoiser(
            fb.pos_t, fb.atom_t, fb.charge_t, fb.bond_t,
            batch.edge_index, fb.t, batch.batch,
        )

        # Compute losses per view
        view = self.diffusion_config.view

        if view == "variational":
            assert fb.noise is not None
            loss_pos = position_loss(pred_pos, fb.noise, batch.batch)
            loss_atom = categorical_loss(pred_atom, batch.atom_type)
            loss_charge = categorical_loss(pred_charge, batch.charge)
            loss_bond = categorical_loss(pred_bond, batch.edge_attr)

        elif view == "score":
            assert fb.noise is not None and fb.std is not None
            loss_pos = score_matching_loss(pred_pos, fb.noise, fb.std)
            loss_atom = categorical_loss(pred_atom, batch.atom_type, fb.atom_mask)
            loss_charge = categorical_loss(pred_charge, batch.charge, fb.charge_mask)
            loss_bond = categorical_loss(pred_bond, batch.edge_attr, fb.bond_mask)

        else:  # flow
            assert fb.target_velocity is not None
            loss_pos = position_loss(pred_pos, fb.target_velocity, batch.batch)
            loss_atom = categorical_loss(pred_atom, batch.atom_type, fb.atom_mask)
            loss_charge = categorical_loss(pred_charge, batch.charge, fb.charge_mask)
            loss_bond = categorical_loss(pred_bond, batch.edge_attr, fb.bond_mask)

        tc = self.training_config
        loss = (
            tc.loss_weight_pos * loss_pos
            + tc.loss_weight_atom * loss_atom
            + tc.loss_weight_charge * loss_charge
            + tc.loss_weight_bond * loss_bond
        )

        log_dict = {
            "loss": loss,
            "loss_pos": loss_pos,
            "loss_atom": loss_atom,
            "loss_charge": loss_charge,
            "loss_bond": loss_bond,
        }
        return loss, log_dict

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def on_before_zero_grad(self, optimizer):  # noqa: ARG002
        self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        tc = self.training_config
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
