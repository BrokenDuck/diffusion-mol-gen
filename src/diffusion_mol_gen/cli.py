import math
import itertools

import torch
import click

from diffusion_mol_gen.training.lightning_module import MolGenLightningModule
from diffusion_mol_gen.sampling.utils import generated_to_rdkit, write_sdf

# Empirical atom count distribution for QM9 (H, C, N, O, F explicit).
# Index i = unnormalised probability of a molecule having i atoms total.
# QM9 contains molecules with 3-29 atoms; distribution peaks ~17-18.
# Hard-coded to avoid data dependency at inference time.
_QM9_ATOM_DIST = [
    0.0, 0.0, 0.0,         # 0-2 (impossible)
    0.001, 0.002, 0.005,   # 3-5
    0.010, 0.020, 0.030,   # 6-8
    0.040, 0.055, 0.070,   # 9-11
    0.085, 0.095, 0.100,   # 12-14
    0.095, 0.090, 0.085,   # 15-17
    0.080, 0.075, 0.060,   # 18-20
    0.040, 0.030, 0.020,   # 21-23
    0.010, 0.005, 0.003,   # 24-26
    0.002, 0.001, 0.001,   # 27-29
]
_QM9_ATOM_WEIGHTS = torch.tensor(_QM9_ATOM_DIST)


def _resolve_num_atoms(spec: str, num_molecules: int) -> list[int]:
    """Parse --num-atoms and return a list of per-molecule atom counts."""
    if spec == "sample":
        return torch.multinomial(_QM9_ATOM_WEIGHTS, num_molecules, replacement=True).tolist()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return torch.randint(int(lo), int(hi) + 1, (num_molecules,)).tolist()
    n = int(spec)
    return [n] * num_molecules


@click.group()
def cli():
    """Molecular generation with diffusion models."""


@cli.command()
@click.option("--view", type=click.Choice(["variational", "score", "flow"]), default="variational", show_default=True)
@click.option("--num-timesteps", type=int, default=1000, show_default=True)
@click.option("--schedule", type=click.Choice(["linear", "cosine"]), default="cosine", show_default=True)
@click.option("--max-epochs", type=int, default=500, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=2e-4, show_default=True)
@click.option("--hidden-channels", type=int, default=256, show_default=True)
@click.option("--num-layers", type=int, default=6, show_default=True)
@click.option("--data-root", type=str, default="./data", show_default=True)
@click.option("--wandb", is_flag=True, help="Enable Weights & Biases logging.")
@click.option("--wandb-project", type=str, default="diffusion-mol-gen", show_default=True)
def train(view, num_timesteps, schedule, max_epochs, batch_size, lr, hidden_channels, num_layers, data_root, wandb, wandb_project):
    """Train a molecule generation model."""
    torch.set_float32_matmul_precision("high")
    import pytorch_lightning as pl
    from pathlib import Path
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger
    from diffusion_mol_gen.data.datamodule import QM9DataModule
    from diffusion_mol_gen.configs import ModelConfig, DiffusionConfig, TrainingConfig  # noqa: PLC0415

    model_config = ModelConfig(hidden_channels=hidden_channels, num_layers=num_layers)
    diffusion_config = DiffusionConfig(view=view, num_timesteps=num_timesteps, schedule_type=schedule)
    training_config = TrainingConfig(dataset_root=Path(data_root), batch_size=batch_size, lr=lr, max_epochs=max_epochs)

    datamodule = QM9DataModule(training_config)
    module = MolGenLightningModule(model_config, diffusion_config, training_config)

    callbacks = [
        ModelCheckpoint(monitor="val/loss", save_top_k=3, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = None
    if wandb:
        logger = WandbLogger(project=wandb_project, name=f"mol-gen-{view}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=training_config.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(module, datamodule)


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True, help="Path to .ckpt file.")
@click.option("--num-molecules", type=int, default=100, show_default=True, help="Number of molecules to generate.")
@click.option("--num-atoms", type=str, default="sample", show_default=True, help="Atoms per molecule: integer, range '5-15', or 'sample' (QM9 distribution).")
@click.option("--output", type=click.Path(), default="generated.sdf", show_default=True, help="Output SDF file path.")
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--batch-size", type=int, default=50, show_default=True, help="Molecules per generation batch.")
@click.option("--num-steps", type=int, default=None, help="Integration steps override (SDE/ODE only).")
@click.option("--corrector-steps", type=int, default=0, show_default=True, help="Langevin corrector steps (SDE only).")
@click.option("--snr", type=float, default=0.1, show_default=True, help="Corrector signal-to-noise ratio (SDE only).")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
def sample(checkpoint, num_molecules, num_atoms, output, device, batch_size, num_steps, corrector_steps, snr, seed):
    """Generate molecules from a trained checkpoint."""
    if seed is not None:
        torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    dev = torch.device(device if cuda_available or device == "cpu" else "cpu")
    if device == "cuda" and not cuda_available:
        click.echo("CUDA not available, falling back to CPU.", err=True)

    click.echo(f"Loading checkpoint: {checkpoint}")
    module = MolGenLightningModule.load_from_checkpoint(checkpoint, map_location=dev)
    module.eval()
    module.ema.copy_to(module.denoiser.parameters())

    sampler = module.diffusion.build_sampler(module.denoiser, num_steps, corrector_steps, snr)
    num_atoms_list = _resolve_num_atoms(num_atoms, num_molecules)

    num_batches = math.ceil(num_molecules / batch_size)
    click.echo(f"Generating {num_molecules} molecules (view={module.diffusion_config.view})...")

    all_mols = []
    for idx, batch_atoms in enumerate(itertools.batched(num_atoms_list, batch_size), 1):
        click.echo(f"  Batch {idx}/{num_batches} ({len(batch_atoms)} molecules)")
        pos, atom_type, charge, bond_order, edge_index, _ = sampler.sample(list(batch_atoms), dev)
        all_mols.extend(generated_to_rdkit(pos, atom_type, charge, bond_order, edge_index, list(batch_atoms)))

    n_written = write_sdf(all_mols, output)
    click.echo(f"\nDone. {n_written}/{num_molecules} valid molecules written to {output}")
    click.echo(f"Validity: {100 * n_written / num_molecules:.1f}%")
