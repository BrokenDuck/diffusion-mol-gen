from diffusion_mol_gen.training.lightning_module import MolGenLightningModule
from diffusion_mol_gen.data.datamodule import QM9DataModule
from diffusion_mol_gen.configs.base import ModelConfig, DiffusionConfig, TrainingConfig


def main() -> None:
    """Train a molecule generation model via CLI."""
    import argparse
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger

    parser = argparse.ArgumentParser(description="Molecular generation with diffusion models")
    parser.add_argument("--view", choices=["variational", "score", "flow"], default="variational")
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="diffusion-mol-gen")
    args = parser.parse_args()

    from pathlib import Path

    model_config = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    )
    diffusion_config = DiffusionConfig(
        view=args.view,
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule,
    )
    training_config = TrainingConfig(
        dataset_root=Path(args.data_root),
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
    )

    datamodule = QM9DataModule(training_config)
    module = MolGenLightningModule(model_config, diffusion_config, training_config)

    callbacks = [
        ModelCheckpoint(monitor="val/loss", save_top_k=3, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = None
    if args.wandb:
        logger = WandbLogger(project=args.wandb_project, name=f"mol-gen-{args.view}")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=training_config.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(module, datamodule)
