from typing import Literal
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Config for the EGNN model.

    See: https://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf
    """

    hidden_channels: int = 256
    num_layers: int = 6
    num_rbf: int = 64
    cutoff: float = 5.0
    max_neighbors: int = 32
    time_embed_dim: int = 128
    num_atom_types: int = 5  # H, C, N, O, F for QM9
    num_charges: int = 6  # -2,-1,0,+1,+2,+3 mapped to 0-5
    num_bond_types: int = 5  # none(0), single(1), double(2), triple(3), aromatic(4)
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    """Config for the diffusion process

    We have three different views:
    - Variational: predict clean sample
    - Score: predict score
    - Flow: predict velocity
    """

    view: Literal["variational", "score", "flow"] = "variational"
    num_timesteps: int = 1000
    schedule_type: Literal["linear", "cosine"] = "cosine"
    # VP-SDE specific
    beta_min: float = 0.1
    beta_max: float = 20.0
    # Flow matching specific
    sigma_min: float = 1e-4
    # Categorical process
    cat_transition: Literal["uniform", "absorbing"] = "absorbing"


@dataclass
class TrainingConfig:
    """Config for the training."""

    dataset_root: Path = field(default_factory=lambda: Path.cwd() / "data")
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-12
    ema_decay: float = 0.999
    max_epochs: int = 500
    gradient_clip_val: float = 1.0
    warmup_steps: int = 1000
    loss_weight_pos: float = 1.0
    loss_weight_atom: float = 1.0
    loss_weight_charge: float = 1.0
    loss_weight_bond: float = 1.0
    num_workers: int = 4
    val_check_interval: float = 1.0
