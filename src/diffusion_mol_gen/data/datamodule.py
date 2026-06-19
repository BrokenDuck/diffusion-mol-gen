import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from diffusion_mol_gen.configs.base import TrainingConfig
from diffusion_mol_gen.data.qm9_dataset import GenQM9
from diffusion_mol_gen.data.transforms import (
    MapAtomTypes,
    MapCharges,
    CenterPositions,
    MakeFullyConnected,
)


class QM9DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for QM9 molecule generation."""

    # Standard QM9 split sizes
    TRAIN_SIZE = 100000
    VAL_SIZE = 18000

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        transform = Compose(
            [
                MapAtomTypes(),
                MapCharges(),
                CenterPositions(),
                MakeFullyConnected(),
            ]
        )

        self.dataset = GenQM9(
            root=str(self.config.dataset_root),
            transform=transform,
        )

        n = len(self.dataset)
        train_end = min(self.TRAIN_SIZE, n)
        val_end = min(train_end + self.VAL_SIZE, n)

        self.train_dataset = self.dataset[:train_end]
        self.val_dataset = self.dataset[train_end:val_end]
        self.test_dataset = self.dataset[val_end:]

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,  # ty:ignore[invalid-argument-type]
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,  # ty:ignore[invalid-argument-type]
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,  # ty:ignore[invalid-argument-type]
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    @property
    def atom_size_distribution(self) -> torch.Tensor:
        """Empirical distribution of number of atoms per molecule."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first")
        sizes = torch.tensor([d.num_nodes for d in self.train_dataset])  # ty:ignore[not-iterable, unresolved-attribute]
        counts = torch.bincount(sizes)
        return counts.float() / counts.sum()
