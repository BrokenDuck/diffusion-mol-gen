import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


# QM9 heavy-atom atomic numbers → 0-indexed type
ATOM_TYPE_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F
CHARGE_OFFSET = 2  # formal charges in [-2, 3] mapped to [0, 5]


class MapAtomTypes(BaseTransform):
    """Map atomic numbers (z) to contiguous atom_type indices."""

    def __call__(self, data: Data) -> Data:
        data.atom_type = torch.tensor(
            [ATOM_TYPE_MAP[z.item()] for z in data.z], dtype=torch.long
        )
        return data


class MapCharges(BaseTransform):
    """Map formal charges to non-negative categorical indices."""

    def __call__(self, data: Data) -> Data:
        data.charge = (data.charge + CHARGE_OFFSET).clamp(0, 5)
        return data


class CenterPositions(BaseTransform):
    """Subtract center of mass so positions have zero mean."""

    def __call__(self, data: Data) -> Data:
        assert data.pos is not None
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data


class NormalizePositions(BaseTransform):
    """Scale positions by a fixed standard deviation for training stability."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, data: Data) -> Data:
        assert data.pos is not None
        data.pos = data.pos / self.scale
        return data
