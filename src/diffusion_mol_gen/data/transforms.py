import torch
from torch_geometric.transforms import BaseTransform

from diffusion_mol_gen.data.qm9_dataset import QM9Data


class MapAtomTypes(BaseTransform):
    """Map atomic numbers (z) to contiguous atom_type indices."""

    # QM9 heavy-atom atomic numbers → 0-indexed type
    ATOM_TYPE_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F

    def __call__(self, data: QM9Data) -> QM9Data:
        data.atom_type = torch.tensor(
            [self.ATOM_TYPE_MAP[z.item()] for z in data.z], dtype=torch.long
        )
        return data


class MapCharges(BaseTransform):
    """Map formal charges to non-negative categorical indices."""

    CHARGE_OFFSET = 2  # formal charges in [-2, 3] mapped to [0, 5]

    def __call__(self, data: QM9Data) -> QM9Data:
        data.charge = (data.charge + self.CHARGE_OFFSET).clamp(0, 5)
        return data


class CenterPositions(BaseTransform):
    """Subtract center of mass so positions have zero mean."""

    def __call__(self, data: QM9Data) -> QM9Data:
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data


class NormalizePositions(BaseTransform):
    """Scale positions by a fixed standard deviation for training stability."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, data: QM9Data) -> QM9Data:
        data.pos = data.pos / self.scale
        return data


class MakeFullyConnected(BaseTransform):
    """Make graph fully connected by adding not bonded edges. The generative model has to predict those too."""

    def __call__(self, data: QM9Data):
        num_nodes = data.num_nodes
        assert num_nodes is not None
        device = data.pos.device

        # Transform to dense adjacacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long, device=device)
        if data.edge_index is not None and data.edge_index.numel() > 0:
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            adj[edge_index[0], edge_index[1]] = edge_attr

        # Transform to tuples and remove self loops
        row, col = torch.meshgrid(
            torch.arange(num_nodes, device=device),
            torch.arange(num_nodes, device=device),
            indexing="ij",
        )
        row = row.flatten()
        col = col.flatten()
        mask = row != col
        fc_edge_index = torch.stack([row[mask], col[mask]], dim=0)
        fc_edge_attr = adj[fc_edge_index[0], fc_edge_index[1]]

        data.edge_index = fc_edge_index
        data.edge_attr = fc_edge_attr
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
