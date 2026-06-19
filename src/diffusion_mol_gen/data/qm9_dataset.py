from torch_geometric.data.data import Data
from tqdm import tqdm
from pathlib import Path
from torch_geometric.datasets import QM9
from rdkit import Chem
from torch_geometric.transforms import BaseTransform
import torch

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


class MakeFullyConnected(BaseTransform):
    def __call__(self, data: Data):
        num_nodes = data.num_nodes
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


class GenQM9(QM9):
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return ["gen_qm9_processed.pt"]

    def process(self):
        path = Path().cwd() / "data" / "qgb9.sdf"
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)

        data_list = []
        for mol in tqdm(supplier, desc="Procesing QM9 Dataset"):
            if mol is None:
                continue

            # Clean up molecule
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)
                cleaned_mol = Chem.RemoveHs(mol, sanitize=False)
            except ValueError:
                continue

            if len(Chem.GetMolFrags(cleaned_mol)) > 1:
                continue

            # Extract node features
            pos = []
            atom_types = []
            formal_charges = []

            conf = cleaned_mol.GetConformer()
            for atom in cleaned_mol.GetAtoms():
                idx = atom.GetIdx()
                pos.append(list(conf.GetAtomPosition(idx)))
                atom_types.append(atom.GetAtomicNum())
                formal_charges.append(atom.GetFormalCharge())

            pos = torch.tensor(pos, dtype=torch.float)
            z = torch.tensor(atom_types, dtype=torch.long)
            charges = torch.tensor(formal_charges, dtype=torch.long)

            # Extract edges and edge features
            edge_indices = []
            edge_attrs = []

            for bond in cleaned_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 1)

                edge_indices.extend([[i, j], [j, i]])
                edge_attrs.extend([bond_type, bond_type])

            if len(edge_indices) > 0:
                edge_index = (
                    torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                )
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
            else:
                # Handle single-heavy-atom molecules (e.g., Methane after H-removal becomes just C)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0,), dtype=torch.long)

            data = Data(
                pos=pos, z=z, charge=charges, edge_index=edge_index, edge_attr=edge_attr
            )

            # Apply optional PyG filters and transforms
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Save the fully processed dataset to disk
        torch.save(self.collate(data_list), self.processed_paths[0])
