from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

import torch
from torch import Tensor
from torch_geometric.data.data import Data
from torch_geometric.datasets import QM9


class QM9Data(Data):
    """Type checking graph class"""

    pos: Tensor  # 3D position of atoms
    z: Tensor  # Type of atoms
    charges: Tensor  # Formal charge of atoms
    edge_index: Tensor  # Bonds of the molecule
    edge_attr: Tensor  # Type of the bonds


class GenQM9(QM9):
    DATA_ROOT = Path.cwd() / "data" / "raw"

    BOND_TYPE_MAP = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }

    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return ["gen_qm9_processed.pt"]

    def _load_uncharacterized(self) -> set[int]:
        uncharacterized_file = self.DATA_ROOT / "uncharacterized.txt"
        indices = set()
        with uncharacterized_file.open() as f:
            for line in f:
                line = line.strip()
                if line and line[0].isdigit():
                    indices.add(int(line.split()[0]))
        return indices

    def process(self):
        logger = RDLogger.logger()
        logger.setLevel(RDLogger.ERROR)

        supplier = Chem.SDMolSupplier(
            str(self.DATA_ROOT / "gdb9.sdf"), sanitize=False, removeHs=False
        )
        uncharacterized = self._load_uncharacterized()

        data_list = []
        for idx, mol in enumerate(
            tqdm(supplier, desc="Procesing QM9 Dataset"), start=1
        ):
            if idx in uncharacterized and mol is None:
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
            pos, atom_types, formal_charges = [], [], []
            conf = cleaned_mol.GetConformer()
            for atom in cleaned_mol.GetAtoms():
                pos.append(list(conf.GetAtomPosition(atom.GetIdx())))
                atom_types.append(atom.GetAtomicNum())
                formal_charges.append(atom.GetFormalCharge())

            pos = torch.tensor(pos, dtype=torch.float)
            z = torch.tensor(atom_types, dtype=torch.long)
            charges = torch.tensor(formal_charges, dtype=torch.long)

            # Extract edges and edge features
            edge_indices, edge_attrs = [], []
            for bond in cleaned_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = self.BOND_TYPE_MAP.get(bond.GetBondType(), 1)

                edge_indices.extend([[i, j], [j, i]])  # Graph is undirected
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
