import torch
from torch import Tensor

from rdkit import Chem


# Inverse mapping: atom type index → atomic symbol
IDX_TO_ATOM = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}
# Bond order index → RDKit bond type
IDX_TO_BOND = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}
# Charge index → formal charge value
IDX_TO_CHARGE = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2, 5: 3}


def build_fully_connected(
    num_atoms_list: list[int], device: torch.device
) -> tuple[Tensor, Tensor]:
    """
    Build a fully-connected (no self-loops) edge index for a batch of molecules.

    Returns:
        edge_index: [2, E]
        edge_batch: [E] graph index per edge
    """
    rows, cols, batches = [], [], []
    offset = 0
    for b, n in enumerate(num_atoms_list):
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(i + offset)
                    cols.append(j + offset)
                    batches.append(b)
        offset += n

    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_batch = torch.tensor(batches, dtype=torch.long, device=device)
    return edge_index, edge_batch


def generated_to_rdkit(
    pos: Tensor,
    atom_type: Tensor,
    charge: Tensor,
    bond_order: Tensor,
    edge_index: Tensor,
    num_atoms_list: list[int],
) -> list[Chem.Mol | None]:
    """
    Convert generated tensors to a list of RDKit Mol objects (one per molecule).
    """
    mols = []
    offset = 0

    for n in num_atoms_list:
        rw = Chem.RWMol()

        # Add atoms
        for i in range(n):
            global_i = offset + i
            sym = IDX_TO_ATOM.get(atom_type[global_i].item(), "C")  # type: ignore[operator]
            atom = Chem.Atom(sym)
            fc = IDX_TO_CHARGE.get(charge[global_i].item(), 0)  # type: ignore[operator]
            atom.SetFormalCharge(fc)
            rw.AddAtom(atom)

        # Add bonds (only upper triangle to avoid duplicates)
        n_edges = (edge_index[0] >= offset) & (edge_index[0] < offset + n)
        local_ei = edge_index[:, n_edges] - offset
        local_bo = bond_order[n_edges]

        added = set()
        for k in range(local_ei.shape[1]):
            i = local_ei[0, k].item()
            j = local_ei[1, k].item()
            bo = local_bo[k].item()
            if i < j and bo > 0 and (i, j) not in added:
                bond_type = IDX_TO_BOND.get(bo)  # type: ignore[operator]
                if bond_type is not None:
                    rw.AddBond(int(i), int(j), bond_type)
                    added.add((i, j))

        # Add 3D conformer
        conf = Chem.Conformer(n)
        for i in range(n):
            p = pos[offset + i].tolist()
            conf.SetAtomPosition(i, (p[0], p[1], p[2]))

        mol = rw.GetMol()
        mol.AddConformer(conf, assignId=True)

        try:
            Chem.SanitizeMol(mol)
            mols.append(mol)
        except Exception:
            mols.append(None)

        offset += n

    return mols


def write_sdf(mols: list[Chem.Mol | None], path: str) -> int:
    """Write valid molecules to an SDF file. Returns the count of written molecules."""
    writer = Chem.SDWriter(path)
    count = 0
    for mol in mols:
        if mol is not None:
            writer.write(mol)
            count += 1
    writer.close()
    return count
