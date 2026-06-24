import py3Dmol
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


def view_molecule_3d(mol: Chem.Mol, style: str = "stick") -> object | None:
    """Render a molecule in 3D using py3Dmol (Jupyter)."""
    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mb, "sdf")
    viewer.setStyle({style: {}})
    viewer.zoomTo()
    return viewer


def mol_grid_image(mols: list[Chem.Mol | None], mols_per_row: int = 4):
    """Return a PIL image grid of 2D molecule depictions."""
    valid = [m for m in mols if m is not None]
    for mol in valid:
        rdDepictor.Compute2DCoords(mol)
    return Draw.MolsToGridImage(
        valid,
        molsPerRow=mols_per_row,
        subImgSize=(300, 300),
    )
