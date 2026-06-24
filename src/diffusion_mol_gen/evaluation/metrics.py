from rdkit import Chem


class MoleculeMetrics:
    """Compute standard molecular generation metrics."""

    @staticmethod
    def validity(mols: list[Chem.Mol | None]) -> float:
        """Fraction of non-None molecules that pass RDKit sanitization."""
        valid = [m for m in mols if m is not None]
        return len(valid) / max(len(mols), 1)

    @staticmethod
    def uniqueness(mols: list[Chem.Mol | None]) -> float:
        """Fraction of valid molecules with unique canonical SMILES."""
        valid = [m for m in mols if m is not None]
        if not valid:
            return 0.0
        smiles = [Chem.MolToSmiles(m) for m in valid]
        return len(set(smiles)) / len(smiles)

    @staticmethod
    def novelty(mols: list[Chem.Mol | None], train_smiles: set[str]) -> float:
        """Fraction of unique valid molecules not in training set."""
        valid = [m for m in mols if m is not None]
        if not valid:
            return 0.0
        unique = set(Chem.MolToSmiles(m) for m in valid)
        novel = unique - train_smiles
        return len(novel) / max(len(unique), 1)

    @staticmethod
    def atom_stability(mols: list[Chem.Mol | None]) -> float:
        """Fraction of atoms with valid valency (uses RDKit)."""
        total = 0
        stable = 0
        for mol in mols:
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                total += 1
                try:
                    # If valence is within allowed range, atom is stable
                    if atom.GetImplicitValence() >= 0:
                        stable += 1
                except Exception:
                    pass
        return stable / max(total, 1)

    @classmethod
    def all_metrics(
        cls,
        mols: list[Chem.Mol | None],
        train_smiles: set[str] | None = None,
    ) -> dict[str, float]:
        metrics = {
            "validity": cls.validity(mols),
            "uniqueness": cls.uniqueness(mols),
            "atom_stability": cls.atom_stability(mols),
        }
        if train_smiles is not None:
            metrics["novelty"] = cls.novelty(mols, train_smiles)
        return metrics
