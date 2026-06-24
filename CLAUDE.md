# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Train
uv run dmg train --view variational --num-timesteps 1000 --schedule cosine \
  --max-epochs 500 --batch-size 64 --lr 2e-4 --hidden-channels 256 \
  --num-layers 6 --data-root ./data

# --view options: variational | score | flow
# --schedule options: linear | cosine
# Add --wandb --wandb-project <name> for W&B logging

# Generate molecules from a trained checkpoint
uv run dmg sample --checkpoint path/to/best.ckpt \
  --num-molecules 100 --output generated.sdf --device cuda

# --num-atoms: integer (e.g. 9), range (e.g. 5-15), or 'sample' (QM9 distribution)
# --num-steps: integration steps override (SDE/ODE only)

# Lint / format
uv run ruff check src/
uv run ruff format src/

# Type check
uv run ty check src/
```

No test suite exists yet.

## Architecture

This is a 3D molecule generation project comparing three diffusion paradigms on the same backbone.

**Molecule representation:** Each molecule is a fully-connected graph with per-node features (3D positions, atom type, formal charge) and per-edge features (bond order). Heavy atoms only (H, C, N, O, F). Dataset: QM9 (~130k small organic molecules).

**Three interchangeable views** (swapped via `DiffusionConfig.view`):

| View | Continuous (positions) | Categorical (atom type, charge, bond order) | Sampling |
|------|----------------------|---------------------------------------------|---------|
| `variational` | DDPM — predicts x̂₀ | D3PM transitions | Posterior sampling |
| `score` | VP-SDE — predicts score ∇log p | Absorbing-state diffusion | Euler-Maruyama / Predictor-Corrector |
| `flow` | OT flow matching — predicts velocity v_θ | CTMC masking | Euler ODE |

**Model:** `Denoiser` = EGNN backbone (equivariant, stacked layers with sinusoidal time conditioning) + four prediction heads (position, atom type, charge, bond order). EGNN preserves E(n) equivariance — rotations/translations of the input molecule produce identically rotated/translated outputs for positions and unchanged scalar features.

**Key files:**
- `src/diffusion_mol_gen/diffusion/unified.py` — dispatches forward/reverse diffusion to the correct process per view
- `src/diffusion_mol_gen/training/lightning_module.py` — PyTorch Lightning module (train step, EMA, losses)
- `src/diffusion_mol_gen/models/denoiser.py` — assembles backbone + all four heads
- `src/diffusion_mol_gen/configs/base.py` — `ModelConfig`, `DiffusionConfig`, `TrainingConfig` dataclasses
- `src/diffusion_mol_gen/sampling/` — one sampler per view plus RDKit molecule conversion utils

**Configuration** flows via three Pydantic dataclasses: `ModelConfig` (GNN architecture), `DiffusionConfig` (view + noise schedule), `TrainingConfig` (batch size, LR, EMA decay, loss weights). The CLI in `src/diffusion_mol_gen/__init__.py:main()` parses args and instantiates these.

**Data pipeline:** `QM9Dataset` → fully-connected graph transform → `CenterPositions` / `NormalizePositions` transforms → `QM9DataModule` (PL DataModule, 100k/18k/13k split).

## Python Tooling

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.
