# Molecule Generation with Three Diffusion Views

## Context

This project implements 3D molecule generation using diffusion/flow models with three interchangeable paradigms (Variational/DDPM, Score-based, Flow-based) sharing a single equivariant GNN backbone. Each molecule is represented as:
- **Positions** x_i: continuous 3D coordinates per atom
- **Atom types** a_i: categorical (H, C, N, O, F for QM9)
- **Charges** c_i: categorical (formal charges mapped to indices)
- **Bond orders** e_{ij}: categorical (none, single, double, triple, aromatic)

The three views differ only in what the network predicts and how noise/interpolation is applied:
1. **Variational (DDPM)**: predicts clean sample x_0
2. **Score-based (SDE)**: predicts score ∇_x log p_t(x)
3. **Flow-based**: predicts velocity dx/dt

Inspired by AGeDi (score-based reference) and FlowMol (flow-based reference).

---

## File Structure

```
src/diffusion_mol_gen/
├── __init__.py
├── configs/
│   └── base.py                      # Dataclass configs
├── data/
│   ├── __init__.py
│   ├── qm9_dataset.py              # QM9 loading + preprocessing
│   ├── datamodule.py               # PL DataModule
│   └── transforms.py               # CenterPositions, NormalizePositions
├── models/
│   ├── __init__.py
│   ├── backbone/
│   │   ├── __init__.py
│   │   ├── egnn.py                  # E(n) Equivariant GNN layers
│   │   ├── time_embedding.py        # Sinusoidal + MLP time embed
│   │   └── mol_gnn.py              # Full backbone (layers + time conditioning)
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── position_head.py         # Equivariant vector output
│   │   ├── atom_type_head.py        # Invariant logits
│   │   ├── charge_head.py           # Invariant logits
│   │   └── bond_order_head.py       # Edge-level invariant logits
│   └── denoiser.py                  # Backbone + all heads combined
├── diffusion/
│   ├── __init__.py
│   ├── noise_schedules.py           # Linear/cosine schedules, SDE coefficients
│   ├── continuous/
│   │   ├── __init__.py
│   │   ├── variational.py           # DDPM for positions (predicts x_0)
│   │   ├── score_sde.py             # VP-SDE for positions (predicts score)
│   │   └── flow_matching.py         # OT interpolant for positions (predicts velocity)
│   ├── categorical/
│   │   ├── __init__.py
│   │   ├── d3pm.py                  # D3PM transitions (predicts p(x_0|x_t))
│   │   ├── absorbing.py             # Absorbing-state diffusion (predicts unmasking score)
│   │   └── ctmc.py                  # CTMC masking flow (predicts unmask probs)
│   └── unified.py                   # Dispatches to correct cont + cat per view
├── training/
│   ├── __init__.py
│   ├── lightning_module.py          # Main PL module
│   ├── losses.py                    # MSE, CE, VLB, score matching losses
│   └── ema.py                       # EMA wrapper
├── sampling/
│   ├── __init__.py
│   ├── variational_sampler.py       # DDPM posterior sampling
│   ├── sde_sampler.py              # Euler-Maruyama / Predictor-Corrector
│   ├── ode_sampler.py              # Euler ODE integration
│   └── utils.py                     # Convert generated tensors → RDKit Mol
└── evaluation/
    ├── __init__.py
    ├── metrics.py                   # Validity, uniqueness, novelty
    └── visualization.py             # py3Dmol + RDKit 2D
```

---

## Architecture Overview

### Shared GNN Backbone: EGNN

Using E(n) Equivariant GNN (Satorras et al.) because it naturally separates invariant scalars from equivariant vectors without spherical harmonics. Built on `torch_geometric.nn.MessagePassing`.

Each layer:
```
m_ij = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)        # message
x_i' = x_i + C · Σ_j (x_i - x_j) · φ_x(m_ij)      # equivariant position update
h_i' = φ_h(h_i, Σ_j m_ij)                           # invariant node update
```

Time conditioning: sinusoidal embedding → MLP → added to node features before first layer.

### Prediction Heads

| Head | Input | Output | Property |
|------|-------|--------|----------|
| PositionHead | coord_delta from backbone + h | [N, 3] vector | Equivariant |
| AtomTypeHead | h (node features) | [N, num_atom_types] logits | Invariant |
| ChargeHead | h (node features) | [N, num_charges] logits | Invariant |
| BondOrderHead | h_i, h_j, e_ij (edge features) | [E, num_bond_types] logits | Invariant |

---

## Three Views: Mathematical Formulations

### View 1: Variational (DDPM)

**Continuous (positions):**
- Forward: q(x_t|x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t)·I)
- Network predicts: x̂_0
- Loss: ||x̂_0 - x_0||²
- Reverse: posterior mean μ̃_t = (√ᾱ_{t-1}·β_t)/(1-ᾱ_t)·x̂_0 + (√α_t·(1-ᾱ_{t-1}))/(1-ᾱ_t)·x_t

**Categorical (atom types, charges, bonds) — D3PM:**
- Forward: q(x_t|x_0) = Cat(x_0 · Q̄_t) with uniform or absorbing transitions
- Network predicts: p̂(x_0|x_t) as logits
- Loss: cross-entropy on x_0 (or full VLB with KL on posterior)
- Reverse: q(x_{t-1}|x_t, x̂_0) ∝ q(x_t|x_{t-1}) · q(x_{t-1}|x̂_0)

### View 2: Score-based (SDE)

**Continuous (positions) — VP-SDE:**
- Forward SDE: dx = -½β(t)x dt + √β(t) dw
- Marginal: q(x_t|x_0) = N(μ(t)·x_0, σ²(t)·I) where μ(t) = exp(-¼t²(β_max-β_min) - ½t·β_min)
- Network predicts: score s_θ(x_t, t) ≈ -ε/σ_t
- Loss: ||s_θ + ε/σ_t||² (denoising score matching)
- Reverse: dx = [f(x,t) - g²(t)·s_θ] dt + g(t) dw̄ (Euler-Maruyama)

**Categorical — Absorbing state:**
- Forward: tokens replaced with [MASK] at rate from noise schedule
- Network predicts: unmasking logits for masked positions
- Loss: score entropy / cross-entropy on masked positions only
- Reverse: unmask tokens using predicted rates and reverse rate formula

### View 3: Flow-based

**Continuous (positions) — OT Flow Matching:**
- Interpolation: x_t = (1-t)·z + t·x_1 where z ~ N(0, I)
- Network predicts: velocity v_θ = x_1 - z (endpoint parameterization: predict x̂_1, derive v = α'_t/(1-α_t)·(x̂_1 - x_t))
- Loss: ||v_θ - (x_1 - z)||²
- Sampling: x_{t+dt} = x_t + v_θ(x_t, t)·dt (Euler ODE)

**Categorical — CTMC masking:**
- At t=0: all tokens are [MASK]; at t=1: all revealed
- Network predicts: p_θ(x_1 | x_t, t) for masked positions
- Loss: cross-entropy on masked positions (ignore already-unmasked)
- Sampling: unmask tokens stochastically at rate dt/(1-t) using predicted probs

---

## Key Implementation Details

### Data Pipeline
- Use PyG's QM9 dataset, extract pos, z (→ atom_type index), formal charges (→ charge index), edge_attr (→ bond_order index)
- Center positions to zero CoM (removes translational freedom)
- For bonds: include "no bond" as category 0; during sampling use fully-connected graph for QM9 (max 29 atoms, feasible)
- Train/val/test: 100k/18k/13k standard split

### Edge Handling During Sampling
- QM9 molecules are small enough (≤29 atoms) for fully-connected graphs
- Predict "no bond" (class 0) for non-bonded pairs
- Post-process: threshold bond predictions, apply valency constraints

### Variable Molecule Sizes
- Sample num_atoms from empirical training distribution at generation time
- Batch via PyG's Batch object

### Noise Schedule
- Shared schedule class producing: betas, alphas, alphas_cumprod, sqrt terms
- For score-based: continuous-time marginal params via β(t) integration
- For flow: linear alpha_t = t (optionally cosine)

---

## Implementation Phases

### Phase 1: Data Pipeline
- `configs/base.py` — ModelConfig, DiffusionConfig, TrainingConfig dataclasses
- `data/qm9_dataset.py` — Wrap PyG QM9, map z→atom_type, extract charges/bonds
- `data/transforms.py` — CenterPositions, NormalizePositions
- `data/datamodule.py` — PL DataModule with splits and loaders

### Phase 2: GNN Backbone
- `models/backbone/time_embedding.py` — Sinusoidal encoding + MLP
- `models/backbone/egnn.py` — EGNNLayer extending MessagePassing
- `models/backbone/mol_gnn.py` — Stack of EGNN layers with input embeddings

### Phase 3: Prediction Heads + Denoiser
- `models/heads/*.py` — Four prediction heads
- `models/denoiser.py` — Combines backbone + heads, single forward pass

### Phase 4: Diffusion Processes
- `diffusion/noise_schedules.py` — Betas, cumulative products, SDE coefficients
- `diffusion/continuous/*.py` — Three continuous processes
- `diffusion/categorical/*.py` — Three categorical processes
- `diffusion/unified.py` — View-based dispatch

### Phase 5: Training
- `training/losses.py` — All loss functions
- `training/ema.py` — EMA integration
- `training/lightning_module.py` — Full training loop

### Phase 6: Sampling
- `sampling/variational_sampler.py`, `sde_sampler.py`, `ode_sampler.py`
- `sampling/utils.py` — Tensor → RDKit molecule conversion

### Phase 7: Evaluation
- `evaluation/metrics.py` — Validity, uniqueness, novelty, stability
- `evaluation/visualization.py` — 3D visualization

---

## Verification Plan

1. **Unit tests per module**: Verify EGNN equivariance (rotate input → output rotates), verify noise schedule properties (alpha_bar goes 0→1), verify categorical forward process recovers uniform at t=T
2. **Integration test**: Train each view for 1 epoch on a small QM9 subset, verify loss decreases
3. **Sampling test**: Generate 100 molecules with each view, check validity > 0 (even if low)
4. **Full training**: Train on QM9, compare validity/uniqueness/novelty across views
5. **Wandb logging**: Verify losses, metrics, and generated molecule visualizations are logged
