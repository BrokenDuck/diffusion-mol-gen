# Diffusion Molecular Generation

Diffusion Molecular Generation implements molecular generation using three different diffusion methods: variational, score and flows.

## Functioning

[Lai et al.](https://arxiv.org/abs/2510.21890) show that variational, score and flow are three different views of the same SDE backbone, respecting the Fokker-Planck equation. We apply those three views to the problem of molecular generation.

We follow the problem formalism of [FlowMol](https://arxiv.org/abs/2508.12629). We predict:
- Atom Positions (Continuous distribution)
- Atom Type and Charge (Categorical distribution)
- Bond type (Categorical distribution)

We use the following methods to model the different atom features.
| View        | Continuous     | Discrete                      |
|-------------|----------------|-------------------------------|
| Variational | [DDPM](https://arxiv.org/abs/2006.11239)           | [D3PM](https://arxiv.org/abs/2107.03006)                          |
| Score       | [Score-matching](https://arxiv.org/abs/2011.13456) | [D3PM (absorbing)](https://arxiv.org/abs/2205.14987)              |
| Flow        | [Flow-matching](https://arxiv.org/abs/2210.02747)  | [Continuous-Time Markov Chains](https://arxiv.org/abs/2407.15595) |

The same model is used to predice the noise/score/velocity and category logits: [EGNN](https://arxiv.org/abs/2102.09844).

## Implementation

```
src/diffusion_mol_gen/
├── cli.py                       # CLI Entrypoint
├── configs.py                   # Configuration classes for model, diffusion and samplers
├── diffusion                    # Diffusion code for training
│   ├── noise_schedules.py       # DDPM and Score Matching noise schedule
│   ├── unified.py               # Classes for variational, score and flow views
│   ├── categorical              # Discrete Diffusion Implementation
│   │   ├── absorbing.py         # Absorbing D3PM
│   │   ├── ctmc.py              # Continuous-time Markov Chains
│   │   └── d3pm.py              # D3PM
│   └── continuous               # Continuous Diffusion Implementation
│       ├── flow_matching.py     # Flow-matching
│       ├── score_sde.py         # Score-matching
│       └── variational.py       # DDPM
├── evaluation                   # Evaluation and Visualization code
│   ├── metrics.py             
│   └── visualization.py
├── models                       # EGNN Model Implementation
│   ├── denoiser.py              # Full model implementation
│   ├── egnn.py                  # Equivariant layer implemenatation
│   └── heads.py                 # Head to read out new positions and category logits
├── sampling                     # Sampling code
│   ├── ode_sampler.py           # Flow-matching sampler
│   ├── sde_sampler.py           # Euler-Maruyama sampler
│   ├── utils.py
│   └── variational_sampler.py   # DDPM sampler
└── training
    ├── lightning_module.py      # Pytorch Ligthning Intergration
    └── losses.py                # Loss functions
```

## Running the code

We need [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

```bash
uv sync
uv sync --extra cu130 # For GPU dependencies
```

For training:
```bash
uv run dmg train --help
uv run dmg train --view variational
uv run dmg train --view score
uv run dmg train --view flow
```

For generation:
```bash
uv run dmg sample --help
uv run dmg sample --view variational
uv run dmg sample --view score
uv run dmg sample --view flow
```