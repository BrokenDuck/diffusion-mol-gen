from typing import Literal
import math
import torch
from torch import Tensor


class NoiseSchedule:
    """
    Precomputed discrete noise schedule for DDPM / D3PM.

    Stores: betas, alphas, alphas_cumprod (ᾱ_t), and derived quantities
    for all T timesteps.
    """

    # Some precomputations
    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    # Forward process (ading noise)
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    log_one_minus_alphas_cumprod: Tensor
    # Sampling process (Generation)
    sqrt_recip_alphas_cumprod: Tensor
    sqrt_recipm1_alphas_cumprod: Tensor
    # Reverse process (Denoising)
    posterior_variance: Tensor
    posterior_log_variance_clipped: Tensor
    posterior_mean_coef1: Tensor
    posterior_mean_coef2: Tensor

    def __init__(
        self, num_timesteps: int, schedule_type: Literal["linear", "cosine"] = "cosine"
    ):
        self.T = num_timesteps

        if schedule_type == "cosine":
            # Nichol & Dhariwal (2021) cosine schedule
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
            alpha_bar = torch.cos((steps + s) / (1.0 + s) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()
        else:
            # Default to linear
            betas = torch.linspace(1e-4, 0.02, num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        # Posterior variance for DDPM reverse step
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)
        )

    def _gather(self, tensor: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
        """Gather per-timestep scalar values and broadcast to target shape."""
        out = tensor.to(t.device)[t]
        while out.dim() < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    def gather(self, name: str, t: Tensor, shape: torch.Size) -> Tensor:
        """Helper function to grab an attribute for a specific timestep and cast it."""
        return self._gather(getattr(self, name), t, shape)


class VPSDEParams:
    """
    Continuous-time VP-SDE marginal parameters.

    Forward: dx = -½β(t)x dt + √β(t) dw  (linear schedule on [0,1])
    β(t) = β_min + t·(β_max - β_min)
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def marginal_params(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns mean_coef μ(t) and std σ(t) such that
        q(x_t | x_0) = N(μ(t)·x_0, σ²(t)·I).
        """
        t = t.float()
        log_mean_coef = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean_coef = torch.exp(log_mean_coef)
        std = torch.sqrt((1 - torch.exp(2 * log_mean_coef)).clamp(min=1e-10))
        return mean_coef, std

    def beta(self, t: Tensor) -> Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def diffusion_coef(self, t: Tensor) -> Tensor:
        """g(t) = √β(t)"""
        return torch.sqrt(self.beta(t))

    def drift_coef(self, t: Tensor) -> Tensor:
        """f(t) = -½β(t)"""
        return -0.5 * self.beta(t)
