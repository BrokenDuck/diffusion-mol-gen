"""
Score matching code

See https://arxiv.org/abs/2011.13456
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_mol_gen.diffusion.noise_schedules import VPSDEParams


class ScoreSDE:
    """
    VP-SDE score-based diffusion for continuous (position) features.

    Forward SDE: dx = -½β(t)x dt + √β(t) dw
    Marginal:    q(x_t|x_0) = N(μ(t)·x_0, σ²(t)·I)

    Network predicts: score s_θ(x_t, t) ≈ ∇_x log p_t(x_t) = -ε/σ_t
    Loss:  ||s_θ + ε/σ_t||²  (denoising score matching)
    Reverse: Euler-Maruyama
    """

    def __init__(self, sde: VPSDEParams):
        self.sde = sde

    def q_sample(
        self, x_0: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x_0: [N, 3]
            t:   [N] float in [0, 1]
        Returns:
            x_t, noise, std — all [N, 3] or [N, 1] for std
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coef, std = self.sde.marginal_params(t)
        mean_coef = mean_coef.view(-1, 1)
        std = std.view(-1, 1)
        x_t = mean_coef * x_0 + std * noise
        return x_t, noise, std

    def loss(self, pred_score: Tensor, noise: Tensor, std: Tensor) -> Tensor:
        """Denoising score matching: ||s_θ + ε/σ||²"""
        target_score = -noise / std
        return F.mse_loss(pred_score, target_score)

    @torch.no_grad()
    def reverse_em_step(
        self, pred_score: Tensor, x_t: Tensor, t: Tensor, dt: float
    ) -> Tensor:
        """
        Single Euler-Maruyama reverse step.
        dx = [f(x,t) - g²(t)·s_θ] dt + g(t) dw̄
        """
        t_tensor = (
            t
            if isinstance(t, Tensor)
            else torch.full((x_t.shape[0],), t, device=x_t.device)
        )
        f = self.sde.drift_coef(t_tensor).view(-1, 1)  # [N,1]
        g = self.sde.diffusion_coef(t_tensor).view(-1, 1)  # [N,1]

        drift = f * x_t - g**2 * pred_score
        diffusion = g * torch.randn_like(x_t)
        return x_t - drift * dt + diffusion * (dt**0.5)
