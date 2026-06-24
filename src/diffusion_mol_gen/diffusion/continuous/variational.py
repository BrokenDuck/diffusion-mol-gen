"""
DDPM Code

See https://arxiv.org/abs/2006.11239
"""

import torch
from torch import Tensor

from diffusion_mol_gen.diffusion.noise_schedules import NoiseSchedule


class VariationalContinuous:
    """
    DDPM forward/reverse for continuous (position) features.

    Forward:  q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)
    Network:  predicts ε (the noise added during forward diffusion)
    Loss:     MSE(ε̂, ε)
    Reverse:  recover x̂_0 from ε̂, then posterior mean of q(x_{t-1} | x_t, x̂_0)
    """

    def __init__(self, schedule: NoiseSchedule):
        self.schedule = schedule

    def q_sample(
        self, x_0: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Forward process of DDPM
        Args:
            x_0: [N, 3] clean positions
            t:   [N] integer timestep per atom (broadcast from graph-level t)
        Returns:
            x_t: [N, 3] noisy positions
            noise: [N, 3] added noise (used as training target)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.schedule.gather("sqrt_alphas_cumprod", t, x_0.shape)
        sqrt_1m = self.schedule.gather("sqrt_one_minus_alphas_cumprod", t, x_0.shape)
        x_t = sqrt_ab * x_0 + sqrt_1m * noise
        return x_t, noise

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, pred_eps: Tensor) -> Tensor:
        """Recover x̂_0 from predicted noise: x̂_0 = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t"""
        sqrt_recip = self.schedule.gather("sqrt_recip_alphas_cumprod", t, x_t.shape)
        sqrt_recipm1 = self.schedule.gather("sqrt_recipm1_alphas_cumprod", t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * pred_eps

    def q_posterior_mean(
        self, pred_x0: Tensor, x_t: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Posterior mean and log variance for the reverse step."""
        coef1 = self.schedule.gather("posterior_mean_coef1", t, x_t.shape)
        coef2 = self.schedule.gather("posterior_mean_coef2", t, x_t.shape)
        mean = coef1 * pred_x0 + coef2 * x_t
        log_var = self.schedule.gather("posterior_log_variance_clipped", t, x_t.shape)
        return mean, log_var

    @torch.no_grad()
    def p_sample(self, pred_eps: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Single DDPM reverse step: predict x̂_0 from ε̂, then sample x_{t-1}."""
        pred_x0 = self.predict_x0_from_eps(x_t, t, pred_eps)
        mean, log_var = self.q_posterior_mean(pred_x0, x_t, t)
        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t > 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
