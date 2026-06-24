"""
Flow matching code

See https://arxiv.org/abs/2210.02747
"""

import torch
import torch.nn.functional as F
from torch import Tensor


class FlowMatchingContinuous:
    """
    Optimal Transport (OT) flow matching for continuous (position) features.

    Interpolation: x_t = (1-t)·z + t·x_1,   z ~ N(0, I)
    Target velocity: u = x_1 - z  (constant along trajectory)
    Network predicts: either u directly OR endpoint x̂_1
      (endpoint param: v = (x̂_1 - x_t) / (1-t))

    Loss: ||v_θ - u||²  (MSE on velocity)
    Sampling: Euler ODE  x_{t+dt} = x_t + v_θ · dt
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def interpolate(
        self, x_1: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x_1: [N, 3] clean positions
            t:   [N] float in [0, 1] per atom
        Returns:
            x_t, noise, target_velocity — all [N, 3]
        """
        if noise is None:
            noise = torch.randn_like(x_1)
        t_ = t.view(-1, 1)
        x_t = (1 - t_) * noise + t_ * x_1
        # Small Gaussian noise for numerical stability
        x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        target_velocity = x_1 - noise
        return x_t, noise, target_velocity

    def loss(self, pred_velocity: Tensor, target_velocity: Tensor) -> Tensor:
        return F.mse_loss(pred_velocity, target_velocity)

    def pred_x0_to_velocity(self, pred_x1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Convert endpoint prediction x̂_1 to velocity via endpoint parameterization."""
        t_ = t.view(-1, 1).clamp(max=1 - 1e-6)
        return (pred_x1 - x_t) / (1 - t_)

    @torch.no_grad()
    def euler_step(self, velocity: Tensor, x_t: Tensor, dt: float) -> Tensor:
        return x_t + velocity * dt
