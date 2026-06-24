"""
Score-matching flavored D3PM implementation

See https://arxiv.org/abs/2205.14987
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_mol_gen.diffusion.noise_schedules import NoiseSchedule


class AbsorbingStateDiffusion:
    """
    Categorical diffusion with an absorbing (mask) state for score-based view.

    Forward: tokens are replaced by [MASK] with probability 1 - ᾱ_t.
    The mask token is an extra class at index `num_classes`.

    Network predicts: unmasking logits (over real classes) for masked tokens.
    Loss: cross-entropy only on masked positions (absorbing-state score matching).
    Reverse: unmask using predicted probabilities weighted by the reverse rate.
    """

    def __init__(self, num_classes: int, schedule: NoiseSchedule):
        self.K = num_classes
        self.mask_idx = num_classes
        self.schedule = schedule

    def q_sample(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Replace tokens with mask at rate (1 - ᾱ_t).
        Args:
            x_0: [N] integer class indices
            t:   [N] integer timestep per token
        Returns:
            x_t: [N] tokens (possibly masked)
            mask: [N] bool — True where token was masked
        """
        alpha_bar = self.schedule.alphas_cumprod.to(x_0.device)[t]  # [N]
        keep = torch.rand_like(alpha_bar) < alpha_bar
        x_t = torch.where(keep, x_0, torch.full_like(x_0, self.mask_idx))
        return x_t, ~keep

    def loss(self, pred_logits: Tensor, x_0: Tensor, mask: Tensor) -> Tensor:
        """
        Cross-entropy only on masked positions.
        pred_logits: [N, K]   logits over real classes
        x_0:         [N]      true class
        mask:        [N] bool True where prediction should be applied
        """
        if mask.sum() == 0:
            return pred_logits.sum() * 0.0
        return F.cross_entropy(pred_logits[mask], x_0[mask])

    @torch.no_grad()
    def p_sample(self, pred_logits: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Reverse step: unmask tokens whose current state is [MASK].

        For unmasked tokens: keep as-is.
        For masked tokens: sample from predicted distribution with probability
            (ᾱ_{t-1} - ᾱ_t) / (1 - ᾱ_t)  (Gillespie-style rate).
        """
        device = x_t.device
        alpha_bar_t = self.schedule.alphas_cumprod.to(device)[t]  # [N]
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_prev = self.schedule.alphas_cumprod.to(device)[t_prev]  # [N]

        is_masked = x_t == self.mask_idx
        if not is_masked.any():
            return x_t

        # Probability of unmasking at this step
        unmask_prob = ((alpha_bar_prev - alpha_bar_t) / (1 - alpha_bar_t + 1e-8)).clamp(
            0, 1
        )
        do_unmask = is_masked & (torch.rand_like(alpha_bar_t) < unmask_prob)

        probs = F.softmax(pred_logits, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)

        x_new = x_t.clone()
        x_new[do_unmask] = sampled[do_unmask]
        return x_new
