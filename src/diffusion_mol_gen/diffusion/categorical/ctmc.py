"""
Discrete Flow Matching

See https://arxiv.org/abs/2407.15595
"""
import torch
import torch.nn.functional as F
from torch import Tensor


class CTMCFlow:
    """
    Continuous-Time Markov Chain masking flow for categorical features (flow-based view).

    At t=0: all tokens are [MASK].
    At t=1: all tokens are revealed (data).

    Schedule: α(t) = t  (linear; token revealed with probability t).

    Network predicts: p_θ(x_1 | x_t, t) — posterior over real classes for masked tokens.
    Loss: cross-entropy on masked positions; tokens already revealed are ignored.
    Sampling: stochastic unmasking via CTMC rate  R = α'(t)/(1-α(t)).
    """

    def __init__(self, num_classes: int):
        self.K = num_classes
        self.mask_idx = num_classes  # extra class

    # ------------------------------------------------------------------
    # Forward: interpolation
    # ------------------------------------------------------------------

    def interpolate(self, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Each token is revealed (=x_1) with probability t, masked otherwise.
        Args:
            x_1: [N] true class indices
            t:   [N] float in [0, 1] per token
        Returns:
            x_t: [N] masked/revealed tokens
            mask: [N] bool — True for still-masked tokens
        """
        reveal = torch.rand_like(t) < t
        x_t = torch.where(reveal, x_1, torch.full_like(x_1, self.mask_idx))
        return x_t, ~reveal

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, pred_logits: Tensor, x_1: Tensor, mask: Tensor) -> Tensor:
        """
        Cross-entropy on masked positions only.
        pred_logits: [N, K]
        x_1:         [N] ground truth classes
        mask:        [N] bool — True where token is masked (should be predicted)
        """
        if mask.sum() == 0:
            return pred_logits.sum() * 0.0
        return F.cross_entropy(pred_logits[mask], x_1[mask])

    # ------------------------------------------------------------------
    # Sampling: CTMC step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_step(
        self, pred_logits: Tensor, x_t: Tensor, t: float, dt: float
    ) -> Tensor:
        """
        Stochastic unmasking step.

        Rate of unmasking: R(t) = α'(t)/(1-α(t)) = 1/(1-t)  [for linear α]
        Probability of unmasking in interval dt: R·dt ≈ dt/(1-t).

        Args:
            pred_logits: [N, K] logits from network
            x_t:         [N] current token states
            t:           current time scalar in [0, 1)
            dt:          integration step size
        Returns:
            x_new: [N] updated token states
        """
        is_masked = x_t == self.mask_idx
        if not is_masked.any():
            return x_t

        # Probability of unmasking a masked token in this step
        unmask_prob = min(dt / max(1 - t, 1e-6), 1.0)
        do_unmask = is_masked & (torch.rand(x_t.shape, device=x_t.device) < unmask_prob)

        probs = F.softmax(pred_logits, dim=-1)  # [N, K]
        sampled = torch.multinomial(probs, 1).squeeze(-1)

        x_new = x_t.clone()
        x_new[do_unmask] = sampled[do_unmask]
        return x_new
