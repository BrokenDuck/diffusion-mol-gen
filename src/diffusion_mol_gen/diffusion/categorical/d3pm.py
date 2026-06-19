import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_mol_gen.diffusion.noise_schedules import NoiseSchedule


class D3PM:
    """
    Discrete Denoising Diffusion Probabilistic Model (Austin et al. 2021).

    Forward: q(x_t | x_0) = Cat(x_0 · Q̄_t)
    Supports:
      - "uniform"   Q_t = (1−β_t)·I + β_t/K · 1·1ᵀ
      - "absorbing" Q_t = (1−β_t)·I + β_t · e_mask·1ᵀ

    Network predicts: p̂(x_0 | x_t) as logits (cross-entropy loss).
    Reverse: posterior q(x_{t−1} | x_t, x̂_0)  ∝  q(x_t | x_{t−1}) · q(x_{t−1} | x̂_0)
    """

    def __init__(
        self,
        num_classes: int,
        schedule: NoiseSchedule,
        transition: str = "absorbing",
    ):
        self.K = num_classes
        self.mask_idx = num_classes  # absorbing/mask token is an extra class
        self.transition = transition
        self._build_transitions(schedule)

    # ------------------------------------------------------------------
    # Transition matrices
    # ------------------------------------------------------------------

    def _build_transitions(self, schedule: NoiseSchedule) -> None:
        T = schedule.T
        K = self.K

        Qt_list = []

        for t in range(T):
            beta_t = schedule.betas[t].double()
            if self.transition == "uniform":
                Qt = (1 - beta_t) * torch.eye(K, dtype=torch.float64) + (beta_t / K) * torch.ones(K, K, dtype=torch.float64)
            else:  # absorbing: K+1 states (0..K-1 real + K mask)
                K_full = K + 1
                Qt_full = (1 - beta_t) * torch.eye(K_full, dtype=torch.float64)
                # Real tokens absorb into mask with probability beta_t
                Qt_full[:K, K] = beta_t
                Qt = Qt_full  # [K+1, K+1]

            Qt_list.append(Qt)

        # Cumulative product Q̄_t = Q_1 · Q_2 · … · Q_t
        if self.transition == "absorbing":
            K_full = K + 1
            qbar = torch.eye(K_full, dtype=torch.float64)
        else:
            qbar = torch.eye(K, dtype=torch.float64)

        Qbar_list = []
        for Qt in Qt_list:
            qbar = qbar @ Qt
            Qbar_list.append(qbar.float())

        # [T, K(+1), K(+1)]
        self.Qt = torch.stack(Qt_list).float()       # single-step
        self.Qbar = torch.stack(Qbar_list).float()   # cumulative

    def _effective_K(self) -> int:
        return self.K + 1 if self.transition == "absorbing" else self.K

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(self, x_0: Tensor, t: Tensor) -> Tensor:
        """
        Sample noisy categorical x_t ~ Cat(one_hot(x_0) · Q̄_t).
        Args:
            x_0: [N] integer class indices  (0..K-1)
            t:   [N] integer timestep per token
        Returns:
            x_t: [N] noisy class indices
        """
        K_eff = self._effective_K()
        Qbar_t = self.Qbar.to(x_0.device)[t]           # [N, K_eff, K_eff]
        x0_oh = F.one_hot(x_0, K_eff).float()           # [N, K_eff]
        probs = torch.bmm(x0_oh.unsqueeze(1), Qbar_t).squeeze(1)  # [N, K_eff]
        probs = probs.clamp(min=0)  # numerical safety
        return torch.multinomial(probs, 1).squeeze(-1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, pred_logits: Tensor, x_0: Tensor) -> Tensor:
        """
        Cross-entropy between predicted clean distribution and true x_0.
        pred_logits: [N, num_atom_types]  (logits over real classes 0..K-1)
        x_0:         [N] true class indices
        """
        return F.cross_entropy(pred_logits, x_0)

    # ------------------------------------------------------------------
    # Reverse step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, pred_logits: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Single reverse step using posterior q(x_{t−1} | x_t, x̂_0).
        """
        device = x_t.device
        T_int = t.long()
        T_prev = (T_int - 1).clamp(min=0)

        # Predicted x̂_0 distribution  [N, K] or [N, K+1]
        if self.transition == "absorbing":
            # Network only predicts real classes; absorbing state is never predicted
            real_probs = F.softmax(pred_logits, dim=-1)           # [N, K]
            # Pad mask token probability as zero
            x0_probs = torch.cat([real_probs, torch.zeros_like(real_probs[:, :1])], dim=-1)  # [N, K+1]
        else:
            x0_probs = F.softmax(pred_logits, dim=-1)             # [N, K]

        # q(x_t | x_{t-1}) = row T_int of Qt
        Qt_t = self.Qt.to(device)[T_int]              # [N, K_eff, K_eff]
        # q(x_{t-1} | x_0) = row of Qbar_{t-1}
        Qbar_prev = self.Qbar.to(device)[T_prev]      # [N, K_eff, K_eff]

        # q(x_t | x_{t-1}) for each possible x_{t-1}  → column Qt[:, x_t]
        # qt_xt_given_xtm1[n, j] = Qt_t[n, j, x_t[n]]
        qt_xt = Qt_t.transpose(1, 2)[torch.arange(len(x_t)), x_t]  # [N, K_eff]

        # q(x_{t-1} | x_0_hat) = x0_probs @ Qbar_prev  → [N, K_eff]
        qtm1_x0 = torch.bmm(x0_probs.unsqueeze(1), Qbar_prev).squeeze(1)  # [N, K_eff]

        # Unnormalised posterior
        posterior = qt_xt * qtm1_x0                   # [N, K_eff]
        posterior = posterior / (posterior.sum(dim=-1, keepdim=True) + 1e-10)

        return torch.multinomial(posterior, 1).squeeze(-1)
