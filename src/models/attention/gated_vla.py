"""
Gated Variational Linear Attention (Gated VLA / VLA v2).

Combines the Sherman-Morrison inverse penalty direction (provably optimal
for tracking inverse covariance) with a learned decay gate on the memory
matrix S, enabling selective forgetting to prevent interference.

S_t = g_t * S_{t-1} + e_t ⊗ normalize(A_t k_t)

where g_t = sigmoid(W_g(x_t)) is a per-head, per-timestep decay gate:
  - g ≈ 1: retains memory (like original VLA)
  - g ≈ 0: aggressive forgetting (fresh state)
  - Learned end-to-end to balance retention vs interference

This gives VLA the best of both worlds:
  - Sherman-Morrison penalty: unique to VLA, provably optimal update direction
  - Selective forgetting: proven in DeltaNet/GLA/Mamba2 to be essential
    for high-capacity associative recall
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedVLAv3(nn.Module):
    """
    Gated Variational Linear Attention v3.

    Single-head implementation (for use in multi-head wrappers or
    heterogeneous-head architectures).

    Args:
        d_model: Dimension of the input (= head dimension when used per-head).
        lambda_0: Initial inverse regularization (A_0 = (1/lambda_0) * I).
        stab_eps: Clamping epsilon for Sherman-Morrison denominator.
        per_eps: Periodic identity nudge magnitude.
        period: Period for identity nudge (every `period` steps).
        gate_init_bias: Initial bias for the gate sigmoid. Positive values
            (e.g., 2.0) make the gate start near 1.0 (retain memory),
            letting the model learn to forget selectively.
    """

    def __init__(
        self,
        d_model: int,
        lambda_0: float = 0.1,
        stab_eps: float = 1e-4,
        per_eps: float = 1e-3,
        period: int = 20,
        gate_init_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.lambda_0 = lambda_0
        self.stab_eps = stab_eps
        self.per_eps = per_eps
        self.period = period

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.u_proj = nn.Linear(d_model, d_model, bias=False)

        # NEW: decay gate projection
        self.W_g = nn.Linear(d_model, d_model)
        # Initialize gate bias so sigmoid starts near 1.0 (mostly retaining)
        nn.init.constant_(self.W_g.bias, gate_init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        d = self.d_model
        device = x.device

        k_raw = self.W_k(x).float()
        k_feat = F.elu(k_raw) + 1.0
        q_feat = F.elu(self.W_q(x).float()) + 1.0
        v_feat = self.W_v(x).float()
        u_dirs = F.normalize(self.u_proj(k_raw), p=2, dim=-1)
        # Decay gate: per-timestep, values in [0, 1]
        gate = torch.sigmoid(self.W_g(x).float())

        eye = torch.eye(d, device=device, dtype=torch.float32)
        a_t = (1.0 / self.lambda_0) * eye.unsqueeze(0).expand(bsz, -1, -1).clone()
        s_t = torch.zeros(bsz, d, d, device=device, dtype=torch.float32)
        z_t = torch.zeros(bsz, d, device=device, dtype=torch.float32)
        inv_sqrt_d = 1.0 / math.sqrt(d)

        outputs = []
        for t in range(seq_len):
            # ── Sherman-Morrison inverse update ──
            u_t = u_dirs[:, t, :] * inv_sqrt_d
            uv = u_t.unsqueeze(-1)
            z_sm = torch.bmm(a_t, uv)
            dot = torch.bmm(uv.transpose(1, 2), z_sm).squeeze(-1).squeeze(-1)
            delta = torch.clamp(1.0 + dot, min=self.stab_eps)
            update = torch.bmm(z_sm, z_sm.transpose(1, 2)) / delta.view(bsz, 1, 1)
            a_t = a_t - update

            if (t + 1) % self.period == 0:
                a_t = a_t + self.per_eps * eye.unsqueeze(0)

            # ── Memory update with decay gate ──
            k_t = k_feat[:, t, :]
            v_t = v_feat[:, t, :]
            q_t = q_feat[:, t, :]
            g_t = gate[:, t, :]  # (bsz, d)

            k_n = F.normalize(k_t, p=2, dim=-1)
            alpha_t = torch.bmm(a_t, k_n.unsqueeze(-1)).squeeze(-1)
            alpha_n = F.normalize(alpha_t, p=2, dim=-1)

            err = v_t - torch.bmm(s_t, k_n.unsqueeze(-1)).squeeze(-1)

            # THE KEY CHANGE: gated decay on S
            s_t = g_t.unsqueeze(-1) * s_t + torch.matmul(
                err.unsqueeze(2), alpha_n.unsqueeze(1)
            )

            z_t = z_t + k_t
            o_t = torch.bmm(s_t, q_t.unsqueeze(-1)).squeeze(-1)
            denom = (z_t * q_t).sum(-1, keepdim=True).clamp(min=1e-6)
            outputs.append(o_t / denom)

        out = torch.stack(outputs, dim=1).to(dtype=x.dtype)
        return self.W_o(self.norm(out))
