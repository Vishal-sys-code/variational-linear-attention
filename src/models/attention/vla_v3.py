import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLAv3(nn.Module):
    """
    Variational Linear Attention v3.

    This module ports the stable formulation validated in
    notebooks/05_VLAv3_Complete_Fix.ipynb:
      - ELU+1 feature map for Q/K
      - full-gradient penalty-direction projection u = normalize(W_u k_raw)
      - Sherman–Morrison inverse tracking for A_t
      - normalized (k, alpha) update in S_t to keep Jacobian spectral norm bounded
      - z·q denominator normalization for calibrated outputs
    """

    def __init__(
        self,
        d_model: int,
        lambda_0: float = 10.0,
        stab_eps: float = 1e-4,
        per_eps: float = 1e-3,
        period: int = 20,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        d_model = self.d_model
        device = x.device

        k_raw = self.W_k(x).float()
        k_feat = F.elu(k_raw) + 1.0
        q_feat = F.elu(self.W_q(x).float()) + 1.0
        v_feat = self.W_v(x).float()
        u_dirs = F.normalize(self.u_proj(k_raw), p=2, dim=-1)

        eye = torch.eye(d_model, device=device, dtype=torch.float32)
        a_t = (1.0 / self.lambda_0) * eye.unsqueeze(0).expand(bsz, -1, -1).clone()
        s_t = torch.zeros(bsz, d_model, d_model, device=device, dtype=torch.float32)
        z_t = torch.zeros(bsz, d_model, device=device, dtype=torch.float32)
        inv_sqrt_d = 1.0 / math.sqrt(d_model)

        outputs = []
        for t in range(seq_len):
            u_t = u_dirs[:, t, :] * inv_sqrt_d
            uv = u_t.unsqueeze(-1)
            z_sm = torch.bmm(a_t, uv)
            dot = torch.bmm(uv.transpose(1, 2), z_sm).squeeze(-1).squeeze(-1)
            delta = torch.clamp(1.0 + dot, min=self.stab_eps)
            update = torch.bmm(z_sm, z_sm.transpose(1, 2)) / delta.view(bsz, 1, 1)
            a_t = a_t - update

            if (t + 1) % self.period == 0:
                a_t = a_t + self.per_eps * eye.unsqueeze(0)

            k_t = k_feat[:, t, :]
            v_t = v_feat[:, t, :]
            q_t = q_feat[:, t, :]

            k_n = F.normalize(k_t, p=2, dim=-1)
            alpha_t = torch.bmm(a_t, k_n.unsqueeze(-1)).squeeze(-1)
            alpha_n = F.normalize(alpha_t, p=2, dim=-1)

            err = v_t - torch.bmm(s_t, k_n.unsqueeze(-1)).squeeze(-1)
            s_t = s_t + torch.matmul(err.unsqueeze(2), alpha_n.unsqueeze(1))

            z_t = z_t + k_t
            o_t = torch.bmm(s_t, q_t.unsqueeze(-1)).squeeze(-1)
            denom = (z_t * q_t).sum(-1, keepdim=True).clamp(min=1e-6)
            outputs.append(o_t / denom)

        out = torch.stack(outputs, dim=1).to(dtype=x.dtype)
        return self.W_o(self.norm(out))
