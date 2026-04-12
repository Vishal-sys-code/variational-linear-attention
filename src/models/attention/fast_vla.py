import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.attention.penalty_builder import PenaltyBuilder

# Try importing Triton
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def _combine(F_l, G_l, F_r, G_r):
    """Associative operator for the (F, G) recurrence."""
    return torch.matmul(F_r, F_l).to(F_l.dtype), (torch.matmul(F_r, G_l) + G_r).to(G_l.dtype)

def parallel_scan(Fs, Gs):
    """
    Inclusive parallel prefix scan for S_t = F_t @ S_{t-1} + G_t,  S_0 = 0.
    Args:
        Fs, Gs : (T, B, d, d)
    Returns:
        S_all  : (T, B, d, d)  where S_all[t] = S_{t+1}
    """
    T, B, d, _ = Fs.shape

    # Pad to next power of 2
    T_pad = 1 << (T - 1).bit_length()
    pad   = T_pad - T
    if pad > 0:
        I_p = torch.eye(d, device=Fs.device, dtype=Fs.dtype) \
                   .unsqueeze(0).unsqueeze(0).expand(pad, B, -1, -1)
        Z_p = torch.zeros(pad, B, d, d, device=Fs.device, dtype=Fs.dtype)
        Fs  = torch.cat([Fs, I_p], dim=0)
        Gs  = torch.cat([Gs, Z_p], dim=0)

    Fc, Gc = Fs.clone().to(Fs.dtype), Gs.clone().to(Gs.dtype)

    # ── Up-sweep ──────────────────────────────────────────────────────────
    stride = 1
    while stride < T_pad:
        r = torch.arange(2 * stride - 1, T_pad, 2 * stride, device=Fs.device)
        l = r - stride
        if r.numel() > 0:
            Fc[r], Gc[r] = _combine(Fc[l], Gc[l], Fc[r], Gc[r])
        stride *= 2

    # ── Down-sweep ────────────────────────────────────────────────────────
    Fc[T_pad - 1] = torch.eye(d, device=Fs.device, dtype=Fs.dtype) \
                         .unsqueeze(0).expand(B, -1, -1)
    Gc[T_pad - 1] = torch.zeros(B, d, d, device=Fs.device, dtype=Fs.dtype)

    stride = T_pad // 2
    while stride >= 1:
        r = torch.arange(2 * stride - 1, T_pad, 2 * stride, device=Fs.device)
        l = r - stride
        if r.numel() > 0:
            Fr, Gr = Fc[r].clone(), Gc[r].clone()
            Fl, Gl = Fc[l].clone(), Gc[l].clone()
            Fc[l], Gc[l] = Fr, Gr
            Fc[r], Gc[r] = _combine(Fl, Gl, Fr, Gr)
        stride //= 2

    # Inclusive: apply original (Fs, Gs) to exclusive prefix
    S_all = torch.matmul(Fs[:T], Gc[:T]) + Gs[:T]
    return S_all

def _compute_A_sequence(U_seq, lambda_0, d, stab_eps=1e-6, per_eps=1e-5, period=50):
    """
    Sequential Sherman-Morrison: A_t for t = 1..T.
    No .item() — fully torch.compile-able.
    """
    B, T, _ = U_seq.shape
    device, dtype = U_seq.device, U_seq.dtype

    I   = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    A_t = (1.0 / lambda_0) * I.clone()
    A_all = []

    for t in range(T):
        u    = U_seq[:, t, :] / math.sqrt(d)          # (B, d)  — normalise
        uv   = u.unsqueeze(-1)                          # (B, d, 1)
        z    = torch.bmm(A_t, uv)                       # (B, d, 1)
        dot  = torch.bmm(uv.transpose(1, 2), z).squeeze(-1).squeeze(-1)  # (B,)
        delt = torch.clamp(1.0 + dot, min=stab_eps)    # (B,)

        upd  = torch.bmm(z, z.transpose(1, 2)) / delt.view(B, 1, 1)  # (B,d,d)
        bad  = (torch.abs(delt) < stab_eps).view(B, 1, 1)
        A_t  = torch.where(bad, A_t + stab_eps * I, A_t - upd)

        if (t + 1) % period == 0:
            A_t = A_t + per_eps * I

        A_all.append(A_t.clone())
        
    return torch.stack(A_all, dim=0)  # (T, B, d, d)


if HAS_TRITON:
    @triton.jit
    def _sm_scan_kernel(
        U_ptr, K_ptr, Out_ptr,
        T,
        stride_Ub, stride_Ut,
        stride_Kb, stride_Kt,
        stride_Ob, stride_Ot,
        inv_lambda0: tl.constexpr,
        inv_sqrt_D:  tl.constexpr,
        stab_eps:    tl.constexpr,
        period:      tl.constexpr,
        per_eps:     tl.constexpr,
        D:           tl.constexpr,   # power-of-2, <= 128
    ):
        pid  = tl.program_id(0)      # one program per batch element
        ids  = tl.arange(0, D)       # [0 .. D-1]

        # Initialise A = (1/lambda_0) * I  in registers
        A = tl.where(
            ids[:, None] == ids[None, :],
            inv_lambda0, 0.0,
        ).to(tl.float32)

        # Pre-build stability / periodic nudge matrices
        stab_mat = tl.where(ids[:, None] == ids[None, :], stab_eps, 0.0).to(tl.float32)
        per_mat  = tl.where(ids[:, None] == ids[None, :], per_eps,  0.0).to(tl.float32)

        U_base = U_ptr   + pid * stride_Ub
        K_base = K_ptr   + pid * stride_Kb
        O_base = Out_ptr + pid * stride_Ob

        for t in tl.range(T):
            u = tl.load(U_base + t * stride_Ut + ids).to(tl.float32) * inv_sqrt_D
            k = tl.load(K_base + t * stride_Kt + ids).to(tl.float32)

            z = tl.sum(A * u[None, :], axis=1)

            delta = tl.sum(u * z) + 1.0
            delta = tl.maximum(delta, stab_eps)

            outer = z[:, None] * z[None, :]
            A_new = A - outer / delta

            A = tl.where(delta <= stab_eps, A + stab_mat, A_new)

            if (t + 1) % period == 0:
                A = A + per_mat

            alpha = tl.sum(A * k[:, None], axis=0)
            tl.store(O_base + t * stride_Ot + ids, alpha)

    def sm_scan_triton(
        U: torch.Tensor, K: torch.Tensor,
        lambda_0: float = 1.0,
        stab_eps: float = 1e-6,
        per_eps:  float = 1e-5,
        period:   int   = 50,
    ) -> torch.Tensor:
        B, T, D = U.shape
        # Check power of 2 and <= 128
        assert D & (D - 1) == 0 and D <= 128, f"D must be power-of-2 <= 128, got {D}"
        assert U.is_contiguous() and K.is_contiguous()

        alpha = torch.empty_like(U)
        _sm_scan_kernel[(B,)](
            U, K, alpha, T,
            U.stride(0), U.stride(1),
            K.stride(0), K.stride(1),
            alpha.stride(0), alpha.stride(1),
            inv_lambda0 = 1.0 / lambda_0,
            inv_sqrt_D  = 1.0 / math.sqrt(D),
            stab_eps    = stab_eps,
            period      = period,
            per_eps     = per_eps,
            D           = D,
        )
        return alpha


class _VLABase(nn.Module):
    """Shared projections + S_t scan logic."""

    def __init__(self, d_model, lambda_0=1.0, fixed_lambda=None,
                 stab_eps=1e-6, per_eps=1e-5, period=50, use_kv_exploding_fix=False):
        super().__init__()
        self.d_model  = d_model
        self.lambda_0 = lambda_0
        self.stab_eps = stab_eps
        self.per_eps  = per_eps
        self.period   = period
        self.use_kv_exploding_fix = use_kv_exploding_fix

        self.W_q      = nn.Linear(d_model, d_model)
        self.W_k      = nn.Linear(d_model, d_model)
        self.W_v      = nn.Linear(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.W_o      = nn.Linear(d_model, d_model)
        self.penalty  = PenaltyBuilder(d_model, fixed_lambda=fixed_lambda)

    def _build_S(self, alpha_all, K, V, d, B, T, device, dtype):
        """Parallel scan for S_t given pre-computed alpha_all (T,B,d)."""
        K_tr  = K.permute(1, 0, 2)   # (T, B, d)
        V_tr  = V.permute(1, 0, 2)
        I_exp = torch.eye(d, device=device, dtype=dtype) \
                     .unsqueeze(0).unsqueeze(0).expand(T, B, -1, -1)

        # Normalise k and alpha before rank-1 products to keep F_t a contraction.
        # Without this, composing F matrices in the binary tree causes
        # exponential blowup -> NaN gradients at long T.
        # This acts as the KV Exploding fix.
        if self.use_kv_exploding_fix:
            k_n = F.normalize(K_tr, dim=-1)
            a_n = F.normalize(alpha_all, dim=-1)
        else:
            k_n = K_tr
            a_n = alpha_all

        k_alpha = torch.matmul(k_n.unsqueeze(-1), a_n.unsqueeze(-2))   # (T,B,d,d)
        v_alpha = torch.matmul(V_tr.unsqueeze(-1), a_n.unsqueeze(-2))  # (T,B,d,d)

        # Clip G to prevent S explosion
        vn = torch.norm(v_alpha, dim=(-2, -1), keepdim=True)
        v_alpha = torch.where(vn > 10.0, v_alpha * 10.0 / (vn + 1e-6), v_alpha)

        Fs = I_exp - k_alpha
        Gs = v_alpha
        return parallel_scan(Fs, Gs)   # (T, B, d, d)

    def _output(self, S_all, Q, d, B, T, dtype):
        Q_tr = Q.permute(1, 0, 2).unsqueeze(-1)             # (T, B, d, 1)
        O    = torch.matmul(S_all, Q_tr).squeeze(-1) \
                    .permute(1, 0, 2).to(dtype)              # (B, T, d)
        return self.W_o(self.out_norm(O))


class VLASequential(_VLABase):
    """Baseline: Python for-loop over T (one CUDA kernel launch per token)."""

    def forward(self, x):
        B, T, _ = x.shape
        d = self.d_model
        device, dtype = x.device, x.dtype

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        
        _, U_seq, _ = self.penalty(K)

        I   = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        A_t = (1.0 / self.lambda_0) * I.clone()
        S_t = torch.zeros(B, d, d, device=device, dtype=dtype)
        outputs = []

        for t in range(T):
            u    = U_seq[:, t, :].float() / math.sqrt(d)
            uv   = u.unsqueeze(-1)
            z    = torch.bmm(A_t, uv)
            dot  = torch.bmm(uv.transpose(1, 2), z).squeeze(-1).squeeze(-1)
            delt = torch.clamp(1.0 + dot, min=self.stab_eps)
            upd  = torch.bmm(z, z.transpose(1, 2)) / delt.view(B, 1, 1)
            bad  = (torch.abs(delt) < self.stab_eps).view(B, 1, 1)
            A_t  = torch.where(bad, A_t + self.stab_eps * I, A_t - upd)
            if (t + 1) % self.period == 0:
                A_t = A_t + self.per_eps * I

            k_t     = K[:, t, :].float()
            v_t     = V[:, t, :].float()
            q_t     = Q[:, t, :].float()
            alpha_t = torch.bmm(A_t, k_t.unsqueeze(-1)).squeeze(-1)
            
            v_hat = torch.bmm(S_t, k_t.unsqueeze(-1)).squeeze(-1)
            e_t   = v_t - v_hat
            en    = torch.norm(e_t, dim=-1, keepdim=True)
            e_t   = torch.where(en > 10.0, e_t * 10.0 / (en + 1e-6), e_t)

            S_t = S_t + torch.matmul(e_t.unsqueeze(2), alpha_t.unsqueeze(1))
            outputs.append(torch.matmul(S_t, q_t.unsqueeze(2)).squeeze(2))

        O = torch.stack(outputs, dim=1).to(dtype)
        return self.W_o(self.out_norm(O))


class VLAParallel(_VLABase):
    """Python A_t loop + parallel prefix scan for S_t."""

    def forward(self, x):
        B, T, _ = x.shape
        d = self.d_model
        device, dtype = x.device, x.dtype

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        
        _, U_seq, _ = self.penalty(K)

        A_all     = _compute_A_sequence(U_seq.float(), self.lambda_0, d,
                                        self.stab_eps, self.per_eps, self.period)
        K_t       = K.permute(1, 0, 2).unsqueeze(-1)
        # Use a_t_scaled computation from VLA base if needed, but here alpha_t = A_t @ k_t
        alpha_all = torch.matmul(A_all, K_t).squeeze(-1)   # (T, B, d)

        S_all = self._build_S(alpha_all, K, V, d, B, T, device, dtype)
        return self._output(S_all, Q, d, B, T, dtype)


class VLATriton(_VLABase):
    """
    Triton kernel fuses entire A_t loop into ONE kernel launch.
    """

    def forward(self, x):
        assert HAS_TRITON and x.device.type == "cuda", \
            "VLATriton requires CUDA + Triton"
        B, T, _ = x.shape
        d = self.d_model
        device, dtype = x.device, x.dtype

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        
        _, U_seq, _ = self.penalty(K)

        U_c = U_seq.float().contiguous()
        K_c = K.float().contiguous()
        alpha_all = sm_scan_triton(
            U_c, K_c,
            lambda_0 = self.lambda_0,
            stab_eps = self.stab_eps,
            per_eps  = self.per_eps,
            period   = self.period,
        )                                     # (B, T, d)
        alpha_all = alpha_all.permute(1, 0, 2)  # -> (T, B, d)

        S_all = self._build_S(alpha_all, K, V, d, B, T, device, dtype)
        return self._output(S_all, Q, d, B, T, dtype)


