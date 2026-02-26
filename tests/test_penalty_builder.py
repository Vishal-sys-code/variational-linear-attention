# tests/test_penalty_builder.py
import torch
import pytest
import sys
import os

# Ensure src is importable (adjust relative path if needed)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.attention.penalty_builder import PenaltyBuilder, KernelPenaltyBuilder
from src.math.primitives import sherman_morrison_update


# -----------------------------------------------------------------------------
# CHECK 1 — DeltaNet Recovery (Exact) using torch.float64 and pure-torch ops
# -----------------------------------------------------------------------------
def test_check_1_deltanet_recovery():
    torch.manual_seed(42)
    dtype = torch.float64     # use double precision for numerical stability in this unit test
    device = torch.device("cpu")

    d = 4
    T = 6

    S_delta = torch.zeros(d, d, dtype=dtype, device=device)

    keys = []
    vals = []
    betas = []

    for t in range(T):
        k_t = torch.randn(d, dtype=dtype, device=device)
        v_t = torch.randn(d, dtype=dtype, device=device)
        q_t = torch.randn(d, dtype=dtype, device=device)

        beta_t = float(0.1 + 0.8 * torch.rand(1).item())  # stable range

        # DeltaNet recurrence (ground truth)
        term1 = beta_t * torch.outer(S_delta @ k_t, k_t)
        term2 = beta_t * torch.outer(v_t, k_t)
        S_delta = S_delta - term1 + term2
        o_delta = S_delta @ q_t

        # append history
        keys.append(k_t)
        vals.append(v_t)
        betas.append(beta_t)

        # Compute alphas using exact recurrence-compatible formula
        alphas = []

        for i in range(t + 1):
            k_i = keys[i]
            beta_i = betas[i]

            # γ_i needs to be a vector to track direction changes
            # We are computing: k_i^T * Prod_{j=i+1}^t (I - beta_j k_j k_j^T) * q_t
            # Let gamma_vec = k_i^T * Prod_{j=i+1}^t (I - beta_j k_j k_j^T)
            gamma_vec = k_i.clone()

            # Apply decay for j = i+1 .. t
            for j in range(i + 1, t + 1):
                k_j = keys[j]
                beta_j = betas[j]
                # gamma_vec = gamma_vec (I - beta_j k_j k_j^T)
                #           = gamma_vec - beta_j * (gamma_vec . k_j) * k_j
                dot_val = torch.dot(gamma_vec, k_j)
                gamma_vec = gamma_vec - beta_j * dot_val * k_j

            alpha_i = beta_i * torch.dot(gamma_vec, q_t)
            alphas.append(alpha_i)

        alpha_vec = torch.stack(alphas, dim=0)         # (t+1,)
        V = torch.stack(vals[: t + 1], dim=1)          # (d, t+1)
        o_alpha = V @ alpha_vec                        # (d,)

        err = torch.norm(o_alpha - o_delta).item()
        assert err < 1e-8, f"DeltaNet mismatch at t={t}, err={err}"

    assert True


# -----------------------------------------------------------------------------
# CHECK 2 — SPD Property (use float64 here too for stability)
# -----------------------------------------------------------------------------
def test_check_2_spd_property():
    torch.manual_seed(0)
    dtype = torch.float64
    device = torch.device("cpu")

    d = 8
    B = 4
    builder = PenaltyBuilder(d_model=d, rank=1).to(dtype=dtype, device=device)

    k = torch.randn(B, d, dtype=dtype, device=device)
    lambda_t, u_t, _ = builder(k)

    for i in range(B):
        lam = float(lambda_t[i].item())
        u = u_t[i]
        M = lam * torch.eye(d, dtype=dtype, device=device) + torch.outer(u, u)

        # symmetry
        assert torch.allclose(M, M.T, atol=1e-10), "M_t not symmetric"

        eigvals = torch.linalg.eigvalsh(M)
        min_eig = float(eigvals.min().item())

        # allow tiny numerical slack
        tol = 1e-10
        assert (min_eig + tol) >= (lam - tol), f"Min eig {min_eig} < lambda {lam}"
        assert min_eig > 0.0, f"M is not PD, min_eig={min_eig}"


# -----------------------------------------------------------------------------
# CHECK 3 — Rank-r shapes (no change except dtype)
# -----------------------------------------------------------------------------
def test_check_3_rank_r_consistency():
    torch.manual_seed(1)
    dtype = torch.float64
    device = torch.device("cpu")

    d = 8
    B = 2
    T = 3

    b1 = PenaltyBuilder(d_model=d, rank=1).to(dtype=dtype, device=device)
    k_seq = torch.randn(B, T, d, dtype=dtype, device=device)
    l1, u1, _ = b1(k_seq)
    assert l1.shape == (B, T, 1)
    assert u1.shape == (B, T, d)

    r = 3
    br = PenaltyBuilder(d_model=d, rank=r).to(dtype=dtype, device=device)
    lr, ur, _ = br(k_seq)
    assert lr.shape == (B, T, 1)
    assert ur.shape == (B, T, r, d)

    k_single = torch.randn(B, d, dtype=dtype, device=device)
    l_single, u_single, _ = b1(k_single)
    assert u_single.shape == (B, d)


# -----------------------------------------------------------------------------
# CHECK 4 — Sherman–Morrison safety (float64)
# -----------------------------------------------------------------------------
def test_check_4_denominator_safety():
    torch.manual_seed(2)
    dtype = torch.float64
    device = torch.device("cpu")

    d = 4
    A_prev = torch.eye(d, dtype=dtype, device=device)

    u_small = torch.randn(d, dtype=dtype, device=device) * 1e-6
    A_new = sherman_morrison_update(A_prev, u_small)
    assert torch.all(torch.isfinite(A_new))

    A_bad = -1.0 * torch.eye(d, dtype=dtype, device=device)
    u_bad = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    try:
        A_safe = sherman_morrison_update(A_bad, u_bad)
        assert torch.all(torch.isfinite(A_safe))
    except Exception as e:
        pytest.fail(f"SM crashed on pathological input: {e}")