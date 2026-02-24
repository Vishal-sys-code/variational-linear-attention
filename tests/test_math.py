import sys
import os

# Robust path setup to ensure src is importable regardless of pytest execution
# This must be done BEFORE importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
import numpy as np

from src.math.primitives import (
    inner_product_score,
    sherman_morrison_update,
    multiple_rank1_updates,
    memory_update,
    recover_alpha
)

# Helper function to generate random SPD matrix
def generate_spd(d: int, seed: int = None, dtype=torch.float64) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    # Generate random matrix
    A = torch.randn(d, d, dtype=dtype)
    # Make it SPD: A A^T + epsilon I
    return A @ A.T + 1e-3 * torch.eye(d, dtype=dtype)

# -----------------------------------------------------------------------------
# Test Set 1: Direct vs Sherman-Morrison Accuracy
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("d", [4, 16, 32])
def test_sherman_morrison_accuracy(d):
    """
    Compare Sherman-Morrison result against torch.inverse(M0 + u u^T).
    Assert relative Frobenius norm error < 1e-6.
    """
    torch.manual_seed(42)
    dtype = torch.float64
    
    # Generate random SPD matrix M0
    M0 = generate_spd(d, dtype=dtype)
    A0 = torch.inverse(M0)
    
    # Generate random update vector u
    u = torch.randn(d, dtype=dtype)
    
    # Compute true inverse of M = M0 + u u^T
    M_new = M0 + torch.outer(u, u)
    A_direct = torch.inverse(M_new)
    
    # Compute using Sherman-Morrison
    A_sm = sherman_morrison_update(A0, u)
    
    # Compute error
    diff = A_direct - A_sm
    norm_diff = torch.norm(diff, p='fro')
    norm_true = torch.norm(A_direct, p='fro')
    rel_error = norm_diff / norm_true
    
    assert rel_error < 1e-6, f"Relative error {rel_error} exceeds 1e-6 for d={d}"

# -----------------------------------------------------------------------------
# Test Set 2: Degenerate Update Stability
# -----------------------------------------------------------------------------
def test_degenerate_update_small_u():
    """
    Test when u is very small or nearly zero.
    """
    d = 16
    dtype = torch.float64
    torch.manual_seed(101)
    M0 = generate_spd(d, dtype=dtype)
    A0 = torch.inverse(M0)
    
    # Very small u
    u = torch.randn(d, dtype=dtype) * 1e-10
    
    # Should be very close to A0
    A_sm = sherman_morrison_update(A0, u)
    
    # Check validity
    assert torch.all(torch.isfinite(A_sm))
    
    # Check SPD property (eigenvalues > 0)
    eigvals = torch.linalg.eigvalsh(A_sm)
    assert torch.all(eigvals > 0), "Resulting matrix is not SPD"

def test_degenerate_update_collinear():
    """
    Test near-collinearity: u approx eigenvector of M0.
    """
    d = 16
    dtype = torch.float64
    torch.manual_seed(102)
    M0 = generate_spd(d, dtype=dtype)
    A0 = torch.inverse(M0)
    
    # Get an eigenvector
    eigvals, eigvecs = torch.linalg.eigh(M0)
    v = eigvecs[:, 0] # First eigenvector
    
    # u is scaled eigenvector
    u = 10.0 * v
    
    # This should be a large update in that direction
    A_sm = sherman_morrison_update(A0, u)
    
    assert torch.all(torch.isfinite(A_sm))
    eigvals_new = torch.linalg.eigvalsh(A_sm)
    assert torch.all(eigvals_new > 0), "Resulting matrix is not SPD"

# -----------------------------------------------------------------------------
# Test Set 3: Reproducibility
# -----------------------------------------------------------------------------
def test_reproducibility():
    """
    Run the same SM update pipeline twice and assert exact match.
    """
    d = 16
    seed = 42
    dtype = torch.float64
    
    # Run 1
    torch.manual_seed(seed)
    M0 = generate_spd(d, dtype=dtype)
    A0 = torch.inverse(M0)
    updates = [torch.randn(d, dtype=dtype) for _ in range(5)]
    
    A_final_1 = multiple_rank1_updates(A0, updates)
    
    # Run 2
    torch.manual_seed(seed)
    M0_2 = generate_spd(d, dtype=dtype)
    A0_2 = torch.inverse(M0_2)
    updates_2 = [torch.randn(d, dtype=dtype) for _ in range(5)]
    
    A_final_2 = multiple_rank1_updates(A0_2, updates_2)
    
    # Check strict equality for CPU
    assert torch.equal(A_final_1, A_final_2), "Reproducibility failed: results not identical"

# -----------------------------------------------------------------------------
# Test Set 4: Memory Update
# -----------------------------------------------------------------------------
def test_memory_update_shapes_and_inplace():
    """
    Confirm shape consistency and no in-place modifications.
    """
    d_v = 8
    d_k = 12
    S = torch.randn(d_v, d_k)
    v = torch.randn(d_v)
    k = torch.randn(d_k)
    alpha = 0.5
    
    # Copy S to check in-place modification
    S_orig = S.clone()
    
    S_new = memory_update(S, alpha, v, k)
    
    # Check shapes
    assert S_new.shape == (d_v, d_k)
    
    # Check not in-place
    assert not torch.equal(S, S_new), "S should be different from S_new"
    assert torch.equal(S, S_orig), "S should not be modified in-place"
    
    # Check alpha vector support
    alpha_vec = torch.randn(d_v) # Matches v
    S_vec = memory_update(S, alpha_vec, v, k)
    assert S_vec.shape == (d_v, d_k)
    
    # Check alpha vector k support
    alpha_k = torch.randn(d_k)
    S_k = memory_update(S, alpha_k, v, k)
    assert S_k.shape == (d_v, d_k)

# -----------------------------------------------------------------------------
# Test Set 5: Recover Alpha
# -----------------------------------------------------------------------------
def test_recover_alpha():
    """
    Validate alpha = A s matches solve(M, s).
    """
    d = 16
    dtype = torch.float64
    torch.manual_seed(2023)
    
    # Random M and s
    M = generate_spd(d, dtype=dtype)
    s = torch.randn(d, dtype=dtype)
    
    # Ground truth
    # alpha_gt = M^-1 s
    # We use solve for numerical stability check
    alpha_gt = torch.linalg.solve(M, s)
    
    # Using our primitives
    # Here we assume A is exactly inverse(M).
    # In practice A is built via SM, but here we test the recovery formula itself.
    A = torch.inverse(M)
    alpha_rec = recover_alpha(A, s)
    
    # Check error
    diff = alpha_rec - alpha_gt
    rel_error = torch.norm(diff) / torch.norm(alpha_gt)
    
    assert rel_error < 1e-6, f"Alpha recovery failed, relative error {rel_error}"

def test_recover_alpha_with_sm_built_A():
    """
    Validate alpha = A s where A is built via Sherman-Morrison.
    """
    d = 8
    dtype = torch.float64
    torch.manual_seed(2024)
    
    # Start with Identity
    M0 = torch.eye(d, dtype=dtype)
    A0 = torch.eye(d, dtype=dtype)
    
    updates = [torch.randn(d, dtype=dtype) * 0.5 for _ in range(5)]
    
    # Build M and A incrementally
    M = M0
    A = A0
    for u in updates:
        M = M + torch.outer(u, u)
        A = sherman_morrison_update(A, u)
        
    s = torch.randn(d, dtype=dtype)
    
    # Ground truth
    alpha_gt = torch.linalg.solve(M, s)
    
    # Recover
    alpha_rec = recover_alpha(A, s)
    
    # Check error
    rel_error = torch.norm(alpha_rec - alpha_gt) / torch.norm(alpha_gt)
    assert rel_error < 1e-6, f"Alpha recovery with SM failed, relative error {rel_error}"

# -----------------------------------------------------------------------------
# Additional Tests
# -----------------------------------------------------------------------------
def test_inner_product_score():
    k = torch.tensor([1.0, 2.0, 3.0])
    q = torch.tensor([0.5, 1.0, -0.5])
    # 0.5 + 2 - 1.5 = 1.0
    s = inner_product_score(k, q)
    assert torch.isclose(s, torch.tensor(1.0))
    
    # With scale
    s_scaled = inner_product_score(k, q, scale=2.0)
    assert torch.isclose(s_scaled, torch.tensor(2.0))
    
    # Finite check
    k_inf = torch.tensor([float('inf'), 0.0])
    q_inf = torch.tensor([0.0, 1.0])
    # dot is NaN or Inf depending on impl? 
    # inf * 0 is NaN.
    with pytest.raises(AssertionError):
        inner_product_score(k_inf, q_inf)

def test_associativity_check():
    """
    (u1 then u2) ~ (u2 then u1) numerically for random vectors.
    Strict associativity implies commutativity of updates if they are rank-1?
    M + u1 u1^T + u2 u2^T is commutative.
    So A should be commutative too.
    """
    d = 8
    torch.manual_seed(777)
    A0 = torch.eye(d)
    u1 = torch.randn(d)
    u2 = torch.randn(d)
    
    # Order 1
    A1 = sherman_morrison_update(A0, u1)
    A12 = sherman_morrison_update(A1, u2)
    
    # Order 2
    A2 = sherman_morrison_update(A0, u2)
    A21 = sherman_morrison_update(A2, u1)
    
    # Should be close
    diff = torch.norm(A12 - A21)
    assert diff < 1e-6, f"Commutativity failed: diff {diff}"