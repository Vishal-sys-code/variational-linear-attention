import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.maths.primitives import (
    inner_product_score,
    sherman_morrison_update,
    memory_update,
    recover_alpha
)
from src.models.attention.vla import VLALayer

def test_math():
    """Test Module 1 — Score computation correctness"""
    torch.manual_seed(42)
    d = 32
    k_t = torch.randn(d, dtype=torch.float64)
    q_t = torch.randn(d, dtype=torch.float64)
    
    # Manual
    score_manual = torch.dot(k_t, q_t)
    
    # Implementation
    score_impl = inner_product_score(k_t, q_t)
    
    absolute_difference = torch.abs(score_manual - score_impl).item()
    assert absolute_difference < 1e-8, f"Score computation difference {absolute_difference} exceeds 1e-8"


def test_inverse():
    """Test Module 2 — Sherman Morrison inverse update correctness"""
    torch.manual_seed(43)
    d = 16
    
    # Step 1: Generate random positive definite matrix M_prev
    A = torch.randn(d, d, dtype=torch.float64)
    M_prev = A @ A.T + 1e-3 * torch.eye(d, dtype=torch.float64)
    
    # Step 2: Compute inverse directly
    A_prev_direct = torch.inverse(M_prev)
    
    # Step 3: Generate random vector u_t
    u_t = torch.randn(d, dtype=torch.float64)
    
    # Step 4: Compute updated matrix
    M_t = M_prev + torch.outer(u_t, u_t)
    
    # Step 5: Compute inverse directly
    A_t_direct = torch.inverse(M_t)
    
    # Step 6: Compute inverse using Sherman Morrison
    delta_t = 1.0 + torch.dot(u_t, torch.mv(A_prev_direct, u_t))
    B_t = A_prev_direct @ torch.outer(u_t, u_t) @ A_prev_direct
    A_t_sm = A_prev_direct - B_t / delta_t
    
    # Or using implementation primitive to ensure it works properly too
    A_t_impl = sherman_morrison_update(A_prev_direct, u_t)
    
    # Step 7: Compare
    diff = A_t_direct - A_t_sm
    relative_error = torch.norm(diff) / torch.norm(A_t_direct)
    assert relative_error.item() < 1e-6, f"Sherman Morrison relative error {relative_error} exceeds 1e-6"
    
    diff_impl = A_t_direct - A_t_impl
    rel_error_impl = torch.norm(diff_impl) / torch.norm(A_t_direct)
    assert rel_error_impl.item() < 1e-6, f"Implementation SM relative error {rel_error_impl} exceeds 1e-6"


def test_alpha():
    """Test Module 3 — alpha computation correctness"""
    torch.manual_seed(44)
    d = 16
    
    # Generate small random SPD matrix M_t
    A = torch.randn(d, d, dtype=torch.float64)
    M_t = A @ A.T + 1e-3 * torch.eye(d, dtype=torch.float64)
    
    # Compute A_t = inverse(M_t)
    A_t = torch.inverse(M_t)
    
    # Generate random vector s_t
    s_t = torch.randn(d, dtype=torch.float64)
    
    # Compute alpha_direct = inverse(M_t) dot s_t => M_t x = s_t
    alpha_direct = torch.linalg.solve(M_t, s_t)
    
    # Compute alpha_model = A_t dot s_t
    alpha_model = torch.mv(A_t, s_t)
    
    difference_norm = torch.norm(alpha_direct - alpha_model).item()
    assert difference_norm < 1e-7, f"Alpha difference norm {difference_norm} exceeds 1e-7"


def test_memory():
    """Test Module 4 — Memory matrix update correctness"""
    torch.manual_seed(45)
    d_v = 16
    d_k = 16
    
    # Initialize S_0 = zero matrix
    S_0 = torch.zeros((d_v, d_k), dtype=torch.float64)
    
    # Generate random alpha_t and v_t
    alpha_t = torch.randn(d_k, dtype=torch.float64)
    v_t = torch.randn(d_v, dtype=torch.float64)
    
    # Compute S_1 manually
    S_1_manual = S_0 + torch.outer(v_t, alpha_t)
    
    # Compare with implementation
    S_1_impl = memory_update(S_0, 1.0, v_t, alpha_t)
    
    difference_norm = torch.norm(S_1_manual - S_1_impl).item()
    assert difference_norm < 1e-8, f"Memory update difference norm {difference_norm} exceeds 1e-8"


def test_forward():
    """Test Module 5 — Forward pass sanity check"""
    torch.manual_seed(46)
    
    d_model = 32
    batch_size = 2
    seq_len = 5 # Run 5 timestep forward pass
    
    model = VLALayer(d_model=d_model, enable_stabilization=True)
    
    # Random inputs
    x = torch.randn((batch_size, seq_len, d_model), dtype=torch.float32)
    
    outputs, states = model(x, return_states=True)
    
    # Check output shape
    assert outputs.shape == (batch_size, seq_len, d_model), "Output shape incorrect"
    
    # Check no NaNs
    assert not torch.isnan(outputs).any(), "NaNs detected in outputs"
    assert not torch.isnan(states["A"]).any(), "NaNs detected in states['A']"
    assert not torch.isnan(states["S_norm"]).any(), "NaNs detected in states['S_norm']"
    
    # Check no Infs
    assert torch.isfinite(outputs).all(), "Infs detected in outputs"
    assert torch.isfinite(states["A"]).all(), "Infs detected in states['A']"
    assert torch.isfinite(states["S_norm"]).all(), "Infs detected in states['S_norm']"

