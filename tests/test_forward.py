import torch
import pytest
import numpy as np
from src.models.attention.vla import VLALayer

# Use CPU for deterministic testing
DEVICE = torch.device('cpu')
DTYPE = torch.float32

def test_vla_forward_determinism():
    """
    Check 1 — Determinism
    Under a fixed random seed and identical inputs, the forward pass MUST produce identical outputs.
    """
    torch.manual_seed(42)
    B, T, d_model = 2, 10, 8
    model = VLALayer(d_model=d_model).to(DEVICE)
    x = torch.randn(B, T, d_model, device=DEVICE)
    
    # Run 1
    model.eval() # ensure dropout is off if any (though none here)
    with torch.no_grad():
        out1 = model(x)
        
    # Run 2
    with torch.no_grad():
        out2 = model(x)
        
    assert torch.allclose(out1, out2), "Forward pass is not deterministic!"

def test_vla_small_t_reference():
    """
    Check 2 — Small-T correctness (reference inversion)
    For sequences of length T <= 64:
    Build M_t directly via accumulation of lambda_t (here assumed lambda_0 * I) and u_t u_t^T.
    Compute alpha_t_ref = inverse(M_t) * s_vector (Note: s_t is scalar, so alpha_t = s_t * A_t * u_t)
    Compute S_t_ref explicitly.
    Compute o_t_ref = S_t_ref * q_t
    Compare o_t vs o_t_ref
    """
    torch.manual_seed(123)
    B, T, d_model = 2, 10, 4
    d_head = 4
    lambda_0 = 1.0
    
    # Instantiate model
    model = VLALayer(d_model=d_model, d_head=d_head, lambda_0=lambda_0).to(DEVICE)
    model.eval()
    
    x = torch.randn(B, T, d_model, device=DEVICE)
    
    # Get model output
    with torch.no_grad():
        model_out = model(x) # (B, T, d_head)
        
    # Reference Implementation
    # We iterate batch items manually or vectorized
    
    # Weights
    W_q = model.W_q
    W_k = model.W_k
    W_v = model.W_v
    penalty_builder = model.penalty_builder
    
    # Loop over batch
    for b in range(B):
        # Initial State
        # M_0 = lambda_0 * I
        M = lambda_0 * torch.eye(d_head, device=DEVICE)
        S = torch.zeros(d_head, d_head, device=DEVICE)
        
        for t in range(T):
            x_t = x[b, t] # (d_model,)
            
            # Projections
            q_t = W_q(x_t) # (d_head,)
            k_t = W_k(x_t) # (d_head,)
            v_t = W_v(x_t) # (d_head,)
            
            # Score
            s_t = torch.dot(k_t, q_t) # Scalar
            
            # Penalty
            # Model uses penalty_builder(k_t)
            # penalty_builder expects (..., d_model/d_head)
            # It returns lambda_t, u_t, stats
            _, u_t, _ = penalty_builder(k_t.unsqueeze(0)) # Input (1, d_head) -> u_t (1, d_head)
            u_t = u_t.squeeze(0) # (d_head,)
            
            # Update M
            # M_t = M_{t-1} + u_t u_t^T
            M = M + torch.outer(u_t, u_t)
            
            # Compute A_t = inv(M)
            # Use float64 for reference inversion stability
            M_64 = M.to(torch.float64)
            A_64 = torch.linalg.inv(M_64)
            A = A_64.to(torch.float32)
            
            # Compute alpha_t
            # alpha_t = s_t * (A * u_t)
            z_t = torch.mv(A, u_t) # (d_head,)
            alpha_t = s_t * z_t # (d_head,)
            
            # Update S
            # S_t = S_{t-1} + v_t alpha_t^T
            S = S + torch.outer(v_t, alpha_t)
            
            # Output o_t
            # o_t = S_t q_t
            o_t_ref = torch.mv(S, q_t) # (d_head,)
            
            # Compare with model output
            o_t_model = model_out[b, t]
            
            # Check relative error
            # tol=1e-4
            if not torch.allclose(o_t_model, o_t_ref, rtol=1e-4, atol=1e-4):
                print(f"Mismatch at batch {b}, step {t}")
                print(f"Ref: {o_t_ref}")
                print(f"Model: {o_t_model}")
                print(f"Diff: {o_t_model - o_t_ref}")
                
            assert torch.allclose(o_t_model, o_t_ref, rtol=1e-4, atol=1e-4), \
                f"Mismatch at b={b}, t={t}"

if __name__ == "__main__":
    test_vla_forward_determinism()
    test_vla_small_t_reference()
    print("All manual tests passed!")
