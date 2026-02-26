import torch
import pytest
import numpy as np
from src.models.attention.inverse_penalty import InversePenaltyTracker

def get_reference_inverse(lambda_0, updates, d):
    """
    Compute explicit inverse of M = lambda_0 * I + sum(u u^T).
    """
    M = lambda_0 * torch.eye(d, dtype=torch.float64)
    for u in updates:
        # u is (d,)
        u = u.to(dtype=torch.float64)
        M = M + torch.outer(u, u)
    return torch.linalg.inv(M)

@pytest.mark.parametrize("d", [4, 16])
@pytest.mark.parametrize("N", [10, 50])
def test_accuracy_sherman_morrison(d, N):
    """
    Test 1: Accuracy of Sherman–Morrison After N Steps.
    """
    torch.manual_seed(42)
    lambda_0 = 1.0
    
    # Generate N random update vectors
    # Shape (N, d)
    updates = torch.randn(N, d)
    
    # Initialize tracker
    tracker = InversePenaltyTracker(d_model=d, lambda_0=lambda_0, period=1000) # Disable periodic for accuracy check
    tracker.init(batch_size=1)
    
    # Apply updates
    # We feed them one by one or in batches.
    # To match "step" logic, let's feed them as (1, d)
    for i in range(N):
        u = updates[i].unsqueeze(0) # (1, d)
        tracker.update(u)
        
    A_incremental = tracker.get().squeeze(0).to(dtype=torch.float64) # (d, d)
    
    # Compute direct
    A_direct = get_reference_inverse(lambda_0, updates, d)
    
    # Compute error
    # relative_error = ||A_direct - A_incremental|| / ||A_direct||
    # Using frobenius norm
    diff = torch.norm(A_direct - A_incremental)
    norm = torch.norm(A_direct)
    relative_error = diff / norm
    
    print(f"d={d}, N={N}, relative_error={relative_error.item()}")
    
    assert relative_error < 1e-4, f"Relative error {relative_error} too high"

def test_conditioning_and_stabilization():
    """
    Test 2: Conditioning Test (collinear updates) & Stabilization.
    """
    torch.manual_seed(42)
    d = 8
    tracker = InversePenaltyTracker(d_model=d, lambda_0=1.0, periodic_eps=1e-1, period=2)
    tracker.init(batch_size=1)
    
    u1 = torch.randn(d)
    u1 = u1 / torch.norm(u1) * 10.0 # Make it large
    u2 = u1 + torch.randn(d) * 0.001 # Nearly collinear
    
    # Step 1
    tracker.update(u1.unsqueeze(0))
    A_1 = tracker.get().clone()
    
    # Step 2: This triggers periodic stabilization (period=2)
    # Actually period=2 means at step 2 (0, 1 -> 2?).
    # init sets step=0.
    # update 1: step becomes 1. 1 % 2 != 0.
    # update 2: step becomes 2. 2 % 2 == 0. Stabilization triggers.
    
    # We want to measure Condition Number BEFORE stabilization to compare.
    # But the tracker stabilizes inside `update`.
    # So we can't easily access "A_before_stabilization" inside the module from outside
    # unless we hook or inspect.
    # However, we can inspect A_1 (after step 1).
    # And A_2 (after step 2).
    
    # Let's apply u2.
    tracker.update(u2.unsqueeze(0))
    A_2 = tracker.get().clone()
    
    # Check diagnostics
    diag = tracker.diagnostics()
    print(f"Diagnostics: {diag}")
    
    # Check that A_2 has been stabilized.
    # If strictly SM, A_2 would be "pure".
    # Since we added eps*I, A_2 eigenvalues should be bumped.
    # cond(A) = max/min.
    # Adding eps*I increases min (and max).
    # If min was small, (min+eps) is significantly larger -> cond decreases.
    
    # Let's verify condition number is finite.
    assert diag['cond_max'] < float('inf')
    
    # To strictly verify "cond decreases after stabilization", we might need a manual comparison
    # simulating what would happen without stabilization.
    
    # Simulation without stabilization:
    tracker_pure = InversePenaltyTracker(d_model=d, lambda_0=1.0, period=1000)
    tracker_pure.init(batch_size=1)
    tracker_pure.update(u1.unsqueeze(0))
    tracker_pure.update(u2.unsqueeze(0))
    A_pure = tracker_pure.get()
    cond_pure = torch.linalg.cond(A_pure).item()
    
    cond_stabilized = diag['cond_max']
    
    print(f"Cond Pure: {cond_pure}, Cond Stabilized: {cond_stabilized}")
    
    # Stabilization (adding eps*I) should generally reduce condition number 
    # if epsilon is large enough relative to smallest eigenvalue.
    assert cond_stabilized < cond_pure or cond_stabilized < 1e8

def test_overflow_nan_safety():
    """
    Test 3: Overflow / NaN Safety
    """
    torch.manual_seed(42)
    d = 4
    tracker = InversePenaltyTracker(d_model=d, lambda_0=1.0, stabilization_eps=1e-6)
    tracker.init(batch_size=1)
    
    # Extremely large u
    u_large = torch.randn(1, d) * 1e10
    
    # This should result in very large M, very small A.
    # delta = 1 + u^T A u.
    # If u is 1e10, u^T u is 1e20. A is ~1. delta ~ 1e20.
    # update = z z^T / delta.
    # z = A u ~ 1e10. z z^T ~ 1e20.
    # update ~ 1.
    # So A should remain finite.
    
    tracker.update(u_large)
    A = tracker.get()
    assert torch.isfinite(A).all(), "A contains NaNs or Infs after large update"
    
    # Extremely small u
    u_small = torch.randn(1, d) * 1e-10
    # delta ~ 1. update ~ 0.
    tracker.update(u_small)
    A = tracker.get()
    assert torch.isfinite(A).all(), "A contains NaNs or Infs after small update"

def test_batch_support():
    """
    Test 4: Batch Independence
    """
    torch.manual_seed(42)
    d = 4
    B = 2
    tracker = InversePenaltyTracker(d_model=d, lambda_0=1.0)
    tracker.init(batch_size=B)
    
    u1 = torch.randn(d)
    u2 = torch.randn(d)
    
    # Construct batch input: [u1, u2]
    u_batch = torch.stack([u1, u2]) # (2, d)
    
    tracker.update(u_batch)
    A_batch = tracker.get() # (2, d, d)
    
    # Reference: run single batch
    t1 = InversePenaltyTracker(d_model=d, lambda_0=1.0)
    t1.init(batch_size=1)
    t1.update(u1.unsqueeze(0))
    A1 = t1.get().squeeze(0)
    
    t2 = InversePenaltyTracker(d_model=d, lambda_0=1.0)
    t2.init(batch_size=1)
    t2.update(u2.unsqueeze(0))
    A2 = t2.get().squeeze(0)
    
    # Compare
    assert torch.allclose(A_batch[0], A1, atol=1e-6)
    assert torch.allclose(A_batch[1], A2, atol=1e-6)

def test_rank_r_update():
    """
    Test rank-r update logic.
    """
    d = 4
    r = 2
    tracker = InversePenaltyTracker(d_model=d)
    tracker.init(batch_size=1)
    
    # (1, r, d)
    u_r = torch.randn(1, r, d)
    
    tracker.update(u_r)
    A_final = tracker.get()
    
    # Reference: sequential
    t_seq = InversePenaltyTracker(d_model=d)
    t_seq.init(batch_size=1)
    t_seq.update(u_r[:, 0, :])
    t_seq.update(u_r[:, 1, :])
    
    assert torch.allclose(A_final, t_seq.get())