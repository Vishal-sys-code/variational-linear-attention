import torch
import pytest
from src.models.attention.memory_matrix import MemoryMatrixManager

@pytest.fixture
def memory_manager():
    d_model = 8
    manager = MemoryMatrixManager(d_model=d_model, enable_renorm=False)
    return manager

def test_rank1_update(memory_manager):
    """
    Test 1 — Correct rank-1 update
    For random vectors v_t and alpha_t, verify that:
    S_t_computed = S_{t-1} + v_t (alpha_t)^T
    matches the manually computed update.
    """
    torch.manual_seed(42)
    B = 2
    d = memory_manager.d_model
    
    memory_manager.reset(batch_size=B)
    
    v_t = torch.randn(B, d)
    alpha_t = torch.randn(B, d)
    
    # Initial S_{t-1} is zeros
    S_prev = memory_manager.get_S().clone()
    
    # Perform update
    stats = memory_manager.update(v_t, alpha_t)
    S_curr = memory_manager.get_S()
    
    # Manual update
    # v_t: (B, d) -> (B, d, 1)
    # alpha_t: (B, d) -> (B, 1, d)
    v_t_f32 = v_t / (torch.norm(v_t, dim=-1, keepdim=True) + 1e-6)
    update = torch.matmul(v_t_f32.unsqueeze(2), alpha_t.unsqueeze(1))
    S_expected = S_prev + update
    
    assert torch.allclose(S_curr, S_expected, atol=1e-6, rtol=1e-6), "Rank-1 update mismatch"
    assert S_curr.dtype == torch.float32

def test_unrolled_sum(memory_manager):
    """
    Test 2 — Correct unrolled accumulation
    Generate a short synthetic sequence of length T.
    Manually compute sum and compare.
    """
    torch.manual_seed(123)
    B = 2
    T = 10
    d = memory_manager.d_model
    
    memory_manager.reset(batch_size=B)
    
    v_seq = torch.randn(T, B, d)
    alpha_seq = torch.randn(T, B, d)
    
    # Run through manager
    for t in range(T):
        memory_manager.update(v_seq[t], alpha_seq[t])
        
    S_final = memory_manager.get_S()
    
    # Manual sum
    S_manual = torch.zeros(B, d, d)
    for t in range(T):
        v_t_f32 = v_seq[t] / (torch.norm(v_seq[t], dim=-1, keepdim=True) + 1e-6)
        update = torch.matmul(v_t_f32.unsqueeze(2), alpha_seq[t].unsqueeze(1))
        S_manual += update
        
    assert torch.allclose(S_final, S_manual, atol=1e-6, rtol=1e-6), "Unrolled sum mismatch"

def test_output_consistency(memory_manager):
    """
    Test 3 — Output consistency
    For randomly generated S_t and q_t:
    o_expected = S_t @ q_t
    o_implementation = S_t_from_module @ q_t
    """
    torch.manual_seed(456)
    B = 3
    d = memory_manager.d_model
    
    memory_manager.reset(batch_size=B)
    
    # Update state once to have non-zero S
    v = torch.randn(B, d)
    alpha = torch.randn(B, d)
    memory_manager.update(v, alpha)
    
    S_t = memory_manager.get_S()
    q_t = torch.randn(B, d)
    
    o_computed = memory_manager.compute_output(q_t)
    
    # Manual computation
    # S_t: (B, d, d)
    # q_t: (B, d) -> (B, d, 1)
    o_expected = torch.matmul(S_t, q_t.unsqueeze(2)).squeeze(2)
    
    assert torch.allclose(o_computed, o_expected, atol=1e-6, rtol=1e-6), "Output consistency mismatch"
    assert o_computed.shape == (B, d)

def test_stability_renorm():
    """
    Test 4 — Stability test with renormalization
    Construct a sequence where v_t and alpha_t have moderately large values.
    Verify renormalization triggers and S_t is scaled down.
    """
    d = 4
    threshold = 10.0
    manager = MemoryMatrixManager(d_model=d, enable_renorm=True, renorm_threshold=threshold)
    
    B = 1
    manager.reset(batch_size=B)
    
    # Create inputs that will definitely exceed threshold
    # Threshold is 10.
    # If we add v @ alpha^T such that norm > 10.
    # Let v = [10, 0...], alpha = [1, 0...] -> update is matrix with 10 at (0,0).
    # Norm is 10. If existing S was small, new S has norm >= 10.
    
    v = torch.zeros(B, d)
    v[0, 0] = 20.0 # Will be normalized
    alpha = torch.zeros(B, d)
    alpha[0, 0] = 20.0
    
    stats = manager.update(v, alpha)
    
    assert stats['renorm_triggered'] == 1.0, "Renormalization should have triggered"
    assert abs(stats['norm_max'] - 20.0) < 1e-4, f"Expected max norm ~20.0 before renorm, got {stats['norm_max']}"
    
    # Verify S_t is renormalized
    S_curr = manager.get_S()
    # After renorm, S = S_old / 20 = 20 / 20 = 1.
    # Norm of S should be 1.
    norm_new = torch.norm(S_curr)
    assert torch.abs(norm_new - 1.0) < 1e-5, f"Expected norm 1.0 after renorm, got {norm_new}"

def test_stability_no_renorm():
    """
    Test stability with renorm disabled.
    Norm should grow.
    """
    d = 4
    threshold = 10.0
    manager = MemoryMatrixManager(d_model=d, enable_renorm=False, renorm_threshold=threshold)
    
    B = 1
    manager.reset(batch_size=B)
    
    v = torch.zeros(B, d)
    v[0, 0] = 20.0
    alpha = torch.zeros(B, d)
    alpha[0, 0] = 20.0
    
    stats = manager.update(v, alpha)
    
    assert stats['renorm_triggered'] == 0.0, "Renormalization should NOT have triggered"
    assert abs(stats['norm_max'] - 20.0) < 1e-4
    
    S_curr = manager.get_S()
    norm_new = torch.norm(S_curr)
    assert torch.abs(norm_new - 20.0) < 1e-4, f"Expected norm 20.0, got {norm_new}"

def test_mixed_batch_renorm():
    """
    Test batch where one element needs renorm and another doesn't.
    """
    d = 4
    threshold = 10.0
    manager = MemoryMatrixManager(d_model=d, enable_renorm=True, renorm_threshold=threshold)
    
    B = 2
    manager.reset(batch_size=B)
    
    v = torch.zeros(B, d)
    alpha = torch.zeros(B, d)
    
    # Batch 0: Large update (20)
    v[0, 0] = 20.0
    alpha[0, 0] = 20.0
    
    # Batch 1: Small update (1)
    v[1, 0] = 1.0
    alpha[1, 0] = 1.0
    
    stats = manager.update(v, alpha)
    
    assert stats['renorm_triggered'] == 1.0
    
    S_curr = manager.get_S()
    
    # Check Batch 0
    # Was 20, should be renormalized to 1 (20/20)
    norm0 = torch.norm(S_curr[0])
    assert torch.abs(norm0 - 1.0) < 1e-5, f"Batch 0 should be renormed to 1.0, got {norm0}"
    
    # Check Batch 1
    # Was 1, should NOT be renormalized (threshold 10)
    # So it stays 1.
    norm1 = torch.norm(S_curr[1])
    assert torch.abs(norm1 - 1.0) < 1e-5, f"Batch 1 should remain 1.0, got {norm1}"
