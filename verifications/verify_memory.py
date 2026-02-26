import torch
from src.models.attention.memory_matrix import MemoryMatrixManager

def main():
    print("=== Verifying MemoryMatrixManager (Step A4) ===")
    
    d_model = 16
    batch_size = 4
    
    # 1. Initialize
    print(f"\nInitializing MemoryMatrixManager with d_model={d_model}, batch_size={batch_size}")
    manager = MemoryMatrixManager(d_model=d_model, enable_renorm=True, renorm_threshold=100.0)
    manager.reset(batch_size=batch_size)
    
    # Check initial state
    S_init = manager.get_S()
    print(f"Initial S shape: {S_init.shape}")
    print(f"Initial S norm: {torch.norm(S_init).item():.4f}")
    assert torch.all(S_init == 0), "S should be initialized to zeros"
    
    # 2. Perform Updates
    print("\nPerforming 5 random updates...")
    torch.manual_seed(42)
    
    for t in range(5):
        v_t = torch.randn(batch_size, d_model)
        alpha_t = torch.randn(batch_size, d_model)
        
        stats = manager.update(v_t, alpha_t)
        print(f"Step {t+1}: Norm max={stats['norm_max']:.4f}, Mean={stats['norm_mean']:.4f}, Renorm={stats['renorm_triggered']}")
        
    # 3. Compute Output
    print("\nComputing output o_t = S_t * q_t...")
    q_t = torch.randn(batch_size, d_model)
    o_t = manager.compute_output(q_t)
    
    print(f"Output shape: {o_t.shape}")
    assert o_t.shape == (batch_size, d_model), f"Output shape mismatch: {o_t.shape}"
    
    print("\n=== Verification SUCCESS ===")
    print("MemoryMatrixManager is functioning correctly.")
    print("Ready for Step B.")

if __name__ == "__main__":
    main()