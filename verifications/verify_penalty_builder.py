import torch
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.attention.penalty_builder import PenaltyBuilder

def verify_penalty_builder():
    """
    Manually verifies PenaltyBuilder output.
    """
    print("--- 1. Initialization ---")
    d_model = 4
    rank = 1
    builder = PenaltyBuilder(d_model=d_model, rank=rank)
    print(f"Created PenaltyBuilder(d_model={d_model}, rank={rank})")
    print(builder)

    print("\n--- 2. Single Step Verification (B, d) ---")
    B = 2
    # Random key (B=2, d=4)
    k = torch.randn(B, d_model)
    print(f"Input k (shape {k.shape}):\n{k}")

    # Forward
    lambda_t, u_t, stats = builder(k)
    print(f"\nOutput lambda_t (shape {lambda_t.shape}):\n{lambda_t}")
    print(f"Output u_t (shape {u_t.shape}):\n{u_t}")
    print(f"Stats:\n{stats}")

    print("\n--- 3. Constructing M_t (Manual Check) ---")
    # For batch element 0
    lam = lambda_t[0].item()
    u = u_t[0]
    
    # M = lam * I + u u^T
    M = lam * torch.eye(d_model) + torch.outer(u, u)
    print(f"Constructed M_t (Batch 0):\n{M}")
    
    eigvals = torch.linalg.eigvalsh(M)
    print(f"Eigenvalues of M_t:\n{eigvals}")
    
    is_pd = torch.all(eigvals > 0)
    print(f"Is Symmetric Positive Definite? {is_pd}")
    if not is_pd:
        print("WARNING: Matrix is not PD!")

    print("\n--- 4. Batch/Sequence Verification (B, T, d) ---")
    T = 3
    k_seq = torch.randn(B, T, d_model)
    print(f"Input k_seq (shape {k_seq.shape})")
    
    l_seq, u_seq, stats_seq = builder(k_seq)
    print(f"Output lambda_seq (shape {l_seq.shape})")
    print(f"Output u_seq (shape {u_seq.shape})")
    print(f"Stats:\n{stats_seq}")
    
    print("\n--- 5. Rank-r Verification (rank=2) ---")
    rank_r = 2
    builder_r = PenaltyBuilder(d_model=d_model, rank=rank_r)
    print(f"Created PenaltyBuilder(rank={rank_r})")
    
    l_r, u_r, stats_r = builder_r(k)
    print(f"Output u_r (shape {u_r.shape})") # Should be (B, r, d)
    print(f"Stats:\n{stats_r}")

if __name__ == "__main__":
    verify_penalty_builder()