import torch
import math
from src.models.attention.vla import VLALayer

def generate_mock_data(B=2, T=10, d=16):
    torch.manual_seed(42)
    x = torch.randn(B, T, d)
    
    # Create a mock adjacency matrix with some block structure (e.g. 0-4 related, 5-9 related)
    A_rel = torch.zeros(B, T, T)
    for b in range(B):
        A_rel[b, :5, :5] = 1.0
        A_rel[b, 5:, 5:] = 1.0
        # self loops
        for i in range(T):
            A_rel[b, i, i] = 1.0
            
    return x, A_rel

def test_1_invariance():
    print("--- Test 1: Invariance when gamma=0 ---")
    B, T, d = 2, 10, 16
    x, A_rel = generate_mock_data(B, T, d)
    
    # Baseline
    vla_base = VLALayer(d_model=d, gamma=0.0)
    vla_base.eval()
    
    # Symbolic with gamma 0
    vla_sym = VLALayer(d_model=d, gamma=0.0)
    vla_sym.load_state_dict(vla_base.state_dict())
    vla_sym.eval()
    
    out_base = vla_base(x)
    out_sym = vla_sym(x, symbolic_adj=A_rel)
    
    diff = torch.abs(out_base - out_sym).max().item()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-6, "Test 1 Failed: Invariance broken"
    print("Test 1 Passed.")

def test_2_and_3():
    print("\n--- Test 2 & 3: Eigenvalues & Sherman-Morrison Equivalence ---")
    B, T, d = 2, 10, 16
    x, A_rel = generate_mock_data(B, T, d)
    gamma = 0.5
    
    vla_base = VLALayer(d_model=d, gamma=0.0)
    vla_base.eval()
    
    vla_sym = VLALayer(d_model=d, gamma=gamma)
    vla_sym.load_state_dict(vla_base.state_dict())
    vla_sym.eval()
    
    # Forward pass to collect states
    # Note: we need to modify VLALayer to return A_t if return_states=True
    # Currently VLALayer returns states["A"] which has shape (B, T, d, d)
    with torch.no_grad():
        out_b, states_b = vla_base(x, return_states=True)
        out_s, states_s = vla_sym(x, return_states=True, symbolic_adj=A_rel)
        
    A_t_base = states_b["A"][:, -1] # (B, d, d) at last timestep
    A_t_sym = states_s["A"][:, -1]  # (B, d, d) at last timestep
    
    # Test 2: Eigenvalues
    # Note: A_t is the inverse penalty matrix. 
    # M_t = inv(A_t)
    try:
        M_base = torch.linalg.inv(A_t_base)
        M_sym = torch.linalg.inv(A_t_sym)
        
        eig_base = torch.linalg.eigvalsh(M_base)
        eig_sym = torch.linalg.eigvalsh(M_sym)
        
        print("M_base min eigval:", eig_base.min().item())
        print("M_sym min eigval:", eig_sym.min().item())
        
        assert eig_sym.min() > 0, "M_sym is not positive definite"
        
        # Test 3: Sherman-Morrison equivalence
        # We compute M_direct by accumulating the exact rank-1 updates manually
        # and checking if inv(M_direct) == A_t_sym
        
        # First, reconstruct M_t_base manually or just use M_base from inverse
        # Actually M_base is exactly lambda_t I + u_t u_t^T (accumulated)
        # Wait, M_t in VLA does NOT accumulate over time! 
        # Ah!! "M_t = lambda_t I + u_t u_t^T + gamma (A_r^T A_r)"
        # InverseTracker in VLA reinitializes? No, InverseTracker maintains A_t which accumulates!
        # Wait, InverseTracker: M_t = M_{t-1} + u_t u_t^T
        
        # Let's verify M_sym == M_base + \gamma \sum a_t a_t^T
        # get keys
        q_s = states_s["q"] # (B, T, d)
        v_s = states_s["v"]
        
        # Need k_t. We didn't explicitly return k_t. Let's just project x exactly as in model
        W_k = vla_sym.W_k
        k_all = W_k(x) # (B, T, d)
        
        D = A_rel.sum(dim=-1) + 1e-6
        D_inv_sqrt = 1.0 / torch.sqrt(D)
        W = A_rel * D_inv_sqrt.unsqueeze(-1) * D_inv_sqrt.unsqueeze(1)
        
        M_direct = M_base.clone()
        for t in range(T):
            W_t = W[:, t, :t+1] # (B, t+1)
            
            # Check if has relation
            has_rel = (torch.abs(W_t).max(dim=-1).values > 1e-9)
            
            # compute a_t
            # k_all[:, :t+1] is (B, t+1, d)
            a_t = (k_all[:, :t+1, :] * W_t.unsqueeze(-1)).sum(dim=1) # (B, d)
            a_t = a_t * math.sqrt(gamma)
            
            mask = has_rel.unsqueeze(-1).float()
            a_t = a_t * mask
            
            # rank 1 update
            a_t_vec = a_t.unsqueeze(-1)
            M_direct = M_direct + torch.bmm(a_t_vec, a_t_vec.transpose(1, 2))
            
        A_direct = torch.linalg.inv(M_direct)
        
        err = torch.norm(A_t_sym - A_direct, p='fro').item()
        print(f"Frobenius error exact SM vs Incremental SM: {err:.4e}")
        assert err < 1e-4, "Test 3 Failed: SM equivalence broken"
        print("Tests 2 & 3 Passed.")
    except Exception as e:
        print("Error during tests:", e)
        raise e

if __name__ == "__main__":
    test_1_invariance()
    test_2_and_3()
