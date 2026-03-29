import os
import torch
import torch.nn.functional as F
from src.models.attention.vla import VLALayer
from src.models.attention.deltanet import DeltaNetLayer

def run_diagnostics():
    print("Generating diagnostic logs...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, D = 1, 128, 64
    
    # 1. Initialize models
    vla = VLALayer(d_model=D, d_head=D, enable_stabilization=True).to(device)
    delta = DeltaNetLayer(d_model=D).to(device)
    
    # Set to eval mode for deterministic behavior
    vla.eval()
    delta.eval()
    
    # Create deterministic input
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(B, T, D, device=device)
    
    # We will simulate the forward loop step-by-step to extract EVERYTHING.
    # We can just manually run the VLA and Delta recurrence loop here!
    
    # --- VLA DIAGNOSTIC LOOP ---
    vla.inverse_tracker.init(batch_size=B, device=device, dtype=torch.float32)
    vla.memory_manager.reset(batch_size=B, device=device, dtype=torch.float32)
    vla.symbolic_tracker.init_sequence(A_rel=None, batch_size=B, max_seq_len=T, device=device, dtype=torch.float32)

    vla_logs = {
        'k': [], 'q': [], 'v': [], 'lambda': [], 'u': [],
        'A': [], 'S': [], 'alpha': [], 'o': []
    }
    
    for t in range(T):
        x_t = x[:, t, :]
        q_t = vla.W_q(x_t)
        k_t = vla.W_k(x_t)
        v_t = vla.W_v(x_t)
        
        s_t = (k_t * q_t).sum(dim=-1, keepdim=True)
        lambda_t, u_t, _ = vla.penalty_builder(k_t)
        
        vla.inverse_tracker.update(u_t)
        A_t = vla.inverse_tracker.get()
        
        if u_t.dim() == 2:
            u_vec = u_t.unsqueeze(-1)
            z_t = torch.bmm(A_t, u_vec).squeeze(-1)
            alpha_t = s_t * z_t
        else:
            u_vec = u_t.sum(dim=1).unsqueeze(-1)
            z_t = torch.bmm(A_t, u_vec).squeeze(-1)
            alpha_t = s_t * z_t
            
        vla.memory_manager.update(v_t, alpha_t)
        S_t = vla.memory_manager.get_S().clone()
        
        # Stabilization checks (simplified for diagnostic)
        o_t = vla.memory_manager.compute_output(q_t)
        
        vla_logs['k'].append(k_t.detach().cpu())
        vla_logs['q'].append(q_t.detach().cpu())
        vla_logs['v'].append(v_t.detach().cpu())
        vla_logs['lambda'].append(lambda_t.detach().cpu())
        vla_logs['u'].append(u_t.detach().cpu())
        vla_logs['A'].append(A_t.detach().cpu())
        vla_logs['S'].append(S_t.detach().cpu())
        vla_logs['alpha'].append(alpha_t.detach().cpu())
        vla_logs['o'].append(o_t.detach().cpu())
        
    for k in vla_logs.keys():
        vla_logs[k] = torch.stack(vla_logs[k], dim=1)
        
    # --- DELTANET DIAGNOSTIC LOOP ---
    S_delta = torch.zeros(B, D, D, device=device)
    delta_logs = {
        'k': [], 'q': [], 'v': [], 'beta': [],
        'S': [], 'o': []
    }
    
    for t in range(T):
        x_t = x[:, t, :]
        q = delta.W_q(x_t)
        k = F.normalize(delta.W_k(x_t), p=2, dim=-1)
        v = delta.W_v(x_t)
        
        beta = torch.sigmoid(delta.W_beta(k))
        
        S_k = torch.bmm(S_delta, k.unsqueeze(2)).squeeze(2)
        term1 = beta.unsqueeze(2) * torch.bmm(S_k.unsqueeze(2), k.unsqueeze(1))
        term2 = beta.unsqueeze(2) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        
        S_delta = S_delta - term1 + term2
        o_t = torch.bmm(S_delta, q.unsqueeze(2)).squeeze(2)
        
        delta_logs['k'].append(k.detach().cpu())
        delta_logs['q'].append(q.detach().cpu())
        delta_logs['v'].append(v.detach().cpu())
        delta_logs['beta'].append(beta.detach().cpu())
        delta_logs['S'].append(S_delta.clone().detach().cpu())
        delta_logs['o'].append(o_t.detach().cpu())

    for k in delta_logs.keys():
        delta_logs[k] = torch.stack(delta_logs[k], dim=1)
        
    # --- SAVE LOGS ---
    os.makedirs('results/forward_pass', exist_ok=True)
    torch.save({'vla': vla_logs, 'delta': delta_logs}, 'results/forward_pass/diagnostics.pt')
    print("Saved diagnostics to results/forward_pass/diagnostics.pt")

if __name__ == '__main__':
    run_diagnostics()
