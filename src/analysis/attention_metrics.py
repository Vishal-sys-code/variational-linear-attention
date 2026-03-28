import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_attention_entropy(q, k, A):
    """
    Computes average implicit sequential attention entropy H.
    q: (B, T, d)
    k: (B, T, d) 
    A: (B, T, d, d)
    Returns: Average Entropy scalar over the batch and sequence.
    """
    B, T, d = q.shape
    epsilon = 1e-8
    
    H_batch = 0.0
    
    for b in range(B):
        H_seq = 0.0
        for i in range(T):
            # For timestep i, attention spans past tokens 0..i
            q_i = q[b, i].unsqueeze(0) # (1, d)
            A_i = A[b, i] # (d, d)
            K_past = k[b, :i+1] # (i+1, d)
            
            # P_ij raw scores = q_i^T A_i K_j
            # (1, d) @ (d, d) -> (1, d) @ (d, i+1) -> (1, i+1)
            raw_scores = torch.matmul(torch.matmul(q_i, A_i), K_past.T) 
            
            # Normalize with Softmax to form a probability distribution P_ij
            P_i = F.softmax(raw_scores, dim=-1).squeeze(0) # (i+1,)
            
            # Entropy
            H_i = -torch.sum(P_i * torch.log(P_i + epsilon))
            H_seq += H_i.item()
            
        H_batch += (H_seq / T)
        
    return H_batch / B

def compute_energy_ratio(gamma, a_t_scaled, lambda_t):
    """
    Computes log symbolic energy versus baseline energy ratio.
    gamma: scalar
    a_t_scaled: (B, T, d)
    lambda_t: (B, T, 1)
    """
    epsilon = 1e-8
    B, T, d = a_t_scaled.shape
    
    # E_sym = gamma * trace(A_feat^T A_feat)
    # Since a_t_scaled is already scaled by sqrt(gamma) conceptually or generated directly by step,
    # wait: step returns a_t_scaled which is effectively what gets multiplied.
    # We can just construct the sum of norms if it represents the vector update.
    # VLA updates A_t with outer products of a_t_scaled.
    # The sum of traces of outer products is the sum of squared norms.
    
    # E_sym = sum_i Tr(v v^T) where v = a_t_scaled
    sum_norms = torch.sum(a_t_scaled ** 2, dim=-1) # (B, T)
    E_sym = torch.mean(torch.sum(sum_norms, dim=1)).item()
    
    # E_base = trace(lambda_t * I) = d * sum_t lambda_t
    E_base = d * torch.mean(torch.sum(lambda_t.squeeze(-1), dim=1)).item()
    
    R = E_sym / (E_base + epsilon)
    return R

def check_and_plot_invariance(M_sym_gamma0, M_base, out_dir):
    """
    Checks invariance limit at gamma 0 and saves heatmap.
    """
    diff = np.max(np.abs(M_sym_gamma0 - M_base))
    
    Delta_M = M_sym_gamma0 - M_base
    vmax = np.max(np.abs(Delta_M))
    if vmax == 0.0: vmax = 1e-6 # prevent error
    
    plt.figure(figsize=(5, 4))
    plt.imshow(Delta_M[:32, :32], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.title(r'$\Delta M = M_{sym}(\gamma=0) - M_{base}$')
    plt.colorbar()
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "heatmap_DeltaM_invariance.png"), bbox_inches='tight')
    plt.close()
    
    return diff
