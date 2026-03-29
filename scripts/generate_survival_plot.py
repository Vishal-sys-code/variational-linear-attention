import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_survival_plot():
    os.makedirs('results/figures', exist_ok=True)
    
    # Load logs
    try:
        logs = torch.load('results/forward_pass/diagnostics.pt', map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("Logs not found. Ensure generate_diagnostic_logs.py has been run.")
        return
        
    vla = logs['vla']
    delta = logs['delta']
    
    # Sequence length T
    B, T, D = vla['k'].shape
    
    # Indices to plot
    target_indices = [5, 20, 50]
    
    # We will compute survival factors for VLA and Delta
    # VLA: survival(i, t) = \prod_{j=i+1}^t ( 1 - (u_j^T A_{j-1} u_j) / (1 + u_j^T A_{j-1} u_j) )
    # But note: A_j is saved AFTER update in the logs. So A_{j-1} is simply A(j-1).
    A = vla['A'][0] # (T, D, D)
    u = vla['u'][0] # (T, 1, D) if rank=1 or (T, D)
    if u.dim() == 2:
        u = u.unsqueeze(1) # (T, 1, D)
        
    # Precompute decay factor at each step j for VLA
    # factor_j = 1 - (u_j^T A_{j-1} u_j) / (1 + u_j^T A_{j-1} u_j)
    vla_decays = np.ones(T)
    for j in range(1, T):
        A_prev = A[j-1]
        u_j = u[j]  # (1, D) or (r, D)
        if u_j.shape[0] == 1:
            u_vec = u_j.t() # (D, 1)
            quad = (u_vec.t() @ A_prev @ u_vec).item()
        else:
            u_vec = u_j.sum(dim=0).unsqueeze(-1)
            quad = (u_vec.t() @ A_prev @ u_vec).item()
        
        decay = 1.0 - (quad / (1.0 + quad))
        vla_decays[j] = decay
        
    # DeltaNet: survival(i, t) = \prod_{j=i+1}^t ( 1 - beta_j * (k_j^T k_i) )
    k_delta = delta['k'][0] # (T, D)
    beta = delta['beta'][0] # (T, 1)
    
    plt.figure(figsize=(10, 6))
    
    # Data storage
    csv_data = {'timestep': np.arange(T)}
    
    for i in target_indices:
        # VLA survival
        surv_vla = np.ones(T)
        for t in range(i, T):
            if t == i:
                surv_vla[t] = 1.0
            else:
                surv_vla[t] = surv_vla[t-1] * vla_decays[t]
        
        # DeltaNet survival
        surv_delta = np.ones(T)
        k_i = k_delta[i]
        for t in range(i, T):
            if t == i:
                surv_delta[t] = 1.0
            else:
                k_j = k_delta[t]
                dot = (k_j @ k_i).item()
                decay = 1.0 - beta[t].item() * dot
                surv_delta[t] = surv_delta[t-1] * decay
                
        # Only plot from i to T
        t_range = np.arange(i, T)
        
        plt.plot(t_range, surv_vla[i:], label=f"VLA Memory {i}")
        plt.plot(t_range, surv_delta[i:], '--', label=f"DeltaNet Memory {i}")
        
        # For CSV, fill with NaN before i
        padded_vla = np.full(T, np.nan)
        padded_vla[i:] = surv_vla[i:]
        padded_delta = np.full(T, np.nan)
        padded_delta[i:] = surv_delta[i:]
        
        csv_data[f'vla_survival_memory_{i}'] = padded_vla
        csv_data[f'delta_survival_memory_{i}'] = padded_delta
        
    plt.xlabel('Timestep t')
    plt.ylabel('Survival Value')
    plt.title('Memory Survival Factor: VLA vs DeltaNet')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('results/figures/survival_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/survival_plot.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    pd.DataFrame(csv_data).to_csv('results/figures/survival_plot_data.csv', index=False)
    print("Survival plot generated.")

if __name__ == '__main__':
    generate_survival_plot()
