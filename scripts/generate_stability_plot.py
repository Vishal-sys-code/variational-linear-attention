import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_stability_plot():
    os.makedirs('results/figures', exist_ok=True)
    
    try:
        logs = torch.load('results/forward_pass/diagnostics.pt', map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("Logs not found.")
        return
        
    vla = logs['vla']
    A = vla['A'][0] # (T, D, D)
    T = A.shape[0]
    
    norms = []
    conds = []
    
    for t in range(T):
        A_t = A[t].numpy()
        # Singular values
        try:
            U, S, V = np.linalg.svd(A_t)
            s_max = S[0]
            s_min = S[-1]
            
            norms.append(s_max)
            cond = s_max / s_min if s_min > 1e-12 else np.nan
            conds.append(cond)
        except Exception:
            norms.append(np.nan)
            conds.append(np.nan)
            
    plt.figure(figsize=(10, 6))
    
    # Dual axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('Spectral Norm ||A_t||_2', color=color)
    ax1.plot(range(T), norms, color=color, label='Spectral Norm')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Condition Number cond(A_t)', color=color)
    ax2.plot(range(T), conds, color=color, linestyle='--', label='Condition Number')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Stability Diagnostic: Inverse Penalty Matrix A_t')
    
    plt.savefig('results/figures/stability_norm.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/stability_norm.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    import csv
    with open('results/figures/stability_norm.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestep', 'spectral_norm', 'condition_number'])
        for t in range(T):
            writer.writerow([t, norms[t], conds[t]])
            
    print("Stability plot generated.")

if __name__ == '__main__':
    generate_stability_plot()
