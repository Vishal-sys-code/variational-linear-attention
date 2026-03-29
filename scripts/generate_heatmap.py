import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap():
    os.makedirs('results/figures', exist_ok=True)
    
    try:
        logs = torch.load('results/forward_pass/diagnostics.pt', map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("Logs not found. Ensure generate_diagnostic_logs.py has been run.")
        return
        
    vla = logs['vla']
    
    # We visualize the penalty strength between recent keys
    # penalty(i, j) = u_i^T u_j
    u = vla['u'][0] # shape: (T, r, D) or (T, D)
    T = u.shape[0]
    
    # Ensure it's (T, D)
    if u.dim() == 3:
        u = u.sum(dim=1) # flatten rank if needed for vector multiplication comparison
        
    # H[i,j] = penalty(i,j)
    # Using torch.matmul
    H = torch.matmul(u, u.t()).numpy() # (T, T)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(H, cmap='magma', interpolation='nearest')
    plt.colorbar(label='Penalty Strength')
    plt.xlabel('Timestep j')
    plt.ylabel('Timestep i')
    plt.title('Penalty Matrix Structure (u_i^T u_j)')
    
    # Save Image
    plt.savefig('results/figures/penalty_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/penalty_heatmap.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # Save CSV
    import csv
    with open('results/figures/penalty_heatmap.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['timestep_i'] + [f'timestep_{j}' for j in range(T)]
        writer.writerow(header)
        for i in range(T):
            row = [i] + H[i].tolist()
            writer.writerow(row)
            
    print("Heatmap generated.")

if __name__ == '__main__':
    generate_heatmap()
