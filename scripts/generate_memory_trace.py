import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_memory_trace():
    os.makedirs('results/figures', exist_ok=True)
    
    try:
        logs = torch.load('results/forward_pass/diagnostics.pt', map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("Logs not found.")
        return
        
    vla = logs['vla']
    v = vla['v'][0] # (T, D)
    o = vla['o'][0] # (T, D)
    
    T, D = v.shape
    
    # Indices to plot
    target_indices = [5, 20, 50, 80]
    
    plt.figure(figsize=(10, 6))
    
    import csv
    with open('results/figures/memory_trace.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['timestep'] + [f'trace_memory_{i}' for i in target_indices]
        writer.writerow(header)
        
        # We need to pre-compute for all T to write row by row
        traces = {i: np.full(T, np.nan) for i in target_indices}
        
        for i in target_indices:
            v_i = v[i]
            for t in range(i, T):
                o_t = o[t]
                dot = (v_i @ o_t).item()
                # Normalized contribution
                norm = torch.norm(o_t).item()
                contribution = dot / (norm + 1e-8)
                traces[i][t] = contribution
                
            plt.plot(range(i, T), traces[i][i:], label=f"Memory {i} Contribution")
            
        for t in range(T):
            row = [t] + [traces[i][t] for i in target_indices]
            writer.writerow(row)
            
    plt.xlabel('Timestep t')
    plt.ylabel('Normalized Contribution (v_i^T o_t) / ||o_t||')
    plt.title('Memory Contribution Trace')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/figures/memory_trace.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/memory_trace.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print("Memory trace plot generated.")

if __name__ == '__main__':
    generate_memory_trace()
