import os
import json
import numpy as np
import matplotlib.pyplot as plt

def apply_neurips_aesthetic():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Palatino', 'serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5
    })

def clean_spines(ax):
    """DeepMind minimalist formatting."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def generate_publication_plots(json_path, out_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    os.makedirs(out_dir, exist_ok=True)
    apply_neurips_aesthetic()
    
    # Standard color palette from cool to warm for increasing gamma
    gammas = sorted([float(k) for k in data.keys()])
    gamma_str = [str(g) if str(g) in data else str(g)+'0' for g in gammas] 
    gamma_str = [k for k in data.keys()] # Keep exact strings
    
    # Create colormap avoiding pure yellow/green for accessibility
    colors = ['#4b5563', '#60a5fa', '#3b82f6', '#f59e0b', '#e11d48']
    
    # -------------------------------------------------------------
    # 1. Convergence Curve (Dual Axis: Loss & Acc)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    
    epochs = np.arange(1, len(data[gamma_str[0]]['train_loss']) + 1)
    
    for i, g in enumerate(gamma_str):
        loss = data[g]['train_loss']
        acc = data[g]['val_acc']
        lbl = f'Baseline' if g == '0.0' else rf'$\gamma={g}$'
        
        ax1.plot(epochs, loss, label=lbl, color=colors[i], zorder=5 if i==0 or i==4 else 3)
        ax2.plot(epochs, np.array(acc)*100, label=lbl, color=colors[i], zorder=5 if i==0 or i==4 else 3)
        
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Convergence Dynamics', pad=10, fontweight='bold')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Relational Reasoning', pad=10, fontweight='bold')
    
    for ax in [ax1, ax2]:
        clean_spines(ax)
    
    ax2.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_convergence_curves.pdf"))
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Eigenvalue Spectrum (Final Epoch)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, g in enumerate(gamma_str):
        # spectrum from last step
        eigs = data[g]['diagnostics'][-1]['eigenvalues']
        eigs_sorted = np.sort(np.array(eigs))[::-1] # descending
        
        lbl = f'Baseline' if g == '0.0' else rf'$\gamma={g}$'
        ax.plot(eigs_sorted, label=lbl, color=colors[i], marker='o' if i==0 or i==4 else '.', markersize=4 if i==0 or i==4 else 2)
        
    ax.set_yscale('log')
    ax.set_xlabel('Principal Component Index')
    ax.set_ylabel('Eigenvalue Magnitude (Log)')
    ax.set_title(r'Penalty Matrix $M_t$ Spectrum Decay', pad=10, fontweight='bold')
    clean_spines(ax)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_eigenvalue_spectrum.pdf"))
    plt.close()

    # -------------------------------------------------------------
    # 3. Dynamic Trace M_t
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    for i, g in enumerate(gamma_str):
        traces = [d['trace_M_t'] for d in data[g]['diagnostics']]
        lbl = f'Baseline' if g == '0.0' else rf'$\gamma={g}$'
        ax.plot(epochs, traces, label=lbl, color=colors[i])
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Trace($M_t$)')
    ax.set_title('Spectral Power Constraint Trajectory', pad=10, fontweight='bold')
    clean_spines(ax)
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    
    plt.xlim(1, len(epochs))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_trace_Mt.pdf"))
    plt.close()
    
    # -------------------------------------------------------------
    # 4. Heatmaps Comparison (Baseline vs Best Gamma)
    # -------------------------------------------------------------
    M_base = np.array(data['0.0']['diagnostics'][-1]['M_t_matrix'])
    M_sym = np.array(data['0.5']['diagnostics'][-1]['M_t_matrix'])
    
    # Focus on upper left 32x32 to see block diagonal
    dim = min(32, M_base.shape[0])
    vmax = max(np.abs(M_base[:dim, :dim]).max(), np.abs(M_sym[:dim, :dim]).max())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    im1 = ax1.imshow(np.abs(M_base[:dim, :dim]), cmap='magma', vmin=0, vmax=vmax)
    ax1.set_title(r"Baseline $M_t$ ($\gamma=0.0$)", pad=10)
    ax1.set_xlabel("Dimension Index")
    ax1.set_ylabel("Dimension Index")
    
    im2 = ax2.imshow(np.abs(M_sym[:dim, :dim]), cmap='magma', vmin=0, vmax=vmax)
    ax2.set_title(r"Symbolic $M_t$ ($\gamma=0.5$)", pad=10)
    ax2.set_xlabel("Dimension Index")
    ax2.set_yticks([])
    
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label(r"Penalty Magnitude $|(M_t)_{i,j}|$")
    
    plt.savefig(os.path.join(out_dir, "paper_heatmap_comparison.pdf"), bbox_inches='tight')
    plt.close()
    
    print("NeurIPS compliant publication plots generated successfully.")

if __name__ == "__main__":
    json_path = "results/symbolic_experiments/logs.json"
    out_dir = "results/symbolic_experiments/neurips_plots"
    generate_publication_plots(json_path, out_dir)
