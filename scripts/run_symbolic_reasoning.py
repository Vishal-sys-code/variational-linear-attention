import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import LRAModel
from src.benchmarks.synthetic.symbolic_dataset import SymbolicReasoningDataset, collate_fn_symbolic
from scripts.run_copy_task import set_seed
import src.benchmarks.synthetic.plots as plots
import matplotlib.pyplot as plt

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, A_rel in loader:
            x, y, A_rel = x.to(device), y.to(device), A_rel.to(device)
            logits = model(x, pool=True, symbolic_adj=A_rel) # pool=True, expecting 1 value
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total

def train_model(config, gamma, train_loader, val_loader, device, out_dir, model_name):
    print(f"\n--- Training Model: {model_name} (gamma={gamma}) ---")
    model = LRAModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        max_len=100,
        attention_type="vla",
        vla_gamma=gamma,
        num_classes=config["vocab_size"] # Output over vocab to match targets (TRUE/FALSE)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accs = []
    conds = []
    
    A_t_sample = None
    
    for epoch in range(config["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}", leave=False)
        epoch_loss = 0.0
        
        for batch_idx, (x, y, A_rel) in enumerate(pbar):
            x, y, A_rel = x.to(device), y.to(device), A_rel.to(device)
            
            return_states = (batch_idx == len(train_loader) - 1)
            
            if return_states:
                logits, states = model(x, pool=True, return_states=True, symbolic_adj=A_rel)
                A_t = states["A"][:, -1] # (B, d, d)
                cond_vals = [torch.linalg.cond(a.to(torch.float64)).item() for a in A_t]
                current_cond = sum(cond_vals)/len(cond_vals)
                conds.append(current_cond)
                A_t_sample = A_t[0].cpu().numpy()
            else:
                logits = model(x, pool=True, symbolic_adj=A_rel)
                
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        train_losses.append(epoch_loss / len(train_loader))
        val_acc = evaluate(model, val_loader, device)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc*100:.1f}%")
        
    M_t_sample = None
    eigenvalues = None
    if A_t_sample is not None:
        try:
            import numpy as np
            M_t_sample = np.linalg.inv(A_t_sample)
            eigenvalues = np.linalg.eigvalsh(M_t_sample)
        except:
            pass
            
    return model, train_losses, val_accs, conds, M_t_sample, eigenvalues

def run_symbolic_reasoning():
    config = {
        "seed": 42,
        "vocab_size": 35,
        "d_model": 64,
        "n_layers": 2,
        "batch_size": 32,
        "epochs": 100,
        "lr": 5e-4
    }
    
    set_seed(config["seed"])
    out_dir = "results/symbolic_experiments"
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scaled datasets for robust Colab GPU training logic
    train_dataset = SymbolicReasoningDataset(num_samples=2000, num_facts=4)
    val_dataset = SymbolicReasoningDataset(num_samples=400, num_facts=4)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_symbolic)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn_symbolic)
    
    # Train Baseline (Model A)
    _, loss_base, acc_base, cond_base, M_base, eig_base = train_model(
        config, 0.0, train_loader, val_loader, device, out_dir, "Model A (Baseline)"
    )
    
    # Train Symbolic (Model B)
    gamma_val = 0.5
    _, loss_sym, acc_sym, cond_sym, M_sym, eig_sym = train_model(
        config, gamma_val, train_loader, val_loader, device, out_dir, "Model B (Symbolic)"
    )
    
    # Save Metrics
    import csv
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss_A", "Loss_B", "ValAcc_A", "ValAcc_B"])
        for e in range(config["epochs"]):
            writer.writerow([e, loss_base[e], loss_sym[e], acc_base[e], acc_sym[e]])
            
    # --- Publication Quality Plotting (NeurIPS Style) ---
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Palatino', 'serif'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
    })
            
    # Save Raw Data for tracking
    import numpy as np
    if M_base is not None and M_sym is not None:
        np.save(os.path.join(out_dir, "M_base.npy"), M_base)
        np.save(os.path.join(out_dir, "M_sym.npy"), M_sym)
        np.save(os.path.join(out_dir, "eig_base.npy"), eig_base)
        np.save(os.path.join(out_dir, "eig_sym.npy"), eig_sym)
            
    # Visualize Heatmap M_t (Publication Quality)
    if M_base is not None and M_sym is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5)) # NeurIPS standard width ~5.5in or spanning
        
        # We plot the absolute values of the penalty matrix for structural clarity
        vmax = max(np.abs(M_base[:32, :32]).max(), np.abs(M_sym[:32, :32]).max())
        
        im1 = axes[0].imshow(np.abs(M_base[:32, :32]), cmap='magma', vmin=0, vmax=vmax)
        axes[0].set_title(r"Baseline $M_t$ (Unconstrained)")
        axes[0].set_xlabel("Head Dimension")
        axes[0].set_ylabel("Head Dimension")
        
        im2 = axes[1].imshow(np.abs(M_sym[:32, :32]), cmap='magma', vmin=0, vmax=vmax)
        axes[1].set_title(r"Symbolic $M_t$ ($\gamma=0.5$)")
        axes[1].set_xlabel("Head Dimension")
        axes[1].set_yticks([]) # remove y-ticks for the second plot
        
        # Add a single colorbar
        cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
        cbar.set_label(r"Penalty Magnitude $|M_{i,j}|$")
        
        plt.savefig(os.path.join(out_dir, "heatmap_Mt_pub.pdf"))
        # Also save a png for quick viewing
        plt.savefig(os.path.join(out_dir, "heatmap_Mt_pub.png"))
        plt.close()
        
    # Visualize Eigenvalues (Publication Quality)
    if eig_base is not None and eig_sym is not None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(eig_base[::-1], label='Baseline', marker='o', markersize=4, linestyle='-', linewidth=1.5, color='#1f77b4')
        ax.plot(eig_sym[::-1], label=r'Symbolic ($\gamma=0.5$)', marker='s', markersize=4, linestyle='--', linewidth=1.5, color='#d62728')
        
        ax.set_title(r'Eigenvalue Spectrum of Penalty Matrix $M_t$')
        ax.set_xlabel('Principal Component Index')
        ax.set_ylabel('Eigenvalue Magnitude (Log Scale)')
        ax.set_yscale('log') # Log scale is much better for spectra
        ax.legend(frameon=True, fancybox=True, edgecolor='black')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        plt.savefig(os.path.join(out_dir, "eigenvalues_plot_pub.pdf"))
        plt.savefig(os.path.join(out_dir, "eigenvalues_plot_pub.png"))
        plt.close()
        
    # Write Error Analysis
    with open(os.path.join(out_dir, "error_analysis.txt"), "w") as f:
        f.write("Symbolic Reasoning Error Analysis\n")
        f.write("=================================\n")
        f.write(f"Final Val Acc Baseline: {acc_base[-1]*100:.1f}%\n")
        f.write(f"Final Val Acc Symbolic: {acc_sym[-1]*100:.1f}%\n")
        if acc_sym[-1] >= acc_base[-1]:
            f.write("Symbolic penalty improved or maintained reasoning accuracy while enforcing structures.\n")
        else:
            f.write("Symbolic penalty slightly degraded accuracy on this specific synthetic task.\n")
            
    # Write a dummy attention_maps.png since true attention doesn't exist in linear attention
    plt.figure(figsize=(4, 4))
    plt.text(0.5, 0.5, "Attention Maps N/A in feature space\nSee Heatmap_Mt.png", ha='center', va='center')
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "attention_maps.png"))
    plt.close()

if __name__ == "__main__":
    run_symbolic_reasoning()
