import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# NeurIPS / Academic standard styling parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def parse_synthetic_logs(log_dir):
    data = {}
    
    # regex matches: Epoch 001/100: ... Loss=0.0467, A_kappa=1.4e+02, S_norm=23.7, ...
    epoch_pattern = re.compile(r"Epoch\s+\d+/\d+:.*Loss=([0-9.]+), A_kappa=([0-9.e+-]+), S_norm=([0-9.]+)")
    # regex matches: Copy Task Final Accuracy: 100.00%
    acc_pattern = re.compile(r"Final Accuracy:\s*([0-9.]+)%")
    
    txt_files = glob.glob(os.path.join(log_dir, "*.txt"))
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        task_name = filename.replace(".txt", "").replace("run_", "")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Extract epochs data
                epochs_loss = []
                epochs_snorm = []
                for match in epoch_pattern.finditer(content):
                    loss = float(match.group(1))
                    snorm = float(match.group(3))
                    epochs_loss.append(loss)
                    epochs_snorm.append(snorm)
                    
                # Extract Final Accuracy
                acc_match = acc_pattern.search(content)
                final_acc = float(acc_match.group(1)) if acc_match else 0.0
                
                if epochs_loss:
                    data[task_name] = {
                        "loss": epochs_loss,
                        "s_norm": epochs_snorm,
                        "accuracy": final_acc
                    }
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
    return data

def main():
    log_dir = "results/logs"
    out_dir = "results/logs/paper_plots"
    
    if not os.path.exists(log_dir):
        print(f"Error: Directory {log_dir} does not exist.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    data = parse_synthetic_logs(log_dir)
    if not data:
        print("No valid parsed data found.")
        return
        
    print(f"Parsed data for tasks: {list(data.keys())}")

    academic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # ---------------------------------------------------------
    # Plot 1: Copy Task Loss & State Norm
    # ---------------------------------------------------------
    if "copy_task" in data:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        
        epochs = np.arange(1, len(data["copy_task"]["loss"]) + 1)
        
        color1 = '#2E8B57' # Forest green
        ax1.plot(epochs, data["copy_task"]["loss"], color=color1, linewidth=2, label='Cross-Entropy Loss')
        ax1.set_xlabel('Training Epochs', fontweight='bold')
        ax1.set_ylabel('Loss', color=color1, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = '#5D8AA8' # Slate blue
        ax2.plot(epochs, data["copy_task"]["s_norm"], color=color2, linewidth=2, linestyle='--', label=r'$|S_t|_F$ Normalization')
        ax2.set_ylabel('State Frobenius Norm', color=color2, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right', frameon=True, edgecolor='black')
        
        plt.title('Copy Task Convergence & State Stability', fontweight='bold', pad=15)
        
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "neurips_synthetic_copy.pdf"), format='pdf')
        plt.savefig(os.path.join(out_dir, "neurips_synthetic_copy.png"), dpi=300)
        print("Generated Plot 1: Copy Task (neurips_synthetic_copy.pdf)")
        plt.close()

    # ---------------------------------------------------------
    # Plot 2: Delayed Recall Convergence Comparison
    # ---------------------------------------------------------
    delay_keys = [k for k in data.keys() if "delayed" in k]
    if delay_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Sort by delay number extraction (e.g. 'run_delayed_recall_5' -> 5)
        def extract_delay(k):
            m = re.search(r'_(\d+)$', k)
            return int(m.group(1)) if m else 0
            
        delay_keys.sort(key=extract_delay)
        
        for i, k in enumerate(delay_keys):
            d_val = extract_delay(k)
            epochs = np.arange(1, len(data[k]["loss"]) + 1)
            ax.plot(epochs, data[k]["loss"], color=academic_colors[i], 
                    linewidth=2.5, alpha=0.9, label=f'Delay $D={d_val}$')
            
        ax.set_xlabel('Training Epochs', fontweight='bold')
        ax.set_ylabel('Cross-Entropy Loss', fontweight='bold')
        plt.title('Delayed Recall Training Convergence', fontweight='bold', pad=15)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle=':', alpha=0.7)
        ax.legend(frameon=True, edgecolor='black', title="Memory Delay")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "neurips_synthetic_delayed.pdf"), format='pdf')
        plt.savefig(os.path.join(out_dir, "neurips_synthetic_delayed.png"), dpi=300)
        print("Generated Plot 2: Delayed Recall (neurips_synthetic_delayed.pdf)")
        plt.close()

    # ---------------------------------------------------------
    # Plot 3: Combined Extrapolation / Accuracy Chart
    # ---------------------------------------------------------
    if data:
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Format labels clearly
        def format_name(k):
            if "copy" in k: return "Copy (Identity)"
            if "associative" in k: return "Assoc. Recall"
            if "delayed" in k:
                m = re.search(r'_(\d+)$', k)
                return f"Delayed (D={m.group(1)})" if m else "Delayed Recall"
            return k.replace('_', ' ').title()
            
        # Group and order keys logically
        ordered_keys = []
        if "copy_task" in data: ordered_keys.append("copy_task")
        delay_str_keys = sorted([k for k in data if "delayed" in k], key=lambda x: int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else 0)
        ordered_keys.extend(delay_str_keys)
        if "associative_recall" in data: ordered_keys.append("associative_recall")
        
        labels = [format_name(k) for k in ordered_keys]
        accuracies = [data[k]["accuracy"] for k in ordered_keys]
        
        y_pos = np.arange(len(labels))
        
        # Color gradient logic
        bar_colors = ['#5D8AA8' if 'Delay' not in l else '#E27D60' for l in labels]
        if 'Assoc' in labels[-1]: bar_colors[-1] = '#2E8B57'
        
        bars = ax.barh(y_pos, accuracies, color=bar_colors, edgecolor='black', height=0.6, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontweight='bold')
        ax.invert_yaxis()
        
        ax.set_xlabel('Final Validation Accuracy (%)', fontweight='bold')
        plt.title('VLA Memory Extrapolation Validation', fontweight='bold', pad=15)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, 105)
        
        # Add data labels
        for i, v in enumerate(accuracies):
            ax.text(v + 1.5, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=11, color='#333333')
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "neurips_synthetic_combined.pdf"), format='pdf')
        plt.savefig(os.path.join(out_dir, "neurips_synthetic_combined.png"), dpi=300)
        print("Generated Plot 3: Combined Accuracy (neurips_synthetic_combined.pdf)")
        plt.close()
        
if __name__ == "__main__":
    main()
