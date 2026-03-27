import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def parse_logs(file_path):
    epochs = 100
    metrics = {
        'Baseline': {'loss': [], 'acc': []},
        'Symbolic': {'loss': [], 'acc': []}
    }
    
    current_model = None
    
    with open(file_path, 'r') as f:
        for line in f:
            if "Model A (Baseline)" in line:
                current_model = 'Baseline'
            elif "Model B (Symbolic)" in line:
                current_model = 'Symbolic'
                
            match = re.search(r"Epoch \d+ \| Loss: ([\d\.]+) \| Val Acc: ([\d\.]+)", line)
            if match and current_model:
                loss = float(match.group(1))
                acc = float(match.group(2))
                metrics[current_model]['loss'].append(loss)
                metrics[current_model]['acc'].append(acc)
                
    return metrics

def create_publication_plot(metrics, out_dir):
    # Aesthetic configuration for NeurIPS / minimalist aesthetic
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Palatino', 'serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'lines.linewidth': 2.0
    })
    
    # Elegant color palette: Baseline = Charcoal Gray, Symbolic = Deep Crimson Red
    color_base = '#4b5563'
    color_sym = '#e11d48'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    
    epochs = range(1, len(metrics['Baseline']['loss']) + 1)
    
    # 1. Plot Loss
    ax1.plot(epochs, metrics['Baseline']['loss'], label='Baseline', color=color_base, linestyle='-')
    ax1.plot(epochs, metrics['Symbolic']['loss'], label=r'Symbolic ($\gamma=0.5$)', color=color_sym, linestyle='-')
    
    ax1.set_xlabel('Epoch', labelpad=8)
    ax1.set_ylabel('Training Loss', labelpad=8)
    ax1.set_title('Convergence Dynamics', pad=12, fontweight='bold')
    
    # 2. Plot Accuracy 
    ax2.plot(epochs, metrics['Baseline']['acc'], label='Baseline', color=color_base, linestyle='-')
    ax2.plot(epochs, metrics['Symbolic']['acc'], label=r'Symbolic ($\gamma=0.5$)', color=color_sym, linestyle='-')
    
    ax2.set_xlabel('Epoch', labelpad=8)
    ax2.set_ylabel('Validation Accuracy (%)', labelpad=8)
    ax2.set_title('Reasoning Accuracy', pad=12, fontweight='bold')
    
    # Despine and refine
    for ax in [ax1, ax2]:
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Faint dashed grid on the y-axis
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#d1d5db')
        # Ticks only bottom and left
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.legend(frameon=False, loc='best')

    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    pdf_path = os.path.join(out_dir, "neurips_convergence.pdf")
    png_path = os.path.join(out_dir, "neurips_convergence.png")
    
    plt.savefig(pdf_path, bbox_inches='tight', transparent=True)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Publication plots saved to {out_dir}")

def main():
    log_file = "results/symbolic_experiments/symbolic_log.txt"
    out_dir = "results/symbolic_experiments"
    
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return
        
    metrics = parse_logs(log_file)
    print(f"Parsed {len(metrics['Baseline']['loss'])} epochs for Baseline.")
    print(f"Parsed {len(metrics['Symbolic']['loss'])} epochs for Symbolic.")
    
    create_publication_plot(metrics, out_dir)

if __name__ == "__main__":
    main()
