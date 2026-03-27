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
    "pdf.fonttype": 42, # TrueType fonts for PDF
    "ps.fonttype": 42
})

def main():
    base_dir = "results/benchmark_suite"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist. Cannot generate paper plots.")
        return
        
    tasks = ["listops", "cqa", "clutrr"]
    task_labels = ["ListOps", "CommonsenseQA", "CLUTRR"]
    models = ["linear_transformer", "deltanet", "vla"]
    
    results = {task: {model: [] for model in models} for task in tasks}
    
    # Regex parser for benchmark output text logs
    pattern = re.compile(r"Run\s+([a-zA-Z0-9_]+)-([a-zA-Z0-9_]+)-seed\d+\s+finished\.\s+Val Acc:\s+([\d.]+)")
    txt_files = glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)
    
    for filepath in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                matches = pattern.findall(content)
                for t, m, acc_str in matches:
                    if t.lower() in results and m.lower() in results[t.lower()]:
                        results[t.lower()][m.lower()].append(float(acc_str) * 100) # Convert to %
        except Exception as e:
            pass
            
    plot_data = {task: {} for task in tasks}
    for task in tasks:
        for model in models:
            accs = results[task][model]
            if len(accs) > 0:
                plot_data[task][model] = (np.mean(accs), np.std(accs))
            else:
                plot_data[task][model] = (0.0, 0.0)

    # ---------------------------------------------------------
    # Publication Plot 1: The Grouped Bar Chart (Per Task)
    # ---------------------------------------------------------
    x = np.arange(len(tasks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Colorblind friendly, high-contrast palette (Academic)
    # Slate Blue, Burnt Orange, Forest Green
    colors = ['#5D8AA8', '#E27D60', '#2E8B57'] 
    hatch_patterns = ['', '////', '\\\\\\\\'] # Add hatching for B&W printing compatibility
    
    model_labels = ["Linear Transformer", "DeltaNet", "VLA (Ours)"]
    
    bars_list = []
    for i, model in enumerate(models):
        means = [plot_data[task][model][0] for task in tasks]
        stds = [plot_data[task][model][1] for task in tasks]
        offset = (i - 1) * width 
        
        bars = ax.bar(x + offset, means, width, yerr=stds, label=model_labels[i], 
                      color=colors[i], edgecolor='black', linewidth=1.0,
                      hatch=hatch_patterns[i], capsize=4, alpha=0.9, error_kw={'capthick':1, 'elinewidth':1})
        bars_list.append(bars)
    
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontweight='bold')
    
    # Remove top and right spines for Tufte-style minimalism
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Beautiful legend placement
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
    plt.tight_layout()
    
    pdf_path_1 = os.path.join(base_dir, "neurips_fig1_per_task.pdf")
    png_path_1 = os.path.join(base_dir, "neurips_fig1_per_task.png")
    plt.savefig(pdf_path_1, format='pdf')
    plt.savefig(png_path_1, dpi=300)
    print(f"Generated NeurIPS Vector PDF (Task Level): {pdf_path_1}")
    
    
    # ---------------------------------------------------------
    # Publication Plot 2: Aggregated / Overall Performance
    # ---------------------------------------------------------
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    overall_means = {}
    overall_stds = {}
    
    for model in models:
        # Aggregate all accuracy samples across all tasks for the model
        all_accs = []
        for task in tasks:
            all_accs.extend(results[task][model])
        overall_means[model] = np.mean(all_accs) if all_accs else 0.0
        # Compute pooled std dev
        overall_stds[model] = np.std(all_accs) if all_accs else 0.0

    y_pos = np.arange(len(models))
    
    # Re-order to trace [Linear, DeltaNet, VLA] -> Top-down aesthetics
    ordered_models = ["linear_transformer", "deltanet", "vla"]
    ordered_labels = ["Linear Transformer", "DeltaNet", "VLA (Ours)"]
    vals = [overall_means[m] for m in ordered_models]
    errs = [overall_stds[m] for m in ordered_models]
    
    ax.barh(y_pos, vals, xerr=errs, align='center', color=colors, 
            edgecolor='black', linewidth=1.0, capsize=5, height=0.6, alpha=0.9)
            
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_labels, fontweight='bold')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Overall Aggregate Accuracy (%)', fontweight='bold')
    
    # Minimalist border formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate precise percentage on the bars for reviewer clarity
    for i, v in enumerate(vals):
        ax.text(v + errs[i] + 0.2, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    pdf_path_2 = os.path.join(base_dir, "neurips_fig2_overall.pdf")
    png_path_2 = os.path.join(base_dir, "neurips_fig2_overall.png")
    plt.savefig(pdf_path_2, format='pdf')
    plt.savefig(png_path_2, dpi=300)
    print(f"Generated NeurIPS Vector PDF (Aggregate): {pdf_path_2}")

if __name__ == "__main__":
    main()
