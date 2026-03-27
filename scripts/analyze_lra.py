import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

def main():
    base_dir = "results/benchmark_suite"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return
        
    tasks = ["listops", "cqa", "clutrr"]
    # Model folder names to actual model string names used in logs
    models = ["linear_transformer", "deltanet", "vla"]
    
    # results[task][model] = [acc1, acc2, acc3]
    results = {
        task: {model: [] for model in models}
        for task in tasks
    }
    
    # Regex to match: "Run listops-vla-seed1 finished. Val Acc: 0.1543"
    pattern = re.compile(r"Run\s+([a-zA-Z0-9_]+)-([a-zA-Z0-9_]+)-seed\d+\s+finished\.\s+Val Acc:\s+([\d.]+)")
    
    # Recursively find all txt files
    txt_files = glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)
    
    print(f"Found {len(txt_files)} log files. Parsing...\n")
    
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # Find all matching accuracies in this file
            matches = pattern.findall(content)
            for task_match, model_match, acc_str in matches:
                task = task_match.lower()
                model = model_match.lower()
                acc = float(acc_str)
                
                if task in results and model in results[task]:
                    results[task][model].append(acc)
    
    # Calculate Mean and Std
    print("=" * 60)
    print(f"{'Task':<12} | {'Model':<20} | {'Seed Count':<10} | {'Val Acc (%)'}")
    print("-" * 60)
    
    plot_data = {task: {} for task in tasks}
    
    for task in tasks:
        for model in models:
            accs = results[task][model]
            if len(accs) > 0:
                accs_pct = [a * 100 for a in accs]
                mean_acc = np.mean(accs_pct)
                std_acc = np.std(accs_pct)
                plot_data[task][model] = (mean_acc, std_acc)
                print(f"{task:<12} | {model:<20} | {len(accs):<10} | {mean_acc:.2f} ± {std_acc:.2f}%")
            else:
                plot_data[task][model] = (0.0, 0.0)
                print(f"{task:<12} | {model:<20} | {0:<10} | N/A")
                
    print("=" * 60)
    
    # Generate Bar Chart
    print("\nGenerating comparative bar chart...")
    
    x = np.arange(len(tasks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors
    colors = {
        "linear_transformer": "#1f77b4", # blue
        "deltanet": "#ff7f0e",           # orange
        "vla": "#2ca02c"                 # green
    }
    
    for i, model in enumerate(models):
        means = [plot_data[task][model][0] for task in tasks]
        stds = [plot_data[task][model][1] for task in tasks]
        
        # Calculate offset for grouped bars
        offset = (i - 1) * width 
        
        ax.bar(x + offset, means, width, yerr=stds, label=model, color=colors[model], capsize=5, alpha=0.9)
    
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('LRA Benchmark Suite Results by Model', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=11)
    ax.legend(title="Architecture", fontsize=10)
    ax.set_ylim(0, max(100, ax.get_ylim()[1])) # Scale up to 100% or slightly above max
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plot_path = os.path.join(base_dir, "lra_accuracy_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Chart successfully saved to: {plot_path}")
    
    # Generate Minimalist Aesthetic Plot (Overall Performance)
    print("\nGenerating minimalist aesthetic overall performance plot...")
    plt.clf()
    
    # Calculate overall means across all tasks
    overall_means = {}
    for model in models:
        means = [plot_data[t][model][0] for t in tasks if plot_data[t][model][0] > 0]
        overall_means[model] = np.mean(means) if means else 0.0
    
    # Sort models by overall performance
    sorted_models = sorted(models, key=lambda m: overall_means[m])
    
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    
    y_pos = np.arange(len(sorted_models))
    values = [overall_means[m] for m in sorted_models]
    
    # Minimalist style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Modern Colors
    modern_colors = {
        "linear_transformer": "#A0B2C6", # Muted slate
        "deltanet": "#FF9F43",           # Vibrant orange
        "vla": "#2ECA96"                 # Emerald green (winner)
    }
    
    # Draw horizontal bars
    bars = ax.barh(y_pos, values, color=[modern_colors[m] for m in sorted_models], height=0.45, alpha=0.9)
    
    # Add annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        model_name = sorted_models[i].replace("_", " ").title()
        if model_name == "Vla": model_name = "Variational Linear Attention (VLA)"
        
        # Label above bar
        ax.text(0.1, bar.get_y() + bar.get_height() + 0.1, model_name, 
                ha='left', va='center', fontsize=12, fontweight='bold', color='#333333')
                
        # Value at end of bar
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}% Avg', 
                ha='left', va='center', fontsize=12, fontweight='bold', color=modern_colors[sorted_models[i]])
    
    ax.set_yticks([])
    ax.set_xlabel('Overall LRA Accuracy (%)', fontsize=11, color='#666666', labelpad=10)
    ax.set_title('Overall Architecture Superiority', fontsize=16, fontweight='bold', color='#222222', pad=20)
    ax.grid(axis='x', linestyle=':', color='#EEEEEE')
    
    plt.tight_layout()
    aesthetic_path = os.path.join(base_dir, "lra_overall_aesthetic.png")
    plt.savefig(aesthetic_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Minimalist Plot successfully saved to: {aesthetic_path}")

if __name__ == "__main__":
    main()
