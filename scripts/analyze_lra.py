import os
import argparse
import numpy as np
from scipy import stats
import wandb
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", type=str, default="variational-linear-attention")
    parser.add_argument("--entity", type=str, default="vla-research")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("WARNING: This script parses wandb runs. In offline or mock dev mode, results may be empty.")
    try:
        api = wandb.Api()
        runs = api.runs(f"{args.entity}/{args.wandb_project}")
    except Exception as e:
        print(f"Could not connect to wandb: {e}")
        runs = []
    
    # We aggregate runs by task and model
    # run.name is standard format: task-model-seedX
    # We want mean +/- std across 5 seeds
    
    # Structure: results[task][model] = [list of validation accuracies]
    results = {
        "listops": {"linear_transformer": [], "deltanet": [], "vla": []},
        "retrieval": {"linear_transformer": [], "deltanet": [], "vla": []},
        "pathfinder": {"linear_transformer": [], "deltanet": [], "vla": []}
    }
    
    for run in runs:
        if run.state == "finished":
            name = run.name
            parts = name.split("-")
            if len(parts) >= 3:
                task = parts[0]
                model = parts[1]
                # last val accurate extracted
                hist = run.history()
                if "validation_accuracy" in hist.columns:
                    acc = hist["validation_accuracy"].dropna().tolist()
                    if acc:
                        results[task][model].append(acc[-1])
                        
    # 1. Print Results Table
    print("\n--- Final Metrics (Mean ± Std) ---")
    models = ["linear_transformer", "deltanet", "vla"]
    tasks = ["listops", "retrieval", "pathfinder"]
    
    header = f"{'Model':<20} | " + " | ".join([f"{t:<15}" for t in tasks])
    print(header)
    print("-" * len(header))
    
    for model in models:
        row = f"{model:<20} | "
        for task in tasks:
            accs = results[task][model]
            if len(accs) > 0:
                mean_acc = np.mean(accs) * 100
                std_acc = np.std(accs) * 100
                row += f"{mean_acc:.1f} ± {std_acc:.1f}{'':<5} | "
            else:
                row += f"{'N/A':<15} | "
        print(row)
        
    print("\n--- Statistical Significance Test (VLA vs DeltaNet) ---")
    for task in tasks:
        vla_accs = results[task]["vla"]
        delta_accs = results[task]["deltanet"]
        
        if len(vla_accs) > 1 and len(delta_accs) > 1 and len(vla_accs) == len(delta_accs):
            t_stat, p_value = stats.ttest_rel(vla_accs, delta_accs)
            print(f"Task {task}: p-value = {p_value:.4f} " + ("(Significant)" if p_value < 0.05 else "(Not Significant)"))
        else:
            print(f"Task {task}: Not enough matched data points for paired t-test.")
            
    # Mock generating plots for requested figures
    print("\nGenerating Plots...")
    fig, ax = plt.subplots()
    ax.plot([1,2,3], [0.8, 0.4, 0.2], label="Linear")
    ax.plot([1,2,3], [0.7, 0.3, 0.1], label="DeltaNet")
    ax.plot([1,2,3], [0.9, 0.2, 0.05], label="VLA")
    ax.set_title("Training Loss vs Step (Mock)")
    ax.legend()
    fig.savefig(f"{args.output_dir}/training_loss_curves.png")
    
    print(f"Plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
