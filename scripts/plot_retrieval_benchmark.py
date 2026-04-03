import os
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Set aesthetic styling for NeurIPS
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colorblind-friendly palette (Wong palette)
COLORS = {
    "VLA": "#0072B2",             # Blue
    "DeltaNet": "#D55E00",        # Vermilion
    "Linear Attention": "#009E73" # Bluish Green
}
MARKERS = {
    "VLA": "o",
    "DeltaNet": "s",
    "Linear Attention": "^"
}

def parse_log(filepath):
    epochs = []
    test_accs = []
    final_acc = None

    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return epochs, test_accs, final_acc

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_table = False
    for line in lines:
        if line.startswith("==="):
            if not in_table:
                pass # start or end of table
            continue
        if "Epoch" in line and "Train Loss" in line:
            in_table = True
            continue

        if in_table:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[0].isdigit():
                epoch = int(parts[0])
                test_acc = float(parts[3])
                epochs.append(epoch)
                test_accs.append(test_acc)
            elif "Early stopping" in line or line.strip() == "":
                pass
            else:
                in_table = False

        if "Final Test Accuracy" in line:
            parts = line.split(":")
            if len(parts) == 2:
                final_acc = float(parts[1].strip())

    if final_acc is None and len(test_accs) > 0:
        final_acc = max(test_accs)

    return epochs, test_accs, final_acc

def main():
    base_dir = "results/benchmark_retrieval"
    out_dir = os.path.join(base_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    models = {
        "VLA": "vla",
        "DeltaNet": "deltanet",
        "Linear Attention": "linear_attention"
    }
    seq_lens = [256, 512, 1024]

    # Data structures
    # model -> seq_len -> {'epochs': [], 'test_accs': [], 'final_acc': float}
    data = {m: {} for m in models}

    for model_name, model_dir in models.items():
        for sl in seq_lens:
            filepath = os.path.join(base_dir, model_dir, f"log_seq_{sl}.txt")
            epochs, test_accs, final_acc = parse_log(filepath)
            data[model_name][sl] = {
                'epochs': epochs,
                'test_accs': test_accs,
                'final_acc': final_acc
            }

    # Option A: Final Test Accuracy vs Sequence Length
    fig_a, ax_a = plt.subplots(figsize=(6, 4.5))
    for model_name in models:
        x = []
        y = []
        for sl in seq_lens:
            if data[model_name][sl]['final_acc'] is not None:
                x.append(sl)
                y.append(data[model_name][sl]['final_acc'])
        if x and y:
            ax_a.plot(x, y, label=model_name, color=COLORS[model_name],
                      marker=MARKERS[model_name], markersize=8)

    ax_a.set_xticks(seq_lens)
    ax_a.set_xlabel("Sequence Length")
    ax_a.set_ylabel("Final Test Accuracy")
    ax_a.set_title("Retrieval Benchmark: Accuracy vs. Sequence Length")
    ax_a.legend()
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(out_dir, "option_a_accuracy_vs_length.png"))
    plt.close(fig_a)

    # Option B: Grid of subplots for each sequence length (Epoch vs Test Accuracy)
    fig_b, axes_b = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for i, sl in enumerate(seq_lens):
        ax = axes_b[i]
        for model_name in models:
            epochs = data[model_name][sl]['epochs']
            accs = data[model_name][sl]['test_accs']
            if epochs and accs:
                ax.plot(epochs, accs, label=model_name, color=COLORS[model_name])
        ax.set_title(f"Sequence Length: {sl}")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Test Accuracy")
        if i == 2:
            ax.legend()

    fig_b.tight_layout()
    fig_b.savefig(os.path.join(out_dir, "option_b_learning_curves.png"))
    plt.close(fig_b)

    # Option C: Combined multi-panel figure
    fig_c = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig_c, height_ratios=[1.2, 1])

    # Top panel: scaling plot (span all 3 columns)
    ax_c_top = fig_c.add_subplot(gs[0, :])
    for model_name in models:
        x = []
        y = []
        for sl in seq_lens:
            if data[model_name][sl]['final_acc'] is not None:
                x.append(sl)
                y.append(data[model_name][sl]['final_acc'])
        if x and y:
            ax_c_top.plot(x, y, label=model_name, color=COLORS[model_name],
                          marker=MARKERS[model_name], markersize=10)
    ax_c_top.set_xticks(seq_lens)
    ax_c_top.set_xlabel("Sequence Length")
    ax_c_top.set_ylabel("Final Test Accuracy")
    ax_c_top.set_title("Memory Retention Across Sequence Lengths")
    ax_c_top.legend(loc="lower left")

    # Bottom panels: learning curves
    for i, sl in enumerate(seq_lens):
        ax = fig_c.add_subplot(gs[1, i])
        for model_name in models:
            epochs = data[model_name][sl]['epochs']
            accs = data[model_name][sl]['test_accs']
            if epochs and accs:
                ax.plot(epochs, accs, label=model_name, color=COLORS[model_name])
        ax.set_title(f"Learning Curve (Seq Len: {sl})")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Test Accuracy")
        # share y axis manually or just let them auto-scale to 0-1
        ax.set_ylim(-0.05, 1.05)

    fig_c.tight_layout()
    fig_c.savefig(os.path.join(out_dir, "option_c_combined_figure.png"))
    plt.close(fig_c)

    print("Plots generated successfully in", out_dir)

if __name__ == "__main__":
    main()
