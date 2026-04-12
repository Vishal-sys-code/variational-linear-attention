import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wong color palette (colorblind friendly)
WONG_PALETTE = {
    "VLA": "#D55E00",              # vermillion
    "DeltaNet": "#0072B2",         # blue
    "Linear": "#E69F00",           # orange
    "VLAMamba": "#56B4E9",         # skyblue
    "VLATriton": "#009E73",        # bluishgreen
    "VLAKVExploding": "#CC79A7",   # reddishpurple
    "VLACombined": "#000000"       # black
}

def main():
    csv_files = glob.glob("results/retrieval_*.csv")
    if not csv_files:
        print("No retrieval CSV files found in results/.")
        return

    # Parse and combine
    all_data = []
    for f in csv_files:
        # e.g. retrieval_VLA_256.csv
        basename = os.path.basename(f)
        parts = basename.replace(".csv", "").split("_")
        if len(parts) >= 3:
            model = parts[1]
            seq_len = int(parts[2])
            df = pd.read_csv(f)
            df["Model"] = model
            df["SeqLen"] = seq_len
            all_data.append(df)
            
    if not all_data:
        print("Could not parse any valid data.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    for model in combined_df["Model"].unique():
        model_data = combined_df[combined_df["Model"] == model]
        # Calculate max accuracy per seq len to plot
        acc_data = model_data.groupby("SeqLen")["test_acc"].max().reset_index()
        acc_data = acc_data.sort_values(by="SeqLen")
        
        color = WONG_PALETTE.get(model, "#000000")
        plt.plot(acc_data["SeqLen"], acc_data["test_acc"], marker='o', label=model, color=color, linewidth=2)

    plt.xlabel("Sequence Length")
    plt.ylabel("Test Accuracy (Exact Match)")
    plt.title("Associative Retrieval Benchmark Accuracy")
    plt.xscale('log', base=2)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("results/benchmark_retrieval/images", exist_ok=True)
    out_path = "results/benchmark_retrieval/images/retrieval_accuracy.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved accuracy plot to {out_path}")

if __name__ == "__main__":
    main()
