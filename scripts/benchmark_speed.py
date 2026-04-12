import time
import torch
import matplotlib.pyplot as plt
from src.models.attention.fast_vla import VLASequential, VLAParallel, VLATriton, HAS_TRITON

# Wong color palette (colorblind friendly)
WONG_PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddishpurple": "#CC79A7"
}

def bench(layer, x, n=20):
    layer.eval()
    with torch.no_grad():
        for _ in range(5):
            layer(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            layer(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000   # ms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, B = 64, 4
    
    print(f"d={d}  B={B}  device={device}\n")
    print(f"{'T':>6}  {'Sequential':>12}  {'Parallel':>10}  {'Triton':>8}  {'Tri/Seq':>8}")
    print("-" * 56)

    seq_lens = [64, 256, 512, 1024, 2048, 4096]
    if device == "cpu":
        seq_lens = [64, 128, 256, 512] # Limit for CPU
    
    results = {
        "Sequential": [],
        "Parallel": [],
        "Triton": []
    }

    for T in seq_lens:
        x   = torch.randn(B, T, d).to(device)
        seq = VLASequential(d_model=d).to(device)
        par = VLAParallel(d_model=d).to(device)

        t_seq = bench(seq, x)
        t_par = bench(par, x)

        results["Sequential"].append(t_seq)
        results["Parallel"].append(t_par)

        if HAS_TRITON and device == "cuda":
            tri   = VLATriton(d_model=d).to(device)
            t_tri = bench(tri, x)
            results["Triton"].append(t_tri)
            print(f"{T:>6}  {t_seq:>10.1f}ms  {t_par:>8.1f}ms  {t_tri:>6.1f}ms  {t_seq/t_tri:>7.2f}x")
        else:
            print(f"{T:>6}  {t_seq:>10.1f}ms  {t_par:>8.1f}ms  {'N/A':>8}  {'N/A':>8}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(seq_lens, results["Sequential"], marker='o', color=WONG_PALETTE["vermillion"], label="Sequential (Baseline)")
    plt.plot(seq_lens, results["Parallel"], marker='s', color=WONG_PALETTE["blue"], label="Parallel Scan (Mamba-like)")
    
    if HAS_TRITON and device == "cuda":
        plt.plot(seq_lens, results["Triton"], marker='^', color=WONG_PALETTE["bluishgreen"], label="Triton Fused")

    plt.xlabel("Sequence Length (T)")
    plt.ylabel("Inference Time (ms)")
    plt.title("VLA Inference Time vs Sequence Length")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/figures/speed_benchmark.png", dpi=300)
    print("\nSaved speed benchmark plot to results/figures/speed_benchmark.png")

if __name__ == "__main__":
    main()
