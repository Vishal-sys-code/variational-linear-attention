import time
import torch
from src.models.attention.vla import VLALayer

def run_benchmark():
    B = 16
    T = 1000
    d_model = 64
    device = torch.device("cpu")

    # Initialize the model
    layer = VLALayer(d_model=d_model).to(device)

    # Create input tensor
    x = torch.randn(B, T, d_model, device=device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = layer(x)

    # Benchmark
    print("Benchmarking...")
    num_iters = 10
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = layer(x)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iters
    print(f"Average time per forward pass: {avg_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
