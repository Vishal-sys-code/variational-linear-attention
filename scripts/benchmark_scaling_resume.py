import os
import json
import time
import torch
from src.models.attention.vla import VLALayer
from src.models.attention.deltanet import DeltaNetLayer
from src.models.attention.linear_transformer import LinearTransformerLayer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 2
    hidden_dimension = 128
    
    # We already have results for 512, 1024, 2048, 4096, 8192
    # Load them first
    with open('results/scaling_results.json', 'r') as f:
        results = json.load(f)
        
    sequence_lengths = [16384, 32768]
    
    models = {
        "VLA": VLALayer(d_model=hidden_dimension).to(device),
        "DeltaNet": DeltaNetLayer(d_model=hidden_dimension).to(device),
        "Linear Attention": LinearTransformerLayer(d_model=hidden_dimension).to(device)
    }
    
    print("Resuming complexity scaling benchmark...")
    
    for N in sequence_lengths:
        print(f"\nSequence length: {N}")
        
        # We might run into OOM at 16384 or 32768, so try-except block
        skip_remaining = False
        
        try:
            X = torch.randn(batch_size, N, hidden_dimension, device=device)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM on sequence length {N} while allocating input. Stopping here.")
                break
            else:
                raise e
            
        N_results = {"VLA": [], "DeltaNet": [], "Linear Attention": []}
        
        for name, model in models.items():
            model.eval()
            
            with torch.no_grad():
                # Warm-up (1 run)
                try:
                    _ = model(X)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"OOM on model {name} with N={N}. Stopping for this N.")
                        skip_remaining = True
                        break
                    else:
                        raise e
                
                # Benchmark (5 runs)
                runtimes = []
                for _ in range(5):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.time()
                    
                    try:
                        _ = model(X)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            skip_remaining = True
                            break
                        else:
                            raise e
                            
                    end_time = time.time()
                    runtimes.append(end_time - start_time)
                    
                if skip_remaining:
                    break
                    
                avg_runtime = sum(runtimes) / len(runtimes)
                tokens_per_sec = (batch_size * N) / avg_runtime
                N_results[name].append(tokens_per_sec)
                
                print(f"{name} tokens/sec: {tokens_per_sec:.0f}")
                
        if skip_remaining:
            print(f"Skipping further sequence lengths due to OOM.")
            break
            
        results["sequence_lengths"].append(N)
        results["VLA_tokens_per_sec"].append(N_results["VLA"][0])
        results["DeltaNet_tokens_per_sec"].append(N_results["DeltaNet"][0])
        results["LinearAttention_tokens_per_sec"].append(N_results["Linear Attention"][0])
        
        # Save after each N so we don't lose progress if it times out
        os.makedirs('results', exist_ok=True)
        with open('results/scaling_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Small sleep to cool down device
        time.sleep(0.5)
        
    print(f"\nSaved final results to results/scaling_results.json")

if __name__ == "__main__":
    main()