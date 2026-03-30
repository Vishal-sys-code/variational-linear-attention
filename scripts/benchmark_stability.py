import os
import json
import torch
from src.models.attention.vla import VLALayer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 1
    hidden_dimension = 128
    sequence_length = 10000
    
    model = VLALayer(d_model=hidden_dimension).to(device)
    model.eval()
    
    print("Starting stability analysis benchmark...")
    
    # Generate random sequence for the forward pass
    # Since we are testing internal states, random input is fine
    X = torch.randn(batch_size, sequence_length, hidden_dimension, device=device)
    
    with torch.no_grad():
        _, states = model(X, return_states=True)
        
    # Extract norms at each timestep
    # Extract explicitly requested lists from states dictionary
    A_norm_list = states["norm_A_t"].squeeze(0).tolist() # (T,)
    S_norm_list = states["norm_S_t"].squeeze(0).tolist() # (T,)
    
    timesteps = list(range(1, sequence_length + 1))
    
    results = {
        "timestep": timesteps,
        "norm_A_t": A_norm_list,
        "norm_S_t": S_norm_list
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/stability_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to results/stability_results.json")

if __name__ == "__main__":
    main()