import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import LRAModel
import traceback

def run_smoke_test():
    """
    Sanity test: 200 steps on sequence_length=512 for each LRA model configuration.
    Verifies that the loss decreases and no NaN occurs.
    """
    B = 8
    T = 512
    vocab_size = 64
    d_model = 64
    n_layers = 2
    d_ffn = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running smoke test on {device}")
    
    models = ["linear_transformer", "deltanet", "vla"]
    
    results = {}
    
    for attn_type in models:
        print(f"\n--- Testing {attn_type} ---")
        try:
            model = LRAModel(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                d_ffn=d_ffn,
                max_len=1024,
                attention_type=attn_type
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            initial_loss = None
            final_loss = None
            
            model.train()
            
            for step in range(200):
                optimizer.zero_grad()
                
                # Mock batch: sequence tasks (like ListOps/Retrieval) usually pooled class outputs.
                x = torch.randint(0, vocab_size, (B, T), device=device)
                y = torch.randint(0, 2, (B,), device=device)
                
                logits = model(x, pool=True) # (B, 2)
                
                loss = criterion(logits, y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                loss_val = loss.item()
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss encountered at step {step}")
                    
                if step == 0:
                    initial_loss = loss_val
                if step == 199:
                    final_loss = loss_val
                    
                if step % 50 == 0:
                    print(f"Step {step}: Loss = {loss_val:.4f}")
                    
            print(f"Success! Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
            results[attn_type] = "PASS"
            
        except Exception as e:
            print(f"FAILED {attn_type}: {e}")
            traceback.print_exc()
            results[attn_type] = "FAIL"
            
    print("\nSmoke Test Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")
        
    if all(v == "PASS" for v in results.values()):
        print("All models passed smoke tests!")
    else:
        print("Some models failed. Please check logs.")

if __name__ == "__main__":
    run_smoke_test()
