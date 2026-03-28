import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer import LRAModel
from scripts.generate_symbolic_dataset import generate_sample

def verify_equivalence():
    print("Running Equivalence Check for Gamma=0")
    # Generate mock sequence
    seq, label, A_rel = generate_sample()
    
    x = torch.tensor(seq, dtype=torch.long).unsqueeze(0) # (1, T)
    A_rel = A_rel.unsqueeze(0) # (1, T, T)
    
    vocab_size = 38
    d_model = 64
    
    # Model Baseline
    model_base = LRAModel(
        vocab_size=vocab_size, d_model=d_model, n_layers=1,
        attention_type="vla", vla_gamma=0.0, num_classes=vocab_size
    )
    model_base.eval()
    
    # Model Symbolic but gamma=0
    model_sym = LRAModel(
        vocab_size=vocab_size, d_model=d_model, n_layers=1,
        attention_type="vla", vla_gamma=0.0, num_classes=vocab_size
    )
    model_sym.load_state_dict(model_base.state_dict())
    model_sym.eval()
    
    with torch.no_grad():
        out_base = model_base(x, pool=True)
        out_sym = model_sym(x, pool=True, symbolic_adj=A_rel)
        
    diff = torch.abs(out_base - out_sym).max().item()
    print(f"Max Diff: {diff:.3e}")
    if diff < 1e-6:
        print("Equivalence Verification PASSED.")
    else:
        print("Equivalence Verification FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    verify_equivalence()
