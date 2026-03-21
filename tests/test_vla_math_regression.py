import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer import LRAModel

def test_vla_math_regression_no_nans():
    """
    A small run CI test: 1 batch, short sequence, check no NaNs in output and states.
    Prevents regressions in core math of VLA.
    """
    vocab_size = 10
    d_model = 16
    n_layers = 1
    
    model = LRAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        max_len=32,
        attention_type="vla"
    )
    
    # 1 batch, 4 timesteps
    x = torch.randint(0, vocab_size, (1, 4))
    
    # Forward with states
    logits, states = model(x, return_states=True)
    
    assert not torch.isnan(logits).any(), "NaNs detected in logits"
    assert "A" in states, "State dict missing 'A'"
    assert "S_norm" in states, "State dict missing 'S_norm'"
    
    assert not torch.isnan(states["A"]).any(), "NaNs detected in A_t"
    assert not torch.isnan(states["S_norm"]).any(), "NaNs detected in S_t norm"
    assert not torch.isnan(states["q"]).any(), "NaNs detected in q_t"
    assert not torch.isnan(states["v"]).any(), "NaNs detected in v_t"
    assert not torch.isnan(states["alpha"]).any(), "NaNs detected in alpha_t"
    
    # Condition number should be finite
    A_T = states["A"][0, -1] # Last timestep A
    cond = torch.linalg.cond(A_T.to(torch.float64))
    assert torch.isfinite(cond), "A_t condition number is infinite or NaN"
    
    print("CI test passed: No NaNs, finite condition numbers.")

if __name__ == "__main__":
    test_vla_math_regression_no_nans()
