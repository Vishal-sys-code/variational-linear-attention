import torch
import pytest
from src.models.attention.vla import VLALayer

def test_vla_layer_forward_shape():
    """Verify VLALayer forward pass output shape matches (B, T, d_model)"""
    # 1. Configuration
    B = 2
    T = 16
    d_model = 8
    d_head = 8  # Enforce d_head = d_model per Step B2 requirements
    
    # 2. Instantiation
    model = VLALayer(d_model=d_model, d_head=d_head)
    model.eval()

    # 3. Input Generation
    x = torch.randn(B, T, d_model)

    # 4. Forward Pass
    with torch.no_grad():
        output = model(x)

    # 5. Output Shape Check
    expected_shape = (B, T, d_model)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_vla_layer_determinism():
    """Verify VLALayer forward pass is deterministic"""
    B = 2
    T = 16
    d_model = 8
    
    model = VLALayer(d_model=d_model, d_head=d_model)
    model.eval()
    
    x = torch.randn(B, T, d_model)
    
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
        
    assert torch.allclose(output1, output2), "Forward pass is not deterministic!"
