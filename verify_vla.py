#!/usr/bin/env python3

import torch
import torch.nn as nn
from src.models.attention.vla import VLALayer

def verify_vla_forward():
    print("=== Verifying VLA Forward Pass ===")
    
    # 1. Configuration
    B = 2
    T = 16
    d_model = 8
    d_head = 8  # Enforce d_head = d_model per Step B2 requirements
    
    print(f"Config: B={B}, T={T}, d_model={d_model}, d_head={d_head}")
    
    # 2. Instantiation
    print("Instantiating VLALayer...")
    try:
        model = VLALayer(d_model=d_model, d_head=d_head)
        model.eval()
        print("✅ VLALayer instantiated successfully.")
    except Exception as e:
        print(f"❌ Failed to instantiate VLALayer: {e}")
        return

    # 3. Input Generation
    x = torch.randn(B, T, d_model)
    print(f"Input shape: {x.shape}")

    # 4. Forward Pass
    print("Running forward pass...")
    try:
        with torch.no_grad():
            output = model(x)
        print("✅ Forward pass completed.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # 5. Output Shape Check
    expected_shape = (B, T, d_model)
    if output.shape == expected_shape:
        print(f"✅ Output shape matches expected: {output.shape}")
    else:
        print(f"❌ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
        return

    # 6. Determinism Check
    print("Checking determinism (running forward pass again)...")
    try:
        with torch.no_grad():
            output2 = model(x)
            
        if torch.allclose(output, output2):
            print("✅ Determinism check passed: Outputs are identical.")
        else:
            diff = (output - output2).abs().max().item()
            print(f"❌ Determinism check failed! Max diff: {diff}")
            return
    except Exception as e:
        print(f"❌ Determinism check crashed: {e}")
        return

    print("\n=== All Checks Passed! Ready for Step B2. ===")

if __name__ == "__main__":
    verify_vla_forward()