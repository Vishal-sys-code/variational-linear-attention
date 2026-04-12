import torch
import pytest
from src.models.attention.fast_vla import VLAParallel, VLATriton, HAS_TRITON

def test_fast_vla_variants():
    d = 64
    B = 2
    T = 16
    device = "cpu"
    
    x = torch.randn(B, T, d)
    
    par1 = VLAParallel(d_model=d, use_kv_exploding_fix=True)
    out1 = par1(x)
    assert out1.shape == (B, T, d)
    
    par2 = VLAParallel(d_model=d, use_kv_exploding_fix=False)
    out2 = par2(x)
    assert out2.shape == (B, T, d)

if __name__ == "__main__":
    pytest.main([__file__])
