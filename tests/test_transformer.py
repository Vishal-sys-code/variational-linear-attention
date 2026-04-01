import torch
import pytest
from src.models.transformer import LRATransformerBlock, LRAModel

DEVICE = torch.device('cpu')

def test_vla_transformer_block_forward():
    """
    Verify that a single VLA Transformer Block:
    1. Accepts input (B, T, d_model)
    2. Returns output (B, T, d_model)
    3. Maintains residual stream shape
    """
    B, T, d_model = 2, 10, 8
    d_ffn = 16
    
    block = LRATransformerBlock(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=0.0
    ).to(DEVICE)
    
    x = torch.randn(B, T, d_model, device=DEVICE)
    
    out = block(x)
    
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"

def test_vla_transformer_full_forward():
    """
    Verify full LRAModel:
    1. Embedding -> Layers -> Head
    2. Output shape (B, T, vocab_size)
    """
    vocab_size = 50
    d_model = 16
    d_ffn = 32
    n_layers = 2
    max_len = 20
    
    model = LRAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len
    ).to(DEVICE)
    
    B, T = 3, 15
    x = torch.randint(0, vocab_size, (B, T), device=DEVICE)
    
    logits = model(x, pool=False)
    
    assert logits.shape == (B, T, vocab_size), f"Expected {(B, T, vocab_size)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaNs"

def test_vla_transformer_backward():
    """
    Verify that gradients flow through the model.
    """
    vocab_size = 20
    d_model = 8
    model = LRAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=1
    ).to(DEVICE)
    
    x = torch.randint(0, vocab_size, (2, 5), device=DEVICE)
    target = torch.randint(0, vocab_size, (2, 5), device=DEVICE)
    
    logits = model(x, pool=False)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    
    loss.backward()
    
    # Check if gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            # lambda_net is unused in this specific VLA variant (Sherman-Morrison rank-1 only)
            if "lambda_net" in name or "cls_head" in name:
                continue
            assert param.grad is not None, f"Parameter {name} has no gradient"
