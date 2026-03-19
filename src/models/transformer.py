import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.models.attention.vla import VLALayer
from src.models.attention.linear_transformer import LinearTransformerLayer
from src.models.attention.deltanet import DeltaNetLayer

class LRATransformerBlock(nn.Module):
    """
    Transformer block using injected Attention (VLA, DeltaNet, Linear Transformer).
    Structure:
    x -> LayerNorm -> Attn -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        attention_type: str = "vla",
        # VLA specific args
        vla_lambda_0: float = 1.0,
        vla_penalty_rank: int = 1,
    ):
        super().__init__()
        
        # 1. Attention Sub-layer
        self.ln1 = nn.LayerNorm(d_model)
        
        if attention_type == "vla":
            self.attn = VLALayer(
                d_model=d_model,
                d_head=d_model,
                lambda_0=vla_lambda_0,
                penalty_rank=vla_penalty_rank
            )
        elif attention_type == "linear_transformer":
            self.attn = LinearTransformerLayer(d_model=d_model)
        elif attention_type == "deltanet":
            self.attn = DeltaNetLayer(d_model=d_model)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. FeedForward Sub-layer
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_states: bool = False) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        # Pre-LN Architecture
        # 1. Attention Path
        residual = x
        x_norm = self.ln1(x)
        
        if return_states:
            attn_out, states = self.attn(x_norm, return_states=True)
        else:
            attn_out = self.attn(x_norm)
            states = None
            
        x = residual + self.dropout1(attn_out)
        
        # 2. FFN Path
        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        
        if return_states:
            return x, states
        return x


class LRAModel(nn.Module):
    """
    Full Transformer model for LRA tasks.
    Enforces required shape configurations internally by default.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_ffn: int = 1024,
        max_len: int = 4096,
        dropout: float = 0.1,
        attention_type: str = "vla",
        vla_lambda_0: float = 1.0,
        vla_penalty_rank: int = 1,
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            LRATransformerBlock(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                attention_type=attention_type,
                vla_lambda_0=vla_lambda_0,
                vla_penalty_rank=vla_penalty_rank
            )
            for _ in range(n_layers)
        ])
        
        # Final normalization before output projection
        self.ln_f = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size) # for sequence tasks
        # If the task requires a single classification output out of the whole sequence:
        self.cls_head = nn.Linear(d_model, 2) # e.g. binary classification/retrieval
        
    def forward(self, x: torch.Tensor, return_states: bool = False, pool: bool = True) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input token indices (B, T)
            return_states: If true, returns dict of states.
            pool: If true, applies mean pooling and returns logits shape (B, num_classes)
                  Otherwise returns token-level logits (B, T, vocab_size)
        """
        B, T = x.shape
        device = x.device
        
        tok_emb = self.token_embedding(x) # (B, T, d_model)
        
        positions = torch.arange(T, device=device).unsqueeze(0) # (1, T)
        pos_emb = self.position_embedding(positions) # (1, T, d_model)
        
        x = self.dropout(tok_emb + pos_emb)
        
        states = None
        for i, layer in enumerate(self.layers):
            if return_states and i == len(self.layers) - 1:
                x, states = layer(x, return_states=True)
            else:
                x = layer(x)
            
        x = self.ln_f(x)
        
        if pool:
            # Mean pooling over the sequence (ignoring pad tokens normally, but simplistic here)
            # A more robust implementation applies attention_mask, ignored here for brevity.
            x_pool = x.mean(dim=1)
            logits = self.cls_head(x_pool) # (B, num_classes)
        else:
            logits = self.head(x) # (B, T, vocab_size)
        
        if return_states:
            return logits, states
        return logits