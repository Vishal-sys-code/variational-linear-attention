import torch
import torch.nn as nn
from typing import Optional

from src.models.attention.vla import VLALayer

class VLATransformerBlock(nn.Module):
    """
    Transformer block using Variational Linear Attention (VLA).
    Structure:
    x -> LayerNorm -> VLA -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        # VLA specific args
        vla_lambda_0: float = 1.0,
        vla_penalty_rank: int = 1,
    ):
        super().__init__()
        
        # 1. Attention Sub-layer
        self.ln1 = nn.LayerNorm(d_model)
        self.vla = VLALayer(
            d_model=d_model,
            d_head=d_model, # Enforced d_head = d_model
            lambda_0=vla_lambda_0,
            penalty_rank=vla_penalty_rank
        )
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)
        Returns:
            Output tensor (B, T, d_model)
        """
        # Pre-LN Architecture
        
        # 1. Attention Path
        residual = x
        x_norm = self.ln1(x)
        attn_out = self.vla(x_norm)
        x = residual + self.dropout1(attn_out)
        
        # 2. FFN Path
        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        
        return x


class VLATransformer(nn.Module):
    """
    Full Transformer model using VLA blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        d_ffn: Optional[int] = None,
        max_len: int = 512,
        dropout: float = 0.1,
        vla_lambda_0: float = 1.0,
        vla_penalty_rank: int = 1,
    ):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = 4 * d_model
            
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            VLATransformerBlock(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                vla_lambda_0=vla_lambda_0,
                vla_penalty_rank=vla_penalty_rank
            )
            for _ in range(n_layers)
        ])
        
        # Final normalization before output projection
        self.ln_f = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices (B, T)
        Returns:
            Logits (B, T, vocab_size)
        """
        B, T = x.shape
        device = x.device
        
        # Embeddings
        tok_emb = self.token_embedding(x) # (B, T, d_model)
        
        # Position embeddings
        positions = torch.arange(T, device=device).unsqueeze(0) # (1, T)
        pos_emb = self.position_embedding(positions) # (1, T, d_model)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Layers
        for layer in self.layers:
            x = layer(x)
            
        # Final Norm
        x = self.ln_f(x)
        
        # Output Head
        logits = self.head(x) # (B, T, vocab_size)
        
        return logits