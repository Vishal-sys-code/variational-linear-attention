import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import LRATransformerBlock
from foundational_model.config import VLAConfig

class VLACausalLM(nn.Module):
    """
    Autoregressive Language Model powered by Variational Linear Attention (VLA).
    Projects next-token probabilities.
    """
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        
        # Token and Position Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Note: VLA does not strictly require rotary embeddings, standard absolute is fine for POC
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # VLA Transformer Layers
        self.layers = nn.ModuleList([
            LRATransformerBlock(
                d_model=config.d_model,
                d_ffn=config.d_ffn,
                dropout=config.dropout,
                attention_type=config.attention_type,
                vla_lambda_0=config.vla_lambda_0,
                vla_penalty_rank=config.vla_penalty_rank,
                vla_gamma=config.vla_gamma,
                vla_fixed_lambda=config.vla_fixed_lambda,
                vla_enable_stabilization=config.vla_enable_stabilization
            )
            for _ in range(config.n_layers)
        ])
        
        # Final Norm and LM Head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (optional, saves parameters)
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        print(f"Initialized VLACausalLM with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_cache: bool = False, start_pos: int = 0):
        """
        Args:
            idx: (B, T) tensor of integer token indices
            targets: (B, T) tensor of target token indices for loss computation
        """
        B, T = idx.shape
        device = idx.device
        
        # Forward embeddings
        tok_emb = self.token_embedding(idx) # (B, T, d_model)
        pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos) # (1, T, d_model)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # VLA explicitly handles causality through its recurrence (since it computes state sequentially)
        # Note: In a causal setting, the current standard VLALayer implementation evaluates states causally.
        for layer in self.layers:
            x = layer(x, use_cache=use_cache)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) and (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Fast O(1) State-Cached Generation.
        1. Pass the initial prompt to build the recurrent state matrix.
        2. Enter a fast loop that only evaluates the *single newest token* using use_cache=True.
        """
        # Crop prompt if needed
        idx_cond = idx if idx.size(1) <= self.config.max_len else idx[:, -self.config.max_len:]
        
        # 1. Prefill Phase: Pass the entire prompt to build up the RNN state
        logits, _ = self(idx_cond, use_cache=False)
        
        # Pluck the final step logits to sample the first new token
        next_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(next_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 2. Decode Phase: Lightning fast O(1) generation
        current_pos = idx_cond.size(1)
        for _ in range(max_new_tokens - 1):
            # Pass ONLY the single newest token and tell the model to use the cached recurrent state
            logits, _ = self(idx_next, use_cache=True, start_pos=current_pos)
            current_pos += 1
            
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
