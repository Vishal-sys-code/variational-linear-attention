import torch
import torch.nn as nn
from typing import Optional, Tuple

class SymbolicPenaltyTracker(nn.Module):
    """
    Manages the symbolic penalty computation and incremental trackings for VLA.
    """
    def __init__(self, d_model: int, gamma: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.gamma = gamma
        self.eps = eps
        
        # State tensors (not persistent because they depend on sequence length/batch)
        self.register_buffer("K_past", None, persistent=False)
        self.register_buffer("W", None, persistent=False)

    def init_sequence(self, A_rel: torch.Tensor, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Precomputes normalized adjacency and initializes key tracking.
        A_rel: (B, T, T) symmetric adjacency matrix for symbolic relations.
        """
        self.K_past = torch.zeros(batch_size, max_seq_len, self.d_model, device=device, dtype=dtype)
        
        if A_rel is not None:
            # Normalize adjacency: D^{-1/2} A_rel D^{-1/2}
            D = A_rel.sum(dim=-1) + self.eps  # (B, T)
            D_inv_sqrt = 1.0 / torch.sqrt(D)  # (B, T)
            
            # W = D^{-1/2} * A_rel * D^{-1/2}
            self.W = A_rel * D_inv_sqrt.unsqueeze(-1) * D_inv_sqrt.unsqueeze(1)
        else:
            self.W = None

    def step(self, k_t: torch.Tensor, t: int) -> Optional[torch.Tensor]:
        """
        Processes step t. Returns the update vector a_t if a new relation applies, else None.
        k_t: (B, d_model)
        """
        if self.K_past is None:
            raise RuntimeError("SymbolicPenaltyTracker not initialized. Call init_sequence() first.")
            
        # Store key
        self.K_past[:, t, :] = k_t
        
        if self.gamma <= 0.0 or self.W is None:
            return None
            
        # Check if token t introduces new relations with past tokens (or itself)
        # We look at the row t of W, up to column t.
        W_t = self.W[:, t, :t+1]  # (B, t+1)
        
        # If there are no non-zero relations, skip update
        has_relation = (torch.abs(W_t).max(dim=-1).values > 1e-9)
        
        if not has_relation.any():
            return None
            
        # Compute a_t = sum_{j<=t} W_{t,j} * k_j
        # K_past[:, :t+1, :] -> (B, t+1, d)
        # W_t.unsqueeze(-1) -> (B, t+1, 1)
        a_t = (self.K_past[:, :t+1, :] * W_t.unsqueeze(-1)).sum(dim=1)  # (B, d)
        
        # Scale by sqrt(gamma) so that (sqrt(gamma) a_t)(sqrt(gamma) a_t)^T = gamma a_t a_t^T
        a_t_scaled = a_t * torch.sqrt(torch.tensor(self.gamma, device=a_t.device, dtype=a_t.dtype))
        
        # We mask out batch elements that didn't have relations
        mask = has_relation.unsqueeze(-1).to(a_t_scaled.dtype)
        a_t_scaled = a_t_scaled * mask
        
        return a_t_scaled
