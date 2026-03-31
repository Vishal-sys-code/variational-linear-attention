import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class DeltaNetLayer(nn.Module):
    """
    DeltaNet Exact Recurrence.
    S_t = S_{t-1} - beta_t * (S_{t-1} k_t) k_t^T + beta_t * v_t k_t^T
    o_t = S_t q_t
    beta_t = sigmoid(W_beta k_t)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_beta = nn.Linear(d_model, 1)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, return_states: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        B, T, D = x.shape
        device = x.device
        
        outputs = []
        
        # S_0 is zero
        S = torch.zeros(B, D, D, device=device)
        
        # Vectorize parameter mappings over the entire sequence T beforehand!
        Q = self.W_q(x)
        K = F.normalize(self.W_k(x), p=2, dim=-1)
        V = self.W_v(x)
        Beta = torch.sigmoid(self.W_beta(K))
        
        for t in range(T):
            q = Q[:, t, :]
            k = K[:, t, :]
            v = V[:, t, :]
            
            # (B, 1)
            beta = Beta[:, t, :]
            
            # S_{t-1} k_t
            # (B, D, D) @ (B, D, 1) -> (B, D, 1) -> (B, D)
            S_k = torch.bmm(S, k.unsqueeze(2)).squeeze(2)
            
            # - beta_t * (S_{t-1} k_t) k_t^T
            # (B, D, 1) @ (B, 1, D) -> (B, D, D)
            term1 = beta.unsqueeze(2) * torch.bmm(S_k.unsqueeze(2), k.unsqueeze(1))
            
            # + beta_t * v_t k_t^T
            term2 = beta.unsqueeze(2) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            
            S = S - term1 + term2
            
            # o_t = S_t q_t
            o_t = torch.bmm(S, q.unsqueeze(2)).squeeze(2)
            outputs.append(o_t)
            
        O = torch.stack(outputs, dim=1)
        O = self.W_o(O)
        
        if return_states:
            return O, {}
        return O
