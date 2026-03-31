import torch
import torch.nn as nn
from typing import Tuple, Dict

class LinearTransformerLayer(nn.Module):
    """
    Standard Linear Transformer (Katharopoulos et al.)
    phi(x) = elu(x) + 1
    attention(q,k,v) = (phi(q)^T * (phi(k) v^T)) / (phi(q)^T * sum(phi(k)))
    Recurrent Formulation.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """ elu(x) + 1 ensures strictly positive feature maps """
        return torch.nn.functional.elu(x) + 1.0
        
    def forward(self, x: torch.Tensor, return_states: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        B, T, D = x.shape
        device = x.device
        
        outputs = []
        
        # S_t: memory matrix (B, D, D)
        # Z_t: normalizer vector (B, D)
        S = torch.zeros(B, D, D, device=device)
        Z = torch.zeros(B, D, device=device)
        
        # Vectorize core linear mappings
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q_phi = self.phi(Q)
        K_phi = self.phi(K)
        
        for t in range(T):
            q_phi = Q_phi[:, t, :]
            k_phi = K_phi[:, t, :]
            v = V[:, t, :]
            
            # S_t = S_{t-1} + k_phi * v^T
            # (B, D, 1) @ (B, 1, D)
            S = S + torch.bmm(k_phi.unsqueeze(2), v.unsqueeze(1))
            
            # Z_t = Z_{t-1} + k_phi
            Z = Z + k_phi
            
            # Numerator: S_t * q_phi
            # (B, D, D) @ (B, D, 1) -> (B, D)
            numerator = torch.bmm(S.transpose(1, 2), q_phi.unsqueeze(2)).squeeze(2)
            
            # Denominator: Z_t * q_phi
            denominator = (Z * q_phi).sum(dim=-1, keepdim=True) + 1e-6
            
            o_t = numerator / denominator
            outputs.append(o_t)
            
        O = torch.stack(outputs, dim=1)
        O = self.W_o(O)
        
        if return_states:
            return O, {}
        return O
