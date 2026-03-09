import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class MemoryMatrixManager(nn.Module):
    """
    Manages the recurrent memory matrix S_t for Variational Linear Attention.
    
    S_t stores the cumulative effect of past value vectors weighted by their
    recurrence coefficients using a rank-1 update rule:
    
        S_t = S_{t-1} + v_t (alpha_t)^T
        
    It also computes the attention output:
    
        o_t = S_t q_t
        
    Attributes:
        d_model (int): Dimension of the model.
        enable_renorm (bool): Whether to enable renormalization of S_t.
        renorm_threshold (float): Threshold for S_t Frobenius norm to trigger renormalization.
    """
    
    def __init__(
        self,
        d_model: int,
        enable_renorm: bool = False,
        renorm_threshold: float = 1e4,
    ):
        super().__init__()
        self.d_model = d_model
        self.enable_renorm = enable_renorm
        self.renorm_threshold = renorm_threshold
        
        # S_t is a buffer (non-persistent state)
        # Shape: (B, d, d)
        self.register_buffer("S_t", None, persistent=False)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long), persistent=False)
        
    def reset(self, batch_size: int = 1, device=None, dtype=torch.float32):
        """
        Resets the memory matrix S_t to zeros.
        
        Args:
            batch_size: Batch dimension B.
            device: Device to initialize S_t on.
            dtype: Data type for S_t (must be float32).
        """
        if device is None:
            # Default to current device if available, else CPU
            device = self.step.device if self.step.device.type != 'cpu' else torch.device('cpu')
            
        # Ensure float32 as per spec
        if dtype != torch.float32:
            # We enforce float32 for stability
            dtype = torch.float32
            
        self.S_t = torch.zeros(
            batch_size, 
            self.d_model, 
            self.d_model, 
            device=device, 
            dtype=dtype
        )
        self.step.zero_()
        
    def update(self, v_t: torch.Tensor, alpha_t: torch.Tensor) -> Dict[str, float]:
        """
        Updates S_t using rank-1 update: S_t = S_{t-1} + v_t (alpha_t)^T.
        
        Args:
            v_t: Value vector at time t, shape (B, d).
            alpha_t: Recurrence coefficient vector at time t, shape (B, d).
            
        Returns:
            Dictionary containing statistics (norm, renorm_triggered).
        """
        if self.S_t is None:
            raise RuntimeError("MemoryMatrixManager not initialized. Call reset() first.")
            
        # Ensure inputs match dimensions
        assert v_t.dim() == 2, f"v_t must be (B, d), got {v_t.shape}"
        assert alpha_t.dim() == 2, f"alpha_t must be (B, d), got {alpha_t.shape}"
        B, d = v_t.shape
        assert d == self.d_model, f"Dimension mismatch: v_t={v_t.shape}, d_model={self.d_model}"
        
        # Cast inputs to float32 for update stability if they aren't already
        v_t_f32 = v_t.to(dtype=torch.float32)
        # Normalize v_t before writing to prevent S_t from exploding
        v_t_f32 = v_t_f32 / (torch.norm(v_t_f32, dim=-1, keepdim=True) + 1e-6)
        alpha_t_f32 = alpha_t.to(dtype=torch.float32)
        
        # Compute outer product: v_t (alpha_t)^T
        # v: (B, d, 1), alpha: (B, 1, d) -> (B, d, d)
        update_term = torch.matmul(v_t_f32.unsqueeze(2), alpha_t_f32.unsqueeze(1))
        
        # Update S_t (out-of-place)
        self.S_t = self.S_t + update_term
        self.step += 1
        
        # Numerical stability checks
        # Compute Frobenius norm per batch element
        # (B, d, d) -> (B,)
        norms = torch.norm(self.S_t, p='fro', dim=(1, 2))
        
        # We handle renormalization logic.
        # If enabled, we renormalize ONLY the batch elements that exceed threshold?
        # The prompt says: "Renormalize S_t by dividing: S_t = S_t / (norm_S_t)".
        # This implies per-sample or global? Usually per-sample in batch.
        # "If the norm exceeds a predefined threshold... take one of the following actions... Renormalize S_t"
        
        renorm_triggered = False
        max_norm = norms.max().item()
        
        if self.enable_renorm:
            # Check which exceed threshold
            mask = norms > self.renorm_threshold
            if mask.any():
                renorm_triggered = True
                
                # We want divisor to be 1.0 where mask is False, and norms where mask is True.
                divisor = torch.ones_like(norms)
                divisor[mask] = norms[mask]
                
                # Reshape for broadcasting (B, 1, 1)
                divisor = divisor.view(-1, 1, 1)
                
                self.S_t = self.S_t / divisor
        
        # Prepare stats
        stats = {
            "norm_max": max_norm,
            "norm_mean": norms.mean().item(),
            "renorm_triggered": float(renorm_triggered),
            "step": self.step.item()
        }
        
        return stats
        
    def compute_output(self, q_t: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention output o_t = S_t q_t.
        
        Args:
            q_t: Query vector at time t, shape (B, d).
            
        Returns:
            Output vector o_t, shape (B, d).
        """
        if self.S_t is None:
            raise RuntimeError("MemoryMatrixManager not initialized. Call reset() first.")
            
        assert q_t.dim() == 2, f"q_t must be (B, d), got {q_t.shape}"
        assert q_t.size(1) == self.d_model, f"Dimension mismatch: q_t={q_t.shape}, d_model={self.d_model}"
        
        # Cast q_t to float32 for computation
        q_t_f32 = q_t.to(dtype=torch.float32)
        
        # o_t = S_t @ q_t
        # (B, d, d) @ (B, d, 1) -> (B, d, 1)
        o_t = torch.matmul(self.S_t, q_t_f32.unsqueeze(2))
        
        # Squeeze back to (B, d)
        o_t = o_t.squeeze(2)
        
        return o_t
        
    def get_S(self) -> torch.Tensor:
        """Returns the current memory matrix S_t."""
        return self.S_t