import torch
import torch.nn as nn
from typing import Dict, Optional

class InversePenaltyTracker(nn.Module):
    """
    Tracks and updates the inverse penalty matrix A_t = (M_t)^{-1} using the
    Sherman-Morrison formula.
    
    Maintain state:
        A_t: (B, d, d)
        step: int (counter)
    
    Logic:
        M_t = M_{t-1} + u_t u_t^T
        A_t = A_{t-1} - (A_{t-1} u_t u_t^T A_{t-1}) / (1 + u_t^T A_{t-1} u_t)
        
    Stabilization:
        - If |delta| < eps, skip update and add epsilon * I (fallback).
        - Every K steps, add epsilon * I to A_t (periodic stabilization).
    """
    
    def __init__(
        self,
        d_model: int,
        lambda_0: float = 1.0,
        stabilization_eps: float = 1e-6,
        periodic_eps: float = 1e-5,
        period: int = 50,
        cond_threshold: float = 1e8,
    ):
        super().__init__()
        self.d_model = d_model
        self.lambda_0 = lambda_0
        self.stabilization_eps = stabilization_eps
        self.periodic_eps = periodic_eps
        self.period = period
        self.cond_threshold = cond_threshold
        
        # Buffers for state
        # A_t is not persistent because its shape depends on batch size.
        self.register_buffer("A_t", None, persistent=False)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("fallback_count", torch.tensor(0, dtype=torch.long), persistent=False)

    def init(self, A_0: Optional[torch.Tensor] = None, batch_size: int = 1, device=None, dtype=None):
        """
        Initialize A_0.
        
        Args:
            A_0: Optional (B, d, d) tensor.
            batch_size: If A_0 is None, create default (B, d, d).
        """
        if device is None:
            # Default to the device of the step buffer
            device = self.step.device
        if dtype is None:
            dtype = torch.float32

        if A_0 is not None:
            self.A_t = A_0.clone().to(device=device, dtype=dtype)
        else:
            # A_0 = (1 / lambda_0) * I
            # Shape (B, d, d)
            I = torch.eye(self.d_model, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
            self.A_t = (1.0 / self.lambda_0) * I
            
        self.step.zero_()
        self.fallback_count.zero_()
        
        # Ensure dimensionality
        assert self.A_t.dim() == 3, f"A_t must be (B, d, d), got {self.A_t.shape}"
        assert self.A_t.size(-1) == self.d_model
        
    def update(self, u_t: torch.Tensor):
        """
        Apply rank-1 or rank-r update.
        
        Args:
            u_t: (B, d) or (B, r, d)
        """
        if self.A_t is None:
            raise RuntimeError("InversePenaltyTracker not initialized. Call init() first.")
            
        # Handle rank-r by sequential updates
        if u_t.dim() == 3:
            # (B, r, d)
            r = u_t.size(1)
            for i in range(r):
                self._update_single(u_t[:, i, :])
        elif u_t.dim() == 2:
            # (B, d)
            self._update_single(u_t)
        else:
            raise ValueError(f"u_t must be (B, d) or (B, r, d), got {u_t.shape}")
            
    def _update_single(self, u: torch.Tensor):
        """
        Internal Sherman-Morrison update for a single vector u per batch.
        u: (B, d)
        """
        # 1. Compute z = A_{t-1} u
        # u is (B, d) -> (B, d, 1) for matmul
        u_vec = u.unsqueeze(-1) 
        z = torch.bmm(self.A_t, u_vec)  # (B, d, d) @ (B, d, 1) -> (B, d, 1)
        
        # 2. Compute delta = 1 + u^T z
        # (B, 1, d) @ (B, d, 1) -> (B, 1, 1)
        u_T = u_vec.transpose(1, 2)
        dot = torch.bmm(u_T, z).squeeze(-1).squeeze(-1) # (B,)
        delta = 1.0 + dot
        
        # 3. Check stability
        # If |delta| < eps or NaN/Inf detected, fallback: A_t = A_{prev} + eps * I
        mask_unstable = (torch.abs(delta) < self.stabilization_eps) | ~torch.isfinite(delta)
        
        # Increment step before applying updates (or after? prompt says "Every K steps". 
        # Usually implies check after update or before. 
        # "Log this at least every K steps". "Every K steps: A_t = ...".
        # I'll increment at the end to match "step count".
        
        if mask_unstable.any():
            self.fallback_count += mask_unstable.sum()
            
            # Prepare fallback update: eps * I
            B = u.size(0)
            d = self.d_model
            I = torch.eye(d, device=u.device, dtype=u.dtype).unsqueeze(0).expand(B, -1, -1)
            fallback_add = self.stabilization_eps * I
            
            # Prepare Sherman-Morrison update term
            numerator = torch.bmm(z, z.transpose(1, 2)) # (B, d, d)
            
            # Safe delta for division (avoid div by zero on unstable paths)
            safe_delta = delta.clone()
            safe_delta[mask_unstable] = 1.0
            
            update_term = numerator / safe_delta.view(-1, 1, 1) # (B, d, d)
            
            # Construct the new A_t
            A_stable = self.A_t - update_term
            A_unstable = self.A_t + fallback_add
            
            # Combine
            mask_expanded = mask_unstable.view(-1, 1, 1).expand_as(self.A_t)
            self.A_t = torch.where(mask_expanded, A_unstable, A_stable)
            
        else:
            # All stable - fast path
            numerator = torch.bmm(z, z.transpose(1, 2))
            update_term = numerator / delta.view(-1, 1, 1)
            self.A_t = self.A_t - update_term

        # Increment step
        self.step += 1
        
        # 4. Periodic Stabilization
        # Every K steps: A_t = A_t + (periodic_eps * I)
        if self.step.item() % self.period == 0:
            B = u.size(0)
            d = self.d_model
            I = torch.eye(d, device=u.device, dtype=u.dtype).unsqueeze(0).expand(B, -1, -1)
            self.A_t = self.A_t + (self.periodic_eps * I)

    def get(self) -> torch.Tensor:
        """Return current A_t."""
        return self.A_t

    def diagnostics(self) -> Dict[str, float]:
        """
        Compute condition number and other stats.
        Costly operation, call sparingly.
        """
        if self.A_t is None:
            return {}
            
        # Condition number: max_eig / min_eig
        try:
            # cond number can be huge, use float64 for stability in calc if possible
            A_64 = self.A_t.to(dtype=torch.float64)
            conds = torch.linalg.cond(A_64) # (B,)
            
            # Check for infinity
            valid = torch.isfinite(conds)
            if valid.any():
                max_cond = conds[valid].max().item()
                mean_cond = conds[valid].mean().item()
            else:
                max_cond = float('inf')
                mean_cond = float('inf')
            
            norms = torch.norm(self.A_t, dim=(1,2))
            
            return {
                "cond_max": max_cond,
                "cond_mean": mean_cond,
                "norm_mean": norms.mean().item(),
                "fallback_count": self.fallback_count.item(),
                "step": self.step.item()
            }
        except RuntimeError:
            return {
                "error": "Linear algebra error in diagnostics"
            }