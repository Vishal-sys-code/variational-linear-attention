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
        enable_stabilization: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.lambda_0 = lambda_0
        self.stabilization_eps = stabilization_eps
        self.periodic_eps = periodic_eps
        self.period = period
        self.cond_threshold = cond_threshold
        self.enable_stabilization = enable_stabilization
        
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
            
        if u_t.dim() == 3:
            # (B, r, d)
            self._update_rank_r(u_t)
        elif u_t.dim() == 2:
            # (B, d)
            self._update_single(u_t)
        else:
            raise ValueError(f"u_t must be (B, d) or (B, r, d), got {u_t.shape}")
            
    def _update_rank_r(self, u_r: torch.Tensor):
        """
        Internal Woodbury update for rank-r block.
        u_r: (B, r, d)
        """
        B, r, d = u_r.shape
        U = u_r.transpose(1, 2)  # (B, d, r)

        # 1. Compute AU = A_{t-1} U
        AU = torch.bmm(self.A_t, U)  # (B, d, d) @ (B, d, r) -> (B, d, r)

        # 2. Compute C = I_r + U^T AU
        I_r = torch.eye(r, device=u_r.device, dtype=u_r.dtype).unsqueeze(0).expand(B, -1, -1)
        UT_AU = torch.bmm(U.transpose(1, 2), AU)  # (B, r, d) @ (B, d, r) -> (B, r, r)
        C = I_r + UT_AU  # (B, r, r)

        # 3. Solve C X = AU^T for X (where X = C^{-1} AU^T)
        # Using torch.linalg.solve(C, AU^T)
        AUT = AU.transpose(1, 2)  # (B, r, d)

        try:
            # Check determinant for instability (similar to delta < eps in rank-1)
            # If abs(det(C)) < stabilization_eps, we consider it unstable.
            det_C = torch.linalg.det(C)
            mask_unstable = (torch.abs(det_C) < self.stabilization_eps) | ~torch.isfinite(det_C)

            # X shape: (B, r, d)
            X = torch.linalg.solve(C, AUT)

            # Also check for NaNs/Infs in X just in case
            mask_unstable = mask_unstable | (~torch.isfinite(X)).reshape(B, -1).any(dim=1)

            if self.enable_stabilization and mask_unstable.any():
                self.fallback_count += mask_unstable.sum()

                # We need to compute stable updates for stable items in batch
                # and fallback for unstable ones
                fallback_add = self.stabilization_eps * torch.eye(d, device=u_r.device, dtype=u_r.dtype).unsqueeze(0).expand(B, -1, -1)

                # Replace unstable items in X with 0 temporarily to avoid NaN propagation
                safe_X = X.clone()
                safe_X[mask_unstable] = 0.0

                update_term = torch.bmm(AU, safe_X)  # (B, d, r) @ (B, r, d) -> (B, d, d)
                A_stable = self.A_t - update_term
                A_unstable = self.A_t + fallback_add

                mask_expanded = mask_unstable.view(-1, 1, 1).expand_as(self.A_t)
                self.A_t = torch.where(mask_expanded, A_unstable, A_stable)
            else:
                update_term = torch.bmm(AU, X)  # (B, d, r) @ (B, r, d) -> (B, d, d)
                self.A_t = self.A_t - update_term

        except RuntimeError:
            # If solve fails (e.g. singular matrix)
            if self.enable_stabilization:
                self.fallback_count += B
                fallback_add = self.stabilization_eps * torch.eye(d, device=u_r.device, dtype=u_r.dtype).unsqueeze(0).expand(B, -1, -1)
                self.A_t = self.A_t + fallback_add
            else:
                raise

        self.step += r

        # 4. Periodic Stabilization
        if self.enable_stabilization:
            # Check if we crossed a period boundary
            prev_step = self.step.item() - r
            curr_step = self.step.item()
            # This triggers if there is at least one multiple of `period` between prev_step (exclusive) and curr_step (inclusive)
            if (curr_step // self.period) > (prev_step // self.period):
                B = u_r.size(0)
                I = torch.eye(d, device=u_r.device, dtype=u_r.dtype).unsqueeze(0).expand(B, -1, -1)
                self.A_t = self.A_t + (self.periodic_eps * I)

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
        
        if self.enable_stabilization and mask_unstable.any():
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
        if self.enable_stabilization and self.step.item() % self.period == 0:
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