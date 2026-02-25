# src/models/attention/penalty_builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class PenaltyBuilder(nn.Module):
    """
    Builds M_t = lambda_t * I + sum_{m=1..r} u_tm u_tm^T.
    Supports rank–1 and rank–r parameterizations.
    """
    def __init__(
        self,
        d_model: int,
        rank: int = 1,
        hidden_dim: Optional[int] = None,
        lambda_min: float = 1e-4,
        use_separate_projections: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lambda_min = lambda_min
        self.use_separate_projections = use_separate_projections

        h = hidden_dim if hidden_dim is not None else 4 * d_model

        self.lambda_net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

        if use_separate_projections:
            self.u_projections = nn.ModuleList(
                [nn.Linear(d_model, d_model, bias=False) for _ in range(rank)]
            )
        else:
            self.u_proj = nn.Linear(d_model, rank * d_model, bias=False)

    def forward(self, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Args:
            k : (..., d_model)

        Returns:
            lambda_t : (..., 1)
            u_t      : (..., d_model)  if rank=1
                       (..., rank, d_model) if rank>1
            stats    : dict
        """
        lambda_raw = self.lambda_net(k)
        lambda_t = F.softplus(lambda_raw)
        lambda_t = torch.clamp(lambda_t, min=self.lambda_min)

        if self.use_separate_projections:
            proj_list = [layer(k) for layer in self.u_projections]
            if self.rank == 1:
                u_t = proj_list[0]
            else:
                u_t = torch.stack(proj_list, dim=-2)
        else:
            raw_u = self.u_proj(k)
            if self.rank == 1:
                u_t = raw_u
            else:
                out_shape = raw_u.shape[:-1] + (self.rank, self.d_model)
                u_t = raw_u.view(out_shape)

        stats = {
            "lambda_mean": lambda_t.mean().item(),
            "lambda_min": lambda_t.min().item(),
            "lambda_max": lambda_t.max().item(),
            "u_norm_mean": torch.norm(u_t, dim=-1).mean().item(),
            "rank": self.rank,
        }

        return lambda_t, u_t, stats


class KernelPenaltyBuilder(nn.Module):
    """
    M_t ≈ lambda_t * I + phi_t phi_t^T
    Used for kernelized low-rank expansions.
    """
    def __init__(
        self,
        d_model: int,
        rank: int = 1,
        hidden_dim: Optional[int] = None,
        lambda_min: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lambda_min = lambda_min

        h = hidden_dim if hidden_dim is not None else 4 * d_model

        self.lambda_net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

        self.phi_proj = nn.Linear(d_model, rank * d_model, bias=False)

    def forward(self, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        lambda_raw = self.lambda_net(k)
        lambda_t = F.softplus(lambda_raw)
        lambda_t = torch.clamp(lambda_t, min=self.lambda_min)

        phi_raw = self.phi_proj(k)
        if self.rank == 1:
            phi_t = phi_raw
        else:
            out_shape = phi_raw.shape[:-1] + (self.rank, self.d_model)
            phi_t = phi_raw.view(out_shape)

        stats = {
            "lambda_mean": lambda_t.mean().item(),
            "lambda_min": lambda_t.min().item(),
            "lambda_max": lambda_t.max().item(),
            "phi_norm_mean": torch.norm(phi_t, dim=-1).mean().item(),
            "rank": self.rank,
        }

        return lambda_t, phi_t, stats