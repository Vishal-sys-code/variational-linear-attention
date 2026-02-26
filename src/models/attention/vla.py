import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from src.models.attention.inverse_penalty import InversePenaltyTracker
from src.models.attention.memory_matrix import MemoryMatrixManager
from src.models.attention.penalty_builder import PenaltyBuilder


class VLALayer(nn.Module):
    """
    Single-Layer Variational Linear Attention (VLA) module.
    Processes sequences token-by-token in a streaming fashion.
    """
    def __init__(
        self,
        d_model: int,
        d_head: Optional[int] = None,
        lambda_0: float = 1.0,
        penalty_rank: int = 1,
        chunk_size: int = 1,  # Not used in token-by-token, but for future compatibility
    ):
        super().__init__()
        self.d_model = d_model
        # Enforce d_head = d_model as per requirements unless explicitly varied,
        # but spec says "Set d_head = d_model Across all layers".
        self.d_head = d_head if d_head is not None else d_model
        
        self.lambda_0 = lambda_0
        self.penalty_rank = penalty_rank

        # Projections
        self.W_q = nn.Linear(d_model, self.d_head)
        self.W_k = nn.Linear(d_model, self.d_head)
        self.W_v = nn.Linear(d_model, self.d_head)
        
        # Output Projection W_o: (d_head -> d_model)
        self.W_o = nn.Linear(self.d_head, self.d_model)
        
        # W_u is handled inside PenaltyBuilder, which takes k_t as input.
        # PenaltyBuilder input dim is d_head (since k_t is d_head).
        self.penalty_builder = PenaltyBuilder(
            d_model=self.d_head,
            rank=penalty_rank,
            lambda_min=1e-4
        )

        # State Managers
        self.inverse_tracker = InversePenaltyTracker(
            d_model=self.d_head,
            lambda_0=lambda_0
        )
        
        self.memory_manager = MemoryMatrixManager(
            d_model=self.d_head,
            enable_renorm=False  # Default per spec (implied standard behavior unless specified)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VLA layer.
        
        Args:
            x: Input tensor of shape (B, T, d_model).
            
        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize state
        # A_0 = (1/lambda_0) * I
        self.inverse_tracker.init(batch_size=B, device=device, dtype=torch.float32)
        # S_0 = 0
        self.memory_manager.reset(batch_size=B, device=device, dtype=torch.float32)

        outputs = []

        # Iterate over tokens
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_model)

            # Step 4.1: Project to q, k, v
            q_t = self.W_q(x_t)  # (B, d_head)
            k_t = self.W_k(x_t)  # (B, d_head)
            v_t = self.W_v(x_t)  # (B, d_head)

            # Step 4.2: Compute score s_t
            # s_t = dot(k_t, q_t)
            # (B, d_head) * (B, d_head) -> sum -> (B, 1)
            s_t = (k_t * q_t).sum(dim=-1, keepdim=True)  # (B, 1)

            # Step 4.3: Build penalty components
            # lambda_t: (B, 1), u_t: (B, d_head) (if rank=1)
            lambda_t, u_t, _ = self.penalty_builder(k_t)

            # Step 4.4: Update A_t using u_t
            # We use the existing tracker which updates A_t internally.
            self.inverse_tracker.update(u_t)
            A_t = self.inverse_tracker.get()  # (B, d_head, d_head)

            # Step 4.5: Compute alpha_t
            # alpha_t = s_t * (A_t * u_t)
            # u_t is (B, d). If rank > 1, u_t is (B, r, d).
            # Spec implies vector u_t. If rank > 1, we might need adjustments.
            # Assuming rank=1 for standard VLA.
            
            # A_t @ u_t: (B, d, d) @ (B, d, 1) -> (B, d, 1)
            if u_t.dim() == 2:
                u_vec = u_t.unsqueeze(-1)
                z_t = torch.bmm(A_t, u_vec).squeeze(-1)  # (B, d_head)
                alpha_t = s_t * z_t  # (B, 1) * (B, d_head) -> (B, d_head)
            else:
                # Fallback for rank > 1 if ever needed, though spec says "vector".
                # If rank > 1, u_t is (B, r, d).
                # This path is likely not hit if rank=1.
                raise NotImplementedError("Rank > 1 not fully specified for alpha calculation in this task.")

            # Step 4.6: Update memory matrix S_t
            self.memory_manager.update(v_t, alpha_t)

            # Step 4.7: Compute output o_t
            o_t = self.memory_manager.compute_output(q_t)  # (B, d_head)
            
            outputs.append(o_t)

        # Step 5: Stack outputs
        O = torch.stack(outputs, dim=1)  # (B, T, d_head)
        
        # Step 6: Output Projection
        O = self.W_o(O)  # (B, T, d_model)
        
        return O