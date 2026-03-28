import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from src.models.attention.inverse_penalty import InversePenaltyTracker
from src.models.attention.memory_matrix import MemoryMatrixManager
from src.models.attention.penalty_builder import PenaltyBuilder
from src.models.attention.symbolic_penalty import SymbolicPenaltyTracker


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
        gamma: float = 0.0,
        chunk_size: int = 1,  # Not used in token-by-token, but for future compatibility
        fixed_lambda: Optional[float] = None,
        enable_stabilization: bool = True,
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
        
        self.enable_stabilization = enable_stabilization
        
        # W_u is handled inside PenaltyBuilder, which takes k_t as input.
        # PenaltyBuilder input dim is d_head (since k_t is d_head).
        self.penalty_builder = PenaltyBuilder(
            d_model=self.d_head,
            rank=penalty_rank,
            lambda_min=1e-4,
            fixed_lambda=fixed_lambda
        )

        # State Managers
        self.inverse_tracker = InversePenaltyTracker(
            d_model=self.d_head,
            lambda_0=lambda_0,
            enable_stabilization=enable_stabilization
        )
        
        self.memory_manager = MemoryMatrixManager(
            d_model=self.d_head,
            enable_renorm=False  # Default per spec (implied standard behavior unless specified)
        )
        
        self.symbolic_tracker = SymbolicPenaltyTracker(
            d_model=self.d_head,
            gamma=gamma
        )

    def forward(self, x: torch.Tensor, return_states: bool = False, symbolic_adj: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Forward pass for VLA layer.
        
        Args:
            x: Input tensor of shape (B, T, d_model).
            return_states: If True, returns a tuple (O, states) with diagnostic states.
            symbolic_adj: Optional shape (B, T, T) adjacency matrix.
            
        Returns:
            Output tensor of shape (B, T, d_model), optionally with states dict.
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize state
        # A_0 = (1/lambda_0) * I
        self.inverse_tracker.init(batch_size=B, device=device, dtype=torch.float32)
        # S_0 = 0
        self.memory_manager.reset(batch_size=B, device=device, dtype=torch.float32)
        
        # Init symbolic tracker
        self.symbolic_tracker.init_sequence(
            A_rel=symbolic_adj, batch_size=B, max_seq_len=T, device=device, dtype=torch.float32
        )

        outputs = []
        if return_states:
            states = {"A": [], "S_norm": [], "q": [], "k": [], "v": [], "alpha": [], "lambda_t": [], "a_t_scaled": [], "u_norm": [], "alpha_norm": []}

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
            
            # Step 4.4b: Update A_t using symbolic relations if provided
            a_t_scaled = self.symbolic_tracker.step(k_t, t)
            if a_t_scaled is not None:
                # Need to handle case where mask zeroed out batch elements without relations
                # InverseTracker applies Sherman-Morrison for the whole batch. If a_t is 0 for some
                # the update is 0 since outer product is 0.
                self.inverse_tracker.update(a_t_scaled)
                
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
                # Rank > 1: u_t is (B, r, d). We use sum over r for alpha retrieval.
                u_vec = u_t.sum(dim=1).unsqueeze(-1) # (B, d, 1)
                z_t = torch.bmm(A_t, u_vec).squeeze(-1)
                alpha_t = s_t * z_t

            # Step 4.6: Update memory matrix S_t
            self.memory_manager.update(v_t, alpha_t)
            
            # Step 4.6b: Norm explosion stabilization
            if self.enable_stabilization:
                S_t = self.memory_manager.get_S()
                # check norm per-batch element
                S_norm = torch.norm(S_t, p='fro', dim=(1,2))
                A_norm = torch.norm(A_t, p='fro', dim=(1,2))
                
                mask_explode = (S_norm > 1000) | (A_norm > 1000)
                if mask_explode.any():
                    I = torch.eye(self.d_head, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
                    fallback_add = 1e-5 * I
                    mask_expanded = mask_explode.view(-1, 1, 1).expand_as(A_t)
                    A_t = torch.where(mask_expanded, A_t + fallback_add, A_t)
                    self.inverse_tracker.A_t = A_t

            # Step 4.7: Compute output o_t
            o_t = self.memory_manager.compute_output(q_t)  # (B, d_head)
            
            outputs.append(o_t)
            
            if return_states:
                states["A"].append(A_t.clone().detach().cpu())
                S_t = self.memory_manager.get_S()
                states["S_norm"].append(torch.norm(S_t, p='fro', dim=(1,2)).clone().detach().cpu())
                states["q"].append(q_t.clone().detach().cpu())
                states["k"].append(k_t.clone().detach().cpu())
                states["v"].append(v_t.clone().detach().cpu())
                states["alpha"].append(alpha_t.clone().detach().cpu())
                states["lambda_t"].append(lambda_t.clone().detach().cpu())
                if a_t_scaled is not None:
                    states["a_t_scaled"].append(a_t_scaled.clone().detach().cpu())
                else:
                    states["a_t_scaled"].append(torch.zeros_like(k_t).cpu())
                
                # Append norms
                states["u_norm"].append(torch.norm(u_t.view(B, -1), dim=-1).clone().detach().cpu())
                states["alpha_norm"].append(torch.norm(alpha_t, dim=-1).clone().detach().cpu())

        # Step 5: Stack outputs
        O = torch.stack(outputs, dim=1)  # (B, T, d_head)
        
        # Step 6: Output Projection
        O = self.W_o(O)  # (B, T, d_model)
        
        if return_states:
            # Stack states along time dimension T
            for k in states.keys():
                states[k] = torch.stack(states[k], dim=1)  # (B, T, ...)
            return O, states
        return O