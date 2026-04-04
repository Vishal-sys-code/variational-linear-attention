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
            states = {"A": [], "S_norm": [], "q": [], "k": [], "v": [], "alpha": [], "lambda_t": [], "a_t_scaled": [], "u_norm": [], "alpha_norm": [], "norm_A_t": [], "norm_S_t": []}

        # Hoist full sequence linear projections
        # This dramatically reduces step-by-step kernel launch overhead!
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Batch construct penalties across the whole sequence T
        Lambda_seq, U_seq, _ = self.penalty_builder(K)

        # Fast inline variables
        I_fallback = torch.eye(self.d_head, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        fallback_add_inv = self.inverse_tracker.stabilization_eps * I_fallback
        periodic_add_inv = self.inverse_tracker.periodic_eps * I_fallback
        fallback_add_explode = 1e-5 * I_fallback
        
        A_t = self.inverse_tracker.A_t
        S_t = self.memory_manager.S_t
        
        inv_step = self.inverse_tracker.step.item()
        mem_step = self.memory_manager.step.item()
        fallback_count = self.inverse_tracker.fallback_count.item()

        # Iterate over tokens
        for t in range(T):
            # Extract current timestep pre-computed vectors
            q_t = Q[:, t, :].to(dtype=torch.float32)  # (B, d_head)
            k_t = K[:, t, :].to(dtype=torch.float32)  # (B, d_head)
            v_t = V[:, t, :].to(dtype=torch.float32)  # (B, d_head)

            # Step 4.2: Compute score s_t
            s_t = (k_t * q_t).sum(dim=-1, keepdim=True)  # (B, 1)

            # Step 4.3: Extract pre-computed penalty components
            lambda_t = Lambda_seq[:, t, :].to(dtype=torch.float32)
            u_t = U_seq[:, t, :].to(dtype=torch.float32)
            
            # Step 4.3b: Scale u_t to prevent large rank-1 updates
            import math
            u_t = u_t / math.sqrt(self.d_head)

            # Step 4.4: Update A_t using u_t
            if u_t.dim() == 2:
                u_vec = u_t.unsqueeze(-1)
                z = torch.bmm(A_t, u_vec)
                dot = torch.bmm(u_vec.transpose(1, 2), z).squeeze(-1).squeeze(-1)
                delta = torch.clamp(1.0 + dot, min=1e-6)
                mask_unstable = (torch.abs(delta) < self.inverse_tracker.stabilization_eps) | ~torch.isfinite(delta)
                
                if self.enable_stabilization and mask_unstable.any():
                    numerator = torch.bmm(z, z.transpose(1, 2))
                    safe_delta = delta.clone()
                    safe_delta[mask_unstable] = 1.0
                    update_term = numerator / safe_delta.view(-1, 1, 1)
                    
                    A_stable = A_t - update_term
                    A_unstable = A_t + fallback_add_inv
                    mask_expanded = mask_unstable.view(-1, 1, 1).expand_as(A_t)
                    A_t = torch.where(mask_expanded, A_unstable, A_stable)
                    fallback_count += mask_unstable.sum().item()
                else:
                    numerator = torch.bmm(z, z.transpose(1, 2))
                    update_term = numerator / delta.view(-1, 1, 1)
                    A_t = A_t - update_term
            else:
                r = u_t.size(1)
                for i in range(r):
                    u_i = u_t[:, i, :]
                    u_vec = u_i.unsqueeze(-1)
                    z = torch.bmm(A_t, u_vec)
                    dot = torch.bmm(u_vec.transpose(1, 2), z).squeeze(-1).squeeze(-1)
                    delta = torch.clamp(1.0 + dot, min=1e-6)
                    mask_unstable = (torch.abs(delta) < self.inverse_tracker.stabilization_eps) | ~torch.isfinite(delta)
                    
                    if self.enable_stabilization and mask_unstable.any():
                        numerator = torch.bmm(z, z.transpose(1, 2))
                        safe_delta = delta.clone()
                        safe_delta[mask_unstable] = 1.0
                        update_term = numerator / safe_delta.view(-1, 1, 1)
                        A_stable = A_t - update_term
                        A_unstable = A_t + fallback_add_inv
                        mask_expanded = mask_unstable.view(-1, 1, 1).expand_as(A_t)
                        A_t = torch.where(mask_expanded, A_unstable, A_stable)
                        fallback_count += mask_unstable.sum().item()
                    else:
                        numerator = torch.bmm(z, z.transpose(1, 2))
                        update_term = numerator / delta.view(-1, 1, 1)
                        A_t = A_t - update_term

            inv_step += 1
            if self.enable_stabilization and inv_step % self.inverse_tracker.period == 0:
                A_t = A_t + periodic_add_inv

            # Step 4.4b: Update A_t using symbolic relations if provided
            a_t_scaled = self.symbolic_tracker.step(k_t, t)
            if a_t_scaled is not None:
                a_t_scaled = a_t_scaled.to(dtype=torch.float32)
                u_vec = a_t_scaled.unsqueeze(-1)
                z = torch.bmm(A_t, u_vec)
                dot = torch.bmm(u_vec.transpose(1, 2), z).squeeze(-1).squeeze(-1)
                delta = torch.clamp(1.0 + dot, min=1e-6)
                mask_unstable = (torch.abs(delta) < self.inverse_tracker.stabilization_eps) | ~torch.isfinite(delta)
                
                if self.enable_stabilization and mask_unstable.any():
                    numerator = torch.bmm(z, z.transpose(1, 2))
                    safe_delta = delta.clone()
                    safe_delta[mask_unstable] = 1.0
                    update_term = numerator / safe_delta.view(-1, 1, 1)
                    A_stable = A_t - update_term
                    A_unstable = A_t + fallback_add_inv
                    mask_expanded = mask_unstable.view(-1, 1, 1).expand_as(A_t)
                    A_t = torch.where(mask_expanded, A_unstable, A_stable)
                    fallback_count += mask_unstable.sum().item()
                else:
                    numerator = torch.bmm(z, z.transpose(1, 2))
                    update_term = numerator / delta.view(-1, 1, 1)
                    A_t = A_t - update_term

                inv_step += 1
                if self.enable_stabilization and inv_step % self.inverse_tracker.period == 0:
                    A_t = A_t + periodic_add_inv

            # Step 4.5: Compute alpha_t
            if u_t.dim() == 2:
                u_vec = u_t.unsqueeze(-1)
                z_t = torch.bmm(A_t, u_vec).squeeze(-1)
                alpha_t = s_t * z_t
            else:
                u_vec = u_t.sum(dim=1).unsqueeze(-1)
                z_t = torch.bmm(A_t, u_vec).squeeze(-1)
                alpha_t = s_t * z_t

            # Step 4.6: Update memory matrix S_t
            v_t_f32 = v_t / (torch.norm(v_t, dim=-1, keepdim=True) + 1e-6)
            
            update_term_S = torch.matmul(v_t_f32.unsqueeze(2), alpha_t.unsqueeze(1))
            S_t = S_t + update_term_S
            mem_step += 1
            
            if self.memory_manager.enable_renorm:
                S_norms = torch.norm(S_t, p='fro', dim=(1, 2))
                mask_renorm = S_norms > self.memory_manager.renorm_threshold
                if mask_renorm.any():
                    divisor = torch.ones_like(S_norms)
                    divisor[mask_renorm] = S_norms[mask_renorm]
                    S_t = S_t / divisor.view(-1, 1, 1)
            
            # Step 4.6b: Norm explosion stabilization
            if self.enable_stabilization:
                S_norm = torch.norm(S_t, p='fro', dim=(1,2))
                A_norm = torch.norm(A_t, p='fro', dim=(1,2))
                
                mask_explode = (S_norm > 1000) | (A_norm > 1000)
                if mask_explode.any():
                    mask_expanded = mask_explode.view(-1, 1, 1).expand_as(A_t)
                    A_t = torch.where(mask_expanded, A_t + fallback_add_explode, A_t)

            # Step 4.7: Compute output o_t
            o_t = torch.matmul(S_t, q_t.unsqueeze(2)).squeeze(2)
            
            outputs.append(o_t)
            
            if return_states:
                states["A"].append(A_t.clone().detach().cpu())
                S_t_norm = torch.norm(S_t, p='fro', dim=(1,2)).clone().detach().cpu()
                states["S_norm"].append(S_t_norm)
                states["norm_S_t"].append(S_t_norm)
                states["norm_A_t"].append(torch.norm(A_t, p='fro', dim=(1,2)).clone().detach().cpu())
                states["q"].append(q_t.clone().detach().cpu())
                states["k"].append(k_t.clone().detach().cpu())
                states["v"].append(v_t.clone().detach().cpu())
                states["alpha"].append(alpha_t.clone().detach().cpu())
                states["lambda_t"].append(lambda_t.clone().detach().cpu())
                if a_t_scaled is not None:
                    states["a_t_scaled"].append(a_t_scaled.clone().detach().cpu())
                else:
                    states["a_t_scaled"].append(torch.zeros_like(k_t).cpu())
                states["u_norm"].append(torch.norm(u_t.view(B, -1), dim=-1).clone().detach().cpu())
                states["alpha_norm"].append(torch.norm(alpha_t, dim=-1).clone().detach().cpu())

        # Save states back
        self.inverse_tracker.A_t = A_t
        self.inverse_tracker.step = torch.tensor(inv_step, device=device, dtype=torch.long)
        self.inverse_tracker.fallback_count = torch.tensor(fallback_count, device=device, dtype=torch.long)
        self.memory_manager.S_t = S_t
        self.memory_manager.step = torch.tensor(mem_step, device=device, dtype=torch.long)

        # Step 5: Stack outputs
        O = torch.stack(outputs, dim=1).to(dtype=dtype)  # (B, T, d_head)
        
        # Step 6: Output Projection
        O = self.W_o(O)  # (B, T, d_model)
        
        if return_states:
            # Stack states along time dimension T
            for k in states.keys():
                states[k] = torch.stack(states[k], dim=1)  # (B, T, ...)
            return O, states
        return O