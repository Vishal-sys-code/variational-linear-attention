import torch

def inner_product_score(k: torch.Tensor, q: torch.Tensor, scale: float = None) -> torch.Tensor:
    """
    Computes the inner-product score s_t = k^T q.
    
    Args:
        k: Key vector of shape (d,).
        q: Query vector of shape (d,).
        scale: Optional scaling factor (e.g. 1/sqrt(d)).
        
    Returns:
        Scalar score s_t.
    """
    assert k.dim() == 1, f"Expected 1D vector k, got {k.dim()}D"
    assert q.dim() == 1, f"Expected 1D vector q, got {q.dim()}D"
    assert k.shape == q.shape, f"Shape mismatch: k={k.shape}, q={q.shape}"
    
    score = torch.dot(k, q)
    
    if scale is not None:
        score = score * scale
        
    assert torch.isfinite(score), f"Score is not finite: {score}"
    
    return score

def sherman_morrison_update(A0: torch.Tensor, u: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Computes the Sherman-Morrison rank-1 inverse update.
    
    Given A0 = inverse(M0) and u, computes A = inverse(M0 + u u^T).
    Formula: A = A0 - (A0 u u^T A0) / (1 + u^T A0 u)
    
    Args:
        A0: Current inverse matrix of shape (d, d).
        u: Update vector of shape (d,).
        epsilon: Numerical stability threshold.
        
    Returns:
        Updated inverse matrix A.
    """
    assert A0.dim() == 2 and A0.size(0) == A0.size(1), f"A0 must be square matrix, got {A0.shape}"
    assert u.dim() == 1, f"u must be a vector, got {u.shape}"
    assert A0.size(0) == u.size(0), f"Dimension mismatch: A0={A0.shape}, u={u.shape}"
    
    # 2.1 Compute denominator (scalar): δ = 1 + u^T (A0 u)
    A0_u = A0 @ u  # Shape (d,)
    u_A0_u = torch.dot(u, A0_u) # Scalar
    delta = 1.0 + u_A0_u
    
    # 2.2 Numerical safety
    # "if abs(δ) < ε: δ = δ + ε # or fallback to δ = ε"
    if torch.abs(delta) < epsilon:
        delta = delta + epsilon
        # If still too small (e.g. delta was -epsilon), force it to epsilon?
        # The prompt implies a simple shift. Let's trust the shift prevents 0 division.
        # But if delta was -epsilon/2, delta+epsilon is epsilon/2.
        # If delta was exactly -epsilon, delta+epsilon is 0. This is risky.
        # "fallback to delta = epsilon" might be safer if result is still small.
        if torch.abs(delta) < epsilon:
             delta = torch.tensor(epsilon, dtype=delta.dtype, device=delta.device) * (torch.sign(delta) if delta != 0 else 1.0)

    # 2.3 Compute intermediate vector: z = A0 u
    z = A0_u
    
    # 2.4 Compute outer product: O = z z^T
    O = torch.outer(z, z)
    
    # 2.5 Final update: A = A0 - O / δ
    A = A0 - O / delta
    
    # 2.6 Validate
    assert torch.all(torch.isfinite(A)), "Resulting matrix A contains non-finite values"
    
    # Optionally symmetrize
    A = 0.5 * (A + A.T)
    
    return A

def multiple_rank1_updates(A0: torch.Tensor, updates: list[torch.Tensor], epsilon: float = 1e-6) -> torch.Tensor:
    """
    Applies multiple Sherman-Morrison updates sequentially.
    
    Args:
        A0: Initial inverse matrix.
        updates: List of update vectors [u1, u2, ...].
        epsilon: Numerical stability threshold.
        
    Returns:
        Final updated inverse matrix.
    """
    A = A0
    for u in updates:
        A = sherman_morrison_update(A, u, epsilon=epsilon)
    return A

def memory_update(S: torch.Tensor, alpha: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Updates the memory matrix S_t = S_{t-1} + alpha v k^T.
    
    Args:
        S: Previous memory matrix (d_v, d_k).
        alpha: Coefficient (scalar or vector).
        v: Value vector (d_v,).
        k: Key vector (d_k,).
        
    Returns:
        Updated memory matrix S_new.
    """
    assert S.dim() == 2, f"S must be a matrix, got {S.shape}"
    d_v, d_k = S.shape
    
    assert v.dim() == 1 and v.size(0) == d_v, f"v shape mismatch: expected ({d_v},), got {v.shape}"
    assert k.dim() == 1 and k.size(0) == d_k, f"k shape mismatch: expected ({d_k},), got {k.shape}"
    
    # Handle alpha scaling
    # We want to compute scaled_update = alpha * (v k^T)
    
    # Case 1: alpha is scalar
    if isinstance(alpha, (float, int)) or (torch.is_tensor(alpha) and alpha.numel() == 1):
        # alpha is scalar
        # Compute update
        update = torch.outer(v, k)
        scaled_update = alpha * update

    # Case 2: alpha is vector
    elif torch.is_tensor(alpha):
        if alpha.shape == v.shape:
             # Assume elementwise scaling of v: (alpha * v) k^T
             scaled_v = alpha * v
             scaled_update = torch.outer(scaled_v, k)
        elif alpha.shape == k.shape:
             # Assume elementwise scaling of k: v (alpha * k)^T
             scaled_k = alpha * k
             scaled_update = torch.outer(v, scaled_k)
        else:
            raise ValueError(f"alpha shape {alpha.shape} incompatible with v {v.shape} or k {k.shape}")
    else:
        raise TypeError(f"alpha must be float or Tensor, got {type(alpha)}")

    # Ensure no in-place modifications
    S_new = S + scaled_update
    
    return S_new

def recover_alpha(A_inv: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Recovers alpha = A s.
    
    Args:
        A_inv: Inverse covariance matrix A (d, d).
        s: Score vector (d,).
        
    Returns:
        alpha vector (d,).
    """
    assert A_inv.dim() == 2, f"A_inv must be 2D, got {A_inv.shape}"
    assert s.dim() == 1, f"s must be 1D, got {s.shape}"
    assert A_inv.size(1) == s.size(0), f"Dimension mismatch: A_inv={A_inv.shape}, s={s.shape}"
    
    alpha = A_inv @ s
    return alpha
