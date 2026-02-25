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

def sherman_morrison_update(A0: torch.Tensor, u: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Robust Sherman-Morrison rank-1 inverse update.
    Given A0 = inverse(M0) and vector u, compute A = inverse(M0 + u u^T)
    via A = A0 - (A0 u u^T A0) / (1 + u^T A0 u)

    This implementation:
    - uses a safe epsilon fallback when denom is too small
    - preserves symmetry to mitigate small numeric asymmetry
    - supports float32/float64 inputs (keeps dtype/device)
    """
    if not (A0.dim() == 2 and A0.size(0) == A0.size(1)):
        raise AssertionError(f"A0 must be square matrix, got {A0.shape}")
    if not (u.dim() == 1):
        raise AssertionError(f"u must be a vector, got {u.shape}")
    if A0.size(0) != u.size(0):
        raise AssertionError(f"Dimension mismatch: A0={A0.shape}, u={u.shape}")

    # Ensure we use the same dtype/device
    device = A0.device
    dtype = A0.dtype

    # Compute A0 @ u
    A0_u = A0 @ u            # shape (d,)
    u_A0_u = torch.dot(u, A0_u)  # scalar tensor

    # Denominator
    delta = (1.0 + u_A0_u).to(dtype=dtype, device=device)

    # Safety: if |delta| is extremely small, set to epsilon (positive)
    if torch.abs(delta) < epsilon:
        delta = torch.tensor(epsilon, dtype=dtype, device=device)

    # Numerator: z z^T where z = A0 u
    z = A0_u
    # Outer product (d,d)
    O = torch.outer(z, z)

    # Update
    A = A0 - O / delta

    # Symmetrize to remove tiny asymmetric numerical noise
    A = 0.5 * (A + A.transpose(-1, -2))

    # Final safety check
    if not torch.all(torch.isfinite(A)):
        raise RuntimeError("sherman_morrison_update produced non-finite entries")

    return A


def multiple_rank1_updates(A0: torch.Tensor, updates: list[torch.Tensor], epsilon: float = 1e-12) -> torch.Tensor:
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
