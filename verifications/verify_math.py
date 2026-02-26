import torch
import sys
import os

# Robust path setup to ensure src is importable
# This must be done BEFORE importing from src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.math.primitives import sherman_morrison_update

# --- Configuration ---
d = 4                 # Matrix dimension (small for easy viewing)
dtype = torch.float64 # Use double precision for accuracy check
torch.manual_seed(42) # For reproducibility

# --- Setup ---
print(f"--- Sherman-Morrison Verification (d={d}, dtype={dtype}) ---")

# 1. Generate a random Symmetric Positive Definite (SPD) matrix M0
print("1. Generating random SPD matrix M0...")
M0 = torch.randn(d, d, dtype=dtype)
M0 = M0 @ M0.T + 1e-3 * torch.eye(d, dtype=dtype) # Make SPD
A0 = torch.inverse(M0) # Compute initial inverse A0 = M0^-1

# 2. Generate a random update vector u
print("2. Generating random update vector u...")
u = torch.randn(d, dtype=dtype)

# --- Computation ---

# 3. Method A: Our Sherman-Morrison Update (Fast, O(d^2))
print("3. Computing Fast Inverse Update (Sherman-Morrison)...")
A_fast = sherman_morrison_update(A0, u)

# 4. Method B: Direct Inversion (Slow/Exact, O(d^3))
print("4. Computing Slow/Exact Inverse (Direct M + uu^T)...")
M_new = M0 + torch.outer(u, u)
A_exact = torch.inverse(M_new)

# --- Verification ---

# 5. Compute the difference
diff = A_fast - A_exact
error_norm = torch.norm(diff)
exact_norm = torch.norm(A_exact)
relative_error = error_norm / exact_norm

print("\n--- Results ---")
print("Fast Inverse (Top-left 2x2 block):")
print(A_fast[:2, :2])
print("\nExact Inverse (Top-left 2x2 block):")
print(A_exact[:2, :2])

print(f"\nMax Absolute Difference: {torch.max(torch.abs(diff)):.2e}")
print(f"Relative Error Norm:     {relative_error:.2e}")

# 6. Conclusion
TOLERANCE = 1e-6
if relative_error < TOLERANCE:
    print(f"\n SUCCESS! The results match within tolerance ({TOLERANCE}).")
    print("   This confirms your Sherman-Morrison implementation is correct.")
else:
    print(f"\n FAILURE! The error ({relative_error:.2e}) is too high.")