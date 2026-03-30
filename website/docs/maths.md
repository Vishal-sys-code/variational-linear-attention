---
id: maths
title: Mathematics & Primitives
sidebar_position: 2
---

# Math Primitives

This document outlines the core mathematical primitives implemented in `src/maths/primitives.py`. These primitives are rigorously designed for numerical stability and form the fundamental backbone of the Variational Linear Attention (VLA) system.

## 1. Inner-Product Score

The attention score $s_t$ at timestep $t$ is computed as the scaled dot product of a key vector $k_t \in \mathbb{R}^d$ and a query vector $q_t \in \mathbb{R}^d$:

$$
s_t = \frac{k_t^\top q_t}{\sqrt{d}}
$$

**Implementation Details:**
- **Returns**: A scalar value indicating query-key relevance.
- **Scaling**: Includes the standard $1/\sqrt{d}$ scaling factor to mathematically constrain gradients from vanishing or exploding, completely analogous to standard scaled dot-product attention.
- **Safety Measures**: Validates that all values are strictly finite (void of `NaN` or `Inf`), which is crucially required before undertaking subsequent inverse operations.

---

## 2. Sherman–Morrison Rank-1 Inverse Update

The core bottleneck of stably solving a linear system at every timestep is the matrix inversion. The Sherman-Morrison formula empowers VLA to surgically update the inverse of the penalty matrix $M_t$ when it is mathematically perturbed by a rank-1 update $u_t u_t^\top$.

### Theoretical Foundation
Given:
- $M_{t-1} \in \mathbb{R}^{d \times d}$: The prior penalty matrix (symmetric positive definite).
- $A_{t-1} = M_{t-1}^{-1}$: The exact inverse of the prior penalty matrix.
- $u_t \in \mathbb{R}^d$: The new update vector proposed by the `PenaltyBuilder`.

The updated penalty matrix naturally takes the form $M_t = M_{t-1} + u_t u_t^\top$. Computing $(M_t)^{-1}$ directly scales at $\mathcal{O}(d^3)$. However, by leveraging the Sherman-Morrison identify, we reconstruct the inverse update securely in $\mathcal{O}(d^2)$:

$$
A_t = A_{t-1} - \frac{A_{t-1} u_t u_t^\top A_{t-1}}{1 + u_t^\top A_{t-1} u_t}
$$

### Algorithmic Execution Steps
1. **Compute Denominator (Scalar):** 
$$
\delta = 1 + u_t^\top \left(A_{t-1} u_t\right)
$$
2. **Numerical Safety Enforcement:** If the absolute value $|\delta| < \epsilon$, a strict fallback is triggered (adding $\epsilon I$) stopping division by zero or gradient detonation. (Default $\epsilon \approx 10^{-6}$).
3. **Compute Intermediate Vector ($z \in \mathbb{R}^d$):** 
$$
z = A_{t-1} u_t
$$
4. **Compute Outer Product ($O \in \mathbb{R}^{d \times d}$):** 
$$
O = z z^\top
$$
5. **Final Rank-1 Update:** 
$$
A_t = A_{t-1} - \frac{O}{\delta}
$$
6. **Periodic Stabilization:** Every $K$ steps, add $\epsilon I$ onto the diagonal tensor of $A_t$ to correct accumulating floating-point drift. This represents a critical fix for extremely long context sequence regimes.

---

## 3. Multiple Rank-1 Updates (Woodbury Generalization)

To securely support higher-rank context parameterizations, we sequence iterating rank-1 updates $\{u_1, u_2, \dots, u_r\}$. This explicitly mirrors the mathematical equivalence of the Woodbury matrix identity for a rank-$r$ update—but iterating sequentially minimizes dangerous intermediate memory spikes.

$$
\begin{aligned}
A^{(0)} &= A_{t-1} \\
A^{(i)} &= \text{ShermanMorrison}\left(A^{(i-1)}, \ u_i\right) \quad \text{for } i \in \{1, \dots, r\} \\
A_t &= A^{(r)}
\end{aligned}
$$

---

## 4. Recovering Optimal Coefficients $\alpha^*$

In standard Linear Attention, the global memory matrix $S_t$ is linearly updated using static $v_t k_t^\top$. In sharp contrast, VLA computes a globally optimal scaling vector $\alpha_t$ that completely minimizes the associative reconstruction error of the value vector $v_t$ bounded strictly by the active penalty $M_t$.

As VLA naturally tracks the inverted formulation $A_t = M_t^{-1}$, generating the theoretical ground-truth optimum $\alpha^* = M_t^{-1} s_t$ simplifies exponentially into a single matrix-vector hardware product:

$$
\alpha_t = A_t s_t
$$

---

## 5. Memory Matrix Update

Having securely arrived at the optimal coefficient map $\alpha_t$, the global memory matrix $S_t$ updates to ingest new contextual information:

$$
S_t = S_{t-1} + \alpha_t \otimes \left(v_t k_t^\top\right)
$$

**Implementation Details:**
- **Batched Outer Products:** Evaluated safely using `v.unsqueeze(2) * alpha.unsqueeze(1)` ensuring fast batched hardware execution without explicit loops.
- **In-place Constraints**: The update strictly operates **out-of-place** ($S_t = S_{t-1} + \Delta$) guarding the PyTorch computational graph graph. (Direct memory overwrites like `+=` are banned.)
- **Renormalization Guard**: In regimes where the Frobenius norm $\|S_t\|_F$ balloons over threshold, $S_t$ rescales safely avoiding numeric overflow during ultra-long runtimes.