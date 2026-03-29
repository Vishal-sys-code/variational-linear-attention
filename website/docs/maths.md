---
id: maths
title: Math Primitives
sidebar_position: 1
---

# Math Primitives

This document outlines the core mathematical primitives implemented in `src/math/primitives.py`. These primitives are designed for numerical stability and are fundamental to the Variational Linear Attention system.

## 1. Inner-Product Score

The score $s_t$ is computed as the dot product of a key vector $k_t$ and a query vector $q_t$:

$$s_t = k_t^T q_t$$

**Implementation Details:**
-   Returns a scalar.
-   Includes an optional scaling factor (e.g., $1/\sqrt{d}$).
-   Validates that the result is finite (no NaNs or Infs).

## 2. Sherman–Morrison Rank-1 Inverse Update

This primitive updates the inverse of a matrix $M$ when $M$ is perturbed by a rank-1 update $u u^T$.

Given:
-   $M_0$: Original matrix (symmetric positive definite).
-   $A_0 = M_0^{-1}$: Inverse of the original matrix.
-   $u$: Update vector.

We compute $A = (M_0 + u u^T)^{-1}$ using the Sherman–Morrison formula:

$$A = A_0 - \frac{A_0 u u^T A_0}{1 + u^T A_0 u}$$

**Algorithm:**
1.  **Compute Denominator:** $\delta = 1 + u^T (A_0 u)$.
2.  **Numerical Safety:** Check if $|\delta| < \epsilon$. If so, apply a fallback (e.g., $\delta \leftarrow \delta + \epsilon$ or clamp to $\epsilon$) to prevent division by zero or numerical instability. Default $\epsilon \approx 10^{-6}$.
3.  **Compute Intermediate Vector:** $z = A_0 u$.
4.  **Compute Outer Product:** $O = z z^T$.
5.  **Update:** $A = A_0 - O / \delta$.
6.  **Validation:** Ensure all entries are finite. Optionally symmetrize $A$.

This update is $O(d^2)$ compared to $O(d^3)$ for full inversion.

## 3. Multiple Rank-1 Updates (Woodbury Generalization)

We support applying a sequence of rank-1 updates $u_1, u_2, \dots, u_r$ by iteratively applying the Sherman–Morrison update:

$$A_i = \text{ShermanMorrison}(A_{i-1}, u_i)$$

This is equivalent to the Woodbury matrix identity for a rank-$r$ update but implemented sequentially for simplicity and stability.

## 4. Memory Matrix Update

The memory matrix $S_t$ is updated with a value vector $v_t$ and key vector $k_t$:

$$S_t = S_{t-1} + \alpha_t v_t k_t^T$$

**Parameters:**
-   $\alpha_t$: Coefficient (scalar or vector). If vector, it scales $v_t$ element-wise (or $k_t$ depending on shape).
-   $v_t$: Value vector.
-   $k_t$: Key vector.

**Constraints:**
-   The update is rank-1.
-   Operations are performed out-of-place to preserve autograd history.

## 5. Recovering $\alpha^*$

We can recover the optimal coefficients $\alpha$ using the inverse covariance matrix $A$ and the score vector $s$:

$$\alpha = A s$$

This solution matches the ground-truth solution $\alpha_{gt} = M^{-1} s$ (where $A = M^{-1}$).
