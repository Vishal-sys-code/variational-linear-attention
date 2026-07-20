---
id: maths
title: Mathematics & Primitives
sidebar_position: 2
---

# Mathematical Primitives

This section details the formal mathematical primitives underpinning the Variational Linear Attention (VLA) architecture, implemented natively in `src/maths/primitives.py`. These operations are formulated to guarantee absolute numerical stability over infinite temporal horizons, forming the computational bedrock of the VLA recurrence formulation.

## 1. Scaled Inner-Product Scoring

At an arbitrary timestep $t$, the localized attention scalar $s_t$ is extracted via the normalized dot product of a projected key vector $k_t \in \mathbb{R}^d$ and a query vector $q_t \in \mathbb{R}^d$:

$$
s_t = \frac{k_t^\top q_t}{\sqrt{d}}
$$

**Structural Constraints:**
- **Asymptotic Gradient Stability**: The $1/\sqrt{d}$ denominator mirrors the standard scaled dot-product attention scaling, mathematically bounding variance to prevent vanishing or exploding gradient artifacts during propagation.
- **Finite State Enforcement**: Strict `NaN`/`Inf` tensor bounds are aggressively enforced prior to evaluation, acting as a crucial safety barrier against matrix singularity in subsequent recurrent steps.

---

## 2. Sherman–Morrison Rank-1 Inverse Tracking

The foundational bottleneck in solving dynamically penalized linear systems temporally is the continuous matrix inversion. Rather than recalculating the exact inverse of the dense penalty matrix $M_t$ in $\mathcal{O}(d^3)$ time, VLA exploits the **Sherman-Morrison** formulation to surgically apply a rank-1 perturbation update ($u_t u_t^\top$), thereby recovering the inverse strictly in $\mathcal{O}(d^2)$ time.

### Theoretical Derivation

Let:
- $M_{t-1} \in \mathbb{R}^{d \times d}$: The symmetric positive definite penalty matrix at step $t-1$.
- $A_{t-1} = M_{t-1}^{-1}$: The exact tracked inverse of the prior penalty matrix.
- $u_t \in \mathbb{R}^d$: The non-linear directional update vector projected by the `PenaltyBuilder`.

The perturbed forward penalty naturally constructs as $M_t = M_{t-1} + u_t u_t^\top$. Leveraging the Sherman-Morrison identity, we recursively maintain the inverse $A_t$ without structural decomposition:

$$
A_t = A_{t-1} - \frac{A_{t-1} u_t u_t^\top A_{t-1}}{1 + u_t^\top A_{t-1} u_t}
$$

### Algorithmic Execution Pipeline

1. **Denominator Evaluation (Scalar $\delta$):**
$$
\delta = 1 + u_t^\top \left(A_{t-1} u_t\right)
$$
2. **Singularity Bound Guard:** If $|\delta| < \epsilon$, a robust fallback projection is instantly triggered (injecting $\epsilon I$), preventing catastrophic division-by-zero artifacts. (Default topological bound: $\epsilon = 10^{-6}$).
3. **Linear Intermediate Projection ($z \in \mathbb{R}^d$):**
$$
z = A_{t-1} u_t
$$
4. **Outer-Product Formulation ($O \in \mathbb{R}^{d \times d}$):**
$$
O = z z^\top
$$
5. **Exact Inverse Update:**
$$
A_t = A_{t-1} - \frac{O}{\delta}
$$
6. **Periodic Spectrum Stabilization:** To counteract accumulated 16-bit floating-point drift over exceptionally long contexts ($T > 10^5$), an $\epsilon I$ diagonal nudging term is periodically applied to the trace of $A_t$.

---

## 3. The Woodbury Matrix Generalization

To natively support multi-head or higher-rank contextual parameterizations, the rank-1 update logic must be generalized for rank-$r$ perturbations. While the Woodbury matrix identity mathematically encapsulates this directly, VLA computationally processes this as a sequence of iterative rank-1 Sherman-Morrison updates $\{u_1, u_2, \dots, u_r\}$. This sequential execution completely avoids intermediate matrix memory spikes and maintains tight $\mathcal{O}(d^2)$ execution constraints.

$$
\begin{aligned}
A^{(0)} &= A_{t-1} \\
A^{(i)} &= \text{ShermanMorrison}\left(A^{(i-1)}, \ u_i\right) \quad \text{for } i \in \{1, \dots, r\} \\
A_t &= A^{(r)}
\end{aligned}
$$

---

## 4. Analytical Optimal Coefficient Recovery ($\alpha^*$)

Standard Linear Attention recursively aggregates the global memory state $S_t$ utilizing a rigid $v_t k_t^\top$ outer-product. VLA, in stark theoretical contrast, computes a dynamically optimal scaling vector $\alpha_t$ that explicitly minimizes the associative reconstruction error for the active value vector $v_t$, bounded exclusively by the evolving inverse penalty $M_t$.

Because VLA inherently computes and tracks the exact inverse $A_t = M_t^{-1}$, deriving the theoretical ground-truth optimum $\alpha^* = M_t^{-1} s_t$ collapses to a highly optimized matrix-vector hardware operation:

$$
\alpha_t = A_t s_t
$$

---

## 5. Recurrent State Modulations

Having securely calculated the exact localized coefficient vector $\alpha_t$, the global state $S_t$ is linearly perturbed to incorporate new sequence signals:

$$
S_t = S_{t-1} + \alpha_t \otimes \left(v_t k_t^\top\right)
$$

**Hardware & Implementation Paradigms:**
- **Batched Tensor Operations:** Executed aggressively via parallelized outer products (e.g., `v.unsqueeze(2) * alpha.unsqueeze(1)`), avoiding explicit iteration loops and maximizing CUDA core saturation.
- **Graph Safety Limits**: Updates operate strictly **out-of-place** ($S_t = S_{t-1} + \Delta$) to prevent backward pass graph corruption inherent in direct memory overwrites (e.g., `+=`).
- **Dynamic Renormalization**: If the Frobenius norm $\|S_t\|_F$ exceeds maximum tolerances during ultra-long rollout operations, $S_t$ is transparently rescaled, ensuring complete stability.