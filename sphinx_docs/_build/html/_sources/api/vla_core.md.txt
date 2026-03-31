---
id: vla_core
title: VLA Core API
sidebar_position: 1
---

# Variational Linear Attention Core

The **Variational Linear Attention Core** (VLA Core) is the flagship module of this project. It is located in `src/models/attention/vla_layer.py`. Unlike baseline linear transformers, VLA actively modulates its memory using data-dependent penalty matrices and Sherman-Morrison rank-1 updates.

**Note**: VLA Core is our primary focus. While `DeltaNet` and `LinearTransformer` are included as robust baselines, VLA Core contains our main research innovations.

## Key Modules

### `VLALayer`

The main computational block implementing the Variational Linear Attention mechanism.

```python
import torch
from src.models.attention.vla_layer import VLALayer

# Initialize the layer
vla = VLALayer(
    d_model=256,
    d_head=256,   # Must equal d_model for single-head VLA
    rank=1,       # Rank of the penalty update (1 or r)
    lambda_init=1e-3,
    eps=1e-6      # Numerical stability threshold
)

# Forward pass (Batch training)
# x: (B, T, d_model)
output, stats = vla(x)
```

#### Important Implementation Details:

1.  **Strict Dimension Matching**: `d_head` must strictly equal `d_model` (single-head VLA). This is a mathematical requirement to ensure the dimensions of the penalty matrix ($d \times d$) match the value vector $v_t$.
2.  **Internal Projections**: `VLALayer` defines `W_q`, `W_k`, `W_v` as `nn.Linear(d_model, d_head)` layers internally. It also includes an output projection matrix `W_o` applied after the VLA computation to map the output back to `d_model`.
3.  **Forward Pass Recurrence**: The recurrence order is strictly enforced:
    1.  Update $A_t$ via `InversePenaltyTracker`
    2.  Compute $\alpha_t$ using the *updated* $A_t$ ($\alpha_t = A_t s_t$)
    3.  Update $S_t$ ($S_t = S_{t-1} + \alpha_t \otimes (v_t k_t^\top)$)
    4.  Compute output $o_t$
4.  **State Reset**: VLA state matrices $A_t$ and $S_t$ are *not* shared between layers. They must be explicitly reset to zero/identity for every new sequence to prevent state contamination.

### `InversePenaltyTracker`

Located in `src/models/attention/inverse_penalty.py`. This module handles the stable computation of $M_t^{-1}$ using the Sherman-Morrison formula.

-   **Precision Requirements**: Updates must be computed in `float32` precision. Using `bf16` or `fp16` will lead to catastrophic numerical instability due to accumulating floating-point errors in $A_t$.
-   **Batch Vectorization**: The tracker maintains $A_t$ as a batched tensor `(B, d, d)`. Operations are heavily vectorized over the batch dimension, avoiding Python loops entirely.
-   **Stability Logic**: If the denominator of the Sherman-Morrison update $|\delta| < \text{eps}$, the update is skipped and $\epsilon I$ is added to $A_t$. Every $K$ steps, $\epsilon I$ is injected into the diagonal for periodic stabilization.

### `MemoryMatrixManager`

Located in `src/models/attention/memory_matrix.py`.

-   **State Representation**: The state matrix $S_t$ must be stored as a `float32` buffer of shape `(B, d, d)`.
-   **Renormalization**: Configurable feature (default `False`). It is only triggered when the Frobenius norm of $S_t$ exceeds a predefined safety threshold, preventing overflow.
-   **Out-of-place Updates**: Modifications to $S_t$ are performed out-of-place (e.g., `S = S + update`) to satisfy PyTorch's autograd constraints regarding recursive state dependencies. In-place ops (`+=`, `add_`) are strictly forbidden.

### `VLATransformer`

The high-level wrapper that constructs a full causal language model using VLA layers.

-   **Positional Embeddings**: Includes learnable positional embeddings (`nn.Embedding`) added to token embeddings prior to the transformer blocks.
-   **Architecture**: Adheres strictly to a **Pre-LN** (Pre-LayerNorm) structure: `x -> LN -> VLA -> Residual -> LN -> FFN -> Residual`. The Feed-Forward Network uses GELU activations.
