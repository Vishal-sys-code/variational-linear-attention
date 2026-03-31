---
id: linear_transformer
title: Linear Transformer Baseline
sidebar_position: 3
---

# Linear Transformer API

The **Linear Transformer** (LT) API serves as our foundational baseline. It implements the standard linear attention mechanism originally proposed by Katharopoulos et al. (2020), which achieves $O(T)$ complexity by decomposing the attention score computation via feature maps.

## Overview

In LT, the softmax operation $\text{softmax}(Q K^\top)$ is replaced with a feature map $\phi(\cdot)$. The sequence is processed by accumulating two running sums: a memory state $S_t = S_{t-1} + \phi(k_t) v_t^\top$ and a normalization state $z_t = z_{t-1} + \phi(k_t)$. The output is computed as $o_t = \frac{\phi(q_t) S_t}{\phi(q_t) z_t}$.

### `LinearAttentionLayer`

Located in `src/models/attention/linear_attention.py`.

```python
import torch
from src.models.attention.linear_attention import LinearAttentionLayer

# Initialize a standard Linear Transformer layer
lt_layer = LinearAttentionLayer(
    d_model=256,
    num_heads=4,
    feature_map='elu'  # e.g., ELU(x) + 1
)

# Forward pass
output, stats = lt_layer(x)
```

#### Key Limitations Compared to VLA

1.  **Uniform Forgetting**: Linear Transformers fundamentally lack a mechanism to unlearn or dynamically forget past tokens. $S_t$ strictly accumulates. This leads to "attention dilution" on long sequences (like the Delayed Recall task), where the magnitude of irrelevant tokens overwhelms the relevant ones.
2.  **No Adaptive Penalty**: VLA solves the exact optimization problem of minimizing reconstruction error under a dynamically learned penalty matrix $M_t$. LT merely computes a fixed running sum.
3.  **Performance Degradation**: As shown in our LRA benchmarks and Synthetic tasks, LT struggles significantly when sequence lengths exceed $10^3$ tokens, while VLA seamlessly scales.

### Feature Maps

The LT API supports various kernel approximations for $\phi(\cdot)$, primarily focusing on $\text{ELU}(x) + 1$ to ensure non-negativity without the computational cost of the RBF kernel approximation.