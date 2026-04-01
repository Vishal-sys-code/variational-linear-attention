---
id: deltanet
title: DeltaNet Baseline
sidebar_position: 2
---

# DeltaNet API

The **DeltaNet** model is implemented as a competitive baseline to Variational Linear Attention. While VLA uses a sophisticated probabilistic penalty framework and exact inverse updates, DeltaNet relies on a simpler, continuous-time inspired linear recurrence.

## Overview

DeltaNet is a variant of linear attention that uses a matrix-valued hidden state updated via a learned gating mechanism. It effectively performs a moving average of value-key outer products, modulated by a data-dependent scalar gate.

### `DeltaNetLayer`

Located in `src/models/attention/deltanet.py`.

```python
import torch
from src.models.attention.deltanet import DeltaNetLayer

# Initialize a DeltaNet baseline layer
deltanet = DeltaNetLayer(
    d_model=256,
    num_heads=4,
    gate_init=-4.0
)

# Forward pass
output, stats = deltanet(x)
```

#### Key Differences from VLA:
- **Vector-Based Recurrence**: DeltaNet's output reconstruction strictly follows a vector-based recurrence (linear operator) and cannot be mathematically simplified to scalar survival products.
- **No Matrix Inversion**: Unlike VLA, which computes $M_t^{-1}$ using Sherman-Morrison, DeltaNet avoids the $O(d^2)$ inverse update entirely. It instead relies on the learned gating parameter to approximate forgetting.
- **Multi-Head Support**: DeltaNet traditionally supports multiple attention heads (`d_head = d_model / num_heads`), whereas our VLA formulation is inherently single-head (`d_head = d_model`) due to the exact covariance tracking requirement.

For detailed theoretical comparisons against VLA, see the [Experiments](../experiments.md) section.