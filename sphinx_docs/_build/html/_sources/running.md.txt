---
id: running
title: Getting Started
sidebar_position: 6
---

# Getting Started & Reproducibility

This guide provides instructions on how to set up the environment, run core experiments, and utilize the Variational Linear Attention (VLA) API in your own projects. All commands should be executed from the root of the repository.

## Environment Setup

The repository is built strictly with `float32` and `float64` operations in PyTorch for maximum numerical stability. We recommend a dedicated Python environment (e.g., `conda` or `venv`).

```bash
# Clone the repository
git clone https://github.com/deepbrain-labs/variational-linear-attention
cd variational-linear-attention

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `torch >= 2.0.0`
- `numpy`, `scipy`
- `pytest` (for unit tests and strict verification)
- `wandb` (for tracking experiments)

---

## 1. Verifying Core Primitives

Before running large-scale models, we highly recommend verifying that your local environment computes the mathematical primitives with the required precision.

Our math primitive tests use `float64` precision (tight tolerance: $10^{-6}$) to ensure the Sherman-Morrison inversion acts correctly.

```bash
# Run the test suite on CPU
pytest tests/
```

**Note:** PyTorch tests must be executed strictly on the CPU to avoid un-reproducible GPU floating-point non-determinism during standard mathematical checks. The codebase itself fully supports `.to(device)`.

---

## 2. Running Synthetic Memory Tasks

To verify the core hypothesis that VLA outperforms standard Linear Attention on long-context memorization tasks, you can run the synthetic verification scripts.

### The Copy Task
This tests if the model can read a sequence of tokens and identically output them without loss of information.

```bash
python -m tests.verify_vla --task copy
```

### Delayed Recall
This tests if the model can remember a key-value pair after observing a massive number of noisy distractors.

```bash
# 10k context delay
python -m tests.verify_vla --task delayed_recall --delay 10000
```

> **Logs**: Training logs and results for synthetic tasks will automatically be saved to `results/synthetic_copy/` with a timestamped filename.

---

## 3. Running LRA Benchmarks

To reproduce our Long Range Arena (LRA) results, use the dedicated benchmarking scripts. You must first ensure the Hugging Face LRA datasets are downloaded.

```bash
# Setup LRA datasets
# This may take a while depending on your internet connection
python scripts/download_lra.py

# Run VLA on the Image task
python src/benchmarks/run_lra.py --model vla --task image

# Run VLA on the Path-X task (Extreme long context: 16k)
python src/benchmarks/run_lra.py --model vla --task pathx
```

---

## 4. Re-generating Paper Plots

The symbolic and diagnostic plots (e.g., Eigenvalue stability, Penalty Matrix Heatmaps) shown in the **Experiments** section can be natively re-generated.

```bash
# Generates figures into results/symbolic_experiments/
python scripts/generate_plots.py --all
```

---

## Utilizing VLA in Your Project

If you wish to drop VLA into an existing PyTorch codebase, simply import `VLALayer` and replace your standard attention blocks.

```python
import torch
from src.models.attention.vla_layer import VLALayer

# Input dimensions: (Batch, Sequence_Length, d_model)
batch_size, seq_len, d_model = 32, 1024, 256
x = torch.randn(batch_size, seq_len, d_model).cuda()

# Initialize VLA. d_head must equal d_model.
vla = VLALayer(d_model=d_model, d_head=d_model, rank=1).cuda()

# Forward pass (stats contains diagnostic info)
output, stats = vla(x)

assert output.shape == (batch_size, seq_len, d_model)
```
