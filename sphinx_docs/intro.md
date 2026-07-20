---
id: intro
title: Introduction
sidebar_position: 1
---

# Variational Linear Attention (VLA)

Welcome to the official research documentation for **Variational Linear Attention (VLA)**, a next-generation sequence modeling architecture developed by DeepBrain Labs. This work introduces a fundamental paradigm shift in attention mechanisms, rigorously designed to bridge the chasm between the expressive capacity of unbounded Softmax Attention and the theoretical $\mathcal{O}(N)$ efficiency of Linear Attention variants.

## The Bottleneck in Sequence Modeling

Standard Transformer architectures fundamentally rely on Softmax Attention, which scales quadratically with respect to sequence length ($\mathcal{O}(T^2)$). This constraint imposes severe computational and memory bounds, rendering them prohibitively expensive for frontier-scale long-context applications.

While subsequent Linear Attention models mathematically approximate the kernel to achieve asymptotic linear complexity ($\mathcal{O}(T)$), they universally suffer from **attention dilution** and catastrophic forgetting. In practice, they severely underperform on targeted long-term memory recall tasks, such as high-density Associative Recall or Delayed Copy tasks, as they lack the structural inductive bias to dynamically prioritize or selectively forget information over extended temporal horizons.

## The VLA Formulation

**Variational Linear Attention (VLA)** reformulates the linear attention mechanism entirely through the lens of a **probabilistic graphical model**. By introducing a mathematically optimal, time-varying, data-dependent penalty tensor ($M_t$), VLA endows the recurrent mechanism with the ability to dynamically modulate its memory state—enabling targeted retention and absolute forgetting natively.

### Key Theoretical Innovations

1.  **Dynamic Penalty Matrix ($M_t$)**: Standard exponential decay sequences force a rigid, untargeted forgetting curve. VLA, conversely, constructs a dynamic, dense penalty matrix over time, actively suppressing irrelevant information based on strictly local and shifting contextual dynamics.
2.  **Stable Sherman-Morrison Inversion**: We leverage the rank-1 **Sherman-Morrison Update** to exactingly track the inverse of the penalty matrix ($A_t = M_t^{-1}$) at every temporal step. This guarantees strictly linear time $\mathcal{O}(T d^2)$ state updates while structurally dodging the exploding eigenvalues that plague baseline state-space models and linear transformers.
3.  **Optimal Coefficient Recovery ($\alpha^*$)**: VLA analytically solves an online optimization problem at every forward pass step, mathematically guaranteeing the coefficient scaling vector $\alpha_t = A_t s_t$ flawlessly minimizes state reconstruction errors.

## Documentation Architecture

This repository serves as both a rigorous theoretical reference and a practical engineering implementation guide for reproducing our experimental suites:

-   **[Theory & Mathematics](./maths.md)**: An in-depth mathematical derivation of the core primitives, including the stable Sherman-Morrison inversions, epsilon-bounded stability mechanisms, and optimal coefficient formulations.
-   **[API Manual](./api/vla_core.md)**: Detailed technical specifications for utilizing the PyTorch bindings of VLA Core, alongside DeltaNet and Linear Transformer baselines.
-   **[Experiments & Benchmarks](./experiments.md)**: Comprehensive empirical analyses spanning symbolic tasks, Long Range Arena (LRA) throughput, and our latest **VLA v3** Multi-Query Associative Recall (MQAR) tests validating state-of-the-art long-context retention.
-   **[Getting Started](./running.md)**: Practical directives for environment orchestration, experiment replication, and plotting pipeline execution.

Navigate via the sidebar to explore how Variational Linear Attention effectively circumvents the long-context bottleneck, delivering pristine memory retention at massive scales.
