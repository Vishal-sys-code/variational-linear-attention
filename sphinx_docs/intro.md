---
id: intro
title: Introduction
sidebar_position: 1
---

# Variational Linear Attention (VLA)

Welcome to the official research documentation for **Variational Linear Attention (VLA)** by DeepBrain Labs. This project introduces a novel approach to attention mechanisms, aiming to bridge the gap between the expressive power of Softmax Attention and the efficiency of Linear Attention.

## The Problem

Standard Transformers rely on Softmax Attention, which scales quadratically with sequence length ($\mathcal{O}(T^2)$). This makes them prohibitively expensive for long-context applications.

Linear Attention models, on the other hand, approximate the attention mechanism to achieve linear complexity ($\mathcal{O}(T)$). However, they often suffer from "attention dilution" and struggle with long-term memory recall tasks, such as the Associative Recall or Delayed Copy tasks. They typically fail to dynamically prioritize relevant information over time.

## Our Solution: VLA

**Variational Linear Attention (VLA)** reformulates the linear attention mechanism through the lens of a **probabilistic graphical model**. By introducing a time-varying, data-dependent penalty term, VLA allows the model to actively modulate its memory.

### Key Innovations

1.  **Dynamic Penalty Matrix ($M_t$)**: Unlike standard linear attention which treats all past tokens equally (or uses fixed decay), VLA learns to construct a penalty matrix based on the current context. This allows it to forget irrelevant information and strongly remember crucial tokens.
2.  **Sherman-Morrison Updates**: We utilize the Sherman-Morrison rank-1 update formula to efficiently compute the inverse of the penalty matrix ($M_t^{-1}$) at each timestep in $\mathcal{O}(d^2)$ time, maintaining the overall linear complexity $\mathcal{O}(T d^2)$.
3.  **Optimal Coefficient Recovery**: VLA solves an online optimization problem at every step to find the optimal memory update coefficients, ensuring the memory matrix is updated in a theoretically grounded manner.

## Structure of this Documentation

This documentation is designed to serve as both a rigorous mathematical reference and a practical API guide:

-   **[Theory & Mathematics](./maths.md)**: Deep dive into the core mathematical primitives, including the Sherman-Morrison updates and optimal coefficient derivations.
-   **[API Manual](./api/vla_core.md)**: Detailed technical specifications for implementing and utilizing VLA Core, DeltaNet, and Linear Transformer baselines.
-   **[Experiments](./experiments.md)**: Comprehensive analysis of our synthetic tasks (Copy, Delayed Recall) and Long Range Arena (LRA) benchmarks, demonstrating VLA's superiority.
-   **[Getting Started](./running.md)**: Practical instructions on how to set up the environment, run experiments, and reproduce our plots.

Explore the sections via the sidebar to understand how Variational Linear Attention redefines efficient sequence modeling.
