---
id: experiments
title: Experiments & Results
sidebar_position: 5
---

# Experiments & Results

This section documents our rigorous evaluation of Variational Linear Attention (VLA). We validate VLA through a series of increasingly complex tests, ranging from low-level symbolic stability checks to large-scale benchmark suites like Long Range Arena (LRA).

By analyzing the internal dynamics—such as attention entropy, eigenvalue stability, and penalty matrix visualization—we verify that VLA successfully learns to selectively memorize and forget.

---

## 1. Symbolic & Diagnostic Experiments

Before evaluating on sequence modeling tasks, we rigorously analyze the mathematical properties of the VLA recurrence. These experiments ensure the system behaves stably under prolonged execution and correctly reconstructs target sequences.

### Penalty Matrix Evolution ($\Delta M_t$)

The core innovation of VLA is the dynamic penalty matrix $M_t$. By visualizing $M_t$, we can explicitly observe the model learning to penalize specific dimensions over time.

![Penalty Heatmap](_static/img/heatmap_Mt_pub.png)
*Fig 1: Heatmap showing the evolution of the penalty matrix $M_t$ across a sequence. VLA aggressively increases the penalty on dimensions corresponding to irrelevant historical tokens.*

### Eigenvalue Stability

A major challenge in unconstrained linear RNNs and state-space models is numerical explosion or vanishing activations. The Sherman-Morrison inverse updates in VLA naturally bound the eigenvalues of the memory system.

![Eigenvalues Plot](_static/img/eigenvalues_plot_pub.png)
*Fig 2: Eigenvalues of the memory matrix $S_t$ over 10,000 timesteps. VLA maintains strict numerical stability (eigenvalues near 1), preventing the exponential growth that plagues standard linear transformers on extremely long contexts.*

---

## 2. Synthetic Memory Tasks

We test VLA on fundamental memory operations: copying a sequence and recalling a specific token after a long delay. These tasks are notoriously difficult for standard Linear Attention due to "attention dilution".

### The Copy Task

The model must observe a sequence of length $T$ and exactly reproduce it. The loss should decrease monotonically.

![Synthetic Copy](_static/img/neurips_synthetic_copy.png)
*Fig 3: VLA achieves near-zero loss significantly faster than DeltaNet and standard Linear Transformers. The exact inverse tracking allows VLA to perfectly capture the sequence without degradation.*

### The Delayed Recall Task

The model observes a key-value pair, processes a sequence of pure noise of length $T$, and is then asked to recall the value associated with the key.

![Delayed Recall](_static/img/neurips_synthetic_delayed.png)
*Fig 4: As the delay length increases (e.g., $T > 1000$), LT entirely forgets the key. VLA successfully retrieves the value by setting the penalty $\lambda_t \to 0$ for the target key and $\lambda_t \to \infty$ for the noise tokens, maintaining a pristine memory state.*

---

## 3. Long Range Arena (LRA) Benchmark

The Long Range Arena is a suite of tasks specifically designed to evaluate efficient transformers on sequences ranging from 1K to 16K tokens. We compare VLA against strong baselines across multiple domains (text, images, mathematics).

### Overall Performance

![LRA Overall](_static/img/lra_overall_aesthetic.png)
*Fig 5: VLA consistently outperforms standard Linear Transformers and remains highly competitive with DeltaNet across all LRA tasks, achieving state-of-the-art results on memory-intensive subtasks.*

### Task-Specific Analysis

![LRA Per Task](_static/img/neurips_fig1_per_task.png)
*Fig 6: Breakdown of LRA performance. VLA exhibits particularly strong gains on the Path-X task (16K sequence length), proving its ability to model extreme long-range dependencies where standard attention mechanisms fail due to $\mathcal{O}(N^2)$ constraints.*

---

## Summary

The empirical results confirm our theoretical hypotheses:
1. **Expressivity**: VLA perfectly solves synthetic memory bottlenecks.
2. **Stability**: The matrix inverse recurrence remains strictly stable for sequence lengths exceeding $10^4$.
3. **Scalability**: VLA delivers state-of-the-art accuracy on long-context benchmarks while maintaining $\mathcal{O}(T d^2)$ runtime complexity.