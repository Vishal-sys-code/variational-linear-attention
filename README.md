<div align="center">
  <h1>Variational Linear Attention (VLA)</h1>
  <p><strong>A Next-Generation Sequence Model</strong></p>
  <p>
    <strong>Vishal S. Pandey</strong><sup>1</sup> &nbsp;&bull;&nbsp; 
    <strong>Gopal Singh</strong><sup>2</sup>
  </p>
  <p>
    <sup>1</sup>DeepBrain Labs &nbsp;&nbsp;&nbsp;
    <sup>2</sup>Metriqual
  </p>
</div>

<br />

> **Abstract:** Standard Linear Attention models achieve $\mathcal{O}(N)$ computational complexity by approximating the Softmax kernel. However, they suffer universally from "attention dilution" due to fixed retention and struggle with targeted historical recall (e.g., Associative Retrieval). **Variational Linear Attention (VLA)** reformulates the linear attention mechanism entirely through the lens of a **probabilistic graphical model**, introducing a mathematically optimal, data-dependent dynamic penalty mechanism ($M_t$). By explicitly learning to *forget* irrelevant tokens via stable rank-1 Sherman-Morrison inverse tracking, VLA naturally maintains pristine long-range dependencies without sequence degradation.

---

## Key Innovations

1. **Dynamic Penalty Matrix ($M_t$):** Unlike standard exponential decay sequences, VLA learns to construct a dynamic, dense penalty matrix over time, strongly suppressing irrelevant information natively based on changing contextual input.
2. **Strict Numerical Stability:** VLA leverages the **Sherman-Morrison Rank-1 Update** to actively maintain the exact inverse of the penalty matrix ($A_t = M_t^{-1}$) throughout the sequence. This guarantees that updating memory stays linear in time $\mathcal{O}(N d^2)$ while entirely dodging $\mathcal{O}(d^3)$ explosive inversions.
3. **Optimally Recovered Memory ($\alpha^*$):** VLA analytically solves an online optimization problem at every forward step, mathematically guaranteeing the coefficient scaling $\alpha_t = A_t s_t$ perfectly minimizes reconstruction errors.

## Theoretical Backbone (The Sherman-Morrison Update)

Maintaining numerical stability over infinitely long contexts requires flawless inversion updating. 
When the penalty matrix is perturbed by a new context vector ($u_t$), we stably update the structural inverse using our $\epsilon$-bounded stabilizer logic:

```math
A_t = A_{t-1} - \frac{A_{t-1} u_t u_t^\top A_{t-1}}{1 + u_t^\top A_{t-1} u_t}
```

This ensures extreme numerical preservation (maintaining stable unity eigenvalues natively over 10M+ tokens without catastrophic failure), bypassing completely the issues dominating baseline standard state-space and linear transformer models.

---

## Performance & Benchmarks

Our empirical evaluations across both Synthetic capabilities and symbolic reasoning scales showcase VLA operating natively at a State-of-the-Art capacity.

- **Synthetic Retrieval:** Hits perfect exact match accuracy on 10,000+ length associative and delayed recall tasks, where standard Linear Transformers drop to baseline 0% due to capacity erasure.
- **Symbolic Reasoning & LRA:** Exhibits powerful dominance against leading Linear-Time variants in memory-intensive logic flows specifically evaluated on **ListOps**, **CLUTRR**, and **CommonsenseQA**.

<div align="center">
  <h3>Experimental Visualizations & Ablations</h3>
  <table>
    <tr>
      <td align="center">
        <b>1. Phase D Ablation Studies</b><br/>
        <img src="website/static/img/ablation_summary.png" alt="Ablation Summaries" width="400"/>
        <br/><i>Comparing learning drivers vs fixed bounds</i>
      </td>
      <td align="center">
        <b>2. Penalty Matrix Heatmap ($M_t$)</b><br/>
        <img src="website/static/img/heatmap_Mt_pub.png" alt="Dynamic Penalty Matrix" width="400"/>
        <br/><i>Visualizing targeted exponential decay</i>
      </td>
    </tr>
    <tr>
      <td align="center">
        <b>3. Recursive Eigenvalue Stability</b><br/>
        <img src="website/static/img/eigenvalues_plot_pub.png" alt="Numerical Stability" width="400"/>
        <br/><i>$\epsilon$-bounded rank-1 inversion preservation</i>
      </td>
      <td align="center">
        <b>4. LRA & Symbolic Task Overviews</b><br/>
        <img src="website/static/img/neurips_fig1_per_task.png" alt="LRA Accuracies" width="400"/>
        <br/><i>Performance spanning 10K+ contexts</i>
      </td>
    </tr>
  </table>
</div>

<br/>

<div align="center">
  <h3>VLA v3 (Triton + Mamba) Benchmarks</h3>
  <table>
    <tr>
      <td align="center">
        <b>1. Multi-Query Associative Recall (MQAR)</b><br/>
        <img src="website/static/img/vla_v3/fig1_mqar_v3.png" alt="MQAR VLA v3" width="400"/>
        <br/><i>Perfect retrieval on 100K+ context lengths</i>
      </td>
      <td align="center">
        <b>2. KV Memory Exploding Norms</b><br/>
        <img src="website/static/img/vla_v3/fig_kv_norms_v3.png" alt="KV Norms Stability" width="400"/>
        <br/><i>Bounding hidden states via stable dynamic retention</i>
      </td>
    </tr>
    <tr>
      <td align="center">
        <b>3. Throughput vs Sequence Length</b><br/>
        <img src="website/static/img/vla_v3/fig2_throughput_v3.png" alt="VLA v3 Throughput" width="400"/>
        <br/><i>High efficiency hardware-aware Triton scanning</i>
      </td>
      <td align="center">
        <b>4. Model Scaling Laws</b><br/>
        <img src="website/static/img/vla_v3/fig3_scaling_v3.png" alt="VLA v3 Scaling" width="400"/>
        <br/><i>Performance as a function of model dimensions</i>
      </td>
    </tr>
  </table>
</div>

*Detailed breakdown and interactive visualization tracking are available directly in our [Documentation Portal](https://variational-linear-attention.vercel.app/).*

---

## Repository Structure

We enforce a strict, isolated architecture separating the pure math primitives from the actual PyTorch NN modules for maximal legibility:

```bash
variational-linear-attention/
├── src/
│   ├── modules/       # High-level PyTorch Neural Network definitions (VLA blocks)
│   ├── maths/         # Core isolated primitive functions (Sherman-Morrison, Woodbury)
│   └── data/          # Synthetic and benchmark data ingestion pipelines
├── benchmarks/        # LRA and complex timing suites
├── experiments/       # Ablation studies and convergence scripts
├── tests/             # Strict CI tests capturing numeric drift regressions
└── website/           # Docusaurus documentation (Math, APIs, Diagrams)
```

## Getting Started

### 1. Installation Environment
Create a clean Conda environment and install torch natively:
```bash
conda create -n vla-env python=3.10
conda activate vla-env
pip install -r requirements.txt
```

### 2. Running the Sub-Tests
DeepBrain Labs enforces absolute numerical precision matching theoretical limits. To verify stability and gradients in your local CUDA configuration:
```bash
pytest tests/ -v
```

### VLA v3 (Research Implementation Notes)
The repository now includes a dedicated `VLAv3` implementation in `src/models/attention/vla_v3.py`, aligned with the stabilized formulation from the v3 notebook:

1. **Positive feature map**: `Q,K = ELU(·)+1` for non-negative linear attention features.
2. **Learned penalty direction**: `u_t = normalize(W_u k_t^{raw})` with full gradient flow.
3. **Stable inverse recursion**: Sherman–Morrison updates for `A_t` with periodic diagonal nudging.
4. **Bounded fast-weight dynamics**: normalized `(k_t, α_t)` outer-product updates to control Jacobian spectrum.
5. **Calibrated readout**: `z·q` denominator normalization at decode time.

This is intended as the canonical code path for v3 ablations and reproducible MQAR-oriented experiments.

#### MQAR Protocol Used in VLA v3 (from Notebook 05)
The VLAv3 experiments are paired with a **Multi-Query Associative Recall (MQAR)** protocol, matching the setup used in `notebooks/05_VLAv3_Complete_Fix.ipynb`:

- **Sequence construction**: a context of interleaved key/value symbols followed by multiple key-only queries.
- **Training target**: each query key must retrieve its associated value token exactly.
- **Canonical default in notebook**: `d_model=64`, `vocab_size=64`, `num_pairs=8`, with cosine LR schedule and warmup.
- **Reported analyses**: (i) MQAR training curves, (ii) scaling vs `d_model`, (iii) scaling vs number of key-value pairs.
- **Why MQAR matters for v3**: it stress-tests selective long-context retrieval under linear-time state updates, where bounded `S_t` and stable `A_t` dynamics are required for robust recall.

If you want to reproduce the paper-style v3 figures, run the full workflow in `notebooks/05_VLAv3_Complete_Fix.ipynb` and compare against the saved artifacts in `notebooks/vla_v3_results/`.


### 3. Local Documentation Server
Want to read the interactive math and architectural deep dives? Boot the local Docusaurus server:
```bash
cd website
npm install
npm start
```

## Citation & Open Source

This repository represents the official implementation payload for the Variational Linear Attention framework initiated by the Research Engineering Core at **DeepBrain Labs**. Check out the issue tracker for upcoming feature releases or integration requests.

```
Paper -> Work in Progress
```
