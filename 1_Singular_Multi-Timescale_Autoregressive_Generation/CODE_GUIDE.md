# Reproducibility & Code Architecture Guide (CODE_GUIDE.md)

> **"Not every token needs the full model."**
> This guide is designed to help peer reviewers, Action Editors, and open-source ML researchers navigate our codebase and establish a direct mapping between our source code and the theoretical/empirical assertions in the paper.

---

## 🛠️ 1. Quick Start: Three-Step Reproduction Pipeline

To explore the codebase from simple prototypes to dense scale-up benchmarks, we recommend the following sequence:

1. **Step 1: Run Google Colab Live Demo (30 Seconds)**
   * Execute the self-contained `singular_colab_demo.py` in the root directory (or click the Colab Badge on our GitHub landing page).
   * It loads a pre-trained `mamba-130m` model and dynamically streams tokens, highlighting System 1 (T3 Surface Core, green tokens) and System 2 (T1 Cognitive Core, red underlined tokens) activations under dynamic predictive surprise (ETCD) gating.
2. **Step 2: Run Mathematical & Control-Theoretic Simulations (3 Minutes)**
   * Navigate to the `simulation/` directory and run `tikhanov_simulation.py` and `iss_stability_test.py`.
   * These scripts evaluate slow-manifold convergence and input-to-state stability trajectories, reproducing the exact control-theoretic plots featured in the paper.
3. **Step 3: Run Full-Scale Token-Consistency Benchmarks (GPU Required)**
   * Navigate to the `benchmarks/` directory and execute `run_real_model_benchmark.py`.
   * This evaluates Mamba sequence consistency over downstream corpora, generating the raw evaluation CSV data mapping directly to Experiments 1–6 in the manuscript.

---

## 📂 2. Directory Mapping & Theoretical Alignment

### 🟢 2.1 Theoretical Control Simulations: `simulation/`

This directory houses the mathematical foundations, non-linear ODE systems, and Lyapunov trajectory evaluations of the Singular framework.

| Filename | Mapped Paper Section / Theorem | Core Functionality & Metrics Evaluated |
| :--- | :--- | :--- |
| **`tikhanov_simulation.py`** | **Chapter 3 (Tikhonov Manifold Separability)** | Simulates high-dimensional non-linear fast-slow coupled dynamics under singular perturbation parameter $\epsilon \to 0$. Validates that fast boundary-layer states converge exponentially to the slow invariant manifold. |
| **`iss_stability_test.py`** | **Section 4.1 (ISS Stability of Fast Subsystem)** | Simulates input-to-state stability (ISS) trajectories under continuous multi-modal perturbations. Validates state boundedness and dissipative Lyapunov decay. |
| **`event_trigger_sim.py`** | **Section 4.2 (Event-Triggered Gate - ETCD)** | Evaluates the dynamic sampling behavior of the self-supervised Event-Triggered Change-Detection (ETCD) gate driven by Normalized Embedding Cosine Deviation (NECD). Outputs wakeup rates and inter-sample intervals. |
| **`film_injection_demo.py`** | **Chapter 5 (FiLM Dynamic Injection)** | Simulates the dynamic parameter injection of slow semantic fields into fast-rate state equations using Feature-wise Linear Modulation (FiLM) circuits, demonstrating transient scaling dynamics. |
| **`generate_figures.py`** | **All Main Figures (Figures 2, 3, 5, 6)** | Integrated simulation control script. Runs all control simulations recursively and renders publication-grade analytical PNGs in the `figures/` directory. |
| **`generate_figure0_conceptual.py`**| **Figure 1 (System Architecture Framework)** | Renders the high-level conceptual multi-scale cascade schematics of the Singular framework. |

---

### 🟢 2.2 Empirical Large Language Model Benchmarks: `benchmarks/`

This directory contains the Mamba sequence evaluation scripts used to measure factual overlap and computational overhead on real-world text generation tasks.

| Filename | Mapped Paper Section / Experiment | Core Functionality & Metrics Evaluated |
| :--- | :--- | :--- |
| **`run_real_model_benchmark.py`** | **Experiments 3 & 4 (Token Overlap & Consistency)** | **Primary Evaluation Script**. Evaluates pre-trained Hugging Face Mamba models under multi-timescale gating, measuring next-token Top-1 consistency and Top-5 token overlap against monolithic sequence baselines. Saves raw output to `experimental_results/`. |
| **`compare_results.py`** | **Experiment 3 (Averaged Deviation Metrics)** | Reads raw CSVs and computes statistical distributions of NECD deviation, mean overlap scores, Cohen's d effect sizes, and p-values to demonstrate statistical significance. |
| **`run_parallel_scan_profiling.py`**| **Section 6.3 (Computational Complexity)** | profiles PyTorch sequential step-by-step decoding vs chunk-based parallel scan under hardware-level CPU/GPU kernel launch overheads to analyze the theoretical 13.95x FLOPs compression vs wall-clock latency. |
| **`run_sandbox_profiling.py`** | **Section 6.4 (Memory & Cache Overhead)** | Measures physical memory footprints of Mamba recurrent states and Transformer KV caches in long-context decoding tasks inside a sandboxed memory tracker. |
| **`continue_experiments.py`** (1 & 2) | **Experimental Utility** | Handles state checkpointing and automatic progress recovery during large-scale corpus evaluation runs. |
| **`find_fp16_mismatch.py`** | **Experimental Utility** | Traces numerical precision mismatches (FP32 vs FP16) in state-space updates. |
| **`generate_final_summary.py`** | **Experimental Utility** | Compiles all generated CSV benchmarks into unified LaTeX-ready table summaries. |
