# Reproducibility & Code Architecture Guide (CODE_GUIDE.md)

> **"Not every token needs the full power."**
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
3. **Step 3: Run Convergence Training & Physical Hardware Profiling (GPU & CUDA Required)**
   * **Dual-Rate Convergence Training**: Navigate to `benchmarks/` and execute:
     ```bash
     python run_dual_rate_mamba_experiment.py --steps 2300 --layer 24
     ```
     This trains the gated model on WikiText-2 next-token prediction, registering validation perplexity (PPL) and saving converged checkpoints (`t3_gated.pt`, `film_gated.pt`).
   * **Physical GPU Latency & Energy Profiling**: Run:
     ```bash
     python run_real_speed_benchmark.py
     ```
     This runs sequential autoregressive text generation, physically bypassing the Cognitive Core ($T_1$) on non-wake steps, while polling physical board-level power draw via NVIDIA NVML (`pynvml`) to yield exact wall-clock speedup (ms/token) and energy savings (mJ/token).
   * **Adversarial Chattering Stress Test**: Run:
     ```bash
     python run_stress_test.py
     ```
     This measures the gate stability under high-surprise out-of-distribution text to verify Zeno-free bounds.

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
| **`run_dual_rate_mamba_experiment.py`** | **Section VI.E (Empirical Dual-Rate Validation)** | **Core Convergence Trainer**. Sweeps hyperparameters, runs convergence training on WikiText-2, tracks validation perplexity (PPL) of Bare T3, Gated, and Oracle baselines, and saves weights. |
| **`run_real_speed_benchmark.py`** | **Section VI.E (Empirical Dual-Rate Validation)** | **Physical Profiler**. Implements true token-by-token physical Cognitive Core bypassing, hooks NVIDIA NVML to profile live GPU power draw, and logs wall-clock speedup and energy savings. |
| **`run_stress_test.py`** | **Section VI.E (Domain-Chattering Stress Test)** | **Robustness Evaluator**. Subjects the ETCD gate to out-of-distribution adversarial text to measure wake frequency ceilings and exclude Zeno behavior. |
| **`run_real_model_benchmark.py`** | **Section V.E (Token Overlap & Consistency)** | Evaluates pre-trained Hugging Face Mamba models under multi-timescale gating, measuring next-token Top-1 consistency and Top-5 token overlap against monolithic sequence baselines. Saves raw output to `experimental_results/`. |
| **`compare_results.py`** | **Experiment 3 (Averaged Deviation Metrics)** | Reads raw CSVs and computes statistical distributions of NECD deviation, mean overlap scores, Cohen's d effect sizes, and p-values to demonstrate statistical significance. |
| **`run_parallel_scan_profiling.py`**| **Section VI.C (Computational Complexity)** | Profiles PyTorch sequential step-by-step decoding vs chunk-based parallel scan under hardware-level CPU/GPU kernel launch overheads to analyze the theoretical 13.95x FLOPs compression vs wall-clock latency. |
| **`run_sandbox_profiling.py`** | **Section VI.C (Memory & Cache Overhead)** | Measures physical memory footprints of Mamba recurrent states and Transformer KV caches in long-context decoding tasks inside a sandboxed memory tracker. |
| **`continue_experiments.py`** (1 & 2) | **Experimental Utility** | Handles state checkpointing and automatic progress recovery during large-scale corpus evaluation runs. |
| **`find_fp16_mismatch.py`** | **Experimental Utility** | Traces numerical precision mismatches (FP32 vs FP16) in state-space updates. |
| **`generate_final_summary.py`** | **Experimental Utility** | Compiles all generated CSV benchmarks into unified LaTeX-ready table summaries. |
