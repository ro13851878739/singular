# Singular: Multi-Timescale Autoregressive Generation

[![Colab Demo](https://img.shields.io/badge/Run%20in%20Colab-Live%20Demo-brightgreen?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ro13851878739/singular/blob/main/1_Singular_Multi-Timescale_Autoregressive_Generation/singular_colab_demo.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **"Not every token needs the full power."**
> This repository hosts the official source code, control-theoretic simulations, and empirical evaluation benchmarks for **"Singular: Multi-Timescale Autoregressive Generation"**. 

---

## 💡 Core Conceptual Overview

Modern Autoregressive Large Language Models (LLMs) operate under a uniform temporal scale, forcing every generated token—regardless of its informational entropy—to evaluate synchronously across the entire dense parametric backbone. 

Drawing on **non-linear singular perturbation theory**, this paper formalizes **Multi-Timescale Autoregressive Generation**—a coupled dual-timescale architecture instantiated as the **Singular State-Space Model (Singular-SSM)**. 
* **T1 Cognitive Core (System 2, ~2 Hz)**: Models high-order logical planning as a discrete-time slow semantic invariant manifold.
* **T3 Surface Core (System 1, 50 Hz)**: Routes high-frequency, routine token transitions through a lightweight selective state-space boundary-layer core.

By separating low-level syntactic execution from macro-semantic anchoring via a self-supervised **Event-Triggered Change-Detection (ETCD)** gate, Singular-SSM dramatically reduces redundant computation while maintaining strict theoretical stability bounds (Input-to-State Stability under Tikhonov manifold contraction).

---

## 📂 1. Repository Structure

All materials for reproducing our findings are organized as follows:

* **`CODE_GUIDE.md`**: The official code blueprint mapping source scripts directly to the paper's mathematical equations, stability theorems, and Figure 2–5.
* **`singular_colab_demo.py`**: A lightweight, self-contained Python script to stream Hugging Face Mamba-130m with colored terminal highlighters showing active System 1/2 switching in real time.
* **`simulation/`**: High-fidelity control simulations validating slow invariant manifolds, Tikhonov convergence, and input-to-state stability.
* **`benchmarks/`**: Empirical evaluation suite measuring next-token consistency, Cohen's d effect sizes, and parallel-scan complexity.
  * **`benchmarks/experimental_results/`**: Stages the complete raw CSV datasets and latex summaries for Experiments 1–6.
* **`figures/`**: Renders all 11 high-resolution analytical PNG figures used in the manuscript.

---

## 🛠️ 2. Quick Start: Three-Step Reproduction Pipeline

### Step 1: Run Live Google Colab Demo (30 Seconds)
You can run the interactive prototype of our event-triggered Singular-SSM directly in your browser with zero installation:
1. Click our **[Run in Colab](https://colab.research.google.com/github/ro13851878739/singular/blob/main/1_Singular_Multi-Timescale_Autoregressive_Generation/singular_colab_demo.py)** badge at the top of this page to open the notebook instantly in Google Colab.
2. Sign in with your Google account and click the **Play/Run** button on the cell.
3. The script will dynamically download `mamba-130m-hf` (fully self-contained, auto-detecting GPU/MPS/CPU), execute text generation, and stream output where:
   * **Green tokens** represent routine generation processed by the lightweight T3 Surface Core (System 1, 50 Hz).
   * **Red underlined tokens** represent cognitive boundaries where the ETCD gate triggered a hardware interrupt to wake the T1 Cognitive Core (System 2, 2 Hz) for semantic re-anchoring.

### Step 2: Run Control Simulations (3 Minutes)
Validate our theoretical stability bounds locally:
```bash
cd 1_Singular_Multi-Timescale_Autoregressive_Generation/simulation
python tikhanov_simulation.py
python iss_stability_test.py
```
These will output numerical convergence diagnostics and reproduce the exact mathematical phase portraits featured in the manuscript.

### Step 3: Run Full-Scale LLM Benchmarks (GPU Required)
Evaluate token-level consistency over downstream corpora:
```bash
cd 1_Singular_Multi-Timescale_Autoregressive_Generation/benchmarks
python run_real_model_benchmark.py
```
This loads Mamba backbones to measure Top-1 consistency and Top-5 token overlap under multi-rate execution, saving results directly to `experimental_results/`.

---

## 📄 3. Citation

If you find our theoretical framework, simulations, or code helpful for your research, please cite our paper:

```bibtex
@article{peng2026singular,
  title={Singular: Multi-Timescale Autoregressive Generation},
  author={Luo, Peng and Chen, Yanan},
  journal={Transactions on Machine Learning Research},
  note={Under review},
  year={2026}
}
```
