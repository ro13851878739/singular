import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────────
# Configurations & Settings
# ──────────────────────────────────────────────────────────────────
MODEL_NAME = "state-spaces/mamba-370m-hf"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32
INJECTION_LAYER = 24  # Default intermediate semantic injection layer (middle of 48 layers)

# Hyperparameters
WINDOW_SIZE = 3
F_TOKEN = 50.0
DT_COG = 0.04

# Paths
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/experimental_results/exp_a_b"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Model & FiLM Definitions (Matching Experiment A)
# ──────────────────────────────────────────────────────────────────
class MambaFiLMModule(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.proj_gamma = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])
        self.proj_delta = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])

    def forward(self, Phi, layer_idx):
        gamma = self.proj_gamma[layer_idx](Phi)
        delta = self.proj_delta[layer_idx](Phi)
        return gamma, delta

CURRENT_PHI = None

def make_film_hook(layer_idx, film_module):
    def hook(module, input_args, output):
        global CURRENT_PHI
        if CURRENT_PHI is None:
            return output
        if isinstance(output, tuple):
            hidden_states, cache = output
            is_tuple = True
        else:
            hidden_states = output
            is_tuple = False
        gamma, delta = film_module(CURRENT_PHI.to(hidden_states.device), layer_idx)
        modulated = (1.0 + torch.tanh(gamma)) * hidden_states + delta
        return (modulated, cache) if is_tuple else modulated
    return hook

# ──────────────────────────────────────────────────────────────────
# ETCD Gating & Interpolation (Matching Experiment A)
# ──────────────────────────────────────────────────────────────────
def compute_interpolated_phi(H_T1, threshold):
    b_sz, seq_len, d_model = H_T1.shape
    Phi = torch.zeros_like(H_T1)
    wake_mask = torch.zeros(b_sz, seq_len, dtype=torch.bool, device=H_T1.device)
    
    for i in range(b_sz):
        h_seq = H_T1[i]
        deltas = torch.zeros(seq_len, device=H_T1.device)
        for t in range(WINDOW_SIZE, seq_len):
            a = h_seq[t]
            b_vec = h_seq[t - WINDOW_SIZE]
            sim = F.cosine_similarity(a, b_vec, dim=0)
            deltas[t] = 1.0 - sim
            
        t_last = 0
        wakes = [0]
        wake_mask[i, 0] = True
        
        for t in range(WINDOW_SIZE, seq_len):
            t_sec = t / F_TOKEN
            t_last_sec = t_last / F_TOKEN
            if deltas[t] > threshold and (t_sec - t_last_sec >= DT_COG):
                wakes.append(t)
                wake_mask[i, t] = True
                t_last = t
                
        if wakes[-1] != seq_len - 1:
            wakes.append(seq_len - 1)
            wake_mask[i, seq_len - 1] = True
            
        # Interpolate/decay causally (no future leak)
        for j in range(len(wakes) - 1):
            t_start = wakes[j]
            t_end = wakes[j+1]
            anchor_start = h_seq[t_start]
            
            for t in range(t_start, t_end + 1):
                dt = (t - t_start) / F_TOKEN
                alpha = math.exp(-2.0 * dt)
                # Purely causal: decays the current anchor over time
                Phi[i, t] = anchor_start * alpha
                
    return Phi, wake_mask

# ──────────────────────────────────────────────────────────────────
# Adversarial Stress Corpus
# ──────────────────────────────────────────────────────────────────
# Level 0 (Predictable): Calm, continuous text
T_L0 = (
    "The capital of France is Paris. Paris is a major European city. "
    "It is known for its culture and history. The city has many museums. "
    "One famous museum is the Louvre. The Louvre contains the Mona Lisa."
)

# Level 1 (Moderate Surprise): Regular text with transitions
T_L1 = (
    "The capital of France is Paris. However, the economic center is shifting. "
    "Meanwhile, London remains a financial hub. In contrast, Berlin focuses on "
    "startups. Furthermore, Amsterdam excels in logistics. Therefore, cities compete."
)

# Level 2 (High Surprise / Adversarial): Text that jumps domains constantly
T_L2 = (
    "The capital of France is Paris. import math; x = math.sqrt(25). "
    "Meanwhile, a woodchuck chucks wood under the quadratic formula ax^2 + bx + c = 0. "
    "Consequently, deep learning transforms baking cookies in a warm climate. "
    "Finally, hello how are you doing today? I am fine, thank you."
)

STRESS_CORPUS = {
    "Level 0 (Predictable)": T_L0,
    "Level 1 (Discourse transitions)": T_L1,
    "Level 2 (Adversarial domain-chattering)": T_L2
}

# ──────────────────────────────────────────────────────────────────
# Main Stress Test
# ──────────────────────────────────────────────────────────────────
def main():
    global CURRENT_PHI
    print("=" * 68)
    print("  EXPERIMENT B: ADVERSARIAL STRESS TESTING")
    print(f"  Device: {DEVICE}  |  Precision: {DTYPE}")
    print("=" * 68)
    
    # 1. Load Tokenizer & Configs
    print("\n[1/5] Loading tokenizer and configuration...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    t1_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    d_model = t1_config.hidden_size
    
    t3_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    t3_config.num_hidden_layers = 4
    
    # 2. Load Cognitive Core (T1)
    print("\n[2/5] Loading and freezing Cognitive Core (T1 Mamba-370M)...")
    t1_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE
    ).to(DEVICE)
    t1_model.eval()
    for p in t1_model.parameters():
        p.requires_grad = False
        
    # 3. Load Gated T3 + FiLM weights from Experiment A
    print("\n[3/5] Loading Gated T3 and FiLM checkpoints from Experiment A...")
    t3_model = AutoModelForCausalLM.from_config(t3_config).to(DEVICE)
    film = MambaFiLMModule(d_model, t3_config.num_hidden_layers).to(DEVICE)
    
    t3_checkpoint_path = f"{RESULTS_DIR}/t3_gated.pt"
    film_checkpoint_path = f"{RESULTS_DIR}/film_gated.pt"
    results_json_path = f"{RESULTS_DIR}/results.json"
    
    if not (os.path.exists(t3_checkpoint_path) and os.path.exists(film_checkpoint_path) and os.path.exists(results_json_path)):
        print("  ❌ ERROR: Trained checkpoints from Experiment A not found! Please run run_dual_rate_mamba_experiment.py first.")
        return
        
    t3_model.load_state_dict(torch.load(t3_checkpoint_path, map_location=DEVICE))
    film.load_state_dict(torch.load(film_checkpoint_path, map_location=DEVICE))
    
    t3_model.eval()
    film.eval()
    print("  Checkpoints loaded successfully.")
    
    # Retrieve the calibrated threshold from results.json
    with open(results_json_path, "r") as f:
        stored_data = json.load(f)
        etcd_threshold = stored_data["etcd_threshold"]
    print(f"  Retrieved calibrated threshold: Γ₀ = {etcd_threshold:.6f}")
    
    # 4. Stress Test Execution
    print("\n[4/5] Running stress test over varying surprise levels...")
    hooks = []
    for l in range(t3_config.num_hidden_layers):
        h_hook = t3_model.backbone.layers[l].register_forward_hook(make_film_hook(l, film))
        hooks.append(h_hook)
        
    stress_results = {}
    
    for label, text in STRESS_CORPUS.items():
        print(f"\n  Evaluating {label}...")
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)
        b_sz, seq_len = ids.shape
        
        with torch.no_grad():
            t1_out = t1_model(ids, output_hidden_states=True, return_dict=True)
            h_t1 = t1_out.hidden_states[INJECTION_LAYER].float()
            
            # Gated forward
            CURRENT_PHI, w_mask = compute_interpolated_phi(h_t1, etcd_threshold)
            out = t3_model(ids, labels=ids)
            
            ppl = math.exp(out.loss.item())
            wake_count = w_mask.sum().item()
            wake_pct = (wake_count / (b_sz * seq_len)) * 100.0
            wake_hz = (wake_pct / 100.0) * F_TOKEN
            
            # GFLOPS Compression Ratio calculation
            p_t1 = sum(p.numel() for p in t1_model.parameters())
            p_t3 = sum(p.numel() for p in t3_model.parameters()) + sum(p.numel() for p in film.parameters())
            
            mono_gflops = 2 * p_t1 * F_TOKEN / 1e9
            t1_gflops = 2 * p_t1 * wake_hz / 1e9
            t3_gflops = 2 * p_t3 * F_TOKEN / 1e9
            ssm_gflops = t1_gflops + t3_gflops
            
            gflops_comp = mono_gflops / ssm_gflops if ssm_gflops > 0 else float("inf")
            
            stress_results[label] = {
                "ppl": ppl,
                "wake_count": wake_count,
                "seq_len": seq_len,
                "wake_pct": wake_pct,
                "wake_hz": wake_hz,
                "gflops_compression": gflops_comp
            }
            
            print(f"    Tokens: {seq_len}  |  PPL: {ppl:.2f}  |  Wake Rate: {wake_hz:.1f} Hz ({wake_pct:.1f}%)  |  Compression: {gflops_comp:.2f}×")
            
    # Clean hooks
    for hook in hooks:
        hook.remove()
        
    # Save stress test results to file
    with open(f"{RESULTS_DIR}/stress_results.json", "w") as f:
        json.dump(stress_results, f, indent=2)
        
    # 5. Plot Stress Test Response
    print("\n[5/5] Plotting stress test curves...")
    levels = list(stress_results.keys())
    wake_rates = [stress_results[l]["wake_hz"] for l in levels]
    compressions = [stress_results[l]["gflops_compression"] for l in levels]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Bar plot for Wake Rate
    color = '#F44336'
    ax1.set_xlabel('Surprise Density Level')
    ax1.set_ylabel('ETCD Wake Rate (Hz)', color=color)
    bars = ax1.bar(levels, wake_rates, color=color, alpha=0.6, width=0.4, label='Wake Rate (Hz)', edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=F_TOKEN, color='black', linestyle='--', alpha=0.5, label='Maximum Rate (50 Hz)')
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f} Hz",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight="bold", color='#B71C1C')

    # Line plot for GFLOPS Compression
    ax2 = ax1.twinx()
    color = '#3F51B5'
    ax2.set_ylabel('GFLOPS Compression Ratio', color=color)
    line = ax2.plot(levels, compressions, color=color, marker='o', linewidth=2.5, markersize=8, label='Compression Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    
    for i, txt in enumerate(compressions):
        ax2.annotate(f"{txt:.2f}×", (levels[i], compressions[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontweight="bold", color='#1A237E')
                    
    plt.title("Gated Dual-Rate Mamba: Dynamic GFLOPS Scaling under Stress")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/stress_test_scaling.png", dpi=150)
    plt.close()
    
    print(f"\n  ✅ Stress test results plot saved to {RESULTS_DIR}")
    print("=" * 68)

if __name__ == "__main__":
    main()
