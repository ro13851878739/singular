"""
PHYSICAL HARDWARE WALL-CLOCK LATENCY AND ENERGY PROFILER
Resolves Bug 2: True physical bypassing of the T1 Cognitive Core during non-wake steps.
Measures wall-clock latency (ms/token) and real GPU energy consumption (Joules/token)
using the NVIDIA Management Library (NVML) via `pynvml`.

Architecture:
  - T1: Mamba-370M (frozen Cognitive Core, 48 layers).
  - T3: Pruned Mamba (4 layers, active Surface Core).
  - Gated Dual-Rate (Singular-SSM): T1 is physically bypassed (zero forward execution)
    on non-wake tokens. ETCD surprise thresholding is evaluated on T3's high-frequency
    hidden-state trajectory.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────────
# Configuration & Hardware Selection
# ──────────────────────────────────────────────────────────────────
MODEL_NAME = "state-spaces/mamba-370m-hf"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

# Gating & Decoding Parameters
WINDOW_SIZE = 3
F_TOKEN = 50.0  # nominal token rate (Hz)
DT_COG = 0.04   # cognitive dwell-time threshold (sec)
INJECTION_LAYER = 24

# ──────────────────────────────────────────────────────────────────
# Modules & Hook Implementations
# ──────────────────────────────────────────────────────────────────
class MambaFiLMModule(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.proj_gamma = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])
        self.proj_delta = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])
        
        # Zero init ensures initial stability
        for l in range(num_layers):
            nn.init.zeros_(self.proj_gamma[l].weight)
            nn.init.zeros_(self.proj_gamma[l].bias)
            nn.init.zeros_(self.proj_delta[l].weight)
            nn.init.zeros_(self.proj_delta[l].bias)

    def forward(self, Phi, layer_idx):
        gamma = self.proj_gamma[layer_idx](Phi)
        delta = self.proj_delta[layer_idx](Phi)
        return gamma, delta

# Global variable to transfer Phi into hooks dynamically
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
# NVML Real-Time Energy Monitor
# ──────────────────────────────────────────────────────────────────
class GPUEnergyMonitor:
    def __init__(self, sample_interval_ms=5):
        self.interval = sample_interval_ms / 1000.0
        self.power_samples = []
        self.timestamps = []
        self.running = False
        self.supported = False
        self.device_handle = None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            # Select default active GPU (index 0)
            self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.supported = True
            print("  [NVML] Successfully initialized. Monitoring NVIDIA GPU 0.")
        except Exception as e:
            print(f"  [NVML] Warning: Could not initialize NVML ({e}). Falling back to latency-only profiling.")

    def _monitor_loop(self):
        import pynvml
        t_start = time.perf_counter()
        while self.running:
            try:
                # Get power usage in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.device_handle)
                power_w = power_mw / 1000.0
                self.power_samples.append(power_w)
                self.timestamps.append(time.perf_counter() - t_start)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if not self.supported:
            return
        self.power_samples = []
        self.timestamps = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.supported or not self.running:
            return 0.0
        self.running = False
        self.thread.join(timeout=1.0)
        
        # Numerical integration to calculate Joules (Watt-seconds)
        if len(self.power_samples) < 2:
            return 0.0
            
        joules = 0.0
        for i in range(len(self.power_samples) - 1):
            dt = self.timestamps[i+1] - self.timestamps[i]
            avg_power = (self.power_samples[i] + self.power_samples[i+1]) / 2.0
            joules += avg_power * dt
        return joules

# ──────────────────────────────────────────────────────────────────
# Decoupled Autoregressive Inference Loop (Bug 2 Fix)
# ──────────────────────────────────────────────────────────────────
def run_gated_autoregressive_generation(
    t1_model, t3_model, film_module, input_ids, max_new_tokens, threshold, d_model
):
    global CURRENT_PHI
    
    t1_model.eval()
    t3_model.eval()
    
    # Register Hooks on T3
    hooks = []
    for l in range(t3_model.config.num_hidden_layers):
        h_hook = t3_model.backbone.layers[l].register_forward_hook(make_film_hook(l, film_module))
        hooks.append(h_hook)

    # Convert prompt to list of token IDs
    generated = input_ids.clone()
    seq_len = generated.shape[1]
    
    # T1/T3 Recurrent Caches are initialized automatically by HF Mamba during sequence forward
    # Warmup with prompt context
    with torch.no_grad():
        t1_out = t1_model(generated, output_hidden_states=True, return_dict=True)
        h_t1_init = t1_out.hidden_states[INJECTION_LAYER][:, -1:, :]  # [batch, 1, d_model]
        
        t3_out = t3_model(generated, output_hidden_states=True, return_dict=True)
        h_t3_seq = t3_out.hidden_states[-1]  # [batch, seq_len, d_model]

    # Initialize ETCD sliding window of T3 hidden states
    t3_window = [h_t3_seq[0, i].float() for i in range(max(0, seq_len - WINDOW_SIZE), seq_len)]
    
    # Gating and timing states
    anchor_start = h_t1_init[0, 0].float()
    t_last = seq_len - 1
    wake_count = 0
    
    # Autoregressive generation steps
    for step in range(max_new_tokens):
        current_step_idx = seq_len + step
        t_sec = current_step_idx / F_TOKEN
        t_last_sec = t_last / F_TOKEN
        
        # Fetch the last token generated
        last_token = generated[:, -1:]
        
        # 1. Evaluate Surface Core (T3) for next token logits
        with torch.no_grad():
            # T3 is ALWAYS evaluated to generate candidate logits and fast representations
            t3_out = t3_model(last_token, output_hidden_states=True, return_dict=True)
            next_logit = t3_out.logits[:, -1:, :]
            next_token = torch.argmax(next_logit, dim=-1)
            h_t3_curr = t3_out.hidden_states[-1][0, 0].float()
            
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Update sliding window
        t3_window.append(h_t3_curr)
        if len(t3_window) > WINDOW_SIZE:
            t3_window.pop(0)
            
        # 2. Check the ETCD Surprise Gate on T3's hidden state trajectory
        should_wake = False
        if len(t3_window) == WINDOW_SIZE:
            # Measure semantic surprise (cosine drift) in Surface representation
            sim = F.cosine_similarity(t3_window[0], t3_window[-1], dim=0).item()
            delta_semantic = 1.0 - sim
            
            # Wake up if surprise exceeds threshold AND dwell-time safety bound is met
            if delta_semantic > threshold and (t_sec - t_last_sec >= DT_COG):
                should_wake = True

        # 3. Handle Cognitive Core (T1) execution (The Bug 2 Physical Bypass)
        if should_wake:
            wake_count += 1
            t_last = current_step_idx
            
            # WAKE STEP: Execute heavy T1 forward pass to retrieve cognitive semantic anchor
            with torch.no_grad():
                t1_out = t1_model(next_token, output_hidden_states=True, return_dict=True)
                anchor_start = t1_out.hidden_states[INJECTION_LAYER][0, 0].float()
                
            CURRENT_PHI = anchor_start
        else:
            # NON-WAKE STEP: Bypasses T1 forward pass entirely! Zero execution overhead!
            dt = (current_step_idx - t_last) / F_TOKEN
            alpha = math.exp(-2.0 * dt)
            
            # Decay slow contextual prior causally without running T1
            CURRENT_PHI = anchor_start * alpha

    # Deregister Hooks
    for hook in hooks:
        hook.remove()
        
    return generated, wake_count

# ──────────────────────────────────────────────────────────────────
# Monolithic Baseline Generation Loop
# ──────────────────────────────────────────────────────────────────
def run_monolithic_generation(t1_model, input_ids, max_new_tokens):
    t1_model.eval()
    generated = input_ids.clone()
    
    for step in range(max_new_tokens):
        last_token = generated[:, -1:]
        with torch.no_grad():
            out = t1_model(last_token, return_dict=True)
            next_logit = out.logits[:, -1:, :]
            next_token = torch.argmax(next_logit, dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)
        
    return generated

# ──────────────────────────────────────────────────────────────────
# Performance & Energy Profiling Suite
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  PHYSICAL WALL-CLOCK & ENERGY PROFILER (BUG 2 RESOLUTION)")
    print(f"  Device: {DEVICE}  |  Precision: {DTYPE}")
    print("=" * 68)
    
    # 1. Initialize Tokenizer and Models
    print("\n[1/4] Loading models and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Loaded frozen T1 (48 layers)
    t1_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE
    ).to(DEVICE)
    d_model = t1_model.config.hidden_size
    
    # Pruned T3 core (4 layers)
    t3_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    t3_config.num_hidden_layers = 4
    t3_model = AutoModelForCausalLM.from_config(t3_config).to(DEVICE)
    
    # FiLM projection module
    film = MambaFiLMModule(d_model, t3_config.num_hidden_layers).to(DEVICE)
    
    # Attempt to load trained checkpoints if they exist
    results_dir = os.path.dirname(os.path.abspath(__file__)) + "/experimental_results/exp_a_b"
    t3_checkpoint = f"{results_dir}/t3_gated.pt"
    film_checkpoint = f"{results_dir}/film_gated.pt"
    
    if os.path.exists(t3_checkpoint) and os.path.exists(film_checkpoint):
        print("  Loading pre-trained checkpoints from exp_a_b...")
        t3_model.load_state_dict(torch.load(t3_checkpoint, map_location=DEVICE))
        film.load_state_dict(torch.load(film_checkpoint, map_location=DEVICE))
    else:
        print("  [Note] Pre-trained checkpoints not found. Operating under initialized state (Step 0 behaviour).")

    # Load experimental gating threshold
    etcd_threshold = 0.05
    results_json = f"{results_dir}/results.json"
    if os.path.exists(results_json):
        try:
            with open(results_json, "r") as f:
                etcd_threshold = json.load(f).get("etcd_threshold", 0.05)
        except Exception as e:
            print(f"  [Warning] Could not load etcd_threshold from results.json ({e}). Using default.")
    print(f"  ETCD Gate Threshold: Γ₀ = {etcd_threshold:.6f}")

    # Set up prompt and generation sizes
    prompt = "The computational complexity of deep neural networks can be compressed"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    max_new_tokens = 60
    
    print(f"\n[2/4] Initializing GPUEnergyMonitor via NVML...")
    monitor = GPUEnergyMonitor(sample_interval_ms=5)

    # 2. Benchmarking Gated Dual-Rate (Physical Bypassing)
    print(f"\n[3/4] Profiling Gated Dual-Rate Generation ({max_new_tokens} tokens)...")
    # Warmup
    _, _ = run_gated_autoregressive_generation(t1_model, t3_model, film, ids, 10, etcd_threshold, d_model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    monitor.start()
    t_start = time.perf_counter()
    _, wake_count = run_gated_autoregressive_generation(t1_model, t3_model, film, ids, max_new_tokens, etcd_threshold, d_model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gated = time.perf_counter() - t_start
    gated_joules = monitor.stop()
    
    wake_pct = (wake_count / max_new_tokens) * 100.0
    print(f"    Gated wall-clock time:  {t_gated * 1000:.1f} ms  ({(t_gated / max_new_tokens) * 1000:.1f} ms/token)")
    print(f"    Gated wake frequency:   {wake_count}/{max_new_tokens} tokens = {wake_pct:.1f}% active waker")
    if monitor.supported:
        print(f"    Gated energy spent:     {gated_joules:.3f} Joules  ({(gated_joules / max_new_tokens) * 1000:.3f} mJ/token)")

    # 3. Benchmarking Monolithic (Always Active)
    print(f"\n[4/4] Profiling Monolithic Mamba-370M Generation ({max_new_tokens} tokens)...")
    # Warmup
    _ = run_monolithic_generation(t1_model, ids, 10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    monitor.start()
    t_start = time.perf_counter()
    _ = run_monolithic_generation(t1_model, ids, max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_mono = time.perf_counter() - t_start
    mono_joules = monitor.stop()
    
    print(f"    Monolithic wall-clock time: {t_mono * 1000:.1f} ms  ({(t_mono / max_new_tokens) * 1000:.1f} ms/token)")
    if monitor.supported:
        print(f"    Monolithic energy spent:    {mono_joules:.3f} Joules  ({(mono_joules / max_new_tokens) * 1000:.3f} mJ/token)")

    # 4. Save and Report Hardware Metrics
    print("\n" + "=" * 68)
    print("  HARDWARE EFFICIENCY ANALYSIS SUMMARY")
    print("=" * 68)
    speedup = t_mono / t_gated
    print(f"  Physical Wall-Clock Speedup:       {speedup:.2f}× speedup")
    print(f"  Active Deliberation Wakes:          {wake_pct:.1f}% of steps")
    
    if monitor.supported:
        energy_saving = (1.0 - (gated_joules / mono_joules)) * 100.0
        print(f"  Physical GPU Energy Savings:       {energy_saving:.1f}% energy saved")
        print(f"  Monolithic Energy Consumption:     {(mono_joules / max_new_tokens) * 1000:.1f} mJ/token")
        print(f"  Gated Energy Consumption:          {(gated_joules / max_new_tokens) * 1000:.1f} mJ/token")
        
        # Save real energy stats to a JSON results file
        real_stats_file = f"{results_dir}/physical_hardware_efficiency.json"
        with open(real_stats_file, "w") as f:
            json.dump({
                "t_mono_ms_token": (t_mono / max_new_tokens) * 1000,
                "t_gated_ms_token": (t_gated / max_new_tokens) * 1000,
                "speedup_factor": speedup,
                "wake_percentage": wake_pct,
                "mono_mj_token": (mono_joules / max_new_tokens) * 1000,
                "gated_mj_token": (gated_joules / max_new_tokens) * 1000,
                "energy_savings_percentage": energy_saving
            }, f, indent=2)
            print(f"\n  ✅ Successfully archived hardware metrics to {real_stats_file}")
    else:
        print("\n  [Note] Energy monitoring requires Nvidia NVML drivers. Latency speedups logged successfully.")
    print("=" * 68)

if __name__ == "__main__":
    main()
