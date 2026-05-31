import math
import time
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
from datasets import load_dataset

# ──────────────────────────────────────────────────────────────────
# Configurations & Settings
# ──────────────────────────────────────────────────────────────────
MODEL_NAME = "state-spaces/mamba-370m-hf"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

# Hyperparameters
SEQ_LEN = 128
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 1
TRAIN_STEPS = 250
EVAL_STEPS = 50
INJECTION_LAYER = 24  # Default intermediate semantic injection layer (middle of 48 layers)

# ETCD Gating hyperparameters
WINDOW_SIZE = 3
F_TOKEN = 50.0
DT_COG = 0.04
THRESHOLD_K = 2.0  # multiplier for threshold calibration

# Output directory for results
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/experimental_results/exp_a_b"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# FiLM Module Definition
# ──────────────────────────────────────────────────────────────────
class MambaFiLMModule(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Projections for scale (gamma) and shift (delta) for each layer
        self.proj_gamma = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])
        self.proj_delta = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)])
        
        # Zero initialization ensures stability at step 0 (behaves like bare model)
        for l in range(num_layers):
            nn.init.zeros_(self.proj_gamma[l].weight)
            nn.init.zeros_(self.proj_gamma[l].bias)
            nn.init.zeros_(self.proj_delta[l].weight)
            nn.init.zeros_(self.proj_delta[l].bias)

    def forward(self, Phi, layer_idx):
        # Phi: [batch, seq_len, d_model]
        gamma = self.proj_gamma[layer_idx](Phi)
        delta = self.proj_delta[layer_idx](Phi)
        return gamma, delta

# Global variable to pass the modulation vector Phi to hooks
CURRENT_PHI = None

def make_film_hook(layer_idx, film_module):
    def hook(module, input_args, output):
        global CURRENT_PHI
        if CURRENT_PHI is None:
            return output
            
        # MambaBlock output can be a tuple of (hidden_states, cache_params) or just hidden_states
        if isinstance(output, tuple):
            hidden_states, cache = output
            is_tuple = True
        else:
            hidden_states = output
            is_tuple = False
            
        # Project current Phi to layer scale/shift parameters
        gamma, delta = film_module(CURRENT_PHI.to(hidden_states.device), layer_idx)
        
        # Apply element-wise modulation: tilde_H = (1 + tanh(gamma)) * H + delta
        modulated = (1.0 + torch.tanh(gamma)) * hidden_states + delta
        
        return (modulated, cache) if is_tuple else modulated
    return hook

# ──────────────────────────────────────────────────────────────────
# Data Processing (WikiText-2)
# ──────────────────────────────────────────────────────────────────
def prepare_data(tokenizer):
    print("[Data] Loading WikiText-2 dataset...")
    ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False)
        
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    # Remove attention_mask column to avoid length mismatch in pyarrow
    tokenized = tokenized.remove_columns(["attention_mask"])
    
    # Concatenate all input_ids and split into chunks of SEQ_LEN
    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
            
        total_len = len(concatenated)
        # Round down to multiple of SEQ_LEN
        total_len = (total_len // SEQ_LEN) * SEQ_LEN
        
        result = []
        for i in range(0, total_len, SEQ_LEN):
            result.append(concatenated[i : i + SEQ_LEN])
        return {"input_ids": result}
        
    grouped = tokenized.map(group_texts, batched=True)
    return grouped["train"], grouped["validation"]

# ──────────────────────────────────────────────────────────────────
# ETCD Gating & Interpolation Algorithms
# ──────────────────────────────────────────────────────────────────
def calibrate_threshold(H_T1):
    """Calibrate the ETCD threshold on predictable slices of text."""
    # H_T1 shape: [batch_size, seq_len, d_model]
    b, n, d = H_T1.shape
    deltas = []
    for i in range(b):
        h_seq = H_T1[i]
        for t in range(WINDOW_SIZE, n):
            a = h_seq[t]
            b_vec = h_seq[t - WINDOW_SIZE]
            sim = F.cosine_similarity(a, b_vec, dim=0).item()
            deltas.append(1.0 - sim)
            
    mu = sum(deltas) / len(deltas)
    sigma = (sum((x - mu) ** 2 for x in deltas) / len(deltas)) ** 0.5
    threshold = mu + THRESHOLD_K * sigma
    return max(0.001, min(0.95, threshold))

def compute_interpolated_phi(H_T1, threshold):
    """Compute gated interpolated modulation field Phi."""
    b_sz, seq_len, d_model = H_T1.shape
    Phi = torch.zeros_like(H_T1)
    wake_mask = torch.zeros(b_sz, seq_len, dtype=torch.bool, device=H_T1.device)
    
    for i in range(b_sz):
        h_seq = H_T1[i]
        # Calculate sliding window cosine similarity deltas
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
                
        # End anchor
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
# Main Experimentation Loop
# ──────────────────────────────────────────────────────────────────
def main():
    global DEVICE, DTYPE, BATCH_SIZE, LR, TRAIN_STEPS, EVAL_STEPS, RESULTS_DIR, INJECTION_LAYER
    
    import argparse
    parser = argparse.ArgumentParser(description="Singular Dual-Rate Mamba Experiment")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS, help="Number of training steps")
    parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS, help="Number of evaluation steps")
    parser.add_argument("--layer", type=int, default=INJECTION_LAYER, help="T1 hidden layer to inject (0-48)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    # Apply arguments
    TRAIN_STEPS = args.steps
    EVAL_STEPS = args.eval_steps
    INJECTION_LAYER = args.layer
    LR = args.lr
    BATCH_SIZE = args.batch_size
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    print("=" * 68)
    print("  EXPERIMENT A: DUAL-RATE MAMBA ON WIKITEXT-2")
    print(f"  Device: {DEVICE}  |  Precision: {DTYPE}  |  Injection Layer: {INJECTION_LAYER}")
    print(f"  Steps: {TRAIN_STEPS}  |  LR: {LR}  |  Batch Size: {BATCH_SIZE}  |  Seed: {args.seed}")
    print("=" * 68)
    
    # 1. Load Tokenizer & Config
    print("\n[1/7] Loading tokenizer and model configuration...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    t1_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    d_model = t1_config.hidden_size
    
    # 2. Load & Prepare WikiText-2 Data
    train_data, val_data = prepare_data(tokenizer)
    print(f"  Train samples: {len(train_data)}  |  Val samples: {len(val_data)}")
    
    # Create simple data loaders
    def get_batches(dataset, size):
        np.random.seed(42)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(dataset), size):
            if i + size > len(dataset):
                break
            batch_indices = indices[i : i + size]
            ids = [dataset[int(idx)]["input_ids"] for idx in batch_indices]
            batches.append(torch.tensor(ids, dtype=torch.long))
        return batches

    train_batches = get_batches(train_data, BATCH_SIZE)[:TRAIN_STEPS]
    val_batches = get_batches(val_data, BATCH_SIZE)[:EVAL_STEPS]
    
    # 3. Load Cognitive Core (T1) and freeze it
    print("\n[3/7] Loading and freezing Cognitive Core (T1 Mamba-370M)...")
    t1_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE
    ).to(DEVICE)
    t1_model.eval()
    for p in t1_model.parameters():
        p.requires_grad = False
    print("  T1 loaded successfully and frozen.")
    
    # 4. Instantiate custom Surface Core (T3)
    print("\n[4/7] Instantiating custom randomly initialized Surface Core (T3 4-layer Mamba)...")
    t3_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    t3_config.num_hidden_layers = 4  # prune to 4 layers for fast surface processing
    t3_model = AutoModelForCausalLM.from_config(t3_config).to(DEVICE)
    print(f"  T3 instantiated: {t3_config.num_hidden_layers} layers, {d_model} hidden size.")
    
    # 5. Calibrate ETCD Threshold
    print("\n[5/7] Calibrating ETCD threshold on a validation batch...")
    t1_val_out = t1_model(val_batches[0].to(DEVICE), output_hidden_states=True, return_dict=True)
    h_t1_val = t1_val_out.hidden_states[INJECTION_LAYER].float()
    etcd_threshold = calibrate_threshold(h_t1_val)
    print(f"  Calibrated ETCD Threshold: Γ₀ = {etcd_threshold:.6f}")
    
    # Define three separate training configurations to compare
    # We will initialize new T3 weights and FiLM weights for each run
    def run_training_experiment(experiment_type):
        global CURRENT_PHI
        print(f"\n--- Training Run: {experiment_type} ---")
        
        # Instantiate model and FiLM
        model_t3 = AutoModelForCausalLM.from_config(t3_config).to(DEVICE)
        film = MambaFiLMModule(d_model, t3_config.num_hidden_layers).to(DEVICE)
        
        # Automatically resume from periodic checkpoint if it exists
        if "Gated" in experiment_type:
            checkpoint_t3 = f"{RESULTS_DIR}/t3_gated_checkpoint.pt"
            checkpoint_film = f"{RESULTS_DIR}/film_gated_checkpoint.pt"
            if os.path.exists(checkpoint_t3) and os.path.exists(checkpoint_film):
                print(f"  [Resume] Found step-level backup checkpoint! Resuming training...")
                try:
                    model_t3.load_state_dict(torch.load(checkpoint_t3, map_location=DEVICE))
                    film.load_state_dict(torch.load(checkpoint_film, map_location=DEVICE))
                except Exception as e:
                    print(f"  [Resume] Warning: Could not load checkpoint ({e}). Training from scratch.")
        
        if experiment_type == "Bare T3 (no conditioning)":
            optimizer = torch.optim.AdamW(model_t3.parameters(), lr=LR)
            film.eval()  # Keep film deactivated
        else:
            optimizer = torch.optim.AdamW(
                list(model_t3.parameters()) + list(film.parameters()), lr=LR
            )
            film.train()
            
        # Register hooks
        hooks = []
        for l in range(t3_config.num_hidden_layers):
            h_hook = model_t3.backbone.layers[l].register_forward_hook(make_film_hook(l, film))
            hooks.append(h_hook)
            
        # Training loop
        losses = []
        t_start = time.perf_counter()
        
        for step, batch in enumerate(train_batches):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Setup CURRENT_PHI
            if experiment_type == "Bare T3 (no conditioning)":
                CURRENT_PHI = None
            else:
                with torch.no_grad():
                    t1_out = t1_model(batch, output_hidden_states=True, return_dict=True)
                    h_t1 = t1_out.hidden_states[INJECTION_LAYER].float()
                    
                if experiment_type == "Oracle Dual-Rate (continuous)":
                    CURRENT_PHI = h_t1
                elif experiment_type == "Gated Dual-Rate (Singular-SSM)":
                    CURRENT_PHI, _ = compute_interpolated_phi(h_t1, etcd_threshold)
                    
            # Forward pass
            out = model_t3(batch, labels=batch)
            loss = out.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Save periodic checkpoints to protect against cloud preemption
            if step % 1000 == 0 and step > 0:
                if "Gated" in experiment_type:
                    torch.save(model_t3.state_dict(), f"{RESULTS_DIR}/t3_gated_checkpoint.pt")
                    torch.save(film.state_dict(), f"{RESULTS_DIR}/film_gated_checkpoint.pt")
                    print(f"  [Checkpoint] Periodic step-level backup saved at step {step}.")
                    
            if step % 50 == 0:
                print(f"  Step {step:3d}/{TRAIN_STEPS}  |  Loss: {loss.item():.4f}")
                
        t_elapsed = time.perf_counter() - t_start
        print(f"  Finished in {t_elapsed:.1f}s. Avg Loss: {np.mean(losses):.4f}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return model_t3, film, losses

    # Run all three training experiments
    tstart = time.perf_counter()
    
    t3_bare, _, loss_bare = run_training_experiment("Bare T3 (no conditioning)")
    t3_oracle, film_oracle, loss_oracle = run_training_experiment("Oracle Dual-Rate (continuous)")
    t3_gated, film_gated, loss_gated = run_training_experiment("Gated Dual-Rate (Singular-SSM)")
    
    # 6. Evaluation Benchmark
    print("\n[6/7] Running Evaluation & Validation PPL Benchmark...")
    results = {}
    
    # Pre-compiled Monolithic 370M PPL
    print("  Evaluating Monolithic Mamba-370M...")
    t1_ppls = []
    with torch.no_grad():
        for batch in val_batches:
            batch = batch.to(DEVICE)
            out = t1_model(batch, labels=batch)
            t1_ppls.append(math.exp(out.loss.item()))
    results["Monolithic 370M (T1 anchor)"] = {
        "ppl": np.mean(t1_ppls), "wake_hz": F_TOKEN, "wake_pct": 100.0, "gflops_compression": 1.0
    }
    print(f"    PPL: {results['Monolithic 370M (T1 anchor)']['ppl']:.2f}")

    def evaluate_model(experiment_type, model_t3, film):
        global CURRENT_PHI
        print(f"  Evaluating {experiment_type}...")
        
        # Register hooks
        hooks = []
        for l in range(t3_config.num_hidden_layers):
            h_hook = model_t3.backbone.layers[l].register_forward_hook(make_film_hook(l, film))
            hooks.append(h_hook)
            
        losses = []
        wake_counts = 0
        total_tokens = 0
        
        # Separated losses for breakdown analysis
        wake_token_losses = []
        non_wake_token_losses = []
        
        with torch.no_grad():
            for batch in val_batches:
                batch = batch.to(DEVICE)
                b_sz, seq_len = batch.shape
                
                if experiment_type == "Bare T3":
                    CURRENT_PHI = None
                    wake_mask = torch.zeros(b_sz, seq_len, dtype=torch.bool, device=DEVICE)
                else:
                    t1_out = t1_model(batch, output_hidden_states=True, return_dict=True)
                    h_t1 = t1_out.hidden_states[INJECTION_LAYER].float()
                    
                    if experiment_type == "Oracle Dual-Rate":
                        CURRENT_PHI = h_t1
                        wake_mask = torch.ones(b_sz, seq_len, dtype=torch.bool, device=DEVICE)
                        wake_counts += b_sz * seq_len
                    elif experiment_type == "Gated Dual-Rate":
                        CURRENT_PHI, w_mask = compute_interpolated_phi(h_t1, etcd_threshold)
                        wake_mask = w_mask
                        wake_counts += w_mask.sum().item()
                        
                out = model_t3(batch, labels=batch)
                losses.append(out.loss.item())
                total_tokens += b_sz * seq_len
                
                # Breakdown PPL calculation
                logits = out.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                shift_mask = wake_mask[..., 1:].contiguous()
                
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_losses = token_losses.view(b_sz, seq_len - 1)
                
                for r in range(b_sz):
                    for c in range(seq_len - 1):
                        val_loss = token_losses[r, c].item()
                        if shift_mask[r, c]:
                            wake_token_losses.append(val_loss)
                        else:
                            non_wake_token_losses.append(val_loss)
                            
        # Compute stats
        avg_ppl = math.exp(np.mean(losses))
        wake_pct = (wake_counts / total_tokens) * 100.0
        wake_hz = (wake_pct / 100.0) * F_TOKEN
        
        # Calculate theoretical GFLOPS compression
        p_t1 = sum(p.numel() for p in t1_model.parameters())
        p_t3 = sum(p.numel() for p in model_t3.parameters()) + sum(p.numel() for p in film.parameters())
        
        # GFLOPS/s = 2 * params * freq
        mono_gflops = 2 * p_t1 * F_TOKEN / 1e9
        
        t1_gflops = 2 * p_t1 * wake_hz / 1e9
        t3_gflops = 2 * p_t3 * F_TOKEN / 1e9
        ssm_gflops = t1_gflops + t3_gflops
        
        gflops_comp = mono_gflops / ssm_gflops if ssm_gflops > 0 else float("inf")
        
        # Clean hooks
        for hook in hooks:
            hook.remove()
            
        results[experiment_type] = {
            "ppl": avg_ppl,
            "wake_hz": wake_hz,
            "wake_pct": wake_pct,
            "gflops_compression": gflops_comp,
            "wake_ppl_breakdown": math.exp(np.mean(wake_token_losses)) if wake_token_losses else avg_ppl,
            "non_wake_ppl_breakdown": math.exp(np.mean(non_wake_token_losses)) if non_wake_token_losses else avg_ppl
        }
        
        print(f"    PPL: {avg_ppl:.2f}  |  Wake Rate: {wake_hz:.2f} Hz ({wake_pct:.1f}%)  |  GFLOPS compression: {gflops_comp:.2f}×")
        return results[experiment_type]

    evaluate_model("Bare T3", t3_bare, nn.Module())
    evaluate_model("Oracle Dual-Rate", t3_oracle, film_oracle)
    evaluate_model("Gated Dual-Rate", t3_gated, film_gated)
    
    # Save model checkpoints
    print("\n  Saving Gated Dual-Rate model checkpoints...")
    torch.save(t3_gated.state_dict(), f"{RESULTS_DIR}/t3_gated.pt")
    torch.save(film_gated.state_dict(), f"{RESULTS_DIR}/film_gated.pt")
    
    # Save results to file
    with open(f"{RESULTS_DIR}/results.json", "w") as f:
        json.dump({
            "results": results,
            "etcd_threshold": etcd_threshold,
            "loss_bare": loss_bare,
            "loss_oracle": loss_oracle,
            "loss_gated": loss_gated
        }, f, indent=2)
        
    # 7. Generate comparative charts
    print("\n[7/7] Plotting comparison figures...")
    
    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_bare, label="Bare T3 (no conditioning)", color="#FF9800", alpha=0.8)
    ax.plot(loss_oracle, label="Oracle Dual-Rate (continuous T1)", color="#4CAF50", alpha=0.8)
    ax.plot(loss_gated, label="Gated Dual-Rate (Singular-SSM)", color="#2196F3", alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title("Training Loss Convergence Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(f"{RESULTS_DIR}/training_loss.png", dpi=150)
    plt.close()
    
    # Perplexity bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(results.keys())
    ppls = [results[m]["ppl"] for m in models]
    colors = ["#9C27B0", "#FF9800", "#4CAF50", "#2196F3"]
    
    bars = ax.bar(models, ppls, color=colors, edgecolor="black", width=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight="bold")
                    
    ax.set_ylabel("Perplexity (PPL) on WikiText-2 Validation")
    ax.set_title("Perplexity Benchmark Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/perplexity_comparison.png", dpi=150)
    plt.close()
    
    # Gated breakdown (PPL breakdown: wake vs non-wake)
    gated_stats = results["Gated Dual-Rate"]
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["Wake Tokens", "Non-Wake Tokens", "Overall PPL"]
    cat_ppls = [gated_stats["wake_ppl_breakdown"], gated_stats["non_wake_ppl_breakdown"], gated_stats["ppl"]]
    
    bars = ax.bar(categories, cat_ppls, color=["#E91E63", "#00BCD4", "#2196F3"], edgecolor="black", width=0.4)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight="bold")
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title("Gated Dual-Rate PPL Breakdown: Wake vs Non-Wake")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/ppl_breakdown.png", dpi=150)
    plt.close()
    
    print(f"\n  ✅ All comparative figures saved to {RESULTS_DIR}")
    print("=" * 68)

if __name__ == "__main__":
    main()
