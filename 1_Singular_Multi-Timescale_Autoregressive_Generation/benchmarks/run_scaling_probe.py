"""
SCALING PROBE: Hidden-State Trajectory Smoothness Across Model Scale and Architecture

Tests whether "larger models develop smoother semantic manifolds" is a genuine
property of scaling, or an artefact of the SSM architecture.

Design:
  - Two model families: Mamba (SSM) and Pythia (Transformer)
  - Both trained on The Pile with the same GPT-NeoX tokenizer → controlled comparison
  - Same Mamba / Pythia parameter sizes intentionally aligned by the Mamba paper
  - Measurement: d-dimensional cosine drift of adjacent last-layer hidden states
    (NO threshold, NO gating — raw trajectory smoothness only)
  - Multiple long text passages to reduce single-sample noise
  - Reports: per-model delta distribution (mean, median, std, p10, p90) + scale trend

What each result means:
  - If BOTH Mamba AND Pythia show delta declining with scale → genuine scaling effect
  - If only Mamba declines → SSM architectural artefact, NOT a scaling law
  - If neither declines → the original Exp1 finding was pure threshold saturation noise

Usage:
    cd benchmarks/
    python run_scaling_probe.py                        # both families
    python run_scaling_probe.py --family mamba         # Mamba only
    python run_scaling_probe.py --family pythia        # Pythia only
    python run_scaling_probe.py --tokens 2000          # longer context

Hardware: runs on MPS (Apple Silicon) — all models fit in 32GB unified memory.
Downloads: Mamba models already cached. Pythia models ~12GB total, downloaded once.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────
MAMBA_MODELS = [
    ("mamba-130m",  "state-spaces/mamba-130m-hf",   129e6),
    ("mamba-370m",  "state-spaces/mamba-370m-hf",   372e6),
    ("mamba-790m",  "state-spaces/mamba-790m-hf",   790e6),
    ("mamba-1.4b",  "state-spaces/mamba-1.4b-hf",  1372e6),
    ("mamba-2.8b",  "state-spaces/mamba-2.8b-hf",  2768e6),
]

PYTHIA_MODELS = [
    ("pythia-160m", "EleutherAI/pythia-160m",        162e6),
    ("pythia-410m", "EleutherAI/pythia-410m",        405e6),
    ("pythia-1b",   "EleutherAI/pythia-1b",         1011e6),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b",       1414e6),
    ("pythia-2.8b", "EleutherAI/pythia-2.8b",       2775e6),
]

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation texts — varied domains to reduce corpus-specific noise
# ──────────────────────────────────────────────────────────────────────────────
EVAL_TEXTS = [
    # Encyclopedic / factual
    ("encyclopedic",
     "The history of computing stretches back to the development of mechanical calculators "
     "in the seventeenth century. Charles Babbage designed the Difference Engine in 1822, "
     "a mechanical device capable of computing polynomial functions. His later Analytical "
     "Engine, conceived in 1837, incorporated many features of modern computers, including "
     "a central processing unit, memory, and conditional branching. Ada Lovelace wrote what "
     "is considered the first algorithm intended for execution on a machine. The twentieth "
     "century saw the transition from mechanical to electronic computation. The ENIAC, "
     "completed in 1945, was among the first general-purpose electronic computers. "
     "Transistors replaced vacuum tubes in the 1950s, dramatically reducing size and power "
     "consumption. The invention of the integrated circuit in 1958 by Jack Kilby and Robert "
     "Noyce enabled further miniaturisation. Gordon Moore observed in 1965 that the number "
     "of components on an integrated circuit doubled approximately every two years, a "
     "prediction that held for over five decades. The microprocessor, introduced in the "
     "early 1970s, placed the central processing unit on a single chip. The personal "
     "computer revolution of the 1980s brought computing into homes and offices worldwide. "
     "The development of the internet transformed communication and commerce. Today, machine "
     "learning systems trained on vast corpora of text and images perform tasks once thought "
     "to require human intelligence. Large language models, trained with billions of "
     "parameters on internet-scale data, exhibit emergent capabilities including in-context "
     "learning, chain-of-thought reasoning, and few-shot generalisation. The mechanisms "
     "underlying these capabilities remain an active area of research. Scaling laws describe "
     "the relationship between model size, training compute, and performance on downstream "
     "tasks. These empirical regularities suggest that performance improves predictably as "
     "models grow larger, though the reasons are not fully understood."),

    # Narrative / literary
    ("narrative",
     "She had not expected the letter to arrive on a Tuesday. Tuesdays were for small "
     "disappointments and minor repairs, not for the kind of news that rearranges the "
     "furniture of a life. She turned the envelope over twice before opening it, noting "
     "the unfamiliar return address and the slightly uneven handwriting that suggested "
     "age or haste or both. The contents were brief. Three sentences, then a signature "
     "she recognised from photographs she had never expected to hold again. Her first "
     "thought was practical: she would need to cancel Thursday. Her second thought was "
     "less orderly and arrived without warning, the way grief always does when it has "
     "been waiting patiently in an adjacent room. She sat down. The kitchen was very "
     "quiet except for the refrigerator's low hum and the distant sound of traffic. "
     "She had built this silence deliberately over many years, had learned to inhabit "
     "it without discomfort. Now it pressed against her in a way she did not recognise. "
     "She read the letter again. The words did not change. They had the stubborn "
     "permanence of facts, sitting there on the page, indifferent to what she needed "
     "them to mean. She thought about calling someone. She could not think who. "
     "The afternoon light moved slowly across the floor and she watched it without "
     "purpose, which was its own kind of comfort, though she would not have said so."),

    # Technical / mathematical reasoning
    ("technical",
     "Consider a sequence of real-valued random variables drawn independently from a "
     "distribution with finite mean and variance. The central limit theorem states that "
     "the normalised sum of these variables converges in distribution to a standard "
     "normal as the sample size grows without bound. This convergence holds regardless "
     "of the shape of the underlying distribution, provided the variance is finite. "
     "The rate of convergence is characterised by the Berry-Esseen theorem, which bounds "
     "the maximum deviation between the empirical cumulative distribution function and "
     "the normal by a quantity proportional to the third absolute moment divided by the "
     "product of the variance raised to the three-halves power and the square root of "
     "the sample size. In practice, the normal approximation is often adequate for "
     "sample sizes exceeding thirty, though this rule of thumb depends heavily on the "
     "skewness of the underlying distribution. Heavy-tailed distributions require "
     "substantially larger samples before the approximation becomes reliable. The "
     "multivariate extension applies to vectors of random variables, converging to a "
     "multivariate normal whose covariance matrix is the population covariance. "
     "Functional versions extend the result to stochastic processes, establishing "
     "convergence to Brownian motion under appropriate conditions. These results "
     "underpin much of frequentist statistical inference, including hypothesis testing "
     "and the construction of confidence intervals. Bayesian methods avoid reliance on "
     "asymptotic approximations by working directly with posterior distributions, though "
     "they require the specification of prior beliefs. The choice between frequentist "
     "and Bayesian approaches remains a subject of methodological debate."),

    # Discourse with logical connectives (high semantic velocity)
    ("high_velocity",
     "The evidence for anthropogenic climate change is now considered overwhelming by "
     "the scientific community. However, translating scientific consensus into policy "
     "action has proven considerably more difficult. While most governments have signed "
     "international agreements to reduce emissions, the pace of implementation has "
     "fallen short of targets. Furthermore, the distribution of costs and benefits "
     "across nations creates substantial coordination problems. Developed nations have "
     "historically contributed the most to cumulative emissions, yet developing nations "
     "bear a disproportionate share of near-term climate impacts. Therefore, questions "
     "of climate justice have become inseparable from technical questions about "
     "mitigation strategies. Nevertheless, technological progress in renewable energy "
     "has been faster than most projections suggested. Solar and wind costs have fallen "
     "by more than ninety percent over the past decade. Consequently, the economic case "
     "for clean energy has strengthened independently of policy incentives. Despite "
     "this progress, the carbon intensity of the global economy has decreased only "
     "modestly, because energy consumption has grown faster than clean capacity. "
     "Moreover, certain sectors such as heavy industry, aviation, and shipping remain "
     "difficult to decarbonise with current technologies. Alternatively, carbon capture "
     "and storage offers a pathway, but has not yet been deployed at scale. In contrast "
     "to energy generation, land use change remains the second largest source of "
     "emissions globally, yet receives comparatively little policy attention. "
     "Ultimately, the transition requires not merely technological substitution but "
     "fundamental changes to infrastructure, land use, and consumption patterns."),
]


# ──────────────────────────────────────────────────────────────────────────────
# Core measurement: d-dimensional cosine drift, no thresholding
# ──────────────────────────────────────────────────────────────────────────────
def measure_trajectory_smoothness(model, tokenizer, texts, device, dtype,
                                  is_mamba=True, max_tokens=512):
    """
    Returns a flat list of per-step cosine-drift values (1 - cos_sim)
    across all texts. No gating, no threshold — raw manifold velocity.

    For Mamba: uses hidden_states[-1] (last SSM layer output).
    For Transformers: uses hidden_states[-1] (last residual stream).
    Both are the d-dimensional vector before the LM head.
    """
    model.eval()
    all_deltas = []

    with torch.no_grad():
        for _, text in texts:
            ids = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_tokens
            )["input_ids"].to(device)

            n = ids.shape[1]
            if n < 4:
                continue

            # Single full forward pass with all hidden states
            out = model(ids, output_hidden_states=True, return_dict=True)
            # Shape: [1, seq_len, d_model]
            hs = out.hidden_states[-1][0].float()  # [seq_len, d_model]

            # Compute adjacent-step cosine drift: delta[t] = 1 - cos(h[t], h[t-1])
            h_curr = hs[1:]   # positions 1..n-1
            h_prev = hs[:-1]  # positions 0..n-2
            cos = F.cosine_similarity(h_curr, h_prev, dim=-1)  # [n-1]
            delta = (1.0 - cos).cpu().numpy()
            all_deltas.extend(delta.tolist())

    return np.array(all_deltas)


def summarise(deltas):
    return {
        "n":      int(len(deltas)),
        "mean":   float(np.mean(deltas)),
        "median": float(np.median(deltas)),
        "std":    float(np.std(deltas)),
        "p10":    float(np.percentile(deltas, 10)),
        "p90":    float(np.percentile(deltas, 90)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def run_family(family_name, model_list, texts, device, dtype, max_tokens,
               results_dir, offline):
    print(f"\n{'='*72}")
    print(f"  FAMILY: {family_name.upper()}")
    print(f"{'='*72}")

    family_results = []
    is_mamba = (family_name == "mamba")

    for label, hf_id, params in model_list:
        print(f"\n  [{label}] Loading {hf_id}  ({params/1e9:.2f}B params)...")
        try:
            tok = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=True,
                local_files_only=offline
            )
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                hf_id, trust_remote_code=True, torch_dtype=dtype,
                local_files_only=offline
            ).to(device)
            model.eval()

        except Exception as e:
            print(f"  SKIP {label}: {e}")
            continue

        t0 = time.perf_counter()
        deltas = measure_trajectory_smoothness(
            model, tok, texts, device, dtype,
            is_mamba=is_mamba, max_tokens=max_tokens
        )
        elapsed = time.perf_counter() - t0

        stats = summarise(deltas)
        stats["label"]   = label
        stats["hf_id"]   = hf_id
        stats["params"]  = params
        stats["elapsed"] = elapsed
        family_results.append(stats)

        print(f"  ✓ {label:15s}  mean_delta={stats['mean']:.4f}  "
              f"median={stats['median']:.4f}  std={stats['std']:.4f}  "
              f"n={stats['n']}  ({elapsed:.1f}s)")

        # Free memory before next model
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    return family_results


def print_trend_table(results, family_name):
    if not results:
        return
    print(f"\n  {family_name.upper()} SCALING TREND (sorted by params)")
    print(f"  {'Model':<15} {'Params':>8}  {'Mean Δ':>8}  {'Median Δ':>9}  "
          f"{'Std':>7}  {'p10':>7}  {'p90':>7}")
    print(f"  {'-'*15} {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}")
    sorted_r = sorted(results, key=lambda x: x["params"])
    for r in sorted_r:
        print(f"  {r['label']:<15} {r['params']/1e9:>7.2f}B  "
              f"{r['mean']:>8.4f}  {r['median']:>9.4f}  "
              f"{r['std']:>7.4f}  {r['p10']:>7.4f}  {r['p90']:>7.4f}")

    # Simple trend: is mean delta monotonically decreasing with params?
    sorted_means = [r["mean"] for r in sorted_r]
    diffs = [sorted_means[i+1] - sorted_means[i]
             for i in range(len(sorted_means)-1)]
    n_down = sum(1 for d in diffs if d < 0)
    n_up   = sum(1 for d in diffs if d > 0)
    print(f"\n  Trend: {n_down} steps DOWN, {n_up} steps UP "
          f"({'monotonically decreasing ✓' if n_up == 0 else 'non-monotonic'})")


def print_verdict(mamba_res, pythia_res):
    print(f"\n{'='*72}")
    print("  VERDICT")
    print(f"{'='*72}")

    def is_downward(results):
        if len(results) < 2:
            return None
        s = sorted(results, key=lambda x: x["params"])
        means = [r["mean"] for r in s]
        # Count fraction of consecutive pairs that decrease
        diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
        n_down = sum(1 for d in diffs if d < 0)
        return n_down / len(diffs)  # fraction of steps that decrease

    m_frac = is_downward(mamba_res)   if mamba_res   else None
    p_frac = is_downward(pythia_res)  if pythia_res  else None

    if m_frac is None and p_frac is None:
        print("  No results to compare.")
        return

    if m_frac is not None:
        print(f"  Mamba:   {m_frac:.0%} of consecutive scale steps show DECREASING mean delta")
    if p_frac is not None:
        print(f"  Pythia:  {p_frac:.0%} of consecutive scale steps show DECREASING mean delta")

    print()
    if m_frac is not None and p_frac is not None:
        if m_frac >= 0.6 and p_frac >= 0.6:
            print("  → BOTH families show decreasing trajectory drift with scale.")
            print("    This is CONSISTENT with a genuine scaling effect on manifold smoothness.")
            print("    NOT conclusive (correlation ≠ causation), but the hypothesis survives.")
        elif m_frac >= 0.6 and p_frac < 0.4:
            print("  → Only Mamba shows the downward trend.")
            print("    This is CONSISTENT with an SSM architectural artefact,")
            print("    NOT a general property of scaling. Hypothesis does not transfer.")
        elif m_frac < 0.4 and p_frac < 0.4:
            print("  → Neither family shows a consistent downward trend.")
            print("    The original Exp1 pattern may be threshold saturation noise.")
            print("    Hypothesis NOT supported.")
        else:
            print("  → Mixed / inconclusive. Inspect the per-family tables above.")
    elif m_frac is not None:
        print(f"  → Mamba-only result. Run with --family pythia to complete the comparison.")
    else:
        print(f"  → Pythia-only result. Run with --family mamba to complete the comparison.")

    print()
    print("  NOTE: This probe measures d-dimensional cosine drift with NO thresholding.")
    print("  This is the clean version of Exp1, fixing the scalar-mean and 0.99-saturation")
    print("  issues from the original benchmarks. Results here supersede Exp1 for the")
    print("  'scaling smoothness' hypothesis.")


def main():
    parser = argparse.ArgumentParser(
        description="Scaling probe: hidden-state smoothness across model scale & architecture"
    )
    parser.add_argument("--family",  choices=["mamba","pythia","both"], default="both")
    parser.add_argument("--tokens",  type=int, default=512,
                        help="Max tokens per text passage (default 512)")
    parser.add_argument("--offline", action="store_true",
                        help="Use only locally cached models (no downloads)")
    args = parser.parse_args()

    device = ("cuda"  if torch.cuda.is_available() else
              "mps"   if torch.backends.mps.is_available() else "cpu")
    dtype  = torch.float16 if device in ("cuda","mps") else torch.float32

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "experimental_results", "scaling_probe")
    os.makedirs(results_dir, exist_ok=True)

    print("="*72)
    print("  SCALING PROBE — Hidden-State Trajectory Smoothness")
    print(f"  Device: {device}  |  Precision: {dtype}")
    print(f"  Texts: {len(EVAL_TEXTS)} passages  |  Max tokens/passage: {args.tokens}")
    print(f"  Families: {args.family}")
    print("="*72)

    mamba_results  = []
    pythia_results = []

    if args.family in ("mamba", "both"):
        mamba_results = run_family(
            "mamba", MAMBA_MODELS, EVAL_TEXTS, device, dtype,
            args.tokens, results_dir, args.offline
        )
        print_trend_table(mamba_results, "mamba")

    if args.family in ("pythia", "both"):
        pythia_results = run_family(
            "pythia", PYTHIA_MODELS, EVAL_TEXTS, device, dtype,
            args.tokens, results_dir, args.offline
        )
        print_trend_table(pythia_results, "pythia")

    print_verdict(mamba_results, pythia_results)

    # Save results
    out = {
        "config": {
            "device": device,
            "dtype":  str(dtype),
            "max_tokens": args.tokens,
            "n_texts": len(EVAL_TEXTS),
            "text_names": [t[0] for t in EVAL_TEXTS],
        },
        "mamba":  mamba_results,
        "pythia": pythia_results,
    }
    out_path = os.path.join(results_dir, "scaling_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("="*72)


if __name__ == "__main__":
    main()
