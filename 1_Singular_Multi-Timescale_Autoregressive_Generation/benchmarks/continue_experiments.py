"""Continuation — Re-run experiments 2-6 with fixed threshold code."""
import sys, json
sys.path.insert(0, ".")
from run_all_experiments import *

print("=" * 64)
print("  CONTINUING EXPERIMENTS 2-6 (threshold cap: 0.5 → 0.99)")
print("=" * 64)

# Exp 1 already fine — wake counts unchanged by threshold (already > 0.5)
with open(RESULTS_ROOT / "exp1_multiscale_wake" / "results.json") as f:
    exp1_results = json.load(f)

# Re-run 2-6
exp2 = run_exp2_threshold_sensitivity()
exp3 = run_exp3_token_consistency()
exp4 = run_exp4_text_types()
exp5 = run_exp5_long_sequence()
exp6 = run_exp6_mamba2_comparison()

# Generate summary
all_results = {
    "exp1": exp1_results,
    "exp2": exp2,
    "exp3": exp3,
    "exp4": exp4,
    "exp5": exp5,
    "exp6": exp6,
}
generate_summary(all_results)

print("\n" + "=" * 64)
print("  ALL EXPERIMENTS COMPLETE")
print(f"  Results: {RESULTS_ROOT}")
print("=" * 64)
