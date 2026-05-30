"""Continuation — Run experiments 3-6."""
import sys, json
sys.path.insert(0, ".")
from run_all_experiments import *

print("=" * 64)
print("  CONTINUING EXPERIMENTS 3-6")
print("=" * 64)

with open(RESULTS_ROOT / "exp1_multiscale_wake" / "results.json") as f:
    exp1 = json.load(f)
with open(RESULTS_ROOT / "exp2_threshold_sensitivity" / "results.json") as f:
    exp2 = json.load(f)

exp3 = run_exp3_token_consistency()
exp4 = run_exp4_text_types()
exp5 = run_exp5_long_sequence()
exp6 = run_exp6_mamba2_comparison()

all_results = {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "exp5": exp5, "exp6": exp6}
generate_summary(all_results)

print("\n" + "=" * 64)
print("  ALL EXPERIMENTS COMPLETE")
print(f"  Results: {RESULTS_ROOT}")
print("=" * 64)
