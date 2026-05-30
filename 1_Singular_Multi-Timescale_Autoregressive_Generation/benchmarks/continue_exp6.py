import sys, json
sys.path.insert(0, ".")
from run_all_experiments import *

exp6 = run_exp6_mamba2_comparison()

with open(RESULTS_ROOT / "exp1_multiscale_wake" / "results.json") as f: exp1 = json.load(f)
with open(RESULTS_ROOT / "exp2_threshold_sensitivity" / "results.json") as f: exp2 = json.load(f)
with open(RESULTS_ROOT / "exp3_token_consistency" / "results.json") as f: exp3 = json.load(f) if isinstance(json.load(f), dict) else {}
with open(RESULTS_ROOT / "exp4_text_types" / "results.json") as f: exp4 = json.load(f)
with open(RESULTS_ROOT / "exp5_long_sequence" / "results.json") as f: exp5 = json.load(f)

all_results = {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "exp5": exp5, "exp6": exp6}
generate_summary(all_results)
print("DONE")
