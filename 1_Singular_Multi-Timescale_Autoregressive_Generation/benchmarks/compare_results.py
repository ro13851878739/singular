import json

print('=' * 72)
print('OLD vs NEW COMPARISON (cache_params bug fix)')
print('=' * 72)

with open('experimental_results/exp1_multiscale_wake/results.json') as f:
    e1 = json.load(f)

print('\nEXP 1: Multi-Scale Wake Rate')
print(f'  {"Model":<14s} {"Old d":>8s} {"New d":>8s} {"Old Wk":>7s} {"New Wk":>7s}')
old_e1 = {129: (0.559, 1.56), 372: (0.700, 1.95), 793: (1.058, 1.95), 1372: (0.917, 1.56), 2768: (0.323, 1.17)}
for r in e1:
    pm = r['params'] // 1_000_000
    od, ow = old_e1.get(pm, (0, 0))
    print(f'  {r["model"]:<14s} {od:>8.3f} {r["delta_mean"]:>8.3f} {ow:>6.2f}Hz {r["wake_hz"]:>6.2f}Hz')

with open('experimental_results/exp3_token_consistency/results.json') as f:
    e3 = json.load(f)
print(f'\nEXP 3: Token Consistency')
print(f'  OLD: Top-1=23.6%  Top-3=29.7%  Top-5=32.1%')
print(f'  NEW: Top-1={e3["top1_match_rate"]}%  Top-3={e3["top3_overlap_avg"]}%  Top-5={e3["top5_overlap_avg"]}%')

with open('experimental_results/exp4_text_types/results.json') as f:
    e4 = json.load(f)
ow4 = {'Predictable': 1.17, 'Transitions': 1.64, 'Mixed': 1.57, 'Code': 1.56, 'Math': 1.17, 'Dialog': 1.17, 'Wikipedia': 1.17}
print(f'\nEXP 4: Multi-Text-Type')
print(f'  {"Type":<15s} {"Old Wk":>7s} {"New Wk":>7s}  Change')
for r in e4:
    ow = ow4.get(r['text_type'], 0)
    d = r['wake_hz'] - ow
    print(f'  {r["text_type"]:<15s} {ow:>6.2f}Hz {r["wake_hz"]:>6.2f}Hz  {d:+.2f}')

with open('experimental_results/exp5_long_sequence/results.json') as f:
    e5 = json.load(f)
ow5 = {128: 1.17, 256: 1.37, 384: 1.56, 512: 1.56}
print(f'\nEXP 5: Long Sequence')
print(f'  {"Len":>5s} {"Old Wk":>7s} {"New Wk":>7s}  Change')
for r in e5:
    ow = ow5.get(r['seq_len'], 0)
    d = r['wake_hz'] - ow
    print(f'  {r["seq_len"]:>5d} {ow:>6.2f}Hz {r["wake_hz"]:>6.2f}Hz  {d:+.2f}')
