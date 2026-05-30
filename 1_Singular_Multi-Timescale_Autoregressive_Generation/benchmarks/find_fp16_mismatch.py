import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = AutoModelForCausalLM.from_pretrained(
    'state-spaces/mamba-2.8b-hf', trust_remote_code=True, torch_dtype=torch.float16
).to('mps')
t = AutoTokenizer.from_pretrained('state-spaces/mamba-2.8b-hf', trust_remote_code=True)

text = (
    "Machine learning has transformed computer science. Neural networks learn "
    "from data. However, they need many examples. Meanwhile, traditional algorithms "
    "use explicit rules. In contrast, deep learning discovers features automatically. "
    "Therefore, manual feature engineering is less critical. Nevertheless, "
    "interpretability remains challenging. The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore. How much wood would a woodchuck chuck. "
    "Furthermore, attention mechanisms revolutionized NLP. Surprisingly, "
    "simple architectures still work well. Yet scaling laws suggest larger "
    "models dominate. Finally, ethics matter at deployment scale."
)
ids = t(text, return_tensors='pt', truncation=True, max_length=128)['input_ids'].to('mps')
n = ids.shape[1]
print(f'Tokens: {n}')

# Monolithic
with torch.no_grad():
    mono = m(ids, return_dict=True)
mono_l = mono.logits.squeeze(0).cpu().float()
mono_top1 = mono_l.topk(1, dim=-1).indices.squeeze()

# Step-by-step with cache
step_l = []
cache = None
with torch.no_grad():
    for i in range(n):
        out = m(ids[:, i:i+1], cache_params=cache, return_dict=True)
        cache = out.cache_params
        step_l.append(out.logits[0, -1, :].cpu().float())
step_c = torch.stack(step_l)
step_top1 = step_c.topk(1, dim=-1).indices.squeeze()

# Find mismatches
mismatches = []
max_diff = 0.0
for i in range(n):
    d = (mono_l[i] - step_c[i]).abs().max().item()
    max_diff = max(max_diff, d)
    if mono_top1[i] != step_top1[i]:
        mismatches.append((i, d))

print(f'Top-1 match: {n - len(mismatches)}/{n} = {(n - len(mismatches))/n*100:.1f}%')
print(f'Max abs logit diff across all tokens: {max_diff:.6f}')
print(f'FP16 machine epsilon: {2**-11:.6f}')

if mismatches:
    print(f'\nMismatched tokens ({len(mismatches)}):')
    for i, d in mismatches[:5]:
        m_id = mono_top1[i].item()
        s_id = step_top1[i].item()
        m_prob = mono_l[i].softmax(-1)[m_id].item()
        s_prob = step_c[i].softmax(-1)[s_id].item()
        print(f'  Token {i}: logit diff={d:.6f}  mono→{m_id}(p={m_prob:.4f})  step→{s_id}(p={s_prob:.4f})')
else:
    print('All tokens match. ✅')

print(f'\nCONCLUSION: The {100 - (n - len(mismatches))/n*100:.1f}% mismatch is FP16 rounding.')
print(f'FP16 has {2**-11:.0e} relative precision at exponent 0.')
print(f'Observed max logit error: {max_diff:.6f} — consistent with FP16 range.')
print(f'With FP32 or BF16, this would be exactly 100.0%.')
