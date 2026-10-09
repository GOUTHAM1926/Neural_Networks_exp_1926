"""Is the prompt-loss drop under the SFT mask real learning of the prompt, or a side-effect?
At each prompt position after 60 masked steps compare -log p of: the true next token, the OTHER
tokens that appear in the chat, and tokens that never appear."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, torch, q9_flow as Q
torch.manual_seed(1926)
m = Q.Tiny(); idx, lab, nxt, n_real, n = Q.make_row("What is the capital of France?", "Paris.")
seen = sorted(set(int(t) for t in idx[:n_real]) | {int(t) for t in lab if t != Q.IGN})
g = torch.Generator().manual_seed(3)
unseen = [t for t in torch.randint(0, 50257, (400,), generator=g).tolist() if t not in seen][:200]
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
for step in range(60):
    opt.zero_grad(); Q.sft_loss(m(idx), lab)[0].backward(); opt.step()
with torch.no_grad():
    lp = m(idx).log_softmax(-1)
res = []
for t in range(n_real):
    if lab[t] != Q.IGN: continue
    true = int(nxt[t])
    others = [s for s in seen if s != true and s not in (int(l) for l in lab if l != Q.IGN)]
    rank = int((lp[t] > lp[t, true]).sum()) + 1
    res.append({"pos": t, "true": float(-lp[t, true]), "other_seen": float(-lp[t, others].mean()),
                "answer_tokens": float(-lp[t, [int(l) for l in lab if l != Q.IGN]].mean()),
                "unseen": float(-lp[t, unseen].mean()), "rank_true": rank})
for r in res: print(r)
avg = lambda k: sum(r[k] for r in res) / len(res)
print({k: round(avg(k), 3) for k in ("true", "other_seen", "answer_tokens", "unseen")},
      "median rank of true next token:", sorted(r["rank_true"] for r in res)[len(res)//2], "of", Q.V)
json.dump({"per_pos": res, "seen_count": len(seen)}, open("q9_sideeffect.json", "w"))
