"""Pad slots filled with tokens that ALSO appear in the chat: do any weight gradients change?"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, q9_flow as Q
torch.manual_seed(1926)
m = Q.Tiny()
idx, lab, nxt, n_real, n = Q.make_row("What is the capital of France?", "Paris.")
def grads(ids):
    m.zero_grad(); Q.sft_loss(m(ids), lab)[0].backward()
    return {k: p.grad.clone() for k, p in m.named_parameters()}
g0 = grads(idx)
same = [int(idx[11]), int(idx[9]), int(idx[21]), int(idx[22]), int(idx[12])]      # ' France',' capital','Paris','.','?'
idx2 = idx.clone(); idx2[n_real:] = torch.tensor(same)
print("pad slots now hold:", [Q.enc.decode([t]) for t in same])
g1 = grads(idx2)
print("largest change in ANY weight gradient:", max(float((g1[k]-g0[k]).abs().max()) for k in g0))
E = "wte.weight"
for t in same:
    print(f"  embedding row {Q.enc.decode([t])!r:11s}: grad before {float(g0[E][t].norm()):.6e}  after {float(g1[E][t].norm()):.6e}  "
          f"diff {float((g1[E][t]-g0[E][t]).abs().max()):.1e}")
with torch.no_grad():
    h0 = m(idx); h1 = m(idx2)
print("logits change at chat positions 0-22:", float((h1[:n_real]-h0[:n_real]).abs().max()),
      "| at pad positions 23-27:", float((h1[n_real:]-h0[n_real:]).abs().max()))
