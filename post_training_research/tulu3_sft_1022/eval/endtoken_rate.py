# End-token rate (dev suite item from the 10-08 plan; 2026-10-23): does the SFT model end its answers with the end
# token (<|endoftext|> = 50256), as it was trained to, and how long are its answers next to the real ones?
#   Prompts = the FIRST user turn of N val chats (val/inputs.bin, labels.bin, seq_lengths.bin, seq_offsets.bin; the
#   prompt is every token up to the position whose label is the first answer token, so it ends with "<|assistant|>\n"
#   exactly as in training). Greedy decoding, up to --max-new tokens. The real answer length = the trained labels from
#   there up to and including its end token.
#   Reports: share of answers that end with 50256 before the limit, mean / median generated length vs the real answer,
#   and share of answers that hit the limit (a model that never stops writes until the limit).
# usage: python endtoken_rate.py <export dir> <val dir> [--n 200] [--max-new 2048] [--batch 32]
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("export"); ap.add_argument("val")
ap.add_argument("--n", type=int, default=200); ap.add_argument("--max-new", type=int, default=2048)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--out", default=None)
a = ap.parse_args()
T, IGNORE, EOT = 4096, 65535, 50256

inp = np.fromfile(f"{a.val}/inputs.bin", np.uint16).reshape(-1, T)
lab = np.fromfile(f"{a.val}/labels.bin", np.uint16).reshape(-1, T)
lens = np.fromfile(f"{a.val}/seq_lengths.bin", np.int32)
offs = np.fromfile(f"{a.val}/seq_offsets.bin", np.int64)

items = []   # (prompt ids, real first-answer length incl. its end token)
for r in range(inp.shape[0]):
    t = 0
    for n in lens[offs[r]:offs[r + 1]]:
        n = int(n)
        L = lab[r, t:t + n]
        i0 = int(np.nonzero(L != IGNORE)[0][0])                 # label i0 = the first answer token
        ends = np.nonzero(L[i0:] == EOT)[0]
        real = int(ends[0]) + 1 if len(ends) else int((L[i0:] != IGNORE).sum())
        items.append((inp[r, t:t + i0 + 1].astype(np.int64), real))
        t += n
rng = np.random.default_rng(1023)
pick = rng.choice(len(items), size=min(a.n, len(items)), replace=False)
items = [items[i] for i in sorted(pick)]
print(f"{len(items)} prompts (first user turn of val chats), prompt length mean {np.mean([len(p) for p, _ in items]):.0f} tokens")

model = AutoModelForCausalLM.from_pretrained(a.export, dtype=torch.bfloat16).cuda().eval()
gen_len, stopped = [], []
items.sort(key=lambda x: len(x[0]))                              # similar lengths per batch: less padding
for b in range(0, len(items), a.batch):
    chunk = items[b:b + a.batch]
    m = max(len(p) for p, _ in chunk)
    ids = torch.full((len(chunk), m), EOT, dtype=torch.long)     # left padding
    att = torch.zeros((len(chunk), m), dtype=torch.long)
    for i, (p, _) in enumerate(chunk):
        ids[i, m - len(p):] = torch.from_numpy(p); att[i, m - len(p):] = 1
    with torch.no_grad():
        out = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=a.max_new, do_sample=False,
                             eos_token_id=EOT, pad_token_id=EOT)
    for row in out[:, m:].tolist():
        k = row.index(EOT) + 1 if EOT in row else len(row)
        gen_len.append(k); stopped.append(EOT in row)
    print(f"  {min(b + a.batch, len(items))}/{len(items)} done")

real = [r for _, r in items]
res = {"n": len(items), "max_new": a.max_new, "end_token_rate": float(np.mean(stopped)),
       "hit_limit_rate": float(np.mean([not s for s in stopped])),
       "gen_len_mean": float(np.mean(gen_len)), "gen_len_median": float(np.median(gen_len)),
       "real_len_mean": float(np.mean(real)), "real_len_median": float(np.median(real))}
print(json.dumps(res, indent=1))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
